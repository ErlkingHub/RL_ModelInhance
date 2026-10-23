"""Offline theta reachability: executable witnesses, analytic exclusions, unknowns.

Optimizes state-level legal controls only. No controller/decoder weight is updated.
A failed bounded search is always unknown, never an impossibility certificate.
"""
from dataclasses import fields
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from mkp_Model import StrategyTheta
from mkp_performance import prepare_state
from mkp_audit_common import diagnostic_provenance, source_model, theta_payload, write_json
from mkp_utils import file_sha256


def _inverse_sigmoid(value):
    value = value.detach().float().clamp(1e-6, 1 - 1e-6)
    return value.log() - torch.log1p(-value)


class LegalTheta(nn.Module):
    """Same bounds, symmetric upper-triangle mass and companion simplex as Actor."""
    def __init__(self, config, anchor):
        super().__init__()
        self.config = config
        m = anchor.resource_prices.shape[-1]
        device = anchor.resource_prices.device
        self.register_buffer("upper", torch.triu_indices(m, m, device=device))
        self.register_buffer("temperature", anchor.temperature.detach().clone())
        c = config
        names = ("profit_exponent", "scarcity_exponent", "bottleneck_weight", "opportunity_weight", "reserve_weight")
        self.bounds = ((c.profit_exponent_min, c.profit_exponent_max), (0, c.scarcity_exponent_max),
                       (0, c.bottleneck_weight_max), (0, c.opportunity_weight_max), (0, c.reserve_weight_max))
        initial = [((getattr(anchor, name) - lo) / (hi - lo) if hi > lo else
                    torch.full_like(anchor.temperature, .5)) for name, (lo, hi) in zip(names, self.bounds)]
        self.scalars = nn.Parameter(_inverse_sigmoid(torch.stack(initial, -1)))
        price = ((anchor.resource_prices - c.price_floor) / ((1 - c.price_floor) * m)).clamp_min(1e-8)
        self.prices = nn.Parameter(price.log())
        pair = anchor.resource_interactions[:, self.upper[0], self.upper[1]].detach()
        self.pair_mix = nn.Parameter((pair / pair.sum(-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8).log())
        self.pair_gate = nn.Parameter(_inverse_sigmoid(pair.sum(-1, keepdim=True) / max(c.interaction_weight_max, 1e-8)))
        companion = anchor.companion_weights.detach()
        self.companion_mix = nn.Parameter((companion / companion.sum(-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8).log())
        self.companion_gate = nn.Parameter(_inverse_sigmoid(companion.sum(-1, keepdim=True) / max(c.companion_weight_max, 1e-8)))

    def forward(self):
        c = self.config
        b, m = self.prices.shape
        scalars = [lo + (hi - lo) * self.scalars[:, i].sigmoid() for i, (lo, hi) in enumerate(self.bounds)]
        prices = c.price_floor + (1 - c.price_floor) * m * self.prices.softmax(-1)
        pair = self.pair_mix.softmax(-1) * c.interaction_weight_max * self.pair_gate.sigmoid()
        interactions = pair.new_zeros(b, m, m)
        interactions[:, self.upper[0], self.upper[1]] = pair
        interactions = interactions + interactions.transpose(1, 2) - torch.diag_embed(interactions.diagonal(dim1=1, dim2=2))
        companion = self.companion_mix.softmax(-1) * c.companion_weight_max * self.companion_gate.sigmoid()
        return StrategyTheta(prices, scalars[0], scalars[1], scalars[2], self.temperature,
                             scalars[3], scalars[4], interactions, companion)


def target_margin(policy, targets):
    valid = torch.isin(policy.items, targets.to(policy.items.device))
    if not valid.any():
        raise ValueError("No target is feasible in the diagnostic state")
    if valid.all():
        return policy.score_terms["score"].sum() * 0, True
    scores = policy.score_terms["score"]
    return scores[valid].amax() - scores[~valid].amax(), False


@torch.no_grad()
def analytic_exclusion(state, config, targets):
    """Conservative all-legal-theta bound against one competitor per target.

    Positive weighted resource sums obey cost_i/cost_j >= min_r(q_ir/q_jr).
    Each remaining bounded scalar or nonnegative total-mass term is maximized
    independently, relaxing parameter coupling. Negative relaxed bounds exclude
    a target; nonnegative bounds do not prove reachability. Scope is the real
    fixed-score formula on its computed inputs, with a float32 safety allowance.
    """
    from mkp_Model import WeakDecoder
    from mkp_basis import interaction_loads, companion_basis
    decoder = WeakDecoder(config)
    ids = state.feasible[0].nonzero(as_tuple=True)[0]
    owners = torch.zeros_like(ids)
    target_list = [int(x) for x in targets.tolist()]
    others = [int(x) for x in ids.tolist() if int(x) not in target_list]
    if not others:
        return None
    if any(i not in ids.tolist() for i in target_list):
        raise ValueError("Infeasible target in theta audit")
    p, w, cap, rem = (x.float() for x in (state.profits, state.weights, state.capacities, state.remaining))
    q = state.profits.shape[1] * w[0] / cap[0]
    logq = torch.where(w[0] > 0, q.clamp_min(config.eps).log(), -torch.inf).double()
    profit_log = (p[0] / p[0].mean().clamp_min(config.eps)).clamp_min(config.eps).log().double()
    bottleneck = (w[0] / rem[0].clamp_min(config.eps)).amax(-1).double()
    blocked = reserve = torch.zeros_like(profit_log)
    if config.opportunity_weight_max or config.reserve_weight_max:
        hard, soft = decoder.opportunity_features(state, owners, ids)
        blocked, reserve = blocked.clone(), reserve.clone()
        blocked[ids], reserve[ids] = hard.double(), soft.double()
    z = torch.zeros((*state.profits.shape, state.capacities.shape[-1]), device=p.device)
    if config.interaction_weight_max:
        z[owners, ids] = interaction_loads(state, owners, ids, config.eps)
    upper = torch.triu_indices(cap.shape[-1], cap.shape[-1], device=p.device)
    quadratic = (z[0][:, upper[0]] * z[0][:, upper[1]]).double()
    companion = torch.zeros((p.shape[-1], len(config.companion_scales)), device=p.device, dtype=torch.float64)
    if config.companion_weight_max:
        companion[ids] = companion_basis(state, owners, ids, config).double()
    exclusions = []
    for i in target_list:
        best = None
        for j in others:
            positive_j = w[0, j] > 0
            if not positive_j.any():
                if (w[0, i] > 0).any():
                    continue  # zero-cost normalization needs a different bound
                cost_bound = 0.
            else:
                ratio_log = (logq[i, positive_j] - logq[j, positive_j]).amin().item()
                if not math.isfinite(ratio_log):
                    continue
                cost_bound = -ratio_log
            delta = float(profit_log[i] - profit_log[j])
            parts = dict(profit=max(config.profit_exponent_min * delta, config.profit_exponent_max * delta),
                cost=cost_bound,
                bottleneck=config.bottleneck_weight_max * max(0., float(bottleneck[j] - bottleneck[i])),
                blocked=config.opportunity_weight_max * max(0., float(blocked[j] - blocked[i])),
                reserve=config.reserve_weight_max * max(0., float(reserve[j] - reserve[i])),
                interaction=config.interaction_weight_max * max(0., float((quadratic[j] - quadratic[i]).amax())),
                companion=config.companion_weight_max * max(0., float((companion[i] - companion[j]).amax())))
            bound = sum(parts.values())
            guard = 1e-4 * (1 + sum(abs(x) for x in parts.values()))
            if best is None or bound + guard < best["guarded_upper_bound"]:
                best = dict(target=i, competitor=j, upper_bound=bound, rounding_allowance=guard,
                            guarded_upper_bound=bound + guard, terms=parts)
        if best is None or best["guarded_upper_bound"] >= 0:
            return None
        exclusions.append(best)
    return dict(type="analytic_score_upper_bound", scope="real fixed-score family; float32 safety allowance",
                all_targets_excluded=True, bounds=exclusions,
                limitation="Local state/action exclusion, not an instance-level optimality or benchmark gap bound")


def reach_state(model, state, targets, *, restarts=4, steps=100, learning_rate=.05, seed=2729):
    if restarts < 1 or steps < 0 or learning_rate <= 0:
        raise ValueError("Invalid theta-search budget")
    c = model.config
    with torch.no_grad():
        anchor = model.actor(state)[0].detach()
        policy = model.decoder(state, anchor)
        margin, trivial = target_margin(policy, targets)
        chosen = int(policy.choose().item())
        if chosen in targets.tolist():
            return dict(status="reachable", witness_source="current_controller", action=chosen,
                        margin=float(margin), all_feasible_actions_are_targets=trivial,
                        theta=theta_payload(anchor), optimizer_steps=0)
        certificate = analytic_exclusion(state, c, targets)
        if certificate is not None:
            return dict(status="analytic_unreachable", certificate=certificate,
                        optimizer_steps=0, action=chosen, margin=float(margin))
    prepare_state(state, c)
    best_margin = float(margin)
    used = 0
    generator = torch.Generator(device=state.profits.device).manual_seed(seed)
    for restart in range(restarts):
        controls = LegalTheta(c, anchor)
        if restart:
            with torch.no_grad():
                for p in controls.parameters():
                    p.add_(torch.randn(p.shape, generator=generator, device=p.device) * 1.5)
        optimizer = torch.optim.Adam(controls.parameters(), lr=learning_rate)
        for step in range(steps + 1):
            theta = controls()
            policy = model.decoder(state, theta)
            margin, _ = target_margin(policy, targets)
            best_margin = max(best_margin, float(margin.detach()))
            action = int(policy.choose().item())
            if action in targets.tolist():
                return dict(status="reachable", witness_source="legal_theta_search", action=action,
                            margin=float(margin.detach()), theta=theta_payload(theta),
                            optimizer_steps=used, restart=restart,
                            tie_witness=bool(float(margin.detach()) <= 0))
            if step == steps:
                break
            loss = F.softplus((1e-3 - margin) / .2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite theta-search objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            used += 1
    return dict(status="unknown", optimizer_steps=used, best_margin=best_margin,
                reason="Bounded nonconvex search found no executable witness; not an impossibility result")


def run_reachability(bank_path, output, *, device="cpu", restarts=4, steps=100, seed=2729):
    from mkp_audit_bank import load_bank, state_from_record
    bank = load_bank(bank_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    models = {m: source_model(bank, m, device)[0] for m in sorted({r["M"] for r in bank["records"]})}
    for model in models.values():
        model.requires_grad_(False)
    results = []
    witnesses = {}
    for index, record in enumerate(bank["records"]):
        result = reach_state(models[record["M"]], state_from_record(record, device), record["target_actions"],
                             restarts=restarts, steps=steps, seed=seed + index)
        if "theta" in result:
            witnesses[record["id"]] = result.pop("theta")
        result.update({k: record[k] for k in ("id", "split", "M", "N", "capacity_ratio", "stage", "label_kind")})
        results.append(result)
        print(f"reach {index + 1}/{len(bank['records'])}: {record['id']} {result['status']}", flush=True)
    counts = {name: sum(r["status"] == name for r in results)
              for name in ("reachable", "analytic_unreachable", "unknown")}
    report = dict(schema_version=1, bank_sha256=file_sha256(bank_path), data_source="synthetic_only",
        phase="theta_reachability", budget=dict(restarts=restarts, steps=steps, seed=seed),
        elapsed_seconds=time.perf_counter()-started, counts=counts, records=results,
        witness_sources={name: sum(r.get("witness_source") == name for r in results)
                         for name in ("current_controller", "legal_theta_search")},
        source_hashes=diagnostic_provenance(),
        interpretation="Search success is an executable witness; analytic exclusions have explicit local scope; failures remain unknown")
    write_json(output / "reachability.json", report)
    torch.save(dict(schema_version=1, artifact_type="diagnostic_theta_witnesses", bank_sha256=report["bank_sha256"],
                    witnesses=witnesses), output / "witnesses.pt")
    return report
