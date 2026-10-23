"""Synthetic-only offline state banks; never imported by production inference.

Exact tail values enumerate every feasible subset. Other labels are explicitly
finite frozen-greedy continuations, not optimal action values. This module has
no dataset argument, optimizer, learned scoring head, or checkpoint mutation.
"""
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from mkp_config import GeneratorConfig, config_dict
from mkp_data import synthetic_batch
from mkp_env import MKPEnv, MKPState
from mkp_utils import atomic_checkpoint, file_sha256, load_checkpoint


SCHEMA_VERSION = 1
STATE_FIELDS = ("profits", "weights", "capacities", "remaining", "selected")
SOURCE_FILES = ("mkp_Model.py", "mkp_basis.py", "mkp_env.py", "mkp_config.py", "mkp_data.py",
                "mkp_attention.py", "mkp_resource.py", "mkp_controller.py", "mkp_performance.py", "mkp_utils.py")
STAGE_FRACTIONS = {"early": 0.2, "middle": 0.5, "late": 0.8}


def state_from_record(record, device="cpu"):
    """Return an independent B=1 state; prepared features are never serialized."""
    state = record["state"]
    return MKPState(*(state[name].detach().clone().to(device) for name in STATE_FIELDS))


def _state_payload(state):
    return {name: getattr(state, name).detach().cpu().clone() for name in STATE_FIELDS}


def _tie_tolerance(values):
    finite = values[torch.isfinite(values)]
    # Synthetic profits are integers: equal optimal sums must remain exact ties.
    if not finite.numel() or torch.equal(finite, finite.round()):
        return 0.0
    return 1e-10 * max(1.0, float(finite.abs().max()))


def _targets(values):
    known = torch.isfinite(values)
    if not known.any():
        return torch.empty(0, dtype=torch.long)
    best = values[known].max()
    return (known & ((values - best).abs() <= _tie_tolerance(values))).nonzero(as_tuple=True)[0]


def exact_action_values(state, limit=16, node_budget=200000):
    """Return (Q[N], status, evaluated_subsets), failing closed on any cutoff.

    Q_i is the maximum profit of a feasible subset containing i. Nonnegative
    weights make this equivalent to forcing i first and optimally completing.
    All subsets, including zero-profit additions, participate. A cache keyed
    only by an ambiguous remaining mask is deliberately avoided.
    """
    if state.profits.shape[0] != 1 or limit < 1 or node_budget < 1:
        raise ValueError("Exact labels require one state and positive budgets")
    n = state.profits.shape[1]
    unknown = torch.full((n,), torch.nan, dtype=torch.float64)
    ids = state.feasible[0].nonzero(as_tuple=True)[0].cpu()
    k = ids.numel()
    if k == 0:
        return unknown, "terminal", 0
    if k > limit:
        return unknown, "feasible_limit", 0
    # The actual generator and true-prefix replay use integer float64 loads.
    # Reject other capacity arithmetic rather than call rounded subset sums an
    # exact certificate for order-sensitive noninteger environment subtraction.
    weights = state.weights[0, ids.to(state.weights.device)].detach().cpu().double().numpy()
    remaining = state.remaining[0].detach().cpu().double().numpy()
    if not (np.equal(weights, np.rint(weights)).all()
            and np.equal(remaining, np.rint(remaining)).all()
            and np.abs(weights).max(initial=0) * max(k, 1) < 2**53
            and np.abs(remaining).max(initial=0) < 2**53):
        return unknown, "noninteger_capacity_arithmetic", 0
    profits = state.profits[0, ids.to(state.profits.device)].detach().cpu().double().numpy()
    if not (np.equal(profits, np.rint(profits)).all()
            and np.abs(profits).max(initial=0) * max(k, 1) < 2**53):
        return unknown, "noninteger_or_unsafe_profit_arithmetic", 0
    size = 1 << k
    # Allocate at most the allowed number of subsets, even with a large limit.
    examined = min(size, node_budget)
    totals = np.zeros((examined, weights.shape[1]), dtype=np.float64)
    gains = np.zeros(examined, dtype=np.float64)
    best = np.full(k, -np.inf, dtype=np.float64)
    for subset in range(1, examined):
        bit = subset & -subset
        item = bit.bit_length() - 1
        previous = subset ^ bit
        totals[subset] = totals[previous] + weights[item]
        gains[subset] = gains[previous] + profits[item]
        if np.all(totals[subset] <= remaining):
            bits = subset
            while bits:
                member = bits & -bits
                index = member.bit_length() - 1
                best[index] = max(best[index], gains[subset])
                bits ^= member
    if examined < size:
        return unknown, "node_budget_exceeded", examined
    values = unknown.clone()
    values[ids] = torch.from_numpy(best.copy())
    if not torch.isfinite(values[ids]).all():
        raise RuntimeError("Complete subset enumeration omitted a feasible singleton")
    return values, "complete", examined


@torch.no_grad()
def _trace(model, data):
    env = MKPEnv(*data)
    states, actions = [], []
    for _ in range(data[0].shape[1]):
        if not env.state.active.any():
            break
        states.append(_state_payload(env.state))
        action = model.policy(env.state)[0].choose()
        actions.append(int(action.item()))
        env.step(action)
    if env.state.active.any():
        raise RuntimeError("Greedy source trajectory exceeded N steps")
    return states, actions, float(env.objective.item())


@torch.no_grad()
def _forced_future(model, state, action):
    prefix = float((state.profits * state.selected).sum())
    env = MKPEnv.from_state(state)
    env.step(torch.tensor([action], dtype=torch.long, device=state.profits.device))
    for _ in range(state.profits.shape[1]):
        if not env.state.active.any():
            return float(env.objective.item()) - prefix
        env.step(model.policy(env.state)[0].choose())
    raise RuntimeError("Forced greedy continuation exceeded N steps")


@torch.no_grad()
def _finite_values(model, state, count, baseline):
    policy = model.policy(state)[0]
    ordered = policy.items[policy.logits.argsort(descending=True)].tolist()
    feasible = state.feasible[0].nonzero(as_tuple=True)[0]
    p = state.profits[0, feasible]
    loads = state.weights[0, feasible] / state.remaining[0].clamp_min(1e-12)
    proposals = [baseline, *ordered[:2], int(feasible[p.argmax()]),
        int(feasible[(p / loads.mean(-1).clamp_min(1e-12)).argmax()]),
        int(feasible[(p / loads.amax(-1).clamp_min(1e-12)).argmax()]),
        int(feasible[loads.amax(-1).argmin()])]
    for resource in loads.sum(0).argsort(descending=True):
        proposals.append(int(feasible[(p / loads[:, resource].clamp_min(1e-12)).argmax()]))
    proposals.extend(ordered)
    selected = list(dict.fromkeys(proposals))[:count]
    values = torch.full((state.profits.shape[1],), torch.nan, dtype=torch.float64)
    for action in selected:
        values[action] = _forced_future(model, state, action)
    return values


def _chosen_steps(length, tail_index):
    """Deduplicate source prefixes; tail is the canonical name when aliased."""
    aliases = {}
    for stage, fraction in STAGE_FRACTIONS.items():
        step = min(length - 1, int(fraction * length))
        aliases.setdefault(step, []).append(stage)
    if tail_index is not None:
        aliases.setdefault(tail_index, []).append("tail")
    return [("tail" if "tail" in names else names[0], step, names)
            for step, names in aliases.items()]


def _validate_bank(bank):
    if bank.get("schema_version") != SCHEMA_VERSION or bank.get("data_source") != "synthetic_only":
        raise ValueError("Expected a version-1 synthetic-only diagnostic bank")
    records = bank.get("records", [])
    ids, split_by_instance = set(), {}
    for record in records:
        if record["id"] in ids:
            raise ValueError("Duplicate diagnostic record id")
        ids.add(record["id"])
        split = record["split"]
        if split not in ("fit", "heldout") or record["stage"] not in (*STAGE_FRACTIONS, "tail"):
            raise ValueError("Invalid bank split/stage")
        instance = record["instance_id"]
        if instance in split_by_instance and split_by_instance[instance] != split:
            raise ValueError("Fit and heldout states share a generated instance")
        split_by_instance[instance] = split
        state = record["state"]
        n, m = record["N"], record["M"]
        shapes = ((1, n), (1, n, m), (1, m), (1, m), (1, n))
        for name, shape in zip(STATE_FIELDS, shapes):
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError("Malformed serialized B=1 state")
            if name != "selected" and (value.dtype != torch.float64 or not torch.isfinite(value).all()):
                raise ValueError("Physical bank tensors must be finite float64")
            if name == "selected" and value.dtype != torch.bool:
                raise ValueError("Selected mask must be boolean")
        values, targets = record["future_values"], record["target_actions"]
        if values.dtype != torch.float64 or values.shape != (n,) or torch.isinf(values).any():
            raise ValueError("Malformed float64 action values")
        if targets.dtype != torch.long or targets.ndim != 1 or not targets.numel():
            raise ValueError("A labeled record needs a nonempty target action set")
        feasible = state_from_record(record).feasible[0]
        known = torch.isfinite(values)
        if (known & ~feasible).any() or not torch.equal(targets, _targets(values)):
            raise ValueError("Labels/targets disagree with feasible known actions")
        if record["label_kind"] not in ("exact", "finite_greedy"):
            raise ValueError("Unknown action-label kind")
        if record["label_kind"] == "exact" and (record["exact_status"] != "complete" or not torch.equal(known, feasible)):
            raise ValueError("Partial enumeration cannot be labeled exact")
        baseline = record["baseline_action"]
        if (not isinstance(baseline, int) or baseline < 0 or baseline >= n
                or not known[baseline] or not math.isfinite(record["baseline_greedy_future"])):
            raise ValueError("Baseline action/value must be known")
        if record.get("known") is not None and not torch.equal(record["known"], known):
            raise ValueError("Known-action mask disagrees with finite values")
        if not math.isclose(float(values[known].max()), record["reference_future_value"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Reference must equal the best known future value")


def save_bank(path, bank):
    """Save a weights_only-safe tensor/primitive artifact after contract checks."""
    _validate_bank(bank)
    path = Path(path)
    if path.suffix != ".pt":
        path = path / "bank.pt"
    atomic_checkpoint(path, bank)
    return path


def load_bank(path, *, verify_sources=True):
    path = Path(path)
    if path.is_dir():
        path = path / "bank.pt"
    bank = torch.load(path, map_location="cpu", weights_only=True)
    _validate_bank(bank)
    if verify_sources:
        root = Path(__file__).resolve().parent
        for filename, expected in bank.get("source_hashes", {}).items():
            if filename not in SOURCE_FILES or file_sha256(root / filename) != expected:
                raise ValueError(f"Source changed since bank creation: {filename}")
    return bank


@torch.no_grad()
def build_bank(checkpoint_dir, output, resources=(5, 10), items=(100,),
               instances_per_group=2, seed=2729, exact_limit=16,
               exact_node_budget=200000, candidates=8, device="cpu"):
    """Create fit/heldout instances independently; output/bank.pt is diagnostic.

    Only synthetic_batch is available as a data source. An actual source-model
    greedy trajectory supplies every prefix and the independent baseline suffix.
    A bounded exact failure falls back to finite labels with the failure status
    retained; it never exposes a partially enumerated exact target.
    """
    resources, items = tuple(resources), tuple(items)
    if (not resources or not items or min(*resources, *items) < 1 or instances_per_group < 1
            or exact_limit < 1 or exact_limit > 20 or exact_node_budget < 1 or candidates < 1 or seed < 0):
        raise ValueError("Invalid bank shape, seed, or bounded labeling budget")
    checkpoint_dir, output = Path(checkpoint_dir).resolve(), Path(output).resolve()
    if output == checkpoint_dir or checkpoint_dir in output.parents:
        raise ValueError("Diagnostic output must be separate from source checkpoints")
    root = Path(__file__).resolve().parent
    bank = dict(schema_version=SCHEMA_VERSION, data_source="synthetic_only", seed=int(seed),
        checkpoints={}, generation_config={}, source_hashes={name: file_sha256(root / name) for name in SOURCE_FILES},
        protocol=dict(items=list(items), resources=list(resources), instances_per_group=instances_per_group,
            splits=["fit", "heldout"], stage_fractions=STAGE_FRACTIONS.copy(), exact_limit=exact_limit,
            exact_node_budget=exact_node_budget, finite_candidates=candidates,
            tail_rule="first actual greedy prefix with 2..exact_limit feasible items",
            value_units="raw future profit excluding the selected prefix"), records=[])
    tail_coverage, instance_count = [], 0
    # Loading initializes an nn.Module before restoring weights. Preserve the
    # caller's Torch RNG instead of perturbing any later diagnostic fit seeds.
    fork_devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        for m in resources:
            path = checkpoint_dir / f"mkp_M{m}_best.pt"
            before_sha = file_sha256(path)
            model, payload = load_checkpoint(path, device, m)
            model.eval()
            generator = GeneratorConfig(**payload["generator_config"])
            bank["checkpoints"][str(m)] = dict(path=str(path), sha256=before_sha,
                model_config=payload["model_config"], train_config=payload["train_config"],
                update=int(payload["update"]), checkpoint_version=int(payload["checkpoint_version"]),
                controller_version=payload["controller_version"], decoder_version=payload["decoder_version"])
            bank["generation_config"][str(m)] = config_dict(generator)
            for n in items:
                for group, ratio in enumerate(generator.capacity_ratios):
                    for split_index, split in enumerate(("fit", "heldout")):
                        for instance_index in range(instances_per_group):
                            entropy = [int(seed), int(m), int(n), group, split_index, instance_index]
                            rng = np.random.default_rng(np.random.SeedSequence(entropy))
                            data = synthetic_batch(1, n, m, replace(generator, capacity_ratios=(ratio,)), rng, device)
                            instance_id = f"M{m}_N{n}_r{ratio:g}_{split}_{instance_index}_s{seed}"
                            states, actions, final = _trace(model, data)
                            instance_count += 1
                            if not states:
                                tail_coverage.append(dict(instance_id=instance_id, status="no_feasible_source_state"))
                                continue
                            tail_index = next((i for i, raw in enumerate(states) if
                                2 <= int(MKPState(*(raw[key] for key in STATE_FIELDS)).feasible.sum()) <= exact_limit), None)
                            tail_coverage.append(dict(instance_id=instance_id,
                                status="covered" if tail_index is not None else "no_actual_multicandidate_tail",
                                tail_step=tail_index, source_steps=len(states)))
                            chosen = _chosen_steps(len(states), tail_index)
                            for stage, step, aliases in chosen:
                                raw = states[step]
                                state = MKPState(*(raw[key].to(device) for key in STATE_FIELDS))
                                values, exact_status, nodes = exact_action_values(state, exact_limit, exact_node_budget)
                                label_kind = "exact" if exact_status == "complete" else "finite_greedy"
                                if label_kind != "exact":
                                    values = _finite_values(model, state, candidates, actions[step])
                                targets = _targets(values)
                                prefix = float((state.profits * state.selected).sum())
                                record = dict(id=f"{instance_id}/{stage}/{step}", instance_id=instance_id,
                                    split=split, stage=stage, M=int(m), N=int(n), capacity_ratio=float(ratio),
                                    stage_aliases=aliases,
                                    generation_entropy=entropy, source_step=step, source_steps=len(states),
                                    state=raw, future_values=values, known=torch.isfinite(values),
                                    target_actions=targets, target_mask=torch.isin(torch.arange(n), targets),
                                    label_kind=label_kind, baseline_action=actions[step], baseline_greedy_future=final-prefix,
                                    reference_future_value=float(values[torch.isfinite(values)].max()), prefix_profit=prefix,
                                    exact_status=exact_status, exact_nodes=nodes, feasible_count=int(state.feasible.sum()))
                                bank["records"].append(record)
                            print(f"bank {instance_id}: source_steps={len(states)}, states={len(chosen)}, tail={tail_index}", flush=True)
            if file_sha256(path) != before_sha:
                raise RuntimeError("Source checkpoint changed during bank creation")
    bank["coverage"] = dict(instances=instance_count, records=len(bank["records"]), tails=tail_coverage,
        labels=dict(Counter(record["label_kind"] for record in bank["records"])),
        exact_statuses=dict(Counter(record["exact_status"] for record in bank["records"])))
    output.mkdir(parents=True, exist_ok=True)
    bank_path = save_bank(output / "bank.pt", bank)
    summary = {key: value for key, value in bank.items() if key != "records"}
    summary.update(bank_path=str(bank_path), bank_sha256=file_sha256(bank_path))
    (output / "bank_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bank
