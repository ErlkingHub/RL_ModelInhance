"""Training-only theta probes -> deduplicated actions -> frozen-policy preferences.

No optimal solutions, theta regression, decoder parameters, or inference search.
All labels are additional raw profit from a copied real state to termination.
"""
from copy import deepcopy
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from mkp_env import MKPEnv, MKPState
from mkp_config import ModelConfig, config_dict
from mkp_Model import CandidatePolicy, StrategyTheta, WeakDecoder


THETA_FIELDS = ("resource_prices", "profit_exponent", "scarcity_exponent", "bottleneck_weight", "temperature")
RANKING_METRICS = ("ranking_states", "ranking_theta_trials", "ranking_unique_actions", "ranking_branches",
                   "ranking_pairs", "ranking_improvement_rate", "ranking_gain_raw", "ranking_seconds",
                   "ranking_loss", "ranking_pair_accuracy", "ranking_candidate_regret",
                   "ranking_greedy_coverage", "ranking_greedy_regret", "ranking_kl", "ranking_kl_stopped")


def empty_ranking_metrics():
    return {name: (float("nan") if name in ("ranking_pair_accuracy", "ranking_candidate_regret",
                                           "ranking_greedy_coverage", "ranking_greedy_regret") else 0.)
            for name in RANKING_METRICS}


@dataclass
class RankingBatch:
    state: MKPState
    actions: torch.Tensor                 # [S,A], baseline first, unique; -1 is padding
    witness_indices: torch.Tensor         # [S,A], first theta trial that induced each action
    proposal_theta: StrategyTheta         # flattened [S*T,...], never a regression target
    proposal_actions: torch.Tensor        # [S,T]
    returns: torch.Tensor                 # [S,A,R], additional raw profit, float64; NaN padding
    sequences: torch.Tensor               # [S,A,R,N], includes forced first action; -1 padding
    pair_states: torch.Tensor
    winners: torch.Tensor                # action slot in actions, not item identity
    losers: torch.Tensor
    gaps: torch.Tensor                   # normalized positive paired mean gaps
    standard_errors: torch.Tensor
    old_log_probs: torch.Tensor           # complete feasible distribution, ragged
    source_times: torch.Tensor
    source_batches: torch.Tensor
    teacher_sha256: str
    model_config: dict
    seed: int
    metrics: dict


@torch.no_grad()
def theta_proposals(current, model_config, count, std, generator):
    """Current theta + alternating local logit perturbations / global legal probes.

    Resource coordinates are exchangeable; no N-dependent or item-position heads.
    Temperature is held fixed because it cannot change a greedy candidate action.
    """
    c = model_config
    b, m = current.resource_prices.shape
    values = {name: getattr(current, name)[:, None].expand(-1, count, *getattr(current, name).shape[1:]).clone()
              for name in THETA_FIELDS}
    simplex = ((current.resource_prices - c.price_floor) / ((1 - c.price_floor) * m)).clamp_min(1e-7)
    price_center = simplex.log() - simplex.log().mean(-1, keepdim=True)
    bounds = (("profit_exponent", c.profit_exponent_min, c.profit_exponent_max),
              ("scarcity_exponent", 0., c.scarcity_exponent_max),
              ("bottleneck_weight", 0., c.bottleneck_weight_max))
    for trial in range(1, count):
        global_probe = trial % 2 == 0
        noise = torch.randn(b, m, device=price_center.device, generator=generator)
        logits = 2 * noise if global_probe else price_center + std * noise
        values["resource_prices"][:, trial] = c.price_floor + (1 - c.price_floor) * m * logits.softmax(-1)
        for name, lo, hi in bounds:
            if hi == lo:
                values[name][:, trial] = lo
                continue
            center = ((getattr(current, name) - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
            if global_probe:
                fraction = torch.rand(b, device=center.device, generator=generator)
            else:
                noise = torch.randn(b, device=center.device, generator=generator)
                fraction = (torch.logit(center) + std * noise).sigmoid()
            values[name][:, trial] = lo + (hi - lo) * fraction
    return StrategyTheta(*(values[name].flatten(0, 1) for name in THETA_FIELDS))


def deduplicate_actions(proposal_actions, max_actions):
    """Keep baseline and first witnesses; bound continuation work before any rollout."""
    result = proposal_actions.new_full((proposal_actions.shape[0], max_actions), -1)
    witnesses = torch.full_like(result, -1)
    for row, trials in enumerate(proposal_actions.tolist()):
        seen = set()
        for trial, action in enumerate(trials):
            if action < 0 or action in seen:
                continue
            slot = len(seen)
            result[row, slot], witnesses[row, slot] = action, trial
            seen.add(action)
            if len(seen) == max_actions:
                break
    return result, witnesses


@torch.no_grad()
def paired_terminal_returns(teacher, state, actions, config, seed):
    """Branch on each unique first action, then recompute theta at every true successor.

    Sampled mode uses common (source state, repeat, step, item)-indexed Gumbel
    noise. Resetting the private generator for each branch chunk preserves pairing
    and makes the result independent of the rollout chunk size, without N^2 storage.
    """
    s, a = actions.shape
    n = state.profits.shape[1]
    repeats = config.ranking_repeats if config.ranking_continuation == "sampled" else 1
    device = state.profits.device
    q = state.profits.new_full((s, a, repeats), float("nan"), dtype=torch.float64)
    sequences = actions.new_full((s, a, repeats, n), -1)
    rows, slots = (actions >= 0).nonzero(as_tuple=True)
    if actions.shape[0] != state.profits.shape[0]:
        raise ValueError("Each action row must have a matching source state")
    if rows.numel() and not state.feasible[rows, actions[rows, slots]].all():
        raise ValueError("Counterfactual first actions must be feasible")
    branches = torch.stack((rows.repeat_interleave(repeats), slots.repeat_interleave(repeats),
                            torch.arange(repeats, device=device).repeat(rows.numel())), -1)
    for chunk in branches.split(config.ranking_rollout_batch_size):
        row, slot, repeat = chunk.unbind(-1)
        env = MKPEnv.from_state(state.index(row))
        past_profit = env.objective.clone()
        first = actions[row, slot]
        env.step(first)
        sequences[row, slot, repeat, 0] = first
        generator = torch.Generator(device=device).manual_seed(seed)
        for depth in range(1, n):
            if not env.state.active.any():
                break
            policy = teacher.policy(env.state)[0]
            noise = None
            if config.ranking_continuation == "sampled":
                uniform = torch.rand(s, repeats, n, device=device, generator=generator).clamp(1e-7, 1 - 1e-7)
                common_gumbel = -(-uniform.log()).log()
                noise = common_gumbel[row[policy.owners], repeat[policy.owners], policy.items]
            chosen = policy.choose(stochastic=noise is not None, gumbel=noise)
            sequences[row, slot, repeat, depth] = chosen
            env.step(chosen)
        if env.state.active.any():
            raise RuntimeError("Counterfactual continuation exceeded N actions")
        # Accumulate objective in float64; do not sum float32 normalized step rewards.
        q[row, slot, repeat] = env.objective - past_profit
    return q, sequences


def preference_pairs(actions, returns, total_profit, config):
    """All reliable unordered pairs, with SE computed on paired return differences."""
    left, right = torch.triu_indices(actions.shape[1], actions.shape[1], 1, device=actions.device)
    delta = (returns[:, left] - returns[:, right]) / total_profit[:, None, None].clamp_min(1e-12)
    mean = delta.mean(-1)
    se = delta.std(-1, unbiased=True) / delta.shape[-1] ** .5 if delta.shape[-1] > 1 else torch.zeros_like(mean)
    threshold = torch.maximum(torch.full_like(mean, config.ranking_min_gap), config.ranking_se_multiplier * se)
    keep = ((actions[:, left] >= 0) & (actions[:, right] >= 0) & torch.isfinite(mean)
            & (mean.abs() > threshold))
    states, pairs = keep.nonzero(as_tuple=True)
    positive = mean[states, pairs] > 0
    return (states, torch.where(positive, left[pairs], right[pairs]),
            torch.where(positive, right[pairs], left[pairs]), mean[states, pairs].abs().float(),
            se[states, pairs].float())


def actor_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.actor.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def collect_ranking_batch(model, data, rollout, config, seed):
    """Call before PPO changes any weight. Only actual visited synthetic states enter."""
    if config.ranking_coefficient == 0:
        return None
    started = time.perf_counter()
    device = data[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    times, batches = rollout["active"].nonzero(as_tuple=True)
    # Sample visited states, not hypothetical states or padded/terminal transitions.
    count = min(config.ranking_states, times.numel())
    if count == 0:
        return None
    ids = torch.randperm(times.numel(), device=device, generator=generator)[:count]
    times, batches = times[ids], batches[ids]
    state = MKPState(data[0][batches].detach().clone(), data[1][batches].detach().clone(),
                     data[2][batches].detach().clone(), rollout["remaining"][times, batches].detach().clone(),
                     rollout["selected"][times, batches].detach().clone())
    teacher = deepcopy(model).eval().requires_grad_(False)
    teacher_hash = actor_hash(teacher)
    policy, current, _ = teacher.policy(state)
    proposals = theta_proposals(current, teacher.config, config.ranking_theta_candidates,
                                 config.ranking_theta_std, generator)
    proposal_state = state.index(torch.arange(count, device=device).repeat_interleave(config.ranking_theta_candidates))
    proposal_actions = teacher.decoder(proposal_state, proposals).choose().view(count, -1)
    actions, witnesses = deduplicate_actions(proposal_actions, config.ranking_max_actions)
    returns, sequences = paired_terminal_returns(teacher, state, actions, config, seed + 1)
    pairs = preference_pairs(actions, returns, state.profits.sum(-1), config)
    mean_returns = returns.mean(-1)
    best = mean_returns.masked_fill(actions < 0, -torch.inf).amax(-1)
    gain = best - mean_returns[:, 0]
    metrics = empty_ranking_metrics()
    metrics.update(ranking_states=count, ranking_theta_trials=count * config.ranking_theta_candidates,
                   ranking_unique_actions=int((actions >= 0).sum()),
                   ranking_branches=int((actions >= 0).sum()) * returns.shape[-1],
                   ranking_pairs=pairs[0].numel(),
                   ranking_improvement_rate=(gain / state.profits.sum(-1) > config.ranking_min_gap).float().mean().item(),
                   ranking_gain_raw=gain.mean().item(), ranking_seconds=time.perf_counter() - started)
    return RankingBatch(state, actions, witnesses, proposals, proposal_actions, returns, sequences,
                        *pairs, policy.log_probs.detach().clone(), times, batches, teacher_hash,
                        config_dict(model.config), seed, metrics)


def ranking_objective(model, batch, config):
    """Actor-only weighted pairwise logistic loss on the unchanged raw decoder scores."""
    theta, _ = model.actor(batch.state)
    owners, items, scores = model.decoder.scores(batch.state, theta)
    size, n = batch.state.profits.shape
    policy = CandidatePolicy(owners, items, scores / theta.temperature[owners], size)
    kl_by_state = scores.new_zeros(size).scatter_add(
        0, owners, batch.old_log_probs.exp() * (batch.old_log_probs - policy.log_probs))
    # A dense lookup only for score differences, never used as an action distribution.
    dense = scores.new_zeros(size * n).scatter(0, owners * n + items, scores).view(size, n)
    rows = batch.pair_states
    if rows.numel():
        gap = (dense[rows, batch.actions[rows, batch.winners]]
               - dense[rows, batch.actions[rows, batch.losers]])
        weights = (batch.gaps / config.ranking_return_scale).clamp(max=1)
        pair_loss = weights * F.softplus(-gap / config.ranking_score_scale)
        counts = scores.new_zeros(size).scatter_add(0, rows, torch.ones_like(pair_loss))
        per_state = scores.new_zeros(size).scatter_add(0, rows, pair_loss) / counts.clamp_min(1)
        loss = per_state[counts > 0].mean()
        accuracy = (gap > 0).float().mean()
    else:
        loss, accuracy = scores.sum() * 0, scores.new_tensor(float("nan"))
    with torch.no_grad():
        selected_scores = dense.gather(1, batch.actions.clamp_min(0)).masked_fill(batch.actions < 0, -torch.inf)
        chosen_slot = selected_scores.argmax(-1)
        returns = batch.returns.mean(-1)
        best = returns.masked_fill(batch.actions < 0, -torch.inf).amax(-1)
        total = batch.state.profits.sum(-1)
        regret = ((best - returns.gather(1, chosen_slot[:, None]).squeeze(1)) / total).mean()
        greedy = policy.choose()
        match = batch.actions == greedy[:, None]
        covered = match.any(-1)
        covered_return = returns.gather(1, match.long().argmax(-1)[:, None]).squeeze(1)
        greedy_regret = ((best - covered_return)[covered] / total[covered]).mean()
    metrics = {"ranking_loss": loss.detach().item(), "ranking_pair_accuracy": accuracy.item(),
               "ranking_candidate_regret": regret.item(), "ranking_greedy_coverage": covered.float().mean().item(),
               "ranking_greedy_regret": greedy_regret.item(), "ranking_kl": kl_by_state.mean().clamp_min(0).detach().item()}
    return loss, metrics


def write_ranking_audit(path, batch, update, num_resources, config):
    """Store enough physical evidence to recheck actions, feasibility and raw profit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trials = batch.proposal_actions.shape[1]
    theta = {name: getattr(batch.proposal_theta, name).view(batch.actions.shape[0], trials, -1).cpu().tolist()
             for name in THETA_FIELDS}
    states = {name: getattr(batch.state, name).cpu().tolist()
              for name in ("profits", "weights", "capacities", "remaining", "selected")}
    with path.open("a", encoding="utf-8") as stream:
        for row in range(batch.actions.shape[0]):
            candidates = []
            for slot, action in enumerate(batch.actions[row].tolist()):
                if action < 0:
                    continue
                witness = batch.witness_indices[row, slot].item()
                candidates.append({"action": action, "theta_trial": witness,
                                   "theta": {name: theta[name][row][witness] for name in THETA_FIELDS},
                                   "additional_profit": batch.returns[row, slot].cpu().tolist(),
                                   "sequences": [[i for i in sequence if i >= 0]
                                                 for sequence in batch.sequences[row, slot].cpu().tolist()]})
            pairs = (batch.pair_states == row).nonzero(as_tuple=True)[0]
            record = {"update": update, "teacher_update": update - 1, "M": num_resources,
                      "source": "synthetic_ppo_visited_state", "source_time": batch.source_times[row].item(),
                      "source_batch": batch.source_batches[row].item(), "seed": batch.seed,
                      "teacher_actor_sha256": batch.teacher_sha256,
                      "model_config": batch.model_config,
                      "min_gap": config.ranking_min_gap, "se_multiplier": config.ranking_se_multiplier,
                      "continuation": config.ranking_continuation,
                      "state": {name: values[row] for name, values in states.items()},
                      "proposal_actions": batch.proposal_actions[row].cpu().tolist(), "candidates": candidates,
                      "pairs": [{"winner": batch.actions[row, batch.winners[k]].item(),
                                 "loser": batch.actions[row, batch.losers[k]].item(),
                                 "normalized_gap": batch.gaps[k].item(), "paired_se": batch.standard_errors[k].item()}
                                for k in pairs.tolist()]}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def verify_ranking_record(record):
    """Replay physical evidence and theta witnesses, without any model/solver training.

    This verifies feasibility, profit accounting and preference labels. It cannot
    reconstruct the teacher network from its hash; sequences are retained evidence.
    """
    c = ModelConfig(**record["model_config"])
    tensors = [torch.tensor([record["state"][name]], dtype=torch.bool if name == "selected" else torch.float64)
               for name in ("profits", "weights", "capacities", "remaining", "selected")]
    state = MKPState(*tensors)
    expected_remaining = state.capacities - (state.weights * state.selected[..., None]).sum(1)
    if not torch.allclose(state.remaining, expected_remaining, atol=1e-7, rtol=1e-12):
        raise ValueError("Audit state remaining capacity disagrees with selected items")
    decoder = WeakDecoder(c)
    returns = {}
    for candidate in record["candidates"]:
        theta = StrategyTheta(*(torch.tensor([candidate["theta"][name]], dtype=torch.float32)
                                if name == "resource_prices" else
                                torch.tensor(candidate["theta"][name], dtype=torch.float32)
                                for name in THETA_FIELDS))
        if not all(torch.isfinite(getattr(theta, name)).all() for name in THETA_FIELDS):
            raise ValueError("Non-finite theta witness")
        if ((theta.resource_prices < c.price_floor - 1e-6).any()
                or not torch.allclose(theta.resource_prices.mean(-1), torch.ones(1), atol=1e-6)):
            raise ValueError("Illegal price witness")
        for name, lo, hi in (("profit_exponent", c.profit_exponent_min, c.profit_exponent_max),
                             ("scarcity_exponent", 0, c.scarcity_exponent_max),
                             ("bottleneck_weight", 0, c.bottleneck_weight_max),
                             ("temperature", c.temperature_min, c.temperature_max)):
            if not lo - 1e-6 <= getattr(theta, name).item() <= hi + 1e-6:
                raise ValueError("Illegal scalar theta witness")
        action = candidate["action"]
        if action in returns or decoder(state, theta).choose().item() != action:
            raise ValueError("Duplicate action or theta witness does not induce the recorded action")
        if len(candidate["sequences"]) != len(candidate["additional_profit"]):
            raise ValueError("Audit repeat count differs")
        profits = []
        for sequence, reported in zip(candidate["sequences"], candidate["additional_profit"]):
            if not sequence or sequence[0] != action:
                raise ValueError("Sequence must start with the witnessed action")
            env = MKPEnv.from_state(state)
            past = env.objective.item()
            for item in sequence:
                env.step(torch.tensor([item], dtype=torch.long))
            actual = env.objective.item() - past
            if env.state.active.any() or not math.isclose(actual, reported, rel_tol=1e-12, abs_tol=1e-7):
                raise ValueError("Sequence is nonterminal or its profit differs from the audit")
            profits.append(actual)
        returns[action] = torch.tensor(profits, dtype=torch.float64)
    for pair in record["pairs"]:
        delta = (returns[pair["winner"]] - returns[pair["loser"]]) / state.profits.sum()
        se = delta.std(unbiased=True).item() / delta.numel() ** .5 if delta.numel() > 1 else 0.
        if (delta.mean().item() <= max(record["min_gap"], record["se_multiplier"] * se)
                or not math.isclose(delta.mean().item(), pair["normalized_gap"], rel_tol=1e-6, abs_tol=1e-10)
                or not math.isclose(se, pair["paired_se"], rel_tol=1e-6, abs_tol=1e-10)):
            raise ValueError("Preference label disagrees with paired terminal profits")
    return len(record["candidates"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify saved counterfactual actions and profits; no training")
    parser.add_argument("--verify-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    records = candidates = 0
    with args.verify_audit.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                candidates += verify_ranking_record(json.loads(line))
                records += 1
    print(f"Verified {records} states and {candidates} theta-induced candidate actions")


if __name__ == "__main__":
    main()
