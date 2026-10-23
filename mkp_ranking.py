"""Training-only critical-pair supervision using frozen-policy terminal returns.

These are policy-continuation estimates, never oracle/Q* labels or PPO samples.
Inference uses only the controller and the fixed decoder.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from mkp_env import MKPEnv, MKPState


@dataclass
class RankingBatch:
    state: MKPState
    left: torch.Tensor
    right: torch.Tensor
    delta: torch.Tensor
    standard_error: torch.Tensor
    old_log_probs: torch.Tensor  # [B,N]; only feasible entries are read
    confidence: torch.Tensor | None = None
    state_weights: torch.Tensor | None = None  # mean-one weights, equal total per labeled source state

    def index(self, indices):
        return RankingBatch(self.state.index(indices), *(x[indices] for x in
            (self.left, self.right, self.delta, self.standard_error, self.old_log_probs)),
            None if self.confidence is None else self.confidence[indices],
            None if self.state_weights is None else self.state_weights[indices])


@torch.no_grad()
def paired_terminal_returns(model, state, left, right, repeats, stochastic, generator=None):
    """Force i/j from the same state, then finish with shared item-indexed Gumbels."""
    if repeats < 1:
        raise ValueError("Continuation repeats must be positive")
    b, n = state.profits.shape
    indices = torch.arange(b, device=left.device).repeat_interleave(repeats * 2)
    env = MKPEnv.from_state(state.index(indices, with_features=False))
    actions = torch.stack((left, right), -1)[:, None].expand(-1, repeats, -1).reshape(-1)
    env.step(actions)
    for _ in range(n - 1):
        if not env.state.active.any():
            break
        policy = model.policy(env.state)[0]
        if stochastic:
            uniform = torch.rand((b, repeats, n), device=left.device, generator=generator)
            gumbel = -(-uniform.clamp(1e-7, 1 - 1e-7).log()).log()
            # Different candidate sets still share noise for each original item.
            noise = gumbel[:, :, None].expand(-1, -1, 2, -1).reshape(-1, n)
            from mkp_Model import CandidatePolicy
            actions = CandidatePolicy(policy.owners, policy.items,
                policy.logits + noise[policy.owners, policy.items], policy.batch_size).choose()
        else:
            actions = policy.choose()
        env.step(actions)
    if env.state.active.any():
        raise RuntimeError("Counterfactual continuation exceeded N steps")
    return (env.objective / env.state.profits.sum(-1)).float().reshape(b, repeats, 2)


def return_targets(greedy, sampled, config):
    greedy_delta = greedy[:, 0, 0] - greedy[:, 0, 1]
    differences = sampled[:, :, 0] - sampled[:, :, 1]
    sample_delta = differences.mean(-1)
    sample_se = differences.std(-1, unbiased=True) / math.sqrt(differences.shape[1])
    weight = config.ranking_greedy_weight
    delta = weight * greedy_delta + (1 - weight) * sample_delta
    se = (1 - weight) * sample_se
    threshold = torch.maximum(torch.full_like(delta, config.ranking_min_gap),
                              config.ranking_se_multiplier * se)
    keep = delta.abs() > threshold
    if 0 < weight < 1:
        # Avoid teaching confidently conflicting greedy/stochastic orderings.
        conflict = ((greedy_delta * sample_delta < 0)
                    & (greedy_delta.abs() > config.ranking_min_gap)
                    & (sample_delta.abs() > config.ranking_se_multiplier * sample_se))
        keep &= ~conflict
    return delta, se, keep


@torch.no_grad()
def _collect_legacy_ranking_batch(model, data, rollout, config, seed):
    """Bounded pool, top-score similar peers, then 3/4 hard + 1/4 random states."""
    metrics = dict(ranking_proposed=0, ranking_pairs=0, ranking_abs_return_gap=0.,
                   ranking_standard_error=0., ranking_label_seconds=0.)
    if config.ranking_coefficient == 0:
        return None, metrics
    import time
    began = time.perf_counter()
    device = data[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    times, batches = rollout["active"].nonzero(as_tuple=True)
    if times.numel() == 0:
        return None, metrics
    ids = torch.randperm(times.numel(), device=device, generator=generator)[:config.ranking_pool_size]
    b, t = batches[ids], times[ids]
    state = MKPState(data[0][b], data[1][b], data[2][b],
                     rollout["remaining"][t, b], rollout["selected"][t, b])
    state = state.index((state.feasible.sum(-1) >= 2).nonzero(as_tuple=True)[0])
    if state.profits.shape[0] == 0:
        return None, metrics
    was_training = model.training
    model.eval()
    try:
        policy, theta, _ = model.policy(state)
        old = state.profits.new_zeros(state.profits.shape, dtype=torch.float32)
        old[policy.owners, policy.items] = policy.log_probs
        left, right, priority = [], [], []
        anchors = policy.choose()
        for row in range(state.profits.shape[0]):
            entries = (policy.owners == row).nonzero(as_tuple=True)[0]
            anchor_entry = entries[policy.items[entries] == anchors[row]]
            others = entries[policy.items[entries] != anchors[row]]
            top = torch.cat((anchor_entry, others[policy.logits[others].argsort(descending=True)
                             [:config.ranking_candidates - 1]]))
            anchor, peers = policy.items[top[0]], policy.items[top[1:]]
            p = state.profits[row]
            profit_distance = ((p[peers] - p[anchor]).abs() / p[anchor].clamp_min(1e-8)).float()
            load_distance = ((state.weights[row, peers] - state.weights[row, anchor]).abs()
                             / state.remaining[row].clamp_min(1e-8)).mean(-1).float()
            # Raw score gaps, invariant to temperature rescaling.
            gaps = (policy.logits[top[0]] - policy.logits[top[1:]]) * theta.temperature[row]
            closeness = profit_distance + load_distance + gaps
            peer = closeness.argmin()
            left.append(anchor)
            right.append(peers[peer])
            priority.append(closeness[peer])
        left, right = torch.stack(left), torch.stack(right)
        ordering = torch.stack(priority).argsort()
        count = min(config.ranking_states, ordering.numel())
        hard_count = max(1, 3 * count // 4)
        rest = ordering[hard_count:]
        random_ids = torch.randperm(rest.numel(), device=device, generator=generator)[:count - hard_count]
        chosen = torch.cat((ordering[:hard_count], rest[random_ids]))
        state, left, right, old = state.index(chosen), left[chosen], right[chosen], old[chosen]
        metrics["ranking_proposed"] = count
        greedy = paired_terminal_returns(model, state, left, right, 1, False)
        sampled = paired_terminal_returns(model, state, left, right, config.ranking_repeats, True, generator)
        delta, se, keep = return_targets(greedy, sampled, config)
        accepted = keep.nonzero(as_tuple=True)[0]
        metrics["ranking_pairs"] = accepted.numel()
        if accepted.numel():
            metrics["ranking_abs_return_gap"] = delta[accepted].abs().mean().item()
            metrics["ranking_standard_error"] = se[accepted].mean().item()
            batch = RankingBatch(state, left, right, delta, se, old).index(accepted)
        else:
            batch = None
        metrics["ranking_label_seconds"] = time.perf_counter() - began
        return batch, metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def candidate_terminal_returns(model, state, candidates, repeats, stochastic,
                               generator=None, chunk_size=128):
    """Frozen continuations [B,R,K], with shared item noise and bounded policy batches.

    Returns stay float64 so a small final-profit difference cannot disappear when
    the selected prefix is large. Candidates are training proposals, not a search
    added to inference. Repeated padding candidates are legal and later masked.
    """
    if repeats < 1 or chunk_size < 1 or candidates.ndim != 2:
        raise ValueError("Invalid candidate continuation dimensions or budget")
    b, n = state.profits.shape
    if candidates.shape[0] != b or candidates.shape[1] < 1:
        raise ValueError("Candidates must have shape [B,K]")
    k = candidates.shape[1]
    rows = torch.arange(b, device=state.profits.device).repeat_interleave(repeats * k)
    env = MKPEnv.from_state(state.index(rows, with_features=False))
    env.step(candidates[:, None].expand(-1, repeats, -1).reshape(-1))
    total = rows.numel()
    for _ in range(n - 1):
        if not env.state.active.any():
            break
        if stochastic:
            uniform = torch.rand((b * repeats, n), device=rows.device, generator=generator)
            noise = -(-uniform.clamp(1e-7, 1 - 1e-7).log()).log()
        actions = torch.full((total,), -1, device=rows.device, dtype=torch.long)
        for start in range(0, total, chunk_size):
            ids = torch.arange(start, min(start + chunk_size, total), device=rows.device)
            policy = model.policy(env.state.index(ids, with_features=False))[0]
            if stochastic:
                from mkp_Model import CandidatePolicy
                shared = noise[(start + policy.owners) // k, policy.items]
                policy = CandidatePolicy(policy.owners, policy.items,
                                         policy.logits + shared, policy.batch_size)
            actions[ids] = policy.choose()
        env.step(actions)
    if env.state.active.any():
        raise RuntimeError("Candidate continuation exceeded N steps")
    return (env.objective / env.state.profits.sum(-1)).reshape(b, repeats, k)


def dense_return_targets(greedy, sampled, selected_return, valid, config):
    """Best sampled candidate under greedy continuation versus every alternative.

    Positive labels use deterministic residual-profit differences. Disagreeing
    stochastic continuations lower their confidence without reversing the greedy
    target or deleting the label. These are finite candidate-policy comparisons,
    never exact action values or globally optimal solutions.
    """
    greedy = greedy[:, 0]
    best = greedy.masked_fill(~valid, -torch.inf).argmax(-1)
    rows = torch.arange(greedy.shape[0], device=greedy.device)
    best_return = greedy[rows, best]
    # The common selected prefix is constant across branches. Removing it keeps
    # fine endgame improvements visible instead of dividing by all initial items.
    scale = (best_return - selected_return).clamp_min(torch.finfo(greedy.dtype).eps)
    delta = (best_return[:, None] - greedy) / scale[:, None]
    differences = (sampled[rows, :, best][:, :, None] - sampled) / scale[:, None, None]
    sample_delta = differences.mean(1)
    standard_error = differences.std(1, unbiased=True) / math.sqrt(sampled.shape[1])
    band = config.ranking_se_multiplier * standard_error
    agreement = torch.where(band > 0, torch.sigmoid(sample_delta / band.clamp_min(1e-12)),
                            torch.where(sample_delta > 0, torch.ones_like(delta),
                                        torch.where(sample_delta < 0, torch.zeros_like(delta),
                                                    torch.full_like(delta, .5))))
    confidence = config.ranking_confidence_floor + (1 - config.ranking_confidence_floor) * agreement
    keep = valid & (delta > config.ranking_min_gap)
    conflicts = keep & (sample_delta < -band)
    return best, delta, standard_error, confidence, keep, conflicts


def _dense_candidates(state, policy, count, generator):
    """Cover score rivals, profit, distinct resource densities, and a random rival."""
    candidates, valid, priority = [], [], []
    anchors = policy.choose()
    for row in range(state.profits.shape[0]):
        entries = (policy.owners == row).nonzero(as_tuple=True)[0]
        ordered = policy.items[entries[policy.logits[entries].argsort(descending=True)]]
        feasible = policy.items[entries]
        p = state.profits[row, feasible]
        load = state.weights[row, feasible] / state.remaining[row].clamp_min(1e-12)
        density = p / load.mean(-1).clamp_min(1e-12)
        bottleneck_density = p / load.amax(-1).clamp_min(1e-12)
        proposals = [anchors[row], *ordered[:2], feasible[p.argmax()],
                     feasible[density.argmax()], feasible[bottleneck_density.argmax()],
                     feasible[load.amax(-1).argmin()]]
        proposals.extend(feasible[torch.randperm(feasible.numel(), device=feasible.device,
                                                generator=generator)[:1]])
        # Resource-specific density rivals expose complementary resource use.
        for resource in load.sum(0).argsort(descending=True):
            proposals.append(feasible[(p / load[:, resource].clamp_min(1e-12)).argmax()])
        proposals.extend(ordered)
        unique = []
        for item in proposals:
            item = int(item)
            if item not in unique:
                unique.append(item)
            if len(unique) == count:
                break
        size = len(unique)
        candidates.append(unique + [unique[0]] * (count - size))
        valid.append([True] * size + [False] * (count - size))
        sorted_logits = policy.logits[entries].sort(descending=True).values
        priority.append(sorted_logits[0] - sorted_logits[1])
    return (torch.tensor(candidates, device=anchors.device),
            torch.tensor(valid, device=anchors.device), torch.stack(priority))


@torch.no_grad()
def collect_ranking_batch(model, data, rollout, config, seed):
    import time
    began = time.perf_counter()
    metrics = dict(ranking_proposed=0, ranking_pairs=0, ranking_abs_return_gap=0.,
                   ranking_standard_error=0., ranking_label_seconds=0.,
                   ranking_states_labeled=0, ranking_candidates_labeled=0,
                   ranking_confidence_mean=0., ranking_greedy_conflicts=0)
    if config.ranking_target_mode == "legacy_mixed":
        batch, legacy_metrics = _collect_legacy_ranking_batch(model, data, rollout, config, seed)
        metrics.update(legacy_metrics)
        metrics.update(ranking_states_labeled=metrics["ranking_proposed"],
                       ranking_candidates_labeled=2 * metrics["ranking_proposed"],
                       ranking_confidence_mean=1. if batch is not None else 0.)
        return batch, metrics
    if config.ranking_coefficient == 0:
        return None, metrics
    device = data[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    times, batches = rollout["active"].nonzero(as_tuple=True)
    if times.numel() == 0:
        return None, metrics
    ids = torch.randperm(times.numel(), device=device, generator=generator)[:config.ranking_pool_size]
    b, t = batches[ids], times[ids]
    state = MKPState(data[0][b], data[1][b], data[2][b],
                     rollout["remaining"][t, b], rollout["selected"][t, b])
    usable = (state.feasible.sum(-1) >= 2).nonzero(as_tuple=True)[0]
    state, t, b = state.index(usable), t[usable], b[usable]
    if state.profits.shape[0] == 0:
        return None, metrics
    was_training = model.training
    model.eval()
    try:
        policy, theta, _ = model.policy(state)
        candidates, valid, priority = _dense_candidates(state, policy,
                                                        config.ranking_candidates, generator)
        priority = priority * theta.temperature
        count = min(config.ranking_states, state.profits.shape[0])
        # Reserve half the state budget for late decisions where unusable residual
        # capacity is decided; choose the other half from ambiguous score margins.
        progress = t / rollout["active"].sum(0)[b].clamp_min(1)
        late = progress.argsort(descending=True)[:count // 2]
        available = torch.ones(state.profits.shape[0], dtype=torch.bool, device=device)
        available[late] = False
        hard = priority.masked_fill(~available, torch.inf).argsort()[:count - late.numel()]
        chosen = torch.cat((late, hard))
        old = state.profits.new_zeros(state.profits.shape, dtype=torch.float32)
        old[policy.owners, policy.items] = policy.log_probs
        state, old = state.index(chosen), old[chosen]
        candidates, valid = candidates[chosen], valid[chosen]
        greedy = candidate_terminal_returns(model, state, candidates, 1, False,
                                            chunk_size=config.ranking_branch_chunk_size)
        sampled = candidate_terminal_returns(model, state, candidates, config.ranking_repeats,
                                             True, generator, config.ranking_branch_chunk_size)
        selected_return = (state.profits * state.selected).sum(-1) / state.profits.sum(-1)
        best, delta, se, confidence, keep, conflicts = dense_return_targets(
            greedy, sampled, selected_return, valid, config)
        rows, peers = keep.nonzero(as_tuple=True)
        labeled_states = keep.any(-1).sum()
        metrics.update(ranking_states_labeled=int(labeled_states),
                       ranking_candidates_labeled=int(valid.sum()),
                       ranking_proposed=int(valid.sum()) - count,
                       ranking_pairs=rows.numel(), ranking_greedy_conflicts=int(conflicts.sum()))
        batch = None
        if rows.numel():
            metrics.update(ranking_abs_return_gap=delta[rows, peers].mean().item(),
                           ranking_standard_error=se[rows, peers].mean().item(),
                           ranking_confidence_mean=confidence[rows, peers].mean().item())
            batch = RankingBatch(state.index(rows), candidates[rows, best[rows]],
                                 candidates[rows, peers], delta[rows, peers], se[rows, peers],
                                 old[rows], confidence[rows, peers],
                                 rows.numel() / (labeled_states * keep.sum(-1)[rows]).to(confidence))
        metrics["ranking_label_seconds"] = time.perf_counter() - began
        return batch, metrics
    finally:
        model.train(was_training)


def regret_margin_losses(margin, delta, return_scale, margin_cap=1.):
    """Bounded return-weighted corrections; satisfied comparisons have no push.

    ``delta`` is a frozen policy-continuation estimate, not an exact Q value.
    Its sign chooses the preferred ordering. Its magnitude sets a bounded
    required score margin and a bounded importance weight. Labels never receive
    gradients, and equal-return comparisons impose no preference.
    """
    if not math.isfinite(return_scale) or return_scale <= 0:
        raise ValueError("Ranking return scale must be finite and positive")
    if not math.isfinite(margin_cap) or margin_cap <= 0:
        raise ValueError("Ranking margin cap must be finite and positive")
    delta = delta.detach().to(margin)
    strength = delta.abs() / return_scale
    required = strength.clamp(max=margin_cap)
    weight = strength.clamp(max=1.)
    violation = F.relu(required - delta.sign() * margin)
    # Huber avoids an extreme wrong score dominating all other comparisons
    # through a quadratic derivative. d(loss)/d(margin) is bounded by weight;
    # network-parameter gradients still use the normal global gradient clip.
    return F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none") * weight


def ranking_objective(model, batch, config):
    policy, theta, _ = model.policy(batch.state)
    # Temperature cannot satisfy ranking supervision merely by sharpening logits.
    margin = ((policy.log_prob(batch.left) - policy.log_prob(batch.right))
              * theta.temperature / config.ranking_score_scale)
    if config.ranking_loss_type == "soft_bce":
        target = torch.sigmoid(batch.delta.detach() / config.ranking_return_scale).to(margin)
        losses = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
    elif config.ranking_loss_type == "regret_margin":
        losses = regret_margin_losses(margin, batch.delta, config.ranking_return_scale,
                                      config.ranking_margin_cap)
    else:
        raise ValueError("Unknown ranking loss type")
    # Mean rather than renormalizing by confidence sum: if every greedy label
    # conflicts with stochastic continuation, their aggregate gradient is smaller.
    weighted = losses if batch.confidence is None else losses * batch.confidence.to(losses)
    # Equal total weight per source state, so late states with few viable rivals
    # are not diluted by early states contributing many comparisons.
    loss = (weighted if batch.state_weights is None else weighted * batch.state_weights.to(weighted)).mean()
    old = batch.old_log_probs[policy.owners, policy.items]
    contributions = old.exp() * (old - policy.log_probs)
    if batch.state_weights is not None:
        contributions = contributions * batch.state_weights.to(contributions)[policy.owners]
    kl = contributions.sum() / batch.left.numel()
    accuracy = ((margin.detach() * batch.delta) > 0).float().mean()
    return loss, kl.clamp_min(0), accuracy


@torch.no_grad()
def ranking_kl(model, batch):
    policy = model.policy(batch.state)[0]
    old = batch.old_log_probs[policy.owners, policy.items]
    contributions = old.exp() * (old - policy.log_probs)
    if batch.state_weights is not None:
        contributions = contributions * batch.state_weights.to(contributions)[policy.owners]
    return (contributions.sum() / batch.left.numel()).clamp_min(0)
