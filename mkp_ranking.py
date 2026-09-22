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

    def index(self, indices):
        return RankingBatch(self.state.index(indices), *(x[indices] for x in
            (self.left, self.right, self.delta, self.standard_error, self.old_log_probs)))


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
def collect_ranking_batch(model, data, rollout, config, seed):
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


def ranking_objective(model, batch, config):
    policy, theta, _ = model.policy(batch.state)
    # Temperature cannot satisfy ranking supervision merely by sharpening logits.
    margin = ((policy.log_prob(batch.left) - policy.log_prob(batch.right))
              * theta.temperature / config.ranking_score_scale)
    target = torch.sigmoid(batch.delta / config.ranking_return_scale)
    loss = F.binary_cross_entropy_with_logits(margin, target)
    old = batch.old_log_probs[policy.owners, policy.items]
    kl = (old.exp() * (old - policy.log_probs)).sum() / batch.left.numel()
    accuracy = ((margin.detach() * batch.delta) > 0).float().mean()
    return loss, kl.clamp_min(0), accuracy


@torch.no_grad()
def ranking_kl(model, batch):
    policy = model.policy(batch.state)[0]
    old = batch.old_log_probs[policy.owners, policy.items]
    return ((old.exp() * (old - policy.log_probs)).sum() / batch.left.numel()).clamp_min(0)
