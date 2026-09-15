"""Training-only paired Monte Carlo action values under the frozen behavior policy.

No solver, learned item scorer, critic bootstrap or inference-time search. Each
branch starts from the same state and runs to termination after a forced action.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from mkp_env import MKPEnv, MKPState


@dataclass
class RankingBatch:
    state: MKPState
    actions: torch.Tensor          # [S,2], distinct feasible items
    q_samples: torch.Tensor        # [S,R,2], complete normalized future returns
    old_log_probs: torch.Tensor    # [S,N], only feasible entries are used
    reliable: torch.Tensor         # [S], gap exceeds noise threshold

    @property
    def delta(self):
        return (self.q_samples[:, :, 0] - self.q_samples[:, :, 1]).mean(-1)


@torch.no_grad()
def paired_terminal_returns(model, state, actions, repeats):
    """Common per-item Gumbels couple the two valid categorical continuations.

    Randomness is independent across states/repeats/steps, shared across branches.
    Returns include the forced action and all subsequent rewards, never the past.
    """
    if actions.shape != (state.profits.shape[0], 2) or repeats < 2:
        raise ValueError("Expected two actions per state and at least two repeats")
    was_training = model.training
    model.eval()
    try:
        count, n = state.profits.shape
        ids = torch.arange(count, device=actions.device).repeat_interleave(repeats * 2)
        branch_state = state.index(ids)
        # Float64 profit differences avoid summation-order noise for tied subsets.
        prior = (branch_state.profits * branch_state.selected).sum(-1)
        env = MKPEnv.from_state(branch_state)
        forced = actions[:, None, :].expand(-1, repeats, -1).reshape(-1)
        env.step(forced)
        for _ in range(n - 1):
            active = env.state.active.nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                break
            policy = model.policy(env.state.index(active))[0]
            uniform = torch.rand(count, repeats, 1, n, device=actions.device).clamp(1e-7, 1 - 1e-7)
            noise = -(-uniform.log()).log()
            noise = noise.expand(-1, -1, 2, -1).reshape(-1, n)[active]
            chosen = torch.full_like(forced, -1)
            chosen[active] = policy.choose(stochastic=True, gumbels=noise)
            env.step(chosen)
        if env.state.active.any():
            raise RuntimeError("Counterfactual continuation exceeded N steps")
        returns = (env.objective - prior) / env.state.profits.sum(-1)
        return returns.reshape(count, repeats, 2)
    finally:
        model.train(was_training)


@torch.no_grad()
def collect_ranking_batch(model, data, rollout, config):
    """Half capacity-pressure states, half uniform coverage; bounded per update.

    Compare the behavior-policy top item to its runner-up or a uniform alternative.
    Pressure only selects states; terminal return alone determines preferences.
    """
    if config.ranking_coefficient == 0:
        return None
    times, batches, pressures = [], [], []
    for t in range(rollout["selected"].shape[0]):
        state = MKPState(*data, rollout["remaining"][t], rollout["selected"][t])
        feasible = state.feasible
        valid = (feasible.sum(-1) >= 2).nonzero(as_tuple=True)[0]
        if valid.numel():
            load = (state.weights[valid] / state.remaining[valid, None, :].clamp_min(1e-8)).amax(-1)
            pressure = (load * feasible[valid]).sum(-1) / feasible[valid].sum(-1)
            times.append(torch.full_like(valid, t))
            batches.append(valid)
            pressures.append(pressure)
    if not times:
        return None
    times, batches, pressures = map(torch.cat, (times, batches, pressures))
    size = min(config.counterfactual_states, times.numel())
    # Randomize ties so the pressure subset does not systematically pick one row.
    order = torch.randperm(times.numel(), device=times.device)
    priority = order[pressures[order].argsort(descending=True, stable=True)[:size // 2]]
    available = torch.ones_like(times, dtype=torch.bool)
    available[priority] = False
    random_ids = order[available[order]][:size - priority.numel()]
    chosen = torch.cat((priority, random_ids))
    t, b = times[chosen], batches[chosen]
    state = MKPState(data[0][b], data[1][b], data[2][b],
                     rollout["remaining"][t, b], rollout["selected"][t, b])
    was_training = model.training
    model.eval()
    try:
        policy = model.policy(state)[0]
        first = policy.choose()
        second = torch.empty_like(first)
        for row in range(size):
            entries = ((policy.owners == row) & (policy.items != first[row])).nonzero(as_tuple=True)[0]
            if row % 2 == 0:
                entry = entries[policy.log_probs[entries].argmax()]
            else:
                entry = entries[torch.randint(entries.numel(), (), device=entries.device)]
            second[row] = policy.items[entry]
        actions = torch.stack((first, second), -1)
        samples = paired_terminal_returns(model, state, actions, config.counterfactual_repeats)
        differences = samples[:, :, 0] - samples[:, :, 1]
        se = differences.std(-1, unbiased=True) / config.counterfactual_repeats ** 0.5
        threshold = (config.ranking_se_multiplier * se).clamp_min(config.ranking_min_gap)
        reliable = differences.mean(-1).abs() > threshold
        old = policy.log_probs.new_zeros(state.profits.shape)
        old[policy.owners, policy.items] = policy.log_probs
        return RankingBatch(state, actions, samples, old, reliable)
    finally:
        model.train(was_training)


def ranking_loss(model, batch, ids, config):
    """Soft preference encodes sign AND size of Q(i)-Q(j); labels are detached.

    d = log pi(i|s)-log pi(j|s). Target sigmoid(delta / return_scale).
    The gradient reaches only theta and the actor through the fixed decoder.
    Full candidate KL protects these additional states alongside PPO's sampled KL.
    """
    state = batch.state.index(ids)
    policy = model.policy(state)[0]
    actions = batch.actions[ids]
    difference = policy.log_prob(actions[:, 0]) - policy.log_prob(actions[:, 1])
    target = (batch.delta[ids].detach() / config.ranking_return_scale).sigmoid().to(difference.dtype)
    losses = F.binary_cross_entropy_with_logits(difference, target, reduction="none")
    weights = batch.reliable[ids].to(losses.dtype)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1)
    old = batch.old_log_probs[ids][policy.owners, policy.items].detach()
    kl = (old.exp() * (old - policy.log_probs)).sum() / ids.numel()
    return loss, kl


@torch.no_grad()
def ranking_metrics(model, batch):
    if batch is None:
        return dict(ranking_states=0, ranking_reliable_pairs=0, ranking_abs_return_gap=0.,
                    ranking_return_gap_se=0., ranking_pair_accuracy=float("nan"),
                    ranking_pair_regret=float("nan"))
    policy = model.policy(batch.state)[0]
    difference = policy.log_prob(batch.actions[:, 0]) - policy.log_prob(batch.actions[:, 1])
    delta, valid = batch.delta, batch.reliable
    signed = difference.sign() * delta.sign()
    accuracy = (signed[valid] + 1) / 2  # exact score ties get half credit
    regret = torch.where(signed < 0, delta.abs(), torch.where(signed == 0, delta.abs() / 2, 0.))
    differences = batch.q_samples[:, :, 0] - batch.q_samples[:, :, 1]
    return dict(ranking_states=delta.numel(), ranking_reliable_pairs=int(valid.sum()),
                ranking_abs_return_gap=delta.abs().mean().item(),
                ranking_return_gap_se=(differences.std(-1) / differences.shape[1] ** 0.5).mean().item(),
                ranking_pair_accuracy=accuracy.mean().item() if valid.any() else float("nan"),
                ranking_pair_regret=regret[valid].mean().item() if valid.any() else float("nan"))
