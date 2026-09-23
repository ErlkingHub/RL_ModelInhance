"""PPO auxiliary action ranking from training-only theta counterfactuals.

These are policy-continuation estimates, never oracle/Q* labels or PPO samples.
Inference uses only the controller and the fixed decoder.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from mkp_env import MKPEnv, MKPState

THETA_RANKING_VERSION = "theta_action_rollout_v1"
THETA_METRICS = ("ranking_states_considered", "ranking_states_labeled", "ranking_theta_evaluations",
                 "ranking_unique_actions", "ranking_action_diversity", "ranking_no_alternative",
                 "ranking_improved_pairs", "ranking_improved_states", "ranking_improvement_mean",
                 "ranking_raw_improvement_mean", "ranking_rollout_branches")


def ranking_protocol(config):
    return THETA_RANKING_VERSION if config.ranking_mode == "theta" else "score_pairs_v1"


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
def candidate_terminal_returns(model, state, owners, actions, repeats, stochastic, generator=None):
    """Raw Q^mu(s,a): first action plus subsequent profit, excluding the prefix.

    The model's parameters stay fixed during this synchronous labeling call.
    Every successor step regenerates theta. Only the initial action is forced;
    proposal theta is never carried into the continuation. Noise is shared by
    (source state, repeat, step, original item), even with ragged action sets.
    """
    if repeats < 1:
        raise ValueError("Continuation repeats must be positive")
    b, n = state.profits.shape
    if owners.ndim != 1 or actions.shape != owners.shape or not actions.numel():
        raise ValueError("Need nonempty owners/actions vectors")
    indices = owners.repeat_interleave(repeats)
    repeat_ids = torch.arange(repeats, device=actions.device).repeat(actions.numel())
    branch = state.index(indices, with_features=False)
    env = MKPEnv.from_state(branch)
    was_training = model.training
    model.eval()
    try:
        env.step(actions.repeat_interleave(repeats))
        for _ in range(n - 1):
            if not env.state.active.any():
                break
            policy = model.policy(env.state)[0]
            if stochastic:
                uniform = torch.rand((b, repeats, n), device=actions.device, generator=generator)
                gumbel = -(-uniform.clamp(1e-7, 1 - 1e-7).log()).log()
                from mkp_Model import CandidatePolicy
                noise = gumbel[indices[policy.owners], repeat_ids[policy.owners], policy.items]
                following = CandidatePolicy(policy.owners, policy.items,
                    policy.logits + noise, policy.batch_size).choose()
            else:
                following = policy.choose()
            env.step(following)
        if env.state.active.any():
            raise RuntimeError("Counterfactual continuation exceeded N steps")
        # Count newly selected items directly to avoid subtracting a large prefix.
        added = env.state.selected & ~branch.selected
        return (env.state.profits * added).sum(-1).reshape(actions.numel(), repeats)
    finally:
        model.train(was_training)


@torch.no_grad()
def paired_terminal_returns(model, state, left, right, repeats, stochastic, generator=None):
    """Legacy pair interface returns normalized *total* terminal profit."""
    owners = torch.arange(state.profits.shape[0], device=left.device).repeat_interleave(2)
    actions = torch.stack((left, right), -1).flatten()
    returns = candidate_terminal_returns(model, state, owners, actions, repeats, stochastic, generator)
    prefix = (state.profits * state.selected).sum(-1)
    total = (returns + prefix[owners, None]) / state.profits.sum(-1)[owners, None]
    return total.float().reshape(-1, 2, repeats).transpose(1, 2)


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
def collect_score_pair_batch(model, data, rollout, config, seed):
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
def action_comparisons(state, induced, greedy, sampled, config):
    """Compare every alternative with the current action, without a best-of-K label."""
    first = induced.counts.cumsum(0) - induced.counts
    ids = torch.arange(induced.items.numel(), device=induced.items.device)
    alternatives = ids[ids != first[induced.owners]]
    owners = induced.owners[alternatives]
    anchors = first[owners]
    scale = state.profits.sum(-1)[owners, None]
    # Compute differences in float64 before normalization, then use the same
    # paired SE/conflict filter as the previous score-pair supervision.
    greedy_diff = (greedy[anchors] - greedy[alternatives]) / scale
    sample_diff = (sampled[anchors] - sampled[alternatives]) / scale
    delta, se, keep = return_targets(torch.stack((greedy_diff, torch.zeros_like(greedy_diff)), -1),
                                    torch.stack((sample_diff, torch.zeros_like(sample_diff)), -1), config)
    return owners, anchors, alternatives, delta.float(), se.float(), keep


@torch.no_grad()
def collect_theta_ranking_batch(model, data, rollout, config, seed, audit_path=None):
    import time
    from dataclasses import fields
    from mkp_theta_candidates import propose_thetas, decode_proposals, deduplicate_actions, index_theta
    metrics = dict(ranking_proposed=0, ranking_pairs=0, ranking_abs_return_gap=0.,
                   ranking_standard_error=0., ranking_label_seconds=0., **dict.fromkeys(THETA_METRICS, 0.))
    if config.ranking_coefficient == 0:
        return None, metrics
    began = time.perf_counter()
    device = data[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    times, batches = rollout["active"].nonzero(as_tuple=True)
    if not times.numel():
        return None, metrics
    ids = torch.randperm(times.numel(), device=device, generator=generator)[:config.ranking_pool_size]
    b, t = batches[ids], times[ids]
    state = MKPState(data[0][b], data[1][b], data[2][b],
                     rollout["remaining"][t, b], rollout["selected"][t, b])
    eligible = (state.feasible.sum(-1) >= 2).nonzero(as_tuple=True)[0]
    state = state.index(eligible)
    source_times, source_batches = t[eligible], b[eligible]
    metrics["ranking_states_considered"] = state.profits.shape[0]
    if not state.profits.shape[0]:
        return None, metrics
    was_training = model.training
    model.eval()
    try:
        policy, theta, _ = model.policy(state)
        old = torch.zeros_like(state.profits, dtype=torch.float32)
        old[policy.owners, policy.items] = policy.log_probs
        proposals = propose_thetas(theta, model.config, config.ranking_theta_candidates,
                                   config.ranking_theta_local_scale, generator)
        decoded = decode_proposals(model, state, proposals)
        metrics["ranking_theta_evaluations"] = decoded.numel()
        # Prefer small current score margins, with 1/4 random visited states.
        raw = torch.zeros_like(old)
        raw[policy.owners, policy.items] = policy.logits * theta.temperature[policy.owners]
        gaps = raw.gather(1, decoded[:, :1]) - raw.gather(1, decoded)
        priority = gaps.masked_fill(decoded == decoded[:, :1], torch.inf).amin(-1)
        eligible = priority.isfinite().nonzero(as_tuple=True)[0]
        metrics["ranking_no_alternative"] = state.profits.shape[0] - eligible.numel()
        if not eligible.numel():
            return None, metrics
        ordering = eligible[priority[eligible].argsort()]
        count = min(config.ranking_states, ordering.numel())
        hard_count = max(1, 3 * count // 4)
        rest = ordering[hard_count:]
        random_ids = torch.randperm(rest.numel(), device=device, generator=generator)[:count - hard_count]
        chosen = torch.cat((ordering[:hard_count], rest[random_ids]))
        state, old, decoded = state.index(chosen), old[chosen], decoded[chosen]
        proposals = [index_theta(proposal, chosen) for proposal in proposals]
        induced = deduplicate_actions(decoded, config.ranking_max_actions, generator)
        metrics.update(ranking_states_labeled=count, ranking_unique_actions=induced.items.numel(),
                       ranking_action_diversity=induced.items.numel() / count,
                       ranking_rollout_branches=induced.items.numel() * (1 + config.ranking_repeats))
        continuation_rng = generator.get_state()
        greedy = candidate_terminal_returns(model, state, induced.owners, induced.items, 1, False)
        sampled = candidate_terminal_returns(model, state, induced.owners, induced.items,
                                             config.ranking_repeats, True, generator)
        owners, anchors, alternatives, delta, se, keep = action_comparisons(state, induced, greedy, sampled, config)
        metrics["ranking_proposed"] = alternatives.numel()
        metrics["ranking_pairs"] = int(keep.sum())
        improved = keep & (delta < 0)
        metrics["ranking_improved_pairs"] = int(improved.sum())
        metrics["ranking_improved_states"] = owners[improved].unique().numel()
        if improved.any():
            metrics["ranking_improvement_mean"] = (-delta[improved]).mean().item()
            metrics["ranking_raw_improvement_mean"] = (
                -delta[improved].double() * state.profits.sum(-1)[owners[improved]]).mean().item()
        batch = None
        if keep.any():
            metrics["ranking_abs_return_gap"] = delta[keep].abs().mean().item()
            metrics["ranking_standard_error"] = se[keep].mean().item()
            batch = RankingBatch(state.index(owners[keep]), induced.items[anchors[keep]],
                                 induced.items[alternatives[keep]], delta[keep], se[keep], old[owners[keep]])
        if audit_path is not None:
            from mkp_config import config_dict, CONTROLLER_VERSION, DECODER_VERSION
            from mkp_utils import atomic_checkpoint
            cpu = lambda x: x.detach().cpu()
            atomic_checkpoint(audit_path, dict(
                ranking_protocol=THETA_RANKING_VERSION, controller_version=CONTROLLER_VERSION,
                decoder_version=DECODER_VERSION, model_config=config_dict(model.config),
                feature_config=config_dict(model.feature_config),
                reuse_state_features=model.reuse_state_features,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                train_config=config_dict(config), seed=seed, torch_version=str(torch.__version__),
                device=str(device), continuation_rng_state=continuation_rng.cpu(),
                model_state={name: cpu(value) for name, value in model.state_dict().items()},
                state={name: cpu(getattr(state, name)) for name in
                       ("profits", "weights", "capacities", "remaining", "selected")},
                source_times=cpu(source_times[chosen]), source_batches=cpu(source_batches[chosen]),
                theta_proposals=[{field.name: cpu(getattr(proposal, field.name)) for field in fields(proposal)}
                                 for proposal in proposals],
                decoded_actions=cpu(decoded), owners=cpu(induced.owners), actions=cpu(induced.items),
                theta_witnesses=cpu(induced.witnesses), action_counts=cpu(induced.counts),
                greedy_q=cpu(greedy), sampled_q=cpu(sampled), anchor_indices=cpu(anchors),
                alternative_indices=cpu(alternatives), delta=cpu(delta), standard_error=cpu(se), keep=cpu(keep)))
        return batch, metrics
    finally:
        model.train(was_training)
        metrics["ranking_label_seconds"] = time.perf_counter() - began


def collect_ranking_batch(model, data, rollout, config, seed, audit_path=None):
    if config.ranking_mode == "score_pairs":
        batch, metrics = collect_score_pair_batch(model, data, rollout, config, seed)
        metrics.update(dict.fromkeys(THETA_METRICS, 0.))
        return batch, metrics
    return collect_theta_ranking_batch(model, data, rollout, config, seed, audit_path)


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
