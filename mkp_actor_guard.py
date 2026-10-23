"""Opt-in Actor step controls on training-rollout prefixes only.

The guard observes complete greedy suffixes through the existing policy and
environment. It supplies no labels, PPO samples, rewards, or critic targets.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from mkp_env import MKPEnv, MKPState


ACTOR_GUARD_FIELDS = (
    "actor_guard_enabled", "actor_guard_states", "actor_guard_before_return",
    "actor_guard_after_return", "actor_guard_delta", "actor_guard_rejected",
    "actor_guard_max_kl", "actor_guard_kl_rejected", "actor_guard_return_rejected",
    "actor_guard_min_delta", "actor_guard_worsened_states",
)


def build_optimizer(model, config):
    """Preserve legacy AdamW unless a separate Actor learning rate is requested."""
    actor_rate = getattr(config, "actor_learning_rate", None)
    if actor_rate is None:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    if not math.isfinite(actor_rate) or actor_rate <= 0:
        raise ValueError("Actor learning rate must be finite and positive")
    actor = list(model.actor.parameters())
    actor_ids = {id(parameter) for parameter in actor}
    critic = [parameter for parameter in model.parameters() if id(parameter) not in actor_ids]
    return torch.optim.AdamW([
        dict(params=actor, lr=actor_rate, name="actor"),
        dict(params=critic, lr=config.learning_rate, name="critic"),
    ], weight_decay=0.)


def _prefix_indices(active, budget):
    """Deterministic instance strata with early/middle/late relative prefixes.

    Prefer different instances when possible. If the budget exceeds the number
    of active instances, distribute it round-robin and spread each instance's
    selections over its own episode. No RNG, labels, or heldout data are read.
    """
    by_batch = {}
    for time, batch in active.nonzero(as_tuple=False).tolist():
        by_batch.setdefault(batch, []).append(time)
    batches = sorted(by_batch)
    count = min(budget, sum(map(len, by_batch.values())))
    if not count:
        return ()
    if len(batches) >= count:
        result = []
        for rank in range(count):
            batch = batches[(2 * rank + 1) * len(batches) // (2 * count)]
            times = by_batch[batch]
            stage = rank % 3
            result.append((times[(2 * stage + 1) * len(times) // 6], batch))
        return tuple(result)
    quotas = {batch: 0 for batch in batches}
    while sum(quotas.values()) < count:
        for batch in batches:
            if quotas[batch] < len(by_batch[batch]):
                quotas[batch] += 1
                if sum(quotas.values()) == count:
                    break
    return tuple((by_batch[batch][(2 * rank + 1) * len(by_batch[batch]) // (2 * quota)], batch)
                 for batch, quota in quotas.items() for rank in range(quota))


@contextmanager
def _probe_mode(model, device):
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        # Deterministic production policies use no RNG. Preserve RNG even if a
        # future eval-time module performs an incidental random operation.
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


class ActorUpdateGuard:
    """Reject Actor drift while retaining the executed independent Critic step.

    Mean non-regression allows individual prefixes to worsen; the metrics expose
    their count and the smallest per-prefix change. Construct before the PPO
    minibatches and call finish after them. Disabled
    construction/finish do not inspect model, optimizer, data, or rollout.
    """
    def __init__(self, model, optimizer, data, rollout, config):
        self.enabled = bool(getattr(config, "actor_continuation_guard", False))
        self._metrics = dict.fromkeys(ACTOR_GUARD_FIELDS, 0)
        self._metrics["actor_guard_enabled"] = int(self.enabled)
        self._finished = False
        if not self.enabled:
            return
        self.model, self.optimizer = model, optimizer
        self.tolerance = config.actor_guard_tolerance
        self.max_kl = config.target_kl
        self.prefix_indices = _prefix_indices(rollout["active"], config.actor_guard_states)
        self._metrics["actor_guard_states"] = len(self.prefix_indices)
        if not self.prefix_indices:
            self.state = None
            return
        times, batches = zip(*self.prefix_indices)
        device = data[0].device
        times = torch.tensor(times, device=device, dtype=torch.long)
        batches = torch.tensor(batches, device=device, dtype=torch.long)
        self.state = MKPState(
            *(value[batches].detach().double().clone() for value in data),
            rollout["remaining"][times, batches].detach().double().clone(),
            rollout["selected"][times, batches].detach().clone(),
        )
        self.actor_state = {key: value.detach().clone() for key, value in model.actor.state_dict().items()}
        self.actor_parameters = list(model.actor.parameters())
        self.actor_optimizer_state = {
            parameter: deepcopy(optimizer.state[parameter]) if parameter in optimizer.state else None
            for parameter in self.actor_parameters
        }
        self.before, self.old_distribution, self.before_rows = self._measure()
        self._metrics["actor_guard_before_return"] = self.before

    @torch.no_grad()
    def _measure(self):
        with _probe_mode(self.model, self.state.profits.device):
            env = MKPEnv.from_state(self.state)
            distribution = None
            for _ in range(self.state.profits.shape[1]):
                if not env.state.active.any():
                    break
                policy = self.model.policy(env.state)[0]
                if distribution is None:
                    distribution = tuple(value.detach().clone() for value in
                                         (policy.owners, policy.items, policy.log_probs.double()))
                if not torch.isfinite(policy.log_probs).all():
                    return float("nan"), distribution, self.state.profits.new_full((len(self.prefix_indices),), float("nan"))
                env.step(policy.choose(stochastic=False))
            if env.state.active.any():
                raise RuntimeError("Actor guard continuation exceeded N decisions")
            # Sum newly selected profits directly in float64, avoiding both
            # float32 rewards and cancellation against a large selected prefix.
            suffix = (self.state.profits * (env.state.selected & ~self.state.selected)).sum(-1)
            normalized = suffix / self.state.profits.sum(-1)
            return float(normalized.mean()), distribution, normalized.detach().clone()

    def _distribution_kl(self, distribution):
        if distribution is None or self.old_distribution is None:
            return float("nan")
        old_owners, old_items, old_log_probs = self.old_distribution
        owners, items, log_probs = distribution
        if not torch.equal(owners, old_owners) or not torch.equal(items, old_items):
            raise ValueError("Actor guard feasible action support changed on a fixed prefix")
        if not torch.isfinite(old_log_probs).all() or not torch.isfinite(log_probs).all():
            return float("nan")
        kl = log_probs.new_zeros(len(self.prefix_indices))
        kl.index_add_(0, owners, old_log_probs.exp() * (old_log_probs - log_probs))
        return float(kl.clamp_min(0).max())

    def _restore(self):
        self.model.actor.load_state_dict(self.actor_state, strict=True)
        for parameter, old_state in self.actor_optimizer_state.items():
            if old_state is None:
                self.optimizer.state.pop(parameter, None)
            else:
                self.optimizer.state[parameter] = deepcopy(old_state)

    def finish(self):
        if self._finished or not self.enabled:
            return dict(self._metrics)
        self._finished = True
        if self.state is None:
            return dict(self._metrics)
        finite_actor = all(torch.isfinite(value).all() for value in self.model.actor.state_dict().values())
        after, distribution, after_rows = (self._measure() if finite_actor else
            (float("nan"), None, torch.full_like(self.before_rows, float("nan"))))
        delta = after - self.before
        row_deltas = after_rows - self.before_rows
        kl = self._distribution_kl(distribution)
        return_rejected = not math.isfinite(delta) or delta < -self.tolerance
        kl_rejected = not math.isfinite(kl) or kl > self.max_kl
        rejected = return_rejected or kl_rejected
        self._metrics.update(actor_guard_after_return=after, actor_guard_delta=delta,
            actor_guard_max_kl=kl, actor_guard_rejected=int(rejected),
            actor_guard_kl_rejected=int(kl_rejected), actor_guard_return_rejected=int(return_rejected),
            actor_guard_min_delta=float(row_deltas.min()),
            actor_guard_worsened_states=int((row_deltas < 0).sum()))
        if rejected:
            self._restore()
        return dict(self._metrics)
