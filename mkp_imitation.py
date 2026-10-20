"""Synthetic teacher collection and actor-only updates, never used at inference.

Exact labels may be replayed. An unproven improved plan is used only in the
collection phase that compared it to the current frozen greedy policy. Neither
kind of label becomes a PPO transition, a GAE return or an independent V target.
"""
import copy
from dataclasses import dataclass
import math
import time

import torch

from mkp_env import MKPEnv, MKPState
from mkp_teacher import solve_teacher


TEACHER_METRICS = ("teacher_labeled", "teacher_exact", "teacher_optimal", "teacher_incumbent",
    "teacher_ppo_states", "teacher_greedy_states", "teacher_plan_states", "teacher_early",
    "teacher_mid", "teacher_late", "teacher_seconds", "teacher_loss", "teacher_regret_before",
    "teacher_regret_after", "teacher_safe_before", "teacher_safe_after", "teacher_updates",
    "teacher_rejected", "teacher_replay_size", "teacher_kl", "teacher_guard_states", "teacher_exact_guard_states")


def empty_teacher_metrics():
    return dict.fromkeys(TEACHER_METRICS, 0.)


def copy_state(state, device="cpu"):
    return MKPState(*(x.detach().to(device).clone() for x in
        (state.profits, state.weights, state.capacities, state.remaining, state.selected)))


@dataclass
class TeacherExample:
    state: MKPState
    target: torch.Tensor  # [1,N], an optimal-action union or one feasible plan
    values: torch.Tensor  # [1,N], exact Q only where certified, NaN otherwise
    objective: float
    certified: bool
    source: str
    origin: str
    stage: int

    def pack(self):
        result = dict(target=self.target, values=self.values, objective=self.objective,
                      certified=self.certified, source=self.source, origin=self.origin, stage=self.stage)
        result["state"] = [getattr(self.state, name) for name in
                           ("profits", "weights", "capacities", "remaining", "selected")]
        return result

    @classmethod
    def unpack(cls, payload):
        return cls(MKPState(*payload["state"]), *(payload[key] for key in
            ("target", "values", "objective", "certified", "source", "origin", "stage")))


def collate(examples, device):
    state = MKPState(*(torch.cat([getattr(x.state, name) for x in examples]).to(device) for name in
                      ("profits", "weights", "capacities", "remaining", "selected")))
    return (state, torch.cat([x.target for x in examples]).to(device),
            torch.cat([x.values for x in examples]).to(device),
            torch.tensor([x.objective for x in examples], device=device, dtype=torch.float64))


def teacher_objective(model, examples, config):
    """Safe-set log mass over ALL feasible raw scores; temperature cannot cheat.

    Exact all-action values also penalize expected regret. Sparse MILP values
    are never filled with invented zeros or treated as full Q supervision.
    """
    device = next(model.actor.parameters()).device
    state, targets, values, objectives = collate(examples, device)
    policy, _, _ = model.policy(state)
    scores = policy.score_terms["score"] / config.teacher_score_scale
    losses, regrets, safe, row_logs, raw_logs = [], [], [], [], []
    actions = policy.choose()
    for row in range(len(examples)):
        ids = (policy.owners == row).nonzero(as_tuple=True)[0]
        items = policy.items[ids]
        wanted = targets[row, items]
        if ids.numel() == 0 or not wanted.any():
            raise ValueError("Teacher row must have at least one feasible target")
        logp = scores[ids].log_softmax(0)
        loss = -torch.logsumexp(logp[wanted], 0)
        q = values[row, items]
        full_exact = bool(torch.isfinite(q).all())
        if full_exact:
            regret = ((objectives[row] - q).clamp_min(0) / objectives[row].clamp_min(1e-12)).float()
            loss = loss + (logp.exp() * regret).sum()
            chosen_q = values[row, actions[row]]
            regrets.append(((objectives[row] - chosen_q).clamp_min(0)
                            / objectives[row].clamp_min(1e-12)).float())
        else:
            regrets.append(scores.new_tensor(float("nan")))
        losses.append(loss)
        safe.append(targets[row, actions[row]])
        row_logs.append(policy.log_probs[ids])
        raw_logs.append(logp)
    return dict(loss=torch.stack(losses).mean(), regrets=torch.stack(regrets),
                safe=torch.stack(safe), log_probs=row_logs, raw_log_probs=raw_logs,
                actions=actions)


def _detach_metrics(metrics):
    return {key: ([x.detach().clone() for x in value] if isinstance(value, list)
                  else value.detach().clone()) for key, value in metrics.items()}


def _kl(reference, current):
    return torch.stack([(a.exp() * (a - b)).sum() for a, b in zip(reference, current)]).mean().clamp_min(0)


def _mean_regret(values):
    known = torch.isfinite(values)
    return values[known].mean().item() if known.any() else 0.


def acceptable_step(before, after, anchor, max_kl):
    """Local guard only; this is not a claim about unseen-state generalization."""
    if not torch.isfinite(after["loss"]):
        return False, float("inf")
    kl = _kl(anchor["log_probs"], after["log_probs"])
    raw_kl = _kl(anchor["raw_log_probs"], after["raw_log_probs"])
    combined = max(kl.item(), raw_kl.item())
    exact = torch.isfinite(before["regrets"])
    worsened = bool((after["regrets"][exact] > before["regrets"][exact] + 1e-8).any())
    lost_safe = bool((before["safe"] & ~after["safe"]).any())
    improves_loss = bool(after["loss"] <= before["loss"] + 1e-7)
    return (math.isfinite(combined) and combined <= max_kl and not worsened
            and not lost_safe and improves_loss), combined


def _sample_states(data, rollout, count, generator, origin):
    """Stratify by actual episode progress, not by absolute step or item count."""
    if count <= 0:
        return []
    times, rows = rollout["active"].nonzero(as_tuple=True)
    if not times.numel():
        return []
    lengths = rollout["active"].sum(0).clamp_min(1)
    progress = times.float() / lengths[rows]
    phases = (progress * 3).long().clamp_max(2)
    used, examples = set(), []
    for index in range(count):
        phase = index % 3
        candidates = (phases == phase).nonzero(as_tuple=True)[0]
        if not candidates.numel():
            candidates = torch.arange(times.numel(), device=times.device)
        order = candidates[torch.randperm(candidates.numel(), device=times.device, generator=generator)]
        selected = next((int(i) for i in order.tolist() if int(i) not in used), None)
        if selected is None:
            continue
        used.add(selected)
        t, b = int(times[selected]), int(rows[selected])
        state = MKPState(data[0][b:b+1], data[1][b:b+1], data[2][b:b+1],
                         rollout["remaining"][t, b:b+1], rollout["selected"][t, b:b+1])
        examples.append((copy_state(state, data[0].device), origin, int(phases[selected])))
    return examples


@torch.no_grad()
def _greedy_states(model, data, count, generator):
    if count <= 0:
        return []
    rows = torch.randperm(data[0].shape[0], device=data[0].device, generator=generator)[:count]
    subdata = tuple(x[rows] for x in data)
    env = MKPEnv(*subdata)
    records = {key: [] for key in ("active", "remaining", "selected")}
    for _ in range(data[0].shape[1]):
        if not env.state.active.any():
            break
        for key in records:
            records[key].append(getattr(env.state, key).clone())
        env.step(model.policy(env.state)[0].choose())
    if not records["active"]:
        return []
    return _sample_states(subdata, {key: torch.stack(value) for key, value in records.items()},
                          count, generator, "greedy")


@torch.no_grad()
def _complete_greedy(model, state):
    env = MKPEnv.from_state(state)
    for _ in range(state.profits.shape[1]):
        if not env.state.active.any():
            break
        env.step(model.policy(env.state)[0].choose())
    if env.state.active.any():
        raise RuntimeError("Teacher baseline exceeded N steps")
    return env.state.selected & ~state.selected


class TeacherTrainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.optimizer = torch.optim.AdamW(model.actor.parameters(), lr=config.teacher_learning_rate,
                                           weight_decay=0.)
        self.replay = []

    def state_dict(self):
        return dict(version=1, optimizer=self.optimizer.state_dict(),
                    replay=[example.pack() for example in self.replay])

    def load_state_dict(self, payload):
        if payload.get("version") != 1:
            raise ValueError("Incompatible teacher replay/optimizer state")
        self.optimizer.load_state_dict(payload["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.teacher_learning_rate
        self.replay = [TeacherExample.unpack(example) for example in payload["replay"]]
        if any(not example.certified for example in self.replay):
            raise ValueError("Uncertified stale plans cannot be replayed")
        self.replay = self.replay[-self.config.teacher_replay_size:]

    @torch.no_grad()
    def collect(self, data, rollout, update, metrics):
        c, model = self.config, self.model
        generator = torch.Generator(device=data[0].device).manual_seed(c.seed + 104729 * update + data[2].shape[-1])
        count = (c.teacher_states + 1) // 2
        states = _sample_states(data, rollout, count, generator, "ppo")
        states += _greedy_states(model, data, c.teacher_states - count, generator)
        if not states:
            return []
        combined = MKPState(*(torch.cat([getattr(state, name) for state, _, _ in states]) for name in
                              ("profits", "weights", "capacities", "remaining", "selected")))
        baselines = _complete_greedy(model, combined)
        examples = []
        for row, (state, origin, stage) in enumerate(states):
            metrics["teacher_" + origin + "_states"] += 1
            metrics[("teacher_early", "teacher_mid", "teacher_late")[stage]] += 1
            result = solve_teacher(state, c, baselines[row])
            if result is None or not result.target_mask.any():
                continue
            example = TeacherExample(copy_state(state), result.target_mask[None].cpu(),
                result.action_values[None].cpu(), result.objective, result.optimality_proven,
                result.source, origin, stage)
            examples.append(example)
            metrics["teacher_labeled"] += 1
            kind = {"exact_enumeration": "exact", "milp_optimal": "optimal", "milp_incumbent": "incumbent"}[result.source]
            metrics["teacher_" + kind] += 1
            # A feasible plan is distinct from the UNION of optimal first actions.
            # Teach a bounded number of actual prefixes to connect first-action
            # supervision to completing that plan. Certified plan tails remain optimal.
            if result.selected_mask is None or c.teacher_plan_prefixes == 0:
                continue
            plan = result.selected_mask.to(state.profits.device).clone()
            env = MKPEnv.from_state(state)
            total = int(plan.sum())
            # Any ordering of a verified feasible set is feasible. A local random
            # order covers different prefixes without another policy rollout.
            plan_ids = plan.nonzero(as_tuple=True)[0]
            order = plan_ids[torch.randperm(total, device=plan.device, generator=generator)]
            stride = max(1, total // (c.teacher_plan_prefixes + 1))
            kept = 0
            for step in range(max(0, total - 1)):
                chosen = order[step]
                env.step(chosen.reshape(1))
                plan[chosen] = False
                if (step + 1) % stride or kept >= c.teacher_plan_prefixes:
                    continue
                objective = float(env.state.profits[0, plan].sum())
                values = torch.full_like(env.state.profits.cpu(), float("nan"))
                if result.optimality_proven:
                    values[0, plan.cpu()] = objective
                examples.append(TeacherExample(copy_state(env.state), plan[None].cpu().clone(), values,
                    objective, result.optimality_proven, result.source, "plan", stage))
                kept += 1
                metrics["teacher_plan_states"] += 1
                if kept == c.teacher_plan_prefixes:
                    break
        return examples

    def update(self, data, rollout, update):
        c, model = self.config, self.model
        metrics = empty_teacher_metrics()
        if c.supervision_mode != "teacher" or c.teacher_steps == 0 or update % c.teacher_every:
            return metrics
        began, was_training = time.perf_counter(), model.training
        model.eval()
        try:
            fresh = self.collect(data, rollout, update, metrics)
            shape = data[0].shape[1], data[2].shape[-1]
            prior = [x for x in self.replay if (x.state.profits.shape[1], x.state.capacities.shape[-1]) == shape]
            # Fresh uncertified targets are not retained beyond this phase.
            candidates = fresh + prior
            self.replay.extend(x for x in fresh if x.certified)
            self.replay = self.replay[-c.teacher_replay_size:]
            metrics["teacher_replay_size"] = len(self.replay)
            if not candidates:
                return metrics
            rng = torch.Generator().manual_seed(c.seed + 65537 * update + shape[1])
            # Bound guard cost and include both fresh and older certified decisions.
            limit = c.teacher_minibatch_size
            fresh_ids = torch.randperm(len(fresh), generator=rng)[:max(1, limit // 2)].tolist()
            guard = [fresh[i] for i in fresh_ids]
            rest = prior if prior else [x for i, x in enumerate(fresh) if i not in fresh_ids]
            ids = torch.randperm(len(rest), generator=rng)[:max(0, limit - len(guard))].tolist()
            guard += [rest[i] for i in ids]
            if not guard:
                return metrics
            with torch.no_grad():
                anchor = _detach_metrics(teacher_objective(model, guard, c))
            before = anchor
            metrics["teacher_guard_states"] = len(guard)
            metrics["teacher_exact_guard_states"] = int(torch.isfinite(before["regrets"]).sum())
            metrics["teacher_regret_before"] = _mean_regret(before["regrets"])
            metrics["teacher_safe_before"] = before["safe"].float().mean().item()
            for _ in range(c.teacher_steps):
                weights = {key: value.detach().clone() for key, value in model.actor.state_dict().items()}
                optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.optimizer.zero_grad(set_to_none=True)
                current = teacher_objective(model, guard, c)
                if not torch.isfinite(current["loss"]):
                    raise FloatingPointError("Non-finite teacher loss")
                current["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), c.teacher_max_grad_norm,
                                               error_if_nonfinite=True)
                try:
                    self.optimizer.step()
                    with torch.no_grad():
                        after = _detach_metrics(teacher_objective(model, guard, c))
                except Exception:
                    model.actor.load_state_dict(weights, strict=True)
                    self.optimizer.load_state_dict(optimizer_state)
                    raise
                accepted, kl = acceptable_step(before, after, anchor, c.teacher_max_kl)
                if accepted:
                    before = after
                    metrics["teacher_updates"] += 1
                    metrics["teacher_kl"] = kl
                else:
                    model.actor.load_state_dict(weights, strict=True)
                    self.optimizer.load_state_dict(optimizer_state)
                    metrics["teacher_rejected"] += 1
                    break
            metrics["teacher_loss"] = before["loss"].item()
            metrics["teacher_regret_after"] = _mean_regret(before["regrets"])
            metrics["teacher_safe_after"] = before["safe"].float().mean().item()
            self.optimizer.zero_grad(set_to_none=True)
            return metrics
        finally:
            metrics["teacher_seconds"] = time.perf_counter() - began
            model.train(was_training)
