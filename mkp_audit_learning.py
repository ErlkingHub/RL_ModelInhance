"""User-run diagnostic fits and production-PPO retention experiments.

Only independent synthetic bank.fit states are fitted. Heldout labels are never
used for updates. Output actor artifacts are deliberately not production
checkpoints and cannot be loaded by Benchmark_zeroShot.
"""
from dataclasses import fields, replace
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from mkp_config import TrainConfig, GeneratorConfig
from mkp_data import synthetic_batch
from mkp_performance import configure_training_runtime
from mkp_Trainer import MKPTrainer, collect_rollout
from mkp_audit_common import (check_report, evaluate_records, source_model,
                              summarize_rows, write_json, diagnostic_provenance, greedy_future)
from mkp_audit_objectives import action_fit_terms, fit_candidate_key
from mkp_utils import file_sha256, seed_everything


def set_action_loss(model, record, device, score_scale=.2):
    """Temperature-free set imitation, not a claim about unknown action values."""
    from mkp_audit_bank import state_from_record
    policy = model.policy(state_from_record(record, device))[0]
    targets = torch.isin(policy.items, record["target_actions"].to(policy.items.device))
    if not targets.any():
        raise ValueError("Diagnostic target is infeasible")
    scores = policy.score_terms["score"] / score_scale
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores[targets], 0)


def eligible_records(bank, reachability, *, scope="both"):
    if scope not in ("exact", "finite_greedy", "both"):
        raise ValueError("Invalid diagnostic label scope")
    reached = {r["id"] for r in reachability["records"] if r["status"] == "reachable"}
    return [r for r in bank["records"] if r["id"] in reached
            and (scope == "both" or r["label_kind"] == scope)]


def run_learning(bank_path, reach_path, output, *, device="cpu", steps=100,
                 learning_rate=1e-4, batch_size=8, seed=4729, scope="both",
                 objective="regret_margin", anchor_coefficient=.1, max_kl=.03, evaluate_every=10):
    from mkp_audit_bank import load_bank, state_from_record
    if (steps < 0 or learning_rate <= 0 or batch_size < 1 or objective not in ("set_ce", "regret_margin")
            or not math.isfinite(anchor_coefficient) or anchor_coefficient < 0
            or not math.isfinite(max_kl) or max_kl <= 0 or evaluate_every < 1):
        raise ValueError("Invalid diagnostic fit budget")
    bank = load_bank(bank_path)
    reach = json.loads(Path(reach_path).read_text(encoding="utf-8"))
    check_report(bank_path, reach)
    records = eligible_records(bank, reach, scope=scope)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = {}
    for m in sorted({r["M"] for r in records}):
        seed_everything(seed + m)
        model, _ = source_model(bank, m, device)
        model.requires_grad_(False)
        model.actor.requires_grad_(True)
        local = [r for r in records if r["M"] == m]
        fit = [r for r in local if r["split"] == "fit"]
        if not fit:
            reports[str(m)] = dict(status="no_reachable_fit_states", fit_states=0)
            continue
        before = evaluate_records(model, local, device)
        anchors = {}
        with torch.no_grad():
            for record in fit:
                policy = model.policy(state_from_record(record, device))[0]
                anchors[record["id"]] = dict(items=policy.items.detach().cpu(), log_probs=policy.log_probs.detach().cpu())

        def terms(record):
            policy = model.policy(state_from_record(record, device))[0]
            return action_fit_terms(policy, record, objective=objective, anchor=anchors[record["id"]])

        @torch.no_grad()
        def measure(step, rows=None):
            model.eval()
            values = [terms(record) for record in fit]
            pair_mean = sum(float(pair) for pair, _ in values) / len(values)
            mean_kl = sum(float(kl) for _, kl in values) / len(values)
            largest_kl = max(float(kl) for _, kl in values)
            if rows is None and largest_kl <= max_kl:
                rows = evaluate_records(model, fit, device)
            key = fit_candidate_key(rows, pair_mean, largest_kl, max_kl) if rows else None
            wrong_before = {r["id"] for r in before if r["split"] == "fit" and not r["target_hit"]}
            log = dict(step=step, label_loss=pair_mean, mean_behavior_kl=mean_kl, max_behavior_kl=largest_kl,
                measured_continuations=rows is not None, selection_key=key,
                target_hit_rate=sum(r["target_hit"] for r in rows) / len(rows) if rows else None,
                corrected_initial_errors=sum(r["target_hit"] and r["id"] in wrong_before for r in rows) if rows else None,
                lost_initial_correct=sum(not r["target_hit"] and r["id"] not in wrong_before for r in rows) if rows else None,
                unknown_actions=sum(not r["action_value_known"] for r in rows) if rows else None)
            return key, log

        best_key, initial = measure(0, [r for r in before if r["split"] == "fit"])
        if best_key is None:
            raise FloatingPointError("Source actor has invalid diagnostic loss or behavior KL")
        best_actor = {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()}
        selected_step, stop_reason = 0, "budget_completed"
        fit_history = [initial]
        optimizer = torch.optim.AdamW(model.actor.parameters(), lr=learning_rate, weight_decay=0.)
        rng = np.random.default_rng(seed + m)
        losses = []
        model.train()
        for step in range(steps):
            ids = rng.choice(len(fit), size=min(batch_size, len(fit)), replace=False)
            pieces = [terms(fit[int(i)]) for i in ids]
            loss = torch.stack([pair + (anchor_coefficient * kl if objective == "regret_margin" else 0)
                                for pair, kl in pieces]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic fit loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(dict(step=step + 1, loss=float(loss.detach()), pre_clip_grad_norm=float(grad)))
            if step == 0 or (step + 1) % 25 == 0:
                print(f"learn M={m} step={step + 1}/{steps} loss={float(loss.detach()):.6g}", flush=True)
            if objective == "regret_margin" and ((step + 1) % evaluate_every == 0 or step + 1 == steps):
                key, measured = measure(step + 1)
                if key is not None and key > best_key:
                    best_key, selected_step = key, step + 1
                    best_actor = {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()}
                    measured["selected"] = True
                fit_history.append(measured)
                if measured["max_behavior_kl"] > max_kl:
                    stop_reason = "fit_behavior_kl_budget_exceeded"
                    break
                model.train()
        if objective == "regret_margin":
            model.actor.load_state_dict(best_actor, strict=True)
        else:
            selected_step = steps  # Explicit reproduction of the original last-iterate CE fit.
        after = evaluate_records(model, local, device)
        actor_file = output / f"actor_M{m}.pt"
        torch.save(dict(artifact_type="diagnostic_actor_fit", schema_version=1,
            bank_sha256=file_sha256(bank_path), source_checkpoint_sha256=bank["checkpoints"][str(m)]["sha256"],
            num_resources=m, fit_record_ids=[r["id"] for r in fit], label_scope=scope,
            fit_objective=objective, selected_step=selected_step,
            actor_state={k: v.detach().cpu() for k, v in model.actor.state_dict().items()}), actor_file)
        old_by_id = {r["id"]: r for r in before}
        transitions = {}
        for split in ("fit", "heldout"):
            part = [r for r in after if r["split"] == split]
            wrong = [r for r in part if not old_by_id[r["id"]]["target_hit"]]
            transitions[split] = dict(initial_errors=len(wrong), corrected_initial_errors=sum(r["target_hit"] for r in wrong),
                lost_initial_correct=sum(not r["target_hit"] and old_by_id[r["id"]]["target_hit"] for r in part))
        reports[str(m)] = dict(status="complete", fit_states=len(fit), heldout_states=len(local)-len(fit),
            before=before, after=after, before_summary=summarize_rows(before), after_summary=summarize_rows(after),
            target_transitions=transitions, loss_history=losses,
            fit_history=fit_history, selected_step=selected_step, stop_reason=stop_reason,
            selection=("fit normalized greedy continuation, then hit rate and label loss; source actor included"
                       if objective == "regret_margin" else "legacy last iterate"),
            actor_artifact=str(actor_file.resolve()), actor_sha256=file_sha256(actor_file))
    report = dict(schema_version=1, bank_sha256=file_sha256(bank_path), phase="controller_fit",
        data_source="synthetic_only", elapsed_seconds=time.perf_counter()-started,
        budget=dict(steps=steps, learning_rate=learning_rate, batch_size=batch_size, seed=seed, scope=scope,
                    objective=objective, anchor_coefficient=anchor_coefficient, max_kl=max_kl, evaluate_every=evaluate_every),
        source_hashes=diagnostic_provenance(),
        warning="Diagnostic actor fit; no Chu training and no claim about benchmark gap. Finite labels are candidate improvement imitation, not Q*.",
        models=reports)
    write_json(output / "learning.json", report)
    return report


class GradientObserver:
    """One real production minibatch per update; autograd.grad leaves .grad intact."""
    def __init__(self, model):
        self.parameters = [p for p in model.actor.parameters() if p.requires_grad]
        self.critic_parameters = ([p for name, p in model.named_parameters()
                                   if not name.startswith("actor.") and p.requires_grad]
                                  if hasattr(model, "named_parameters") else [])
        self.rows = []
        self.stops = []
        self.captured = False

    def __call__(self, event):
        if event["event"] == "kl_stop":
            self.stops.append({k: v for k, v in event.items() if k != "event"})
            return
        if self.captured or event["event"] != "before_backward":
            return
        self.captured = True
        vectors = {}
        for name in ("policy", "ranking", "feasibility", "entropy"):
            loss = event["losses"].get(name)
            if loss is None or not loss.requires_grad:
                continue
            gradients = torch.autograd.grad(loss, self.parameters, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                       for p, g in zip(self.parameters, gradients)])
        norms = {name: float(value.norm()) for name, value in vectors.items()}
        combined = float(torch.stack(list(vectors.values())).sum(0).norm()) if vectors else 0.
        critic_norm = None
        value_loss = event["losses"].get("value")
        if self.critic_parameters and value_loss is not None and value_loss.requires_grad:
            gradients = torch.autograd.grad(value_loss, self.critic_parameters, retain_graph=True, allow_unused=True)
            critic_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
        cosine = None
        if "policy" in vectors and "ranking" in vectors:
            denominator = vectors["policy"].norm() * vectors["ranking"].norm()
            if denominator > 0:
                cosine = float(torch.dot(vectors["policy"], vectors["ranking"]) / denominator)
        self.rows.append(dict(actor_gradient_norms=norms, actor_combined_gradient_norm=combined,
                             critic_value_gradient_norm=critic_norm, policy_ranking_cosine=cosine,
                             scope="first executed production minibatch of this PPO update"))


def retention_metrics(before, after):
    old = {r["id"]: r for r in before}
    if len(old) != len(before) or len(after) != len(before) or {r["id"] for r in after} != set(old):
        raise ValueError("Retention snapshots refer to different states")
    result = {}
    for split in ("fit", "heldout"):
        rows = [r for r in after if r["split"] == split]
        if not rows:
            continue
        correct = [r for r in rows if old[r["id"]]["target_hit"]]
        deltas = [r["greedy_future"] - old[r["id"]]["greedy_future"] for r in rows]
        result[split] = dict(states=len(rows), initially_correct=len(correct),
            target_retention_rate=(sum(r["target_hit"] for r in correct) / len(correct) if correct else None),
            current_target_hit_rate=sum(r["target_hit"] for r in rows)/len(rows),
            mean_future_delta_from_fit=sum(deltas)/len(deltas),
            worse_future_states=sum(d < -1e-8 for d in deltas))
    return result


@contextmanager
def _measurement_rng(device):
    """Independent validation must not perturb subsequent PPO randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _instance_tensor_sha(record):
    """Fingerprint generated instance data, excluding its mutable prefix."""
    digest = hashlib.sha256()
    for name in ("profits", "weights", "capacities"):
        value = record["state"][name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_independent_bank(bank, validation):
    """Reject reused generation identities/data even when record IDs differ.

    A normal bank contains fit/heldout splits; here *all* of its instances are
    measurement-only heldout data. No reachability result is consulted.
    """
    if not isinstance(bank.get("seed"), int) or not isinstance(validation.get("seed"), int):
        raise ValueError("Independent validation requires explicit integer bank seeds")
    if bank["seed"] == validation["seed"]:
        raise ValueError("Independent validation must use a different bank seed")
    if not bank.get("records") or not validation.get("records"):
        raise ValueError("Independent validation requires nonempty banks")
    scope = lambda b: {(r["M"], r["N"]) for r in b["records"]}
    if scope(bank) != scope(validation):
        raise ValueError("Independent validation must have the same M/N scope")
    for m, _ in scope(bank):
        source = bank["checkpoints"].get(str(m), {}).get("sha256")
        other = validation["checkpoints"].get(str(m), {}).get("sha256")
        if not source or source != other:
            raise ValueError("Independent validation must use the same source checkpoints")

    def identities(value):
        instances = {}
        for row in value["records"]:
            entropy = row.get("generation_entropy")
            if (not isinstance(entropy, (list, tuple)) or not entropy
                    or any(not isinstance(x, int) for x in entropy) or entropy[0] != value["seed"]):
                raise ValueError("Independent validation requires explicit consistent generation entropy")
            identity = (tuple(entropy), _instance_tensor_sha(row))
            old = instances.setdefault(row["instance_id"], identity)
            if old != identity:
                raise ValueError("An instance has inconsistent generation entropy or tensors")
        return {v[0] for v in instances.values()}, {v[1] for v in instances.values()}, set(instances)

    entropy, tensors, ids = identities(bank)
    other_entropy, other_tensors, other_ids = identities(validation)
    if entropy & other_entropy or tensors & other_tensors or ids & other_ids:
        raise ValueError("Independent validation overlaps fit-bank generation identities or instance tensors")
    return dict(seed=validation["seed"], records=len(validation["records"]), instances=len(other_ids),
                instance_tensor_sha256=sorted(other_tensors),
                scope="All independent-bank instances are measurement-only heldout; no reachability filtering")


@torch.no_grad()
def evaluate_validation_instances(model, records, device):
    """Full greedy objective from each unique instance's empty initial state."""
    from mkp_env import MKPEnv
    unique = {r["instance_id"]: r for r in records}
    result = []
    for instance_id, record in sorted(unique.items()):
        data = [record["state"][name].detach().clone().to(device)
                for name in ("profits", "weights", "capacities")]
        env = MKPEnv(*data)
        objective = greedy_future(model, env.state)
        result.append(dict(instance_id=instance_id, M=record["M"], N=record["N"],
                           capacity_ratio=record["capacity_ratio"], objective=objective,
                           total_initial_profit=float(data[0].sum())))
    return result


def _measure_validation(model, records, device):
    modes = [(module, module.training) for module in model.modules()]
    try:
        with _measurement_rng(device):
            rows = evaluate_records(model, records, device)
            original_splits = {r["id"]: r.get("validation_original_split", r["split"]) for r in records}
            rows = [dict(r, validation_original_split=original_splits[r["id"]]) for r in rows]
            return dict(states=rows,
                        instances=evaluate_validation_instances(model, records, device))
    finally:
        for module, training in modes:
            module.training = training


def _relative_validation(reference, current):
    old = {r["id"]: r for r in reference}
    if len(old) != len(reference) or len(current) != len(old) or {r["id"] for r in current} != set(old):
        raise ValueError("Independent-validation snapshots refer to different states")
    differences = [r["greedy_future"] - old[r["id"]]["greedy_future"] for r in current]
    correct = [r for r in current if old[r["id"]]["target_hit"]]
    regrets = [(r["exact_action_regret"], old[r["id"]]["exact_action_regret"])
               for r in current if r.get("exact_action_regret") is not None
               and old[r["id"]].get("exact_action_regret") is not None]
    return dict(states=len(current), initially_correct=len(correct),
                retained_targets=sum(r["target_hit"] for r in correct),
                target_retention_rate=sum(r["target_hit"] for r in correct) / len(correct) if correct else None,
                corrected_targets=sum(r["target_hit"] and not old[r["id"]]["target_hit"] for r in current),
                lost_targets=sum(not r["target_hit"] for r in correct),
                mean_future_delta=sum(differences) / len(current),
                mean_normalized_future_delta=sum(r["greedy_delta_over_initial_profit"]
                    - old[r["id"]]["greedy_delta_over_initial_profit"] for r in current) / len(current),
                profit_dropout_states=sum(d < -1e-8 for d in differences),
                profit_improved_states=sum(d > 1e-8 for d in differences),
                exact_states=len(regrets),
                mean_exact_action_regret_change=sum(a - b for a, b in regrets) / len(regrets) if regrets else None,
                exact_regret_increased_states=sum(a > b + 1e-12 for a, b in regrets))


def _validation_snapshot(measured, source, fitted):
    source_instances = {r["instance_id"]: r for r in source["instances"]}
    fitted_instances = {r["instance_id"]: r for r in fitted["instances"]}
    if ({r["instance_id"] for r in measured["instances"]} != set(source_instances)
            or set(source_instances) != set(fitted_instances)):
        raise ValueError("Independent-validation instances changed")
    instances = []
    for row in measured["instances"]:
        original, initial = source_instances[row["instance_id"]], fitted_instances[row["instance_id"]]
        a, b = row["objective"] - original["objective"], row["objective"] - initial["objective"]
        denominator = max(row["total_initial_profit"], 1e-12)
        instances.append(dict(row, source_objective=original["objective"], fitted_objective=initial["objective"],
                              source_delta=a, fitted_delta=b, source_normalized_delta=a / denominator,
                              fitted_normalized_delta=b / denominator))
    count = len(instances)
    summary = dict(instances=count, mean_objective=sum(r["objective"] for r in instances) / count)
    for baseline in ("source", "fitted"):
        summary[f"mean_delta_{baseline}"] = sum(r[f"{baseline}_delta"] for r in instances) / count
        summary[f"mean_normalized_delta_{baseline}"] = sum(r[f"{baseline}_normalized_delta"] for r in instances) / count
        summary[f"profit_dropout_instances_{baseline}"] = sum(r[f"{baseline}_delta"] < -1e-8 for r in instances)
    return dict(states=measured["states"], prefix_summary=summarize_rows(measured["states"]),
                prefix_relative_to_source=_relative_validation(source["states"], measured["states"]),
                prefix_relative_to_fitted=_relative_validation(fitted["states"], measured["states"]),
                instances=instances, instance_summary=summary)


def _validation_trajectory(history, initial):
    summaries = [h["validation"]["instance_summary"] for h in history]

    def average(part):
        return {key: sum(s[key] for s in part) / len(part)
                for key in part[0] if key != "instances"} if part else None

    return dict(selection="Fixed final update, last 10/20 windows and all executed updates; no best-update selection",
                updates=len(history), final=(history[-1]["validation"] if history else initial),
                whole_trajectory_instance_mean=average(summaries),
                last10_updates=min(10, len(history)), last10_instance_mean=average(summaries[-10:]),
                last20_updates=min(20, len(history)), last20_instance_mean=average(summaries[-20:]))


def run_retention(bank_path, reach_path, learning_path, output, *, device="cpu", updates=5,
                  arms=("ppo_only", "ppo_rank", "ppo_regret"), seed=5729,
                  control_mode="original", actor_learning_rate=None, actor_continuation_guard=False,
                  actor_guard_states=8, actor_guard_tolerance=0., validation_bank_path=None):
    """Continue a fitted Actor diagnostically; never select a best update.

    Original/guarded retain legacy arm keys. Matched runs the fixed four
    original_only/original_regret/guarded_only/guarded_regret groups. An optional
    independent bank is measurement-only, including every original split and
    every prefix without reachability filtering; its full returns weight unique
    instances equally. It is never passed to PPO or the training-rollout guard.
    """
    from mkp_audit_bank import load_bank
    from mkp_actor_guard import build_optimizer
    if (updates < 0 or seed < 0 or not arms or len(set(arms)) != len(arms)
            or any(a not in ("ppo_only", "ppo_rank", "ppo_regret") for a in arms)
            or control_mode not in ("original", "guarded", "matched")
            or (actor_learning_rate is not None and (not math.isfinite(actor_learning_rate) or actor_learning_rate <= 0))
            or actor_guard_states < 1 or not math.isfinite(actor_guard_tolerance) or actor_guard_tolerance < 0):
        raise ValueError("Invalid retention experiment")
    if control_mode == "original" and (actor_learning_rate is not None or actor_continuation_guard):
        raise ValueError("Use guarded or matched control mode to change Actor training controls")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Retention output already contains files; use a new directory")
    bank_path = Path(bank_path)
    if bank_path.is_dir():
        bank_path = bank_path / "bank.pt"
    bank = load_bank(bank_path)
    reach = json.loads(Path(reach_path).read_text(encoding="utf-8"))
    learn = json.loads(Path(learning_path).read_text(encoding="utf-8"))
    check_report(bank_path, reach)
    check_report(bank_path, learn)
    records = eligible_records(bank, reach, scope=learn["budget"]["scope"])
    validation, validation_records, validation_metadata = None, [], None
    if validation_bank_path is not None:
        validation_bank_path = Path(validation_bank_path)
        if validation_bank_path.is_dir():
            validation_bank_path = validation_bank_path / "bank.pt"
        validation = load_bank(validation_bank_path)
        validation_metadata = validate_independent_bank(bank, validation)
        validation_metadata.update(bank_path=str(validation_bank_path.resolve()),
                                   bank_sha256=file_sha256(validation_bank_path))
        validation_records = [dict(r, split="heldout", validation_original_split=r["split"])
                              for r in validation["records"]]
    variants = [(arm, arm, control_mode) for arm in arms]
    if control_mode == "matched":
        variants = [(f"{control}_{suffix}", arm, control) for control in ("original", "guarded")
                    for suffix, arm in (("only", "ppo_only"), ("regret", "ppo_regret"))]
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for m_text, fitted in learn["models"].items():
        if fitted["status"] != "complete":
            continue
        m = int(m_text)
        if file_sha256(fitted["actor_artifact"]) != fitted["actor_sha256"]:
            raise ValueError("Diagnostic fit actor changed after learning report")
        artifact = torch.load(fitted["actor_artifact"], map_location="cpu", weights_only=True)
        if artifact.get("artifact_type") != "diagnostic_actor_fit":
            raise ValueError("Expected a diagnostic actor artifact")
        check_report(bank_path, artifact)
        if artifact.get("source_checkpoint_sha256") != bank["checkpoints"][m_text]["sha256"]:
            raise ValueError("Diagnostic fit actor has different source weights")
        local = [r for r in records if r["M"] == m]
        independent = [r for r in validation_records if r["M"] == m]
        validation_source = validation_fitted = None
        if independent:
            with _measurement_rng(device):
                measurement_model, measurement_saved = source_model(bank, m, device)
                known_fields = {f.name for f in fields(TrainConfig)}
                measurement_settings = {k: v for k, v in measurement_saved["train_config"].items()
                                        if k in known_fields}
                measurement_settings.setdefault("ranking_loss_type", "soft_bce")
                measurement_train = replace(TrainConfig(**measurement_settings), num_resources=(m,))
                configure_training_runtime(measurement_model, measurement_train, device)
                validation_source = _measure_validation(measurement_model, independent, device)
                measurement_model.actor.load_state_dict(artifact["actor_state"], strict=True)
                validation_fitted = _measure_validation(measurement_model, independent, device)
                del measurement_model
        for label, arm, control in variants:
            seed_everything(seed + m)
            model, saved = source_model(bank, m, device)
            model.actor.load_state_dict(artifact["actor_state"], strict=True)
            model.requires_grad_(True)
            fields_known = {f.name for f in fields(TrainConfig)}
            old_train = {k: v for k, v in saved["train_config"].items() if k in fields_known}
            old_train.setdefault("ranking_loss_type", "soft_bce")
            train = TrainConfig(**old_train)
            train = replace(train, num_resources=(m,), ranking_coefficient=0. if arm == "ppo_only" else train.ranking_coefficient)
            if arm == "ppo_regret":
                train = replace(train, ranking_loss_type="regret_margin")
            train = replace(train, actor_learning_rate=actor_learning_rate if control == "guarded" else None,
                            actor_continuation_guard=actor_continuation_guard if control == "guarded" else False,
                            actor_guard_states=actor_guard_states if control == "guarded" else 8,
                            actor_guard_tolerance=actor_guard_tolerance if control == "guarded" else 0.)
            runtime = configure_training_runtime(model, train, device)
            trainer = MKPTrainer.__new__(MKPTrainer)
            trainer.config, trainer.device = train, torch.device(device)
            optimizer = build_optimizer(model, train)
            # Both arms start from the same fitted actor and original critic with
            # fresh optimizer state. These are diagnostic continuations, not resume.
            before = evaluate_records(model, local, device)
            validation_initial = (_validation_snapshot(validation_fitted, validation_source, validation_fitted)
                                  if independent else None)
            history = []
            generator = GeneratorConfig(**saved["generator_config"])
            for update in range(1, updates + 1):
                seed_everything(seed + 100003 * update + m)
                rng = np.random.default_rng(seed + 100003 * update + m)
                n = int(rng.choice(train.train_num_items))
                data = synthetic_batch(train.batch_size, n, m, generator, rng, device)
                rollout = collect_rollout(model, data)
                observer = GradientObserver(model)
                stats = trainer.ppo_update(model, optimizer, data, rollout, update,
                                           audit_observer=observer)
                snapshot = evaluate_records(model, local, device)
                measured = dict(update=update, training_metrics=stats,
                    gradients=observer.rows, kl_stop_events=observer.stops,
                    retention=retention_metrics(before, snapshot), states=snapshot)
                if independent:
                    measured["validation"] = _validation_snapshot(
                        _measure_validation(model, independent, device), validation_source, validation_fitted)
                history.append(measured)
                print(f"retention M={m} arm={label} update={update}/{updates}", flush=True)
            experiment = dict(initial=before, initial_summary=summarize_rows(before),
                history=history, runtime=runtime, ranking_loss_type=train.ranking_loss_type,
                ranking_coefficient=train.ranking_coefficient,
                optimizer_initialization="fresh diagnostic AdamW, original source critic",
                training_controls=dict(control_mode=control, arm=arm, actor_learning_rate=train.actor_learning_rate,
                    effective_actor_learning_rate=train.actor_learning_rate if train.actor_learning_rate is not None else train.learning_rate,
                    critic_learning_rate=train.learning_rate, actor_continuation_guard=train.actor_continuation_guard,
                    actor_guard_states=train.actor_guard_states, actor_guard_tolerance=train.actor_guard_tolerance),
                source_checkpoint_sha256=bank["checkpoints"][m_text]["sha256"], fitted_actor_sha256=fitted["actor_sha256"])
            if independent:
                experiment["validation"] = dict(source=validation_source, fitted=validation_fitted,
                    initial=validation_initial, **_validation_trajectory(history, validation_initial))
            results[f"M{m}/{label}"] = experiment
    effective_arms = list(dict.fromkeys(arm for _, arm, _ in variants))
    report = dict(schema_version=1, bank_sha256=file_sha256(bank_path), phase="ppo_retention",
        data_source="synthetic_only", budget=dict(updates=updates, arms=effective_arms, seed=seed),
        scope="Production ppo_update on fresh synthetic batches; fixed diagnostic bank never enters PPO samples",
        warning="Common data seeds, not identical policy trajectories after arms diverge. Observer gradients cover one minibatch/update.",
        source_hashes=diagnostic_provenance(), experiments=results)
    report["training_controls"] = dict(control_mode=control_mode,
        matched_groups=[v[0] for v in variants], selection="Fixed final update and full trajectory; no best-update selection")
    if validation_metadata is not None:
        report["independent_validation"] = validation_metadata
    write_json(output / "retention.json", report)
    return report
