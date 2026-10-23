"""Retention entry-point contracts with tiny fixtures and mocked PPO/model I/O.

No checkpoint model, training rollout, or benchmark evaluation is executed.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import mkp_audit_learning as audit
from mkp_utils import file_sha256


@dataclass
class TinyTrain:
    learning_rate: float = 1e-4
    ranking_coefficient: float = .1
    ranking_loss_type: str = "soft_bce"
    train_num_items: tuple = (3,)
    batch_size: int = 1
    num_resources: tuple = (1,)
    actor_learning_rate: float | None = None
    actor_continuation_guard: bool = False
    actor_guard_states: int = 8
    actor_guard_tolerance: float = 0.


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(1, 1, bias=False)
        self.critic = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.actor.weight.fill_(1.)


def record(seed, instance, split="fit", stage="early", offset=0.):
    return dict(id=f"s{seed}/{instance}/{stage}", instance_id=f"s{seed}/{instance}",
                M=1, N=3, split=split, stage=stage, capacity_ratio=.5, label_kind="finite_greedy",
                generation_entropy=[seed, 1, 3, 0, int(split == "heldout"), instance],
                state=dict(profits=torch.tensor([[2. + offset, 3., 5.]], dtype=torch.float64),
                           weights=torch.ones((1, 3, 1), dtype=torch.float64),
                           capacities=torch.tensor([[2.]], dtype=torch.float64),
                           remaining=torch.tensor([[1.]], dtype=torch.float64),
                           selected=torch.tensor([[True, False, False]])))


def bank(seed, offset=0.):
    return dict(seed=seed, checkpoints={"1": {"sha256": "same-source-checkpoint"}},
                records=[record(seed, 0, offset=offset), record(seed, 1, "heldout", offset=offset + 1.)])


class RetentionValidationTests(unittest.TestCase):
    def test_independent_bank_rejects_seed_scope_provenance_and_tensor_reuse(self):
        fit, independent = bank(10), bank(20, 20.)
        summary = audit.validate_independent_bank(fit, independent)
        self.assertEqual(summary["instances"], 2)
        self.assertEqual(summary["records"], 2)
        variants = []
        wrong = deepcopy(independent)
        wrong["seed"] = fit["seed"]
        variants.append(wrong)
        wrong = deepcopy(independent)
        wrong["records"][0]["N"] = 4
        variants.append(wrong)
        wrong = deepcopy(independent)
        wrong["checkpoints"]["1"]["sha256"] = "other-checkpoint"
        variants.append(wrong)
        wrong = deepcopy(independent)
        wrong["records"][0]["state"] = deepcopy(fit["records"][0]["state"])
        variants.append(wrong)
        wrong = deepcopy(independent)
        del wrong["records"][0]["generation_entropy"]
        variants.append(wrong)
        wrong = deepcopy(independent)
        duplicate = deepcopy(wrong["records"][0])
        duplicate["state"]["profits"][0, 0] += 1
        wrong["records"].append(duplicate)
        variants.append(wrong)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    audit.validate_independent_bank(fit, value)

    def test_empty_initial_instance_measurement_deduplicates_prefixes(self):
        rows = bank(20, 20.)["records"]
        duplicate = deepcopy(rows[0])
        duplicate["stage"] = "middle"
        rows.append(duplicate)
        seen = []

        def measure(model, state):
            self.assertFalse(bool(state.selected.any()))
            torch.testing.assert_close(state.remaining, state.capacities)
            seen.append(state)
            return 7.

        with patch.object(audit, "greedy_future", side_effect=measure):
            measured = audit.evaluate_validation_instances(object(), rows, "cpu")
        self.assertEqual(len(seen), 2)
        self.assertEqual([r["objective"] for r in measured], [7., 7.])
        self.assertTrue(bool(rows[0]["state"]["selected"].any()))

    def _entry(self, root, *, validation=False, mode="original", updates=2, **controls):
        root.mkdir(parents=True, exist_ok=True)
        fit_path, validation_path = root / "fit.pt", root / "validation.pt"
        fit_path.write_bytes(b"fit bank")
        validation_path.write_bytes(b"independent bank")
        fit, independent = bank(10), bank(20, 20.)
        # A repeated prefix must not give this instance extra full-return weight.
        duplicate = deepcopy(independent["records"][0])
        duplicate.update(id=duplicate["id"] + "/middle", stage="middle")
        independent["records"].append(duplicate)
        banks = {fit_path: fit, validation_path: independent}
        actor_path = root / "actor.pt"
        actor = TinyModel().actor
        with torch.no_grad():
            actor.weight.fill_(2.)
        torch.save(dict(artifact_type="diagnostic_actor_fit", bank_sha256=file_sha256(fit_path),
                        source_checkpoint_sha256="same-source-checkpoint", actor_state=actor.state_dict()), actor_path)
        reach_path, learn_path = root / "reach.json", root / "learn.json"
        reach_path.write_text(json.dumps(dict(bank_sha256=file_sha256(fit_path),
            records=[dict(id=r["id"], status="reachable") for r in fit["records"]])))
        learn_path.write_text(json.dumps(dict(bank_sha256=file_sha256(fit_path), budget=dict(scope="both"),
            models={"1": dict(status="complete", actor_artifact=str(actor_path), actor_sha256=file_sha256(actor_path))})))
        train_calls, batch_calls, instance_calls = [], [], []

        def source_model(*args):
            return TinyModel(), dict(train_config={}, generator_config={})

        def evaluate(model, rows, device):
            self.assertTrue(model.audit_runtime_configured)
            # Deliberately consume all RNGs: the independent-measurement wrapper
            # must isolate even an implementation that later adds randomness.
            torch.rand(1)
            np.random.random()
            random.random()
            value = float(model.actor.weight.detach().item())
            return [dict(id=r["id"], instance_id=r["instance_id"], M=r["M"], N=r["N"],
                         capacity_ratio=r["capacity_ratio"], stage=r["stage"], label_kind=r["label_kind"],
                         split=r["split"], target_hit=value >= 1., action_value_known=True,
                         exact_action_regret=None, greedy_future=10. + value,
                         greedy_future_delta=value - 1., greedy_delta_over_initial_profit=(value - 1.) / 10.) for r in rows]

        def instances(model, rows, device):
            instance_calls.append({r["instance_id"] for r in rows})
            torch.rand(1)
            np.random.random()
            random.random()
            value = float(model.actor.weight.detach().item())
            return [dict(instance_id=instance, M=1, N=3, capacity_ratio=.5,
                         objective=20. + value, total_initial_profit=30.)
                    for instance in sorted({r["instance_id"] for r in rows})]

        def synthetic(*args):
            marker = float(torch.rand(1)) + float(np.random.random()) + random.random()
            data = dict(training_only=True, marker=marker)
            batch_calls.append(data)
            return data

        def ppo(trainer, model, optimizer, data, rollout, update, **kwargs):
            self.assertIs(data, rollout)
            self.assertTrue(data["training_only"])
            train_calls.append(trainer.config)
            with torch.no_grad():
                model.actor.weight.add_(.1 + data["marker"])
            return dict(minibatches=1)

        with patch("mkp_audit_bank.load_bank", side_effect=lambda path: banks[Path(path)]), \
             patch.object(audit, "source_model", side_effect=source_model), \
             patch.object(audit, "TrainConfig", TinyTrain), \
             patch.object(audit, "configure_training_runtime",
                          side_effect=lambda model, *args: setattr(model, "audit_runtime_configured", True) or {}), \
             patch.object(audit, "evaluate_records", side_effect=evaluate), \
             patch.object(audit, "evaluate_validation_instances", side_effect=instances), \
             patch.object(audit, "synthetic_batch", side_effect=synthetic), \
             patch.object(audit, "collect_rollout", side_effect=lambda model, data: data), \
             patch.object(audit.MKPTrainer, "ppo_update", new=ppo), \
             patch("mkp_actor_guard.build_optimizer", return_value=object()), \
             patch.object(audit, "diagnostic_provenance", return_value={}):
            result = audit.run_retention(fit_path, reach_path, learn_path, root / "result", updates=updates,
                arms=("ppo_only", "ppo_regret"), control_mode=mode,
                validation_bank_path=validation_path if validation else None, **controls)
        return result, train_calls, batch_calls, instance_calls

    def test_validation_does_not_change_training_or_filter_independent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            plain, _, batches, _ = self._entry(Path(directory) / "plain")
            validated, _, other_batches, instance_calls = self._entry(Path(directory) / "validated", validation=True)
        self.assertEqual(batches, other_batches)
        for key, experiment in plain["experiments"].items():
            other = validated["experiments"][key]
            self.assertEqual(experiment["initial"], other["initial"])
            self.assertEqual([h["states"] for h in experiment["history"]], [h["states"] for h in other["history"]])
            initial = other["validation"]["initial"]
            self.assertEqual(len(initial["states"]), 3)
            self.assertTrue(all(r["split"] == "heldout" for r in initial["states"]))
            self.assertEqual({r["validation_original_split"] for r in initial["states"]}, {"fit", "heldout"})
            self.assertEqual(initial["instance_summary"]["instances"], 2)
            self.assertEqual(other["validation"]["updates"], 2)
            self.assertEqual(other["validation"]["last10_updates"], 2)
            self.assertEqual(other["validation"]["last20_updates"], 2)
            self.assertEqual(other["validation"]["last10_instance_mean"], other["validation"]["whole_trajectory_instance_mean"])
            self.assertEqual(other["validation"]["final"], other["history"][-1]["validation"])
        # Source/fitted once per M, followed by each arm's executed updates.
        self.assertEqual(len(instance_calls), 2 + 2 * 2)

    def test_matched_mode_runs_four_groups_and_keeps_controls_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls, _, _ = self._entry(Path(directory), mode="matched", updates=1,
                actor_learning_rate=1e-5, actor_continuation_guard=True,
                actor_guard_states=2, actor_guard_tolerance=.5)
        self.assertEqual(set(result["experiments"]), {"M1/original_only", "M1/original_regret", "M1/guarded_only", "M1/guarded_regret"})
        self.assertEqual(len(calls), 4)
        for train in calls[:2]:
            self.assertIsNone(train.actor_learning_rate)
            self.assertFalse(train.actor_continuation_guard)
        for train in calls[2:]:
            self.assertEqual(train.actor_learning_rate, 1e-5)
            self.assertTrue(train.actor_continuation_guard)
            self.assertEqual(train.actor_guard_states, 2)
        self.assertEqual([x.ranking_loss_type for x in calls], ["soft_bce", "regret_margin", "soft_bce", "regret_margin"])
        self.assertEqual([x.ranking_coefficient for x in calls], [0., .1, 0., .1])

    def test_validation_trajectory_uses_fixed_latest_windows(self):
        history = [dict(validation=dict(instance_summary=dict(instances=2, mean_objective=float(i))))
                   for i in range(1, 26)]
        report = audit._validation_trajectory(history, {})
        self.assertEqual(report["final"], history[-1]["validation"])
        self.assertEqual(report["whole_trajectory_instance_mean"]["mean_objective"], 13.)
        self.assertEqual(report["last10_instance_mean"]["mean_objective"], 20.5)
        self.assertEqual(report["last20_instance_mean"]["mean_objective"], 15.5)
        self.assertEqual(report["last10_updates"], 10)
        self.assertEqual(report["last20_updates"], 20)

    def test_nonempty_outputs_and_ambiguous_original_controls_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep.txt").write_text("evidence")
            with self.assertRaises(FileExistsError):
                audit.run_retention("unused", "unused", "unused", path)
            self.assertEqual((path / "keep.txt").read_text(), "evidence")
            with self.assertRaises(ValueError):
                audit.run_retention("unused", "unused", "unused", path / "new", actor_learning_rate=1e-5)


if __name__ == "__main__":
    unittest.main()
