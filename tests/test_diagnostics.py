"""Synthetic metric fixtures and mocked orchestration only; no PPO steps or Chu inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from Benchmark_zeroShot import training_diagnostics
from mkp_config import ModelConfig, TrainConfig
from mkp_diagnostics import convergence_report, plot_training
from mkp_Trainer import MKPTrainer


def metric_rows(values, step=25, start=25):
    best = -float("inf")
    rows = []
    for i, value in enumerate(values):
        best = max(best, value)
        rows.append(dict(M=5, N=100, update=start + step * i, validation_return=value,
                         best_validation_return=best, normalized_return=value - .02,
                         entropy=.6 - i * .001, policy_loss=-.003, value_loss=.01,
                         constraint_aux_loss=.2, approx_kl=.005, clip_fraction=.1,
                         grad_norm=.2, explained_variance=.8, utilization=.9, kl_stopped=0))
    return rows


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainConfig(convergence_min_updates=100, convergence_patience_updates=100,
                                  convergence_min_evaluations=5, convergence_tolerance_pct=.05)

    def report(self, values, **kwargs):
        return convergence_report(metric_rows(values), self.config, complete=True, **kwargs)

    def test_plateau_start_and_confirmation_are_different(self):
        result = self.report([.48, .49] + [.5, .5001, .5, .5001, .5, .5])
        self.assertEqual(result["status"], "converged")
        self.assertEqual(result["plateau_start_update"], 75)
        self.assertEqual(result["confirmation_update"], 175)
        self.assertEqual(result["best_update"], 100)

    def test_continuing_improvement_does_not_converge(self):
        self.assertEqual(self.report([.4 + .001 * i for i in range(50)])["status"], "not_converged")

    def test_late_improvement_invalidates_early_plateau(self):
        values = [.5] * 8
        early = convergence_report(metric_rows(values), self.config)
        self.assertEqual(early["status"], "provisional")
        self.assertEqual(self.report(values + [.51])["status"], "not_converged")
        late = self.report(values + [.51] * 6)
        self.assertEqual(late["plateau_start_update"], 225)
        self.assertEqual(late["confirmation_update"], 325)

    def test_regression_and_noisy_returns_do_not_converge(self):
        self.assertEqual(self.report([.5] * 8 + [.48] * 20)["status"], "not_converged")
        self.assertEqual(self.report([.5, .499] * 10)["status"], "not_converged")

    def test_short_history_is_not_evidence(self):
        self.assertEqual(self.report([.5] * 3)["status"], "insufficient_history")
        self.assertEqual(self.report([])["status"], "insufficient_history")

    def test_patience_is_in_updates_not_observation_count(self):
        rows = metric_rows([.5] * 20, step=1, start=1000)
        self.assertEqual(convergence_report(rows, self.config)["status"], "insufficient_history")

    def test_non_validation_rows_do_not_count(self):
        rows = metric_rows([.5] * 10)
        for row in rows[1:]:
            row["validation_return"] = None
        self.assertEqual(convergence_report(rows, self.config)["tail_evaluations"], 1)

    def test_resume_without_history_respects_old_best(self):
        rows = metric_rows([.5] * 10, start=2000)
        for row in rows:
            row["best_validation_return"] = .51
        result = convergence_report(rows, self.config)
        self.assertEqual(result["status"], "not_converged")
        self.assertIsNone(result["best_update"])

    def test_invalid_history_and_configuration(self):
        with self.assertRaises(ValueError):
            self.report([.5, float("nan")])
        with self.assertRaises(ValueError):
            convergence_report(metric_rows([.5] * 10, step=0), self.config)
        for kwargs in ({"convergence_tolerance_pct": 0}, {"convergence_tolerance_pct": float("nan")},
                       {"convergence_patience_updates": 0}, {"convergence_min_evaluations": 1}):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    def test_plot_outputs_for_converged_and_unconfirmed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for m, values in [(5, [.5] * 10), (10, [.5, .51, .52])]:
                rows = metric_rows(values)
                report = convergence_report(rows, self.config, complete=True)
                paths = plot_training(rows, report, self.config, directory, m)
                self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
                svg = paths[1].read_text(encoding="utf-8")
                self.assertIn("Policy entropy", svg)
                self.assertIn("plateau start: 25" if m == 5 else "no confirmed convergence point", svg)

    def test_mocked_training_resume_and_benchmark_link(self):
        """Exercise real logging/checkpoint orchestration with deterministic fake rollout metrics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainConfig(num_resources=(5,), updates=5, batch_size=2, train_num_items=(4,),
                                 validation_every=1, validation_instances_per_shape=2, checkpoint_every=1,
                                 convergence_min_updates=3, convergence_patience_updates=2,
                                 convergence_min_evaluations=3, runs_dir=root / "runs",
                                 checkpoint_dir=root / "checkpoints", device="cpu")
            model = ModelConfig(embed_dim=8, num_heads=2, num_layers=1)

            def fake_solution(data, model):
                self.assertIsNotNone(model)
                return {"objective": data[0].sum(-1) * .5}

            def fake_rollout(model, data):
                return {"active": torch.ones(1, 2, dtype=torch.bool),
                        "objective": data[0].sum(-1) * .5, "utilization": torch.tensor(.9)}

            stats = {key: .1 for key in ("policy_loss", "value_loss", "constraint_aux_loss", "entropy",
                                        "approx_kl", "clip_fraction", "grad_norm", "explained_variance")}
            stats.update(minibatches=1, kl_stopped=0, ppo_seconds=.1,
                         feature_cache_gib=0., feature_cache_seconds=0.)
            with patch("mkp_Trainer.construct_solution", side_effect=fake_solution), \
                 patch("mkp_Trainer.collect_rollout", side_effect=fake_rollout), \
                 patch.object(MKPTrainer, "ppo_update", return_value=stats), \
                 patch("mkp_Trainer.plot_training", return_value=[]):
                trainer = MKPTrainer(config, model)
                trainer.train()
                report = json.loads((trainer.run_dir / "convergence_M5.json").read_text())
                self.assertEqual(report["status"], "converged")
                log = (trainer.run_dir / "train.log").read_text()
                self.assertIn("CONVERGENCE_FINAL", log)
                self.assertIn("entropy=", log)
                self.assertIn("SPEED |", log)
                checkpoint = root / "checkpoints" / "mkp_M5_last.pt"
                payload = torch.load(checkpoint, weights_only=True)
                self.assertEqual(len(payload["diagnostics_history"]), 5)
                self.assertNotIn("validation_improvement_pct", trainer.fields)
                self.assertTrue(all("validation_improvement_pct" not in row
                                    for row in payload["diagnostics_history"]))
                self.assertEqual(report["scope"], "synthetic_learned_dynamic_greedy_validation_only")
                self.assertEqual(payload["training_runtime"]["profile"], "reference")
                self.assertEqual(training_diagnostics(payload, 5)["report_status"], "available")
                config.updates = 8
                resumed = MKPTrainer(config, model)
                resumed.train(checkpoint)
                history = json.loads((resumed.run_dir / "diagnostics_history_M5.json").read_text())
                self.assertEqual([row["update"] for row in history], list(range(1, 9)))
                resumed_report = json.loads((resumed.run_dir / "convergence_M5.json").read_text())
                self.assertEqual(resumed_report["plateau_start_update"], 1)
                self.assertEqual(resumed_report["observed_through_update"], 8)
            self.assertEqual(training_diagnostics({"update": 1}, 5)["report_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
