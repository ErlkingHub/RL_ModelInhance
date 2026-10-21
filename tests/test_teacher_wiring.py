"""Actual orchestration and checkpoint I/O with mocked PPO and teacher updates."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_config import ModelConfig, TrainConfig, TRAINING_SUPERVISION_VERSION
from mkp_imitation import empty_teacher_metrics
from mkp_Trainer import MKPTrainer, main


class TeacherWiringTests(unittest.TestCase):
    def test_teacher_runs_after_ppo_kl_stop_and_resumes_its_own_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainConfig(num_resources=(2,), train_num_items=(4,), updates=1, batch_size=2,
                validation_instances_per_shape=2, validation_every=1, teacher_every=1,
                checkpoint_every=1, checkpoint_dir=root / "checkpoints", runs_dir=root / "runs", device="cpu")
            model = ModelConfig(embed_dim=8, num_heads=2, num_layers=1)
            events = []
            def fake_solution(data, *args, **kwargs):
                return dict(objective=data[0].sum(-1) * .5)
            def fake_rollout(model, data):
                return dict(active=torch.ones(1, 2, dtype=torch.bool), objective=data[0].sum(-1) * .5,
                            utilization=torch.tensor(.9))
            def fake_ppo(*args, **kwargs):
                events.append("ppo_stopped")
                stats = {key: .1 for key in ("policy_loss", "value_loss", "constraint_aux_loss", "entropy",
                    "approx_kl", "clip_fraction", "grad_norm", "explained_variance", "ppo_seconds",
                    "feature_cache_gib", "feature_cache_seconds")}
                stats.update(minibatches=0, kl_stopped=1)
                return stats
            def fake_teacher(*args, **kwargs):
                events.append("teacher")
                stats = empty_teacher_metrics()
                stats.update(teacher_labeled=3, teacher_updates=2)
                return stats
            with patch("mkp_Trainer.construct_solution", side_effect=fake_solution), \
                 patch("mkp_Trainer.collect_rollout", side_effect=fake_rollout), \
                 patch.object(MKPTrainer, "ppo_update", side_effect=fake_ppo), \
                 patch("mkp_Trainer.TeacherTrainer.update", side_effect=fake_teacher), \
                 patch("mkp_Trainer.plot_training", return_value=[]), \
                 patch("torch.optim.AdamW.step", side_effect=AssertionError("No actual optimization allowed")):
                trainer = MKPTrainer(config, model)
                trainer.train()
                checkpoint = config.checkpoint_dir / "mkp_M2_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                self.assertIsNotNone(saved["teacher_state"])
                self.assertEqual(saved["training_supervision_version"], TRAINING_SUPERVISION_VERSION)
                self.assertEqual(saved["diagnostics_history"][-1]["teacher_updates"], 2)
                resumed = MKPTrainer(replace(config, updates=2), model)
                resumed.train(checkpoint)
                self.assertEqual(events, ["ppo_stopped", "teacher", "ppo_stopped", "teacher"])
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["update"], 2)
                history = json.loads((resumed.run_dir / "diagnostics_history_M2.json").read_text())
                self.assertEqual([x["update"] for x in history], [1, 2])

    def test_cli_teacher_settings_and_ablation_flags(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--teacher-states", "6", "--teacher-exact-items", "12", "--teacher-steps", "3",
                  "--teacher-time-limit", ".5", "--teacher-every", "10", "--efficiency-gate-initial", ".8",
                  "--no-terminal-profit-override"])
        train, model = trainer.call_args.args
        self.assertEqual((train.teacher_states, train.teacher_exact_items, train.teacher_steps,
                          train.teacher_time_limit, train.teacher_every), (6, 12, 3, .5, 10))
        self.assertEqual(model.efficiency_gate_initial, .8)
        self.assertFalse(model.terminal_profit_override)
        self.assertEqual(train.supervision_mode, "teacher")
        self.assertEqual(train.checkpoint_dir.name, "decision_teacher_v9")


if __name__ == "__main__":
    unittest.main()
