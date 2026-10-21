"""Warm-start checkpoint orchestration with fake metrics; no optimization or inference."""
from contextlib import ExitStack
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import torch

from mkp_config import ModelConfig, TrainConfig, config_dict
from mkp_Model import MKPModel
from mkp_Trainer import MKPTrainer, main
from mkp_utils import file_sha256


class ResourceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model_config = ModelConfig(embed_dim=8, num_heads=2, num_layers=1)

    def config(self, **changes):
        values = dict(teacher_steps=0, num_resources=(3,), train_num_items=(4,), updates=2,
                      batch_size=2, minibatch_size=2, validation_every=1,
                      validation_instances_per_shape=2, checkpoint_every=1,
                      convergence_min_updates=2, convergence_patience_updates=2,
                      convergence_min_evaluations=2, runs_dir=self.root / "runs",
                      checkpoint_dir=self.root / "new_checkpoints", device="cpu")
        values.update(changes)
        return TrainConfig(**values)

    def trainer(self, config):
        trainer = MKPTrainer(config, self.model_config)
        # Also close handles if a test stops before invoking train().
        def close_handlers():
            for handler in trainer.logger.handlers[:]:
                handler.close()
                trainer.logger.removeHandler(handler)
        self.addCleanup(close_handlers)
        return trainer

    def legacy_checkpoint(self, m=3):
        config = replace(self.model_config, actor_resource_geometry=False,
                         controller_resource_readout=False)
        serialized = config_dict(config)
        for key in ("actor_resource_geometry", "controller_resource_readout", "efficiency_gate_initial",
                    "terminal_profit_override"):
            serialized.pop(key)
        payload = dict(checkpoint_version=5, controller_version="combination_theta_v5",
                       decoder_version="second_order_density_v5", model_config=serialized,
                       model_state={k: v for k, v in MKPModel(config).state_dict().items()
                                    if not k.startswith("actor.efficiency_head.")}, num_resources=m,
                       training_data_source="synthetic_only", update=987,
                       best_validation_return=.99,
                       optimizer_state={"legacy_optimizer_must_not_be_loaded": True},
                       diagnostics_history=[{"update": 987, "validation_return": .99}])
        path = self.root / "source_v5" / f"mkp_M{m}_best.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(payload, path)
        return path

    def fake_execution(self):
        stack = ExitStack()
        dynamic_returns = iter([.5, .45, .44, .43])

        def solution(data, model=None, **kwargs):
            fraction = .4 if model is None else next(dynamic_returns)
            return {"objective": data[0].sum(-1) * fraction}

        def rollout(model, data):
            return {"active": torch.ones(1, 2, dtype=torch.bool),
                    "objective": data[0].sum(-1) * .4,
                    "utilization": torch.tensor(.9)}

        stats = {key: .1 for key in ("policy_loss", "value_loss", "constraint_aux_loss",
                                     "entropy", "approx_kl", "clip_fraction", "grad_norm",
                                     "explained_variance")}
        stats.update(minibatches=1, kl_stopped=0, ppo_seconds=.1,
                     feature_cache_gib=0., feature_cache_seconds=0.)
        stack.enter_context(patch("mkp_Trainer.construct_solution", side_effect=solution))
        stack.enter_context(patch("mkp_Trainer.collect_rollout", side_effect=rollout))
        stack.enter_context(patch.object(MKPTrainer, "ppo_update", return_value=stats))
        stack.enter_context(patch("mkp_Trainer.prepare_plotting"))
        stack.enter_context(patch("mkp_Trainer.plot_training", return_value=[]))
        stack.enter_context(patch.object(torch.optim.AdamW, "step",
                                        side_effect=AssertionError("No optimizer step is permitted")))
        return stack

    def test_warm_start_retains_initial_best_and_provenance_on_resume(self):
        source = self.legacy_checkpoint()
        source_hash = file_sha256(source)
        config = self.config()
        trainer = self.trainer(config)
        with self.fake_execution():
            trainer.train(warm_start=source)
            paths = {tag: config.checkpoint_dir / f"mkp_M3_{tag}.pt"
                     for tag in ("initial", "best", "last")}
            saved = {tag: torch.load(path, weights_only=True) for tag, path in paths.items()}
            self.assertEqual(saved["initial"]["update"], 0)
            self.assertEqual(saved["best"]["update"], 0)
            self.assertEqual(saved["last"]["update"], 2)
            self.assertAlmostEqual(saved["last"]["best_validation_return"], .5)
            self.assertEqual([r["update"] for r in saved["initial"]["diagnostics_history"]], [0])
            self.assertEqual([r["update"] for r in saved["last"]["diagnostics_history"]], [0, 1, 2])
            self.assertEqual(saved["initial"]["optimizer_state"]["state"], {})
            self.assertNotIn("legacy_optimizer_must_not_be_loaded", saved["initial"]["optimizer_state"])
            provenance = saved["last"]["training_initialization"]
            self.assertEqual(provenance["mode"], "v5_warm_start")
            self.assertEqual(provenance["source_update"], 987)
            self.assertEqual(provenance["source_path"], str(source.resolve()))
            self.assertEqual(provenance["source_sha256"], source_hash)
            self.assertTrue(provenance["optimizer_reset"])
            self.assertFalse(provenance["policy_outputs_preserved"])
            self.assertEqual(provenance["efficiency_gate_initial"], self.model_config.efficiency_gate_initial)
            self.assertEqual(json.loads((trainer.run_dir / "initialization_M3.json").read_text()), provenance)
            with (trainer.run_dir / "metrics.csv").open(newline="") as stream:
                self.assertEqual([int(row["update"]) for row in csv.DictReader(stream)], [0, 1, 2])
            baseline_hashes = {tag: file_sha256(paths[tag]) for tag in ("initial", "best")}

            resumed = self.trainer(replace(config, updates=3))
            resumed.train(resume=paths["last"])
            continued = torch.load(paths["last"], weights_only=True)
            self.assertEqual(continued["update"], 3)
            self.assertEqual([r["update"] for r in continued["diagnostics_history"]], [0, 1, 2, 3])
            self.assertEqual(continued["training_initialization"], provenance)
            self.assertAlmostEqual(continued["best_validation_return"], .5)
            for tag, expected in baseline_hashes.items():
                self.assertEqual(file_sha256(paths[tag]), expected)
        self.assertEqual(file_sha256(source), source_hash)

    def test_directory_dispatches_matching_resource_checkpoints(self):
        source = self.legacy_checkpoint(3).parent
        self.legacy_checkpoint(5)
        trainer = self.trainer(self.config(num_resources=(3, 5)))
        with patch.object(trainer, "_train_shape") as train_shape, \
             patch("mkp_Trainer.prepare_plotting"):
            trainer.train(warm_start=source)
        self.assertEqual(train_shape.call_args_list, [
            call(3, None, source / "mkp_M3_best.pt"),
            call(5, None, source / "mkp_M5_best.pt"),
        ])

    def test_file_with_multiple_resource_counts_fails_before_execution(self):
        source = self.legacy_checkpoint()
        trainer = self.trainer(self.config(num_resources=(3, 5)))
        with patch.object(trainer, "_train_shape") as train_shape:
            with self.assertRaisesRegex(ValueError, "file requires one"):
                trainer.train(warm_start=source)
        train_shape.assert_not_called()

    def test_source_directory_is_protected_for_file_and_directory_inputs(self):
        source = self.legacy_checkpoint()
        original_hash = file_sha256(source)
        for argument in (source, source.parent):
            with self.subTest(argument=argument):
                trainer = self.trainer(self.config(checkpoint_dir=source.parent / "."))
                with patch.object(trainer, "_train_shape") as train_shape:
                    with self.assertRaisesRegex(ValueError, "different checkpoint directory"):
                        trainer.train(warm_start=argument)
                train_shape.assert_not_called()
        self.assertEqual(file_sha256(source), original_hash)

    def test_direct_api_rejects_resume_and_warm_start_together(self):
        trainer = self.trainer(self.config())
        with patch.object(trainer, "_train_shape") as train_shape:
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                trainer.train(resume="saved.pt", warm_start="legacy.pt")
        train_shape.assert_not_called()

    def test_cli_passes_resource_flags_and_initialization_arguments(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--resources", "3", "5", "--device", "cpu",
                  "--warm-start", "source_v5", "--checkpoint-dir", "new_v8",
                  "--no-actor-resource-geometry", "--controller-resource-readout"])
        config, model_config = trainer.call_args.args
        self.assertEqual(config.num_resources, (3, 5))
        self.assertEqual(config.checkpoint_dir, Path("new_v8"))
        self.assertFalse(model_config.actor_resource_geometry)
        self.assertTrue(model_config.controller_resource_readout)
        trainer.return_value.train.assert_called_once_with(None, warm_start=Path("source_v5"))

        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--resources", "3", "--resume", "last.pt", "--no-controller-resource-readout"])
        self.assertTrue(trainer.call_args.args[1].actor_resource_geometry)
        self.assertFalse(trainer.call_args.args[1].controller_resource_readout)
        trainer.return_value.train.assert_called_once_with("last.pt", warm_start=None)

    def test_cli_rejects_both_initialization_modes_before_trainer_creation(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                main(["--resume", "last.pt", "--warm-start", "legacy.pt"])
        self.assertEqual(raised.exception.code, 2)
        trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
