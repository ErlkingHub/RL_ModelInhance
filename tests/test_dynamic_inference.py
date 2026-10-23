"""Tiny greedy inference and mocked report contracts; no dataset evaluation."""
import contextlib
import csv
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import Benchmark_zeroShot as evaluation
from mkp_config import BenchmarkConfig, ModelConfig
from mkp_data import MKPInstance
from mkp_Model import MKPModel
from mkp_utils import construct_solution


class DynamicInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.model = MKPModel(ModelConfig(embed_dim=16, num_heads=4, num_layers=1))
        self.data = (torch.tensor([[8., 6., 1.]], dtype=torch.float64),
                     torch.tensor([[[2., 2.], [2., 1.], [1., 2.]]], dtype=torch.float64),
                     torch.tensor([[5., 5.]], dtype=torch.float64))

    def test_controller_is_recomputed_on_each_successor(self):
        visits = []
        original = self.model.actor.forward

        def actor(state):
            visits.append((state.selected.clone(), state.remaining.clone()))
            return original(state)

        with patch.object(self.model.actor, "forward", side_effect=actor):
            result = construct_solution(self.data, self.model)
        self.assertEqual([int(selected.sum()) for selected, _ in visits], [0, 1, 2])
        self.assertEqual(result["objective"].item(), 15.)
        self.assertTrue(result["selected"].all())
        repeated = construct_solution(self.data, self.model)
        torch.testing.assert_close(result["sequence"], repeated["sequence"], atol=0, rtol=0)

    def test_constructive_api_has_only_data_and_model(self):
        self.assertEqual(list(inspect.signature(construct_solution).parameters), ["data", "model"])
        with self.assertRaises(TypeError):
            construct_solution(self.data)
        with self.assertRaises(ValueError):
            construct_solution(self.data, None)
        for kwargs in (dict(mode="learned_dynamic"), dict(stochastic=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                construct_solution(self.data, self.model, **kwargs)

    def test_single_instance_construction_is_called_once(self):
        instance = MKPInstance("tiny", np.array([8., 6., 1.]),
                               np.array([[2., 2.], [2., 1.], [1., 2.]]), np.array([5., 5.]), 15.)
        batched = dict(objective=torch.tensor([15.]), selected=torch.ones(1, 3, dtype=torch.bool),
                       remaining=torch.zeros(1, 2), sequence=torch.tensor([[0, 1, 2]]),
                       theta_mean=torch.zeros(1, 9), decoder_mean=torch.zeros(1, 4))
        with patch.object(evaluation, "instance_batch", return_value=self.data) as batch, \
             patch.object(evaluation, "construct_solution", return_value=batched) as construct:
            result, elapsed = evaluation.solve_instance(instance, self.model, torch.device("cpu"))
        batch.assert_called_once_with(instance, 1, torch.device("cpu"))
        construct.assert_called_once_with(self.data, self.model)
        self.assertEqual(result["selected"].shape, (3,))
        self.assertGreaterEqual(elapsed, 0.)

    def test_mocked_reporting_writes_one_dynamic_row_and_solution(self):
        instance = MKPInstance("tiny", np.array([4., 3.]), np.array([[1.], [1.]]), np.array([2.]), 7.)
        result = dict(objective=torch.tensor(7.), selected=torch.ones(2, dtype=torch.bool),
                      remaining=torch.zeros(1), sequence=torch.tensor([0, 1]),
                      theta_mean=torch.zeros(9), decoder_mean=torch.zeros(4))
        payload = dict(checkpoint_version=5, controller_version="combination_theta_v5",
                       decoder_version="second_order_density_v5", update=3, model_config={},
                       train_config={"train_num_items": [2]}, training_data_source="synthetic_only")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "fixture.mat"
            dataset.write_bytes(b"mock dataset")
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "mkp_M1_best.pt").write_bytes(b"mock weights")
            config = BenchmarkConfig(dataset=dataset, num_resources=(1,), num_items=(2,),
                                     checkpoint_dir=checkpoints, results_dir=root / "reports", device="cpu")
            with patch.object(evaluation, "load_benchmark", return_value=[instance]), \
                 patch.object(evaluation, "load_checkpoint", return_value=(self.model, payload)), \
                 patch.object(evaluation, "solve_instance", return_value=(result, .02)) as solve, \
                 contextlib.redirect_stdout(io.StringIO()):
                output = evaluation.benchmark(config)
            solve.assert_called_once()
            with (output / "instances.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "learned_dynamic")
            self.assertEqual(rows[0]["rollouts"], "1")
            self.assertFalse(any("improvement_over" in field or "sample_" in field for field in rows[0]))
            solutions = [json.loads(line) for line in (output / "solutions.jsonl").read_text().splitlines()]
            self.assertEqual(len(solutions), 1)
            self.assertEqual(solutions[0]["method"], "learned_dynamic")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["completed_rows"], manifest["instance_count"])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["inference"], "learned_dynamic_single_greedy")
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual({row["method"] for row in summary}, {"learned_dynamic"})
            self.assertFalse(any("improvement_over" in field or "wins_over" in field for field in summary[0]))

    def test_cli_rejects_removed_sampling_flags(self):
        for flag in ("--sample-rollouts", "--sample-batch-size"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluation.main([flag, "2"])
            self.assertEqual(error.exception.code, 2)
        with patch.object(evaluation, "benchmark") as benchmark:
            evaluation.main(["--resources", "5", "--items", "100", "--device", "cpu"])
        config = benchmark.call_args.args[0]
        self.assertFalse(hasattr(config, "sample_rollouts"))
        self.assertFalse(hasattr(config, "sample_batch_size"))


if __name__ == "__main__":
    unittest.main()
