"""Small tensor and mocked reporting tests; no training or benchmark model rollouts."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from mkp_config import ModelConfig, BenchmarkConfig, CHECKPOINT_VERSION, config_dict, decoder_version, variant_name
from mkp_bgnn import BGNNEncoder, BipartiteMessage, dual_pressure_proxy
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, MaskedItemPolicy, CandidatePolicy
from mkp_utils import atomic_checkpoint, load_checkpoint


class BGNNContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    def state(self, device="cpu"):
        return MKPEnv(torch.tensor([[5., 3., 2.], [1., 2., 3.]], device=device),
                      torch.tensor([[[2., 1.], [1., 2.], [1., 1.]],
                                    [[3., 3.], [4., 4.], [5., 5.]]], device=device),
                      torch.tensor([[3., 3.], [1., 1.]], device=device)).state

    def test_bgnn_has_no_attention_and_edges_carry_messages(self):
        encoder = BGNNEncoder(self.config)
        self.assertFalse(any(isinstance(m, torch.nn.MultiheadAttention) for m in encoder.modules()))
        message = BipartiteMessage(16)
        items, resources = torch.zeros(1, 3, 16), torch.zeros(1, 2, 16)
        first = message(items, resources, torch.zeros(1, 3, 2, 3), True)
        second = message(items, resources, torch.ones(1, 3, 2, 3), True)
        self.assertFalse(torch.allclose(first, second))
        missing = message(items, resources, torch.ones(1, 3, 2, 3), True,
                          torch.zeros(1, 3, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(missing, torch.zeros_like(missing)))
        out = encoder(self.state())
        self.assertEqual(out[0].shape, (2, 3, 16))
        self.assertEqual(out[1].shape, (2, 2, 16))

    def test_proxy_tracks_remaining_capacity_and_permutations(self):
        state = self.state()
        proxy = dual_pressure_proxy(state, 1e-8)
        state.remaining[0, 0] *= 0.1
        changed = dual_pressure_proxy(state, 1e-8)
        self.assertGreater(changed[0, 0], proxy[0, 0])
        state.remaining[0, 0] = 0
        self.assertTrue(torch.isfinite(dual_pressure_proxy(state, 1e-8)).all())
        state.selected.fill_(True)
        self.assertTrue(torch.equal(dual_pressure_proxy(state, 1e-8), torch.zeros(2, 2)))

    def test_masked_distribution_matches_ragged_and_handles_terminal_rows(self):
        logits = torch.tensor([[1., -1., 2.], [3., 4., 5.]], requires_grad=True)
        feasible = torch.tensor([[True, False, True], [False, False, False]])
        rows, cols = feasible.nonzero(as_tuple=True)
        ragged = CandidatePolicy(rows, cols, logits[rows, cols], 2)
        masked = MaskedItemPolicy(logits, feasible)
        actions = masked.choose()
        self.assertEqual(actions.tolist(), [2, -1])
        self.assertTrue(torch.allclose(masked.log_prob(actions), ragged.log_prob(actions)))
        self.assertTrue(torch.allclose(masked.entropy(), ragged.entropy()))
        (-masked.log_prob(actions).sum() - 0.1 * masked.entropy().sum()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 1], 0)

    def test_direct_head_gradient_permutation_and_checkpoint(self):
        config = replace(self.config, policy_head="direct_item")
        model = MKPModel(config)
        state = self.state()
        policy, value, theta, aux = model(state)
        self.assertIsNone(theta)
        actions = policy.choose()
        (-policy.log_prob(actions).sum() + value.square().mean() + aux.square().mean()).backward()
        self.assertGreater(model.actor.item_head[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.actor.encoder.blocks[0].item_to_constraint.edge.weight.grad.abs().sum(), 0)
        order = torch.tensor([2, 0, 1])
        reordered = MKPState(state.profits[:, order], state.weights[:, order], state.capacities,
                             state.remaining, state.selected[:, order])
        self.assertTrue(torch.allclose(model.policy(reordered)[0].distribution.probs,
                                       policy.distribution.probs[:, order], atol=1e-6))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.pt"
            atomic_checkpoint(path, dict(checkpoint_version=CHECKPOINT_VERSION,
                decoder_version=decoder_version(config), model_config=config_dict(config),
                model_state=model.state_dict(), num_resources=2, training_data_source="synthetic_only"))
            loaded, _ = load_checkpoint(path, "cpu", 2)
            self.assertEqual(loaded.config.policy_head, "direct_item")

    def test_matched_initial_encoder_and_independent_critic_across_heads(self):
        torch.manual_seed(10)
        weak = MKPModel(self.config)
        torch.manual_seed(10)
        direct = MKPModel(replace(self.config, policy_head="direct_item"))
        for attr in ("critic_encoder",):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(getattr(weak, attr).parameters(), getattr(direct, attr).parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(weak.actor.encoder.parameters(), direct.actor.encoder.parameters())))

    def test_benchmark_reporting_uses_only_dynamic_heads_mocked(self):
        from Benchmark_zeroShot import benchmark
        from mkp_data import MKPInstance
        instance = MKPInstance("toy", [5, 3], [[2, 1], [1, 2]], [3, 3], 8)
        payload = {"update": 1, "train_config": {"train_num_items": [2]}, "training_data_source": "synthetic_only"}
        result = dict(objective=torch.tensor(8.), selected=torch.tensor([True, True]),
                      remaining=torch.tensor([0., 0.]), sequence=torch.tensor([0, 1]), theta_mean=torch.ones(5))
        calls = []
        def load(path, device, m):
            head = "direct_item" if "direct_item" in str(path) else "weak"
            return MKPModel(replace(self.config, policy_head=head)), payload
        def solve(instance, model, method, config, device):
            calls.append(method)
            return result, 0.01, [8.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "stub.mat"
            dataset.write_bytes(b"stub dataset; reader is mocked")
            for head in ("weak", "direct_item"):
                folder = root / "checkpoints" / variant_name(replace(self.config, policy_head=head))
                folder.mkdir(parents=True)
                (folder / "mkp_M2_best.pt").write_bytes(b"stub checkpoint; loader is mocked")
            config = BenchmarkConfig(dataset=dataset, num_resources=(2,), num_items=(2,),
                                     checkpoint_dir=root / "checkpoints", results_dir=root / "results",
                                     compare_heads=True, device="cpu")
            with patch("Benchmark_zeroShot.load_benchmark", return_value=[instance]), \
                 patch("Benchmark_zeroShot.load_checkpoint", side_effect=load), \
                 patch("Benchmark_zeroShot.solve_instance", side_effect=solve):
                output = benchmark(config)
            with (output / "instances.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(calls, ["learned_dynamic", "direct_item_masked"])
            self.assertEqual(len(rows), 2)
            self.assertNotIn("improvement_over_fixed_pct", rows[0])
            self.assertTrue((output / "head_comparison.csv").is_file())
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
