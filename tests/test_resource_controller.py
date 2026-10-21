"""Resource perception and warm-start contracts; no optimizer steps or benchmarks."""
from dataclasses import fields, replace
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_config import DECODER_VERSION, ModelConfig, config_dict
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, state_features
from mkp_performance import cached_features, prepare_state
from mkp_resource import resource_geometry
from mkp_utils import atomic_checkpoint, file_sha256, warm_start_checkpoint


def resource_state():
    return MKPEnv(
        torch.tensor([[10., 8., 6., 3., 2.]]),
        torch.tensor([[[5., 2., 1.], [1., 4., 2.], [2., 1., 5.],
                       [3., 3., 1.], [.5, 1., 2.]]]),
        torch.tensor([[9., 8., 7.]])).state


def reordered_state(state, items=None, resources=None):
    if items is None:
        items = torch.arange(state.profits.shape[1])
    if resources is None:
        resources = torch.arange(state.capacities.shape[1])
    return MKPState(state.profits[:, items], state.weights[:, items][:, :, resources],
                    state.capacities[:, resources], state.remaining[:, resources],
                    state.selected[:, items])


class ResourceGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1,
                                  actor_resource_geometry=True, controller_resource_readout=True)

    def test_equal_marginal_demands_keep_different_resource_coupling(self):
        profit, capacity = torch.ones(1, 4), torch.tensor([[10., 10.]])
        aligned = MKPEnv(profit, torch.tensor([[[8., 8.], [8., 8.], [2., 2.], [2., 2.]]]), capacity).state
        opposed = MKPEnv(profit, torch.tensor([[[8., 2.], [8., 2.], [2., 8.], [2., 8.]]]), capacity).state
        # The original per-resource summaries cannot distinguish this witness.
        torch.testing.assert_close(state_features(aligned, self.config)[1],
                                   state_features(opposed, self.config)[1])
        first, second = resource_geometry(aligned, self.config), resource_geometry(opposed, self.config)
        torch.testing.assert_close(first.nodes, second.nodes)
        self.assertGreater(first.pairs[0, 0, 1, 0].item(), second.pairs[0, 0, 1, 0].item())
        self.assertAlmostEqual(first.pairs[0, 0, 1, 2].item(), 1., places=6)
        self.assertAlmostEqual(second.pairs[0, 0, 1, 2].item(), -1., places=6)
        self.assertEqual(first.pairs[0, 0, 1, 4].item(), 0.)
        self.assertGreater(second.pairs[0, 0, 1, 4].item(), 0.)

    def test_profit_pressure_distinguishes_identical_total_demand(self):
        state = MKPEnv(torch.tensor([[10., 1.]]),
                       torch.tensor([[[8., 2.], [2., 8.]]]), torch.tensor([[10., 10.]])).state
        geometry = resource_geometry(state, self.config)
        torch.testing.assert_close(geometry.nodes[0, :, 0], torch.tensor([.5, .5]))
        torch.testing.assert_close(geometry.nodes[0, :, 3], torch.tensor([8.2 / 11., 2.8 / 11.]))
        self.assertGreater(geometry.nodes[0, 0, 7].item(), geometry.nodes[0, 1, 7].item())
        self.assertGreater(geometry.nodes[0, 0, 9].item(), geometry.nodes[0, 1, 9].item())
        state.profits = state.profits.flip(1)
        reversed_profit = resource_geometry(state, self.config)
        torch.testing.assert_close(geometry.nodes[:, [1, 0]], reversed_profit.nodes)

    def test_geometry_item_resource_permutations_and_independent_units(self):
        state = resource_state()
        items, resources = torch.tensor([4, 2, 0, 3, 1]), torch.tensor([2, 0, 1])
        original = resource_geometry(state, self.config)
        reordered = resource_geometry(reordered_state(state, items, resources), self.config)
        torch.testing.assert_close(reordered.nodes, original.nodes[:, resources])
        torch.testing.assert_close(reordered.pairs, original.pairs[:, resources][:, :, resources])
        scale = torch.tensor([1e-5, 7e5, 3.], dtype=state.weights.dtype)
        scaled = MKPState(state.profits, state.weights * scale, state.capacities * scale,
                          state.remaining * scale, state.selected)
        actual = resource_geometry(scaled, self.config)
        torch.testing.assert_close(actual.nodes, original.nodes)
        torch.testing.assert_close(actual.pairs, original.pairs)

    def test_only_feasible_items_contribute_and_terminal_demand_is_zero(self):
        state = MKPEnv(torch.tensor([[10., 8., 999., 999.]]),
                       torch.tensor([[[5., 2.], [1., 4.], [2., 1.], [20., 30.]]]),
                       torch.tensor([[9., 8.]])).state
        state.selected[0, 2] = True
        original = resource_geometry(state, self.config)
        state.profits[:, 2:] *= 100
        state.weights[:, 2:] *= 100
        changed = resource_geometry(state, self.config)
        torch.testing.assert_close(original.nodes, changed.nodes, rtol=0, atol=0)
        torch.testing.assert_close(original.pairs, changed.pairs, rtol=0, atol=0)
        state.selected.fill_(True)
        terminal = resource_geometry(state, self.config)
        self.assertEqual(terminal.nodes.abs().sum().item(), 0.)
        self.assertEqual(terminal.pairs.abs().sum().item(), 0.)

    def test_zero_demand_resource_and_zero_profit_candidates_are_finite(self):
        state = MKPEnv(torch.tensor([[0., 0., 1.]]),
                       torch.tensor([[[0., 2.], [0., 3.], [0., 1.]]]),
                       torch.tensor([[1., 8.]])).state
        state.selected[0, 2] = True
        state.remaining[0, 0] = 0
        geometry = resource_geometry(state, self.config)
        self.assertTrue(torch.isfinite(geometry.nodes).all())
        self.assertTrue(torch.isfinite(geometry.pairs).all())
        self.assertEqual(geometry.nodes[0, 0].abs().sum().item(), 0.)
        self.assertGreater(geometry.nodes[0, 1, 0].item(), 0.)
        # Geometry is detached arithmetic, never a learned item-scoring path.
        self.assertFalse(geometry.nodes.requires_grad or geometry.pairs.requires_grad)


class ResourceControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1,
                                  actor_resource_geometry=True, controller_resource_readout=True,
                                  actor_relation_chunk_size=2, combination_chunk_size=2)

    @staticmethod
    def make_heads_live(model):
        # Simulate nonzero pretrained controller heads without optimization.
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
            model.actor.strategy_head.weight.normal_(0, .1)
            model.actor.combination_head.resource_pair[-1].weight.normal_(0, .1)
            model.actor.combination_head.interaction_gate.weight.normal_(0, .1)
            model.actor.combination_head.companion_mix.weight.normal_(0, .1)
            model.actor.combination_head.companion_gate.weight.normal_(0, .1)

    def assert_actor_only_gradients(self, model):
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value_head.parameters()))
        self.assertNotIsInstance(model.decoder, torch.nn.Module)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_residuals_preserve_disabled_baseline_with_common_weights(self):
        baseline = MKPModel(replace(self.config, actor_resource_geometry=False,
                                    controller_resource_readout=False)).eval()
        self.make_heads_live(baseline)
        enhanced = MKPModel(self.config).eval()
        missing, unexpected = enhanced.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all("resource_geometry_encoder" in k or "resource_readout" in k for k in missing))
        expected, actual = baseline(resource_state()), enhanced(resource_state())
        torch.testing.assert_close(actual[0].logits, expected[0].logits, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        for field in fields(expected[2]):
            torch.testing.assert_close(getattr(actual[2], field.name), getattr(expected[2], field.name),
                                       rtol=0, atol=0)

    def test_cached_policy_and_all_actor_gradients_match_with_active_residuals(self):
        model = MKPModel(self.config)
        # Nonzero new residuals ensure the cached geometry is actually consumed.
        with torch.no_grad():
            for p in model.actor.parameters():
                p.add_(torch.randn_like(p) * .03)
        plain, cached = resource_state(), prepare_state(resource_state(), self.config)
        self.assertIn("resource_node_geometry", cached._prepared.tensors)
        self.assertIn("resource_pair_geometry", cached._prepared.tensors)
        expected = model.policy(plain)[0]
        (-expected.log_prob(torch.tensor([0])).mean()).backward()
        gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        actual = model.policy(cached)[0]
        torch.testing.assert_close(actual.logits, expected.logits)
        (-actual.log_prob(torch.tensor([0])).mean()).backward()
        for key, p in model.named_parameters():
            if key in gradients:
                torch.testing.assert_close(p.grad, gradients[key])
            else:
                self.assertIsNone(p.grad)
        self.assert_actor_only_gradients(model)
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in cached._prepared.tensors.values()))
        self.assertIsNone(cached_features(cached, replace(self.config, actor_resource_geometry=False)))

    def test_zero_residual_outputs_receive_first_gradient_from_live_heads(self):
        model = MKPModel(self.config)
        self.make_heads_live(model)
        encoder, readout = model.actor.encoder.resource_geometry_encoder, model.actor.resource_readout
        outputs = (encoder.node_adapter[-1], encoder.relation_layers[0].output,
                   readout.pooled_output[-1], readout.resource_output[-1])
        self.assertTrue(all(layer.weight.count_nonzero().item() == 0 for layer in outputs))
        policy = model.policy(resource_state())[0]
        (-policy.log_prob(torch.tensor([0])).mean()).backward()
        for layer in outputs:
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0.)
        self.assert_actor_only_gradients(model)

    def test_active_residuals_propagate_actor_gradient_through_geometry_and_readout(self):
        model = MKPModel(self.config)
        self.make_heads_live(model)
        encoder, readout = model.actor.encoder.resource_geometry_encoder, model.actor.resource_readout
        with torch.no_grad():
            for layer in (encoder.node_adapter[-1], encoder.relation_layers[0].output,
                          readout.pooled_output[-1], readout.resource_output[-1]):
                layer.weight.normal_(0, .05)
        policy = model.policy(resource_state())[0]
        (-policy.log_prob(torch.tensor([0])).mean()).backward()
        for parameter in (encoder.node_adapter[0].weight, encoder.relation_layers[0].relation_bias.weight,
                          encoder.relation_layers[0].relation_value.weight,
                          readout.node_input[0].weight, readout.query.weight, readout.key.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        self.assert_actor_only_gradients(model)

    def legacy_checkpoint(self):
        source = MKPModel(replace(self.config, actor_resource_geometry=False,
                                  controller_resource_readout=False))
        self.make_heads_live(source)
        config = config_dict(source.config)
        del config["actor_resource_geometry"], config["controller_resource_readout"]
        del config["efficiency_gate_initial"], config["terminal_profit_override"]
        payload = dict(checkpoint_version=5, controller_version="combination_theta_v5",
                       decoder_version="second_order_density_v5", num_resources=3, update=123,
                       training_data_source="synthetic_only", model_config=config,
                       model_state={k: v for k, v in source.state_dict().items()
                                    if not k.startswith("actor.efficiency_head.")})
        return source, payload

    def test_legacy_warm_start_preserves_parameters_and_source_checkpoint(self):
        source, payload = self.legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            atomic_checkpoint(path, payload)
            checksum = file_sha256(path)
            target, provenance = warm_start_checkpoint(path, self.config, "cpu", expected_m=3)
            self.assertEqual(file_sha256(path), checksum)
            self.assertEqual(provenance["source_sha256"], checksum)
            self.assertEqual(provenance["source_update"], 123)
            self.assertTrue(provenance["optimizer_reset"])
            self.assertFalse(provenance["policy_outputs_preserved"])
            for name, value in payload["model_state"].items():
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
            # This fixture's source model already has the same initialized v9
            # gate; equality below tests the resource residual transfer only.
            expected, actual = source(resource_state()), target(resource_state())
            torch.testing.assert_close(actual[0].logits, expected[0].logits, rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            for field in fields(expected[2]):
                torch.testing.assert_close(getattr(actual[2], field.name), getattr(expected[2], field.name),
                                           rtol=0, atol=0)
            (-expected[0].log_prob(torch.tensor([0])).mean()).backward()
            (-actual[0].log_prob(torch.tensor([0])).mean()).backward()
            target_params = dict(target.named_parameters())
            for name, parameter in source.named_parameters():
                if parameter.grad is None:
                    self.assertIsNone(target_params[name].grad)
                else:
                    torch.testing.assert_close(target_params[name].grad, parameter.grad, rtol=0, atol=0)

    def test_warm_start_rejects_other_architectures_or_incomplete_weight_transfer(self):
        _, original = self.legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            for change in ("resource_count", "extra_config", "missing_weight", "unexpected_weight", "shape"):
                payload = dict(original, model_config=dict(original["model_config"]),
                               model_state=dict(original["model_state"]))
                if change == "resource_count":
                    payload["num_resources"] = 5
                elif change == "extra_config":
                    payload["model_config"]["controller_readout_mode"] = "reverted_variant"
                elif change == "missing_weight":
                    del payload["model_state"]["actor.price_head.weight"]
                elif change == "unexpected_weight":
                    payload["model_state"]["actor.old_readout.weight"] = torch.zeros(1)
                else:
                    payload["model_state"]["actor.price_head.weight"] = torch.zeros(2, 16)
                atomic_checkpoint(path, payload)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    warm_start_checkpoint(path, self.config, "cpu", expected_m=3)
            atomic_checkpoint(path, original)
            with self.assertRaisesRegex(ValueError, "unchanged v5 model settings"):
                warm_start_checkpoint(path, replace(self.config, price_floor=.2), "cpu", expected_m=3)

    def test_cli_resource_ablation_and_warm_start_do_not_resume_optimizer(self):
        from mkp_Trainer import main
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--no-actor-resource-geometry", "--no-controller-resource-readout",
                  "--warm-start", "existing_v5_weights"])
        _, config = trainer.call_args.args
        self.assertFalse(config.actor_resource_geometry or config.controller_resource_readout)
        trainer.return_value.train.assert_called_once_with(None, warm_start=Path("existing_v5_weights"))
        with patch("mkp_Trainer.MKPTrainer") as trainer, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--warm-start", "existing_v5_weights", "--resume", "new_v8_weights"])
            trainer.assert_not_called()

    def test_active_controller_is_equivariant_and_decoder_formula_is_unchanged(self):
        model = MKPModel(self.config).eval()
        with torch.no_grad():
            for p in model.actor.parameters():
                p.add_(torch.randn_like(p) * .03)
        state = resource_state()
        items, resources = torch.tensor([4, 2, 0, 3, 1]), torch.tensor([2, 0, 1])
        policy, theta, aux = model.policy(state)
        reordered, reordered_theta, reordered_aux = model.policy(reordered_state(state, items, resources))
        torch.testing.assert_close(reordered.logits, policy.logits[items], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(reordered_aux, aux[:, items])
        torch.testing.assert_close(reordered_theta.resource_prices, theta.resource_prices[:, resources])
        torch.testing.assert_close(reordered_theta.resource_interactions,
                                   theta.resource_interactions[:, resources][:, :, resources])
        for field in fields(theta):
            if field.name not in ("resource_prices", "resource_interactions"):
                torch.testing.assert_close(getattr(reordered_theta, field.name), getattr(theta, field.name))
        baseline = MKPModel(replace(self.config, actor_resource_geometry=False,
                                    controller_resource_readout=False))
        torch.testing.assert_close(baseline.decoder(state, theta).logits, policy.logits, rtol=0, atol=0)

    def test_independent_flags_and_terminal_single_resource_backward(self):
        for geometry in (False, True):
            for readout in (False, True):
                model = MKPModel(replace(self.config, actor_resource_geometry=geometry,
                                         controller_resource_readout=readout))
                state = MKPEnv(torch.tensor([[1.], [1.]]), torch.tensor([[[0.]], [[2.]]]),
                               torch.ones(2, 1)).state
                policy, value, theta, aux = model(state)
                self.assertEqual(policy.choose().tolist(), [0, -1])
                (theta.profit_exponent.sum() + value.square().sum() + aux.square().sum()).backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                critic_keys = model.critic_encoder.state_dict()
                self.assertFalse(any("resource_geometry" in key or "resource_readout" in key for key in critic_keys))


if __name__ == "__main__":
    unittest.main()
