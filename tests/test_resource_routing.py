"""CPU contracts for resource routing; no optimization or benchmark data."""
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from mkp_config import (ModelConfig, TrainConfig, config_dict, CHECKPOINT_VERSION,
                        CONTROLLER_VERSION, DECODER_VERSION)
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, WeakDecoder
from mkp_performance import prepare_state
from mkp_routing import routing_geometry, ResourceRoutingBlock, StrategyRoleReadout
from mkp_utils import atomic_checkpoint, load_checkpoint
from test_opportunity_ranking import critical_state, labeled_batch
from mkp_ranking import ranking_objective


def example():
    return MKPEnv(torch.tensor([[10., 8., 6., 3., 2.]]),
                  torch.tensor([[[5., 2., 1.], [1., 4., 2.], [2., 1., 5.],
                                 [3., 3., 1.], [.5, 1., 2.]]]),
                  torch.tensor([[9., 8., 7.]])).state


class ResourceRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(204)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=2)

    def test_resource_coupling_distinguishes_equal_marginal_demands(self):
        p, cap = torch.ones(1, 4), torch.tensor([[10., 10.]])
        aligned = MKPEnv(p, torch.tensor([[[8., 8.], [8., 8.], [2., 2.], [2., 2.]]]), cap).state
        opposed = MKPEnv(p, torch.tensor([[[8., 2.], [8., 2.], [2., 8.], [2., 8.]]]), cap).state
        first, second = routing_geometry(aligned, self.config), routing_geometry(opposed, self.config)
        torch.testing.assert_close(first.resources[..., :5], second.resources[..., :5])
        self.assertAlmostEqual(first.pairs[0, 0, 1, 1].item(), 1, places=6)
        self.assertAlmostEqual(second.pairs[0, 0, 1, 1].item(), -1, places=6)
        self.assertGreater(first.pairs[0, 0, 1, 0].item(), second.pairs[0, 0, 1, 0].item())

    def test_profit_weighted_pressure_and_unit_invariance(self):
        state = MKPEnv(torch.tensor([[10., 1.]]), torch.tensor([[[8., 2.], [2., 8.]]]),
                       torch.tensor([[10., 10.]])).state
        geometry = routing_geometry(state, self.config)
        self.assertGreater(geometry.resources[0, 0, 3].item(), geometry.resources[0, 1, 3].item())
        scale = torch.tensor([1e-4, 1e5], dtype=torch.float64)
        scaled = MKPState(state.profits * 31, state.weights * scale, state.capacities * scale,
                          state.remaining * scale, state.selected)
        actual = routing_geometry(scaled, self.config)
        for f in fields(geometry):
            torch.testing.assert_close(getattr(actual, f.name), getattr(geometry, f.name))

    def test_nonuniform_theta_and_logits_are_permutation_equivariant(self):
        model = MKPModel(self.config).eval()
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.add_(torch.randn_like(parameter) * .04)
        state = example()
        ip, rp = torch.tensor([4, 2, 0, 3, 1]), torch.tensor([2, 0, 1])
        perm = MKPState(state.profits[:, ip], state.weights[:, ip][:, :, rp],
                        state.capacities[:, rp], state.remaining[:, rp], state.selected[:, ip])
        a, ta, _ = model.policy(state)
        b, tb, _ = model.policy(perm)
        torch.testing.assert_close(tb.resource_prices, ta.resource_prices[:, rp])
        torch.testing.assert_close(tb.resource_interactions, ta.resource_interactions[:, rp][:, :, rp])
        torch.testing.assert_close(tb.companion_weights, ta.companion_weights)
        torch.testing.assert_close(b.logits, a.logits[ip])

    def test_invalid_candidates_cannot_change_actor_representation(self):
        state = example()
        state.selected[0, 3] = True
        state.weights[0, 4] = 20.
        model = MKPModel(self.config)
        first = model.actor.encoder(state)
        state.profits[:, 3:] *= 1000
        state.weights[:, 3:] *= 1000
        second = model.actor.encoder(state)
        for original, changed in zip(first, second):
            torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_gather_changes_broadcast_within_the_same_round(self):
        block = ResourceRoutingBlock(self.config)
        state = example()
        geometry = routing_geometry(state, self.config)
        i, r = torch.randn(1, 5, 16), torch.randn(1, 3, 16)
        first = block(i, r, geometry, state.feasible)[0]
        with torch.no_grad():
            block.gather.output.bias.add_(torch.linspace(-1, 1, 16))
        second = block(i, r, geometry, state.feasible)[0]
        self.assertGreater((first - second).abs().max().item(), 1e-3)

    def test_ranking_gradient_reaches_routing_and_roles_without_critic(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
            model.actor.strategy_head.weight.normal_(0, .1)
            model.actor.combination_head.resource_pair[-1].weight.normal_(0, .1)
            model.actor.combination_head.companion_mix.weight.normal_(0, .1)
        loss = ranking_objective(model, labeled_batch(model, critical_state()), TrainConfig())[0]
        loss.backward()
        block = model.actor.encoder.blocks[0]
        for p in (block.gather.edge_value.weight, block.compete.bias[0].weight,
                  block.compete.edge_value.weight, block.broadcast.edge_value.weight,
                  model.actor.encoder.resource_input.weight, model.actor.role_readout.queries,
                  model.actor.role_readout.resource_k.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertNotIsInstance(model.decoder, torch.nn.Module)

    def test_readout_population_replication_and_empty_rows(self):
        readout = StrategyRoleReadout(self.config)
        i, r, p = torch.randn(2, 4, 16), torch.randn(2, 3, 16), torch.randn(2, 16)
        valid = torch.tensor([[True, False, True, True], [False] * 4])
        first = readout(i, r, p, valid)
        second = readout(i.repeat_interleave(7, dim=1), r, p, valid.repeat_interleave(7, dim=1))
        torch.testing.assert_close(first, second)
        third = readout(i, r.repeat_interleave(3, dim=1), p, valid)
        torch.testing.assert_close(first, third)
        self.assertTrue(torch.isfinite(first).all())
        first.square().mean().backward()
        self.assertTrue(all(x.grad is None or torch.isfinite(x.grad).all() for x in readout.parameters()))

    def test_actor_cache_has_no_item_pairs_and_handles_large_item_counts(self):
        model = MKPModel(self.config)
        for n, m in ((1, 1), (100, 5), (250, 10), (500, 10)):
            state = MKPEnv(torch.rand(2, n) + .1, torch.rand(2, n, m), torch.ones(2, m) * 3).state
            state.selected[1].fill_(True)
            with patch("mkp_attention.candidate_relations", side_effect=AssertionError("quadratic actor")), \
                 patch("mkp_Model.candidate_relations", side_effect=AssertionError("quadratic actor")):
                prepare_state(state, self.config)
                theta, aux = model.actor(state)
            self.assertNotIn("relation_pairs", state._prepared.tensors)
            self.assertEqual(state._prepared.tensors["routing_pairs"].shape, (2, m, m, 5))
            self.assertEqual(theta.resource_prices.shape, (2, m))
            self.assertTrue(torch.isfinite(aux).all())
            (aux.square().mean() + theta.profit_exponent.mean()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            model.zero_grad(set_to_none=True)

    def test_both_encoders_and_readout_ablations_keep_fixed_theta_scores(self):
        state = example()
        theta = WeakDecoder(self.config).fixed_theta(state)
        expected = WeakDecoder(self.config)(state, theta).logits
        for encoder in ("legacy", "resource_routing_v1"):
            for readout in (False, True):
                c = replace(self.config, actor_encoder=encoder, actor_strategy_readout=readout)
                model = MKPModel(c)
                self.assertTrue(torch.isfinite(model.policy(state)[0].logits).all())
                torch.testing.assert_close(model.decoder(state, theta).logits, expected, rtol=0, atol=0)

    def test_legacy_checkpoint_without_new_fields_strictly_restores_legacy(self):
        c = replace(self.config, actor_encoder="legacy", actor_strategy_readout=False)
        model = MKPModel(c)
        values = config_dict(c)
        del values["actor_encoder"], values["actor_strategy_readout"]
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, controller_version=CONTROLLER_VERSION,
                       decoder_version=DECODER_VERSION, training_data_source="synthetic_only",
                       num_resources=3, model_config=values, model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            atomic_checkpoint(path, payload)
            restored, _ = load_checkpoint(path, "cpu", 3)
        self.assertEqual(restored.config.actor_encoder, "legacy")
        self.assertIsNone(restored.actor.role_readout)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(model.policy(example())[0].logits,
                                   restored.policy(example())[0].logits, rtol=0, atol=0)

    def test_cli_records_architecture_and_rejects_invalid_dimensions(self):
        from mkp_Trainer import main
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--actor-encoder", "legacy", "--no-actor-strategy-readout",
                  "--embed-dim", "32", "--num-heads", "4", "--num-layers", "3"])
        train, c = trainer.call_args.args
        self.assertEqual((c.actor_encoder, c.actor_strategy_readout, c.embed_dim, c.num_layers),
                         ("legacy", False, 32, 3))
        self.assertEqual(train.checkpoint_dir.name, "resource_routing_v1")
        for kwargs in (dict(actor_encoder="unknown"), dict(embed_dim=17, num_heads=4)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
