"""Resource/strategy integration and historical checkpoint contracts; CPU tensors."""
from dataclasses import fields, replace
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_config import (CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION,
                        ModelConfig, TrainConfig, config_dict)
from mkp_controller import StrategyReadout, TypeBalancedAttention
from mkp_env import MKPState
from mkp_Model import MKPModel, WeakDecoder
from mkp_performance import prepare_state
from mkp_Trainer import main
from mkp_utils import atomic_checkpoint, load_checkpoint
from test_contracts import toy_state
from test_opportunity_ranking import critical_state, labeled_batch
from mkp_ranking import ranking_objective


class StrategyReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(80)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    def test_type_balance_survives_candidate_duplication_and_masks_invalid_items(self):
        attention = TypeBalancedAttention(self.config)
        query, items, resources = torch.randn(2, 3, 16), torch.randn(2, 5, 16), torch.randn(2, 2, 16)
        valid = torch.tensor([[1, 1, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
        expected = attention(query, items, resources, valid)
        duplicated = attention(query, items.repeat(1, 7, 1), resources, valid.repeat(1, 7))
        torch.testing.assert_close(duplicated, expected, atol=1e-6, rtol=1e-5)
        changed = items.clone()
        changed[~valid] = 1000 * torch.randn_like(changed[~valid])
        torch.testing.assert_close(attention(query, changed, resources, valid), expected)
        expected.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in attention.parameters()))

    def test_roles_are_bounded_and_order_invariant_for_both_attention_ablations(self):
        items, resources, pooled = torch.randn(2, 4, 16), torch.randn(2, 3, 16), torch.randn(2, 16)
        valid = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
        order, resource_order = torch.tensor([3, 1, 0, 2]), torch.tensor([2, 0, 1])
        for balanced in (False, True):
            readout = StrategyReadout(replace(self.config, actor_type_balanced_pooling=balanced))
            roles = readout(items, resources, pooled, valid)
            self.assertEqual(roles.shape, (2, 9, 16))
            self.assertLessEqual((roles - pooled[:, None]).abs().max().item(), .100001)
            self.assertGreater((roles[:, 0] - roles[:, 1]).abs().sum().item(), 0.)
            permuted = readout(items[:, order], resources[:, resource_order], pooled, valid[:, order])
            torch.testing.assert_close(permuted, roles, atol=1e-6, rtol=1e-5)

    def test_all_architecture_ablations_keep_pair_channel_critic_and_fixed_score(self):
        state = critical_state()
        theta = WeakDecoder(self.config).fixed_theta(state)
        theta.companion_weights.fill_(.3)
        expected = WeakDecoder(self.config)(state, theta).logits
        for refinement, balanced, tokens in itertools.product((False, True), repeat=3):
            config = replace(self.config, actor_resource_refinement=refinement,
                             actor_type_balanced_pooling=balanced, actor_strategy_tokens=tokens)
            model = MKPModel(config)
            policy, value, actual, _ = model(state)
            self.assertEqual(len(fields(actual)), 9)
            self.assertEqual(actual.resource_interactions.shape, (1, 2, 2))
            torch.testing.assert_close(model.decoder(state, theta).logits, expected, rtol=0, atol=0)
            self.assertIsNotNone(model.actor.encoder.contrast_pool)
            self.assertEqual(model.actor.encoder.item_input.in_features, 12)
            self.assertTrue(model.actor.encoder.blocks[0].pair_attention)
            self.assertIsNone(model.critic_encoder.resource_refinement)
            self.assertIsNone(model.critic_encoder.balanced_attention)
            self.assertTrue(torch.isfinite(value).all() and torch.isfinite(policy.logits).all())

    def test_full_controller_permutations_after_heads_are_nonuniform(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            for p in model.actor.parameters():
                p.add_(.05 * torch.randn_like(p))
        state = toy_state().state
        items, resources = torch.tensor([2, 0, 3, 1]), torch.tensor([1, 0])
        permuted = MKPState(state.profits[:, items], state.weights[:, items][:, :, resources],
                            state.capacities[:, resources], state.remaining[:, resources], state.selected[:, items])
        first, second = model.actor(state)[0], model.actor(permuted)[0]
        for field in fields(first):
            expected = getattr(first, field.name)
            if field.name == "resource_prices":
                expected = expected[:, resources]
            elif field.name == "resource_interactions":
                expected = expected[:, resources][:, :, resources]
            torch.testing.assert_close(getattr(second, field.name), expected, atol=2e-6, rtol=1e-5)

    def test_actor_only_ranking_gradients_reach_resource_and_role_paths(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
            model.actor.strategy_head.weight.normal_(0, .1)
            model.actor.combination_head.companion_mix.weight.normal_(0, .1)
        loss = ranking_objective(model, labeled_batch(model, critical_state()), TrainConfig())[0]
        loss.backward()
        for module in (model.actor.encoder.resource_refinement, model.actor.readout,
                       model.actor.encoder.balanced_attention):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_legacy_checkpoint_disables_missing_modules_and_new_roundtrip_is_strict(self):
        for legacy in (True, False):
            config = replace(self.config, actor_resource_refinement=not legacy,
                             actor_type_balanced_pooling=not legacy, actor_strategy_tokens=not legacy)
            model = MKPModel(config)
            stored_config = config_dict(config)
            if legacy:
                for key in ("actor_resource_refinement", "actor_resource_gate_init", "actor_resource_gate_max",
                            "actor_type_balanced_pooling", "actor_strategy_tokens", "actor_readout_initial_scale"):
                    stored_config.pop(key)
            payload = dict(checkpoint_version=CHECKPOINT_VERSION, decoder_version=DECODER_VERSION,
                controller_version=CONTROLLER_VERSION, num_resources=2, training_data_source="synthetic_only",
                model_config=stored_config, model_state=model.state_dict())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                atomic_checkpoint(path, payload)
                restored, _ = load_checkpoint(path, "cpu", 2)
                torch.testing.assert_close(restored.policy(critical_state())[0].logits,
                                           model.policy(critical_state())[0].logits, rtol=0, atol=0)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                self.assertEqual(restored.actor.readout is None, legacy)
                if legacy:
                    payload["model_config"]["actor_strategy_tokens"] = True
                    atomic_checkpoint(path, payload)
                    with self.assertRaises(RuntimeError):
                        load_checkpoint(path, "cpu", 2)

    def test_cached_resource_geometry_and_cli_match_architecture_contract(self):
        model = MKPModel(self.config)
        state = prepare_state(toy_state().state, self.config)
        self.assertTrue({"resource_geometry", "resource_relations", "resource_pressure"} <= state._prepared.tensors.keys())
        with patch("mkp_Model.resource_geometry", side_effect=AssertionError("must reuse cache")):
            model.policy(state)
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--no-actor-resource-refinement", "--no-actor-type-balanced-pooling", "--no-actor-strategy-tokens",
                  "--actor-resource-gate-init", ".15", "--actor-readout-initial-scale", ".2",
                  "--ranking-target-mode", "legacy_mixed", "--ranking-candidates", "6"])
        train, config = trainer.call_args.args
        self.assertEqual(train.checkpoint_dir.name, "resource_strategy_v2")
        self.assertEqual((train.ranking_target_mode, train.ranking_candidates), ("legacy_mixed", 6))
        self.assertFalse(config.actor_resource_refinement or config.actor_type_balanced_pooling or config.actor_strategy_tokens)
        self.assertEqual((config.actor_resource_gate_init, config.actor_readout_initial_scale), (.15, .2))
        for kwargs in (dict(actor_resource_gate_init=0), dict(actor_resource_gate_init=.6),
                       dict(actor_resource_gate_max=2), dict(actor_resource_gate_max=float("nan")),
                       dict(actor_readout_initial_scale=1)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
