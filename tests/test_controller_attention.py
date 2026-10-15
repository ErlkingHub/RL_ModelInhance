"""Controller perception/gradient contracts; no optimizer steps or benchmarks."""
from dataclasses import replace
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_attention import candidate_relations, PairRelationAttention, CriticalContrastPool
from mkp_config import (CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION,
                        ModelConfig, TrainConfig, config_dict)
from mkp_env import MKPEnv, MKPState
from mkp_Model import BipartiteBlock, MKPModel, WeakDecoder
from mkp_ranking import ranking_objective
from mkp_utils import atomic_checkpoint, load_checkpoint
from test_opportunity_ranking import critical_state, labeled_batch


class ControllerAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1,
                                  actor_relation_chunk_size=2)

    def test_near_equal_items_have_different_future_and_resource_features(self):
        state = critical_state()
        relations = candidate_relations(state, self.config)
        # A blocks B+C; B blocks only A. Resource 0 causes A's loss of C.
        torch.testing.assert_close(relations.post[0, :2, 0], torch.tensor([15 / 25.05, 10.05 / 25.05]))
        self.assertEqual(relations.post[0, 0, 2].item(), 0)
        self.assertEqual(relations.post[0, 1, 2].item(), .5)
        self.assertEqual(relations.pairs[0, 0, 2, 3].item(), 1)
        self.assertEqual(relations.pairs[0, 1, 2, 3].item(), 0)
        self.assertGreater(relations.resource_post[0, 0, 0, 0], relations.resource_post[0, 0, 1, 0])
        # Perception follows current capacity, not just original capacities.
        state.remaining *= 2
        self.assertEqual(candidate_relations(state, self.config).post[..., 0].sum().item(), 0)

    def test_precision_at_capacity_boundary_and_zero_capacity(self):
        env = MKPEnv(torch.tensor([[10., 10., 5.]], dtype=torch.float64),
                     torch.tensor([[[5.00000001, 0.], [5., 0.], [5., 0.]]], dtype=torch.float64),
                     torch.tensor([[10., 1.]], dtype=torch.float64))
        env.state.remaining[0, 1] = 0
        relations = candidate_relations(env.state, self.config)
        self.assertEqual(relations.pairs[0, 0, 2, 3].item(), 1)
        self.assertEqual(relations.pairs[0, 1, 2, 3].item(), 0)
        self.assertTrue(all(torch.isfinite(x).all() for x in
                            (relations.pairs, relations.post, relations.resource_post)))

    def test_excludes_self_selected_and_infeasible_from_all_relation_features(self):
        state = MKPEnv(torch.tensor([[10.05, 10., 5., 100., 200.]]),
                       torch.tensor([[[5.2, 5.], [5., 5.2], [5., 1.], [1., 1.], [20., 20.]]]),
                       torch.tensor([[10., 10.]])).state
        state.selected[0, 3] = True
        first = candidate_relations(state, self.config)
        self.assertFalse(first.valid.diagonal(dim1=1, dim2=2).any())
        self.assertFalse(first.valid[:, 3:].any() or first.valid[:, :, 3:].any())
        self.assertEqual(first.post[:, 3:].abs().sum().item(), 0)
        self.assertEqual(first.resource_post[:, 3:].abs().sum().item(), 0)
        state.profits[:, 3:] *= 100
        second = candidate_relations(state, self.config)
        for name in ("pairs", "post", "resource_post"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_pair_geometry_and_attention_are_chunk_and_permutation_equivariant(self):
        state = critical_state()
        items_order, resource_order = torch.tensor([2, 0, 1]), torch.tensor([1, 0])
        permuted = MKPState(state.profits[:, items_order], state.weights[:, items_order][:, :, resource_order],
                            state.capacities[:, resource_order], state.remaining[:, resource_order],
                            state.selected[:, items_order])
        original = candidate_relations(state, self.config)
        reordered = candidate_relations(permuted, replace(self.config, actor_relation_chunk_size=1))
        torch.testing.assert_close(original.pairs[:, items_order][:, :, items_order], reordered.pairs)
        torch.testing.assert_close(original.post[:, items_order], reordered.post)
        torch.testing.assert_close(original.resource_post[:, items_order][:, :, resource_order], reordered.resource_post)
        attention = PairRelationAttention(self.config)
        inputs = torch.randn(1, 3, 16, requires_grad=True)
        output = attention(inputs, original)
        attention.chunk_size = 1
        torch.testing.assert_close(attention(inputs[:, items_order], reordered), output[:, items_order])
        first_grad = torch.autograd.grad(output.square().sum(), inputs)[0]
        second_grad = torch.autograd.grad(attention(inputs, original).square().sum(), inputs)[0]
        torch.testing.assert_close(first_grad, second_grad)

    def test_relation_values_distinguish_candidates_with_identical_node_embeddings(self):
        relations = candidate_relations(critical_state(), self.config)
        attention = PairRelationAttention(self.config)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            attention.relation_value.weight[:, 3] = 1  # blocked relation only
            attention.output.weight.copy_(torch.eye(16))
        output = attention(torch.zeros(1, 3, 16), relations)
        self.assertGreater((output[:, 0] - output[:, 1]).abs().sum().item(), 0)
        with torch.no_grad():
            attention.relation_value.weight.zero_()
        self.assertEqual(attention(torch.zeros(1, 3, 16), relations).abs().sum().item(), 0)

    def test_ineligible_tokens_cannot_pollute_feasible_attention_messages(self):
        config = replace(self.config, actor_pair_attention=False, actor_postdecision_features=False)
        block = BipartiteBlock(config, strategy=True).eval()
        items, resources, edges = torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 4, 2, 3)
        valid = torch.tensor([[True, True, False, False]])
        first_i, first_r = block(items, resources, edges, valid)
        changed_i, changed_e = items.clone(), edges.clone()
        changed_i[:, 2:] = torch.randn_like(changed_i[:, 2:]) * 100
        changed_e[:, 2:] = torch.randn_like(changed_e[:, 2:]) * 100
        next_i, next_r = block(changed_i, resources, changed_e, valid)
        torch.testing.assert_close(first_i[:, :2], next_i[:, :2])
        torch.testing.assert_close(first_r, next_r)

    def test_critical_contrast_retains_signed_differences_and_ignores_item_order(self):
        relations = candidate_relations(critical_state(), self.config)
        pool = CriticalContrastPool(self.config)
        captured = []
        handle = pool.message[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        items, pooled = torch.zeros(1, 3, 16), torch.randn(1, 16)
        result = pool(items, pooled, relations)
        handle.remove()
        # The same item embeddings still carry nonzero, signed post-decision differences.
        differences = captured[0][0, :, -4:]
        self.assertGreater(differences[0, 0].item(), 0)
        self.assertLess(differences[1, 0].item(), 0)
        changed = replace(relations, post=torch.zeros_like(relations.post))
        self.assertFalse(torch.allclose(result, pool(items, pooled, changed)))
        order = torch.tensor([2, 0, 1])
        reordered = replace(relations, pairs=relations.pairs[:, order][:, :, order],
                            valid=relations.valid[:, order][:, :, order], post=relations.post[:, order],
                            resource_post=relations.resource_post[:, order])
        torch.testing.assert_close(result, pool(items[:, order], pooled, reordered))

    def test_ranking_gradient_reaches_new_paths_without_critic_or_decoder_parameters(self):
        model = MKPModel(self.config)
        # Existing zero-head initialization is intentionally preserved. Nonzero
        # heads test the downstream path after the head's first update, without training.
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
            model.actor.strategy_head.weight.normal_(0, .1)
        loss = ranking_objective(model, labeled_batch(model, critical_state()), TrainConfig())[0]
        loss.backward()
        encoder = model.actor.encoder
        for parameter in (encoder.item_input.weight,
                          encoder.blocks[0].item_cross.edge_value.weight,
                          encoder.blocks[0].item_self.relation_bias.weight,
                          encoder.blocks[0].item_self.relation_value.weight,
                          encoder.contrast_pool.message[0].weight, encoder.contrast_pool.query.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(encoder.item_input.weight.grad[:, -4:].abs().sum().item(), 0)
        self.assertGreater(encoder.blocks[0].item_cross.edge_value.weight.grad[:, -2:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertNotIsInstance(model.decoder, torch.nn.Module)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_terminal_singleton_and_mixed_rows_have_finite_backward(self):
        for pair_attention in (False, True):
            model = MKPModel(replace(self.config, actor_pair_attention=pair_attention))
            state = MKPEnv(torch.tensor([[1.], [1.]]), torch.tensor([[[0., 0.]], [[2., 2.]]]),
                           torch.tensor([[1., 1.], [1., 1.]])).state
            for all_terminal in (False, True):
                if all_terminal:
                    state.selected.fill_(True)
                policy, value, theta, aux = model(state)
                self.assertEqual(policy.choose().tolist(), [-1, -1] if all_terminal else [0, -1])
                (value.square().mean() + aux.square().mean() + theta.profit_exponent.mean()).backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(torch.isfinite(theta.resource_prices).all())

    def test_independent_ablations_and_unchanged_decoder_theta_contract(self):
        state = critical_state()
        expected_decoder = WeakDecoder(self.config)
        theta = expected_decoder.fixed_theta(state)
        theta.opportunity_weight.fill_(.7)
        theta.reserve_weight.fill_(.3)
        expected = expected_decoder(state, theta)
        # All combinations must have valid dimensions, including pair attention
        # without post-feature injection and post features without pair attention.
        for flags in itertools.product((False, True), repeat=3):
            config = replace(self.config, actor_postdecision_features=flags[0],
                             actor_pair_attention=flags[1], actor_contrast_pooling=flags[2])
            model = MKPModel(config)
            policy, _, actual_theta, _ = model(state)
            self.assertTrue(torch.isfinite(policy.logits).all())
            self.assertEqual(actual_theta.resource_prices.shape, (1, 2))
            torch.testing.assert_close(model.decoder(state, theta).logits, expected.logits, rtol=0, atol=0)
        disabled = replace(self.config, actor_postdecision_features=False, actor_pair_attention=False,
                           actor_contrast_pooling=False, actor_feasible_attention=False)
        with patch("mkp_Model.candidate_relations", side_effect=AssertionError("disabled")):
            MKPModel(disabled).policy(state)
        critic = MKPModel(self.config).critic_encoder
        self.assertEqual(critic.item_input.in_features, 8)
        self.assertIsNone(critic.contrast_pool)
        self.assertIsInstance(critic.blocks[0].item_self, torch.nn.MultiheadAttention)

    def test_checkpoint_versions_cli_and_config(self):
        from mkp_Trainer import main
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--no-actor-postdecision-features", "--no-actor-pair-attention",
                  "--no-actor-contrast-pooling", "--no-actor-feasible-attention",
                  "--actor-relation-chunk-size", "3", "--actor-relation-bandwidth", ".07",
                  "--actor-similarity-bandwidth", ".2"])
        train, config = trainer.call_args.args
        self.assertEqual(train.checkpoint_dir.name, "profit_efficiency_v6")
        self.assertFalse(config.actor_postdecision_features or config.actor_pair_attention
                         or config.actor_contrast_pooling or config.actor_feasible_attention)
        self.assertEqual(config.actor_relation_chunk_size, 3)
        self.assertEqual(config.actor_relation_bandwidth, .07)
        self.assertEqual(config.actor_similarity_bandwidth, .2)
        for kwargs in (dict(actor_relation_chunk_size=0), dict(actor_relation_bandwidth=0),
                       dict(actor_similarity_bandwidth=float("nan"))):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        model = MKPModel(self.config)
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, decoder_version=DECODER_VERSION,
                       controller_version=CONTROLLER_VERSION, num_resources=2,
                       training_data_source="synthetic_only", model_config=config_dict(self.config),
                       model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_checkpoint(path, payload)
            restored, _ = load_checkpoint(path, "cpu", 2)
            torch.testing.assert_close(model.policy(critical_state())[0].logits,
                                       restored.policy(critical_state())[0].logits)
            payload["controller_version"] = "wrong-controller"
            atomic_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "high-level controller"):
                load_checkpoint(path, "cpu", 2)

    def test_variable_item_and_resource_counts_forward_only(self):
        model = MKPModel(self.config).eval()
        for n, m in ((7, 1), (100, 5), (250, 10), (500, 5)):
            state = MKPEnv(torch.rand(1, n) + .1, torch.rand(1, n, m), torch.ones(1, m) * 3).state
            with torch.no_grad():
                policy, theta, _ = model.policy(state)
            self.assertEqual(theta.resource_prices.shape, (1, m))
            self.assertTrue(torch.isfinite(policy.logits).all())


if __name__ == "__main__":
    unittest.main()
