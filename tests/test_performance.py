"""Execution/cache equivalence only; no optimizer updates or dataset evaluation."""
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch
import torch

from mkp_config import ModelConfig, TrainConfig, config_dict
from mkp_env import MKPEnv
from mkp_Model import MKPModel
from mkp_performance import (prepare_state, cached_features, build_rollout_cache,
                             configure_training_runtime)
from mkp_Trainer import MKPTrainer, main
from test_contracts import toy_state


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1,
                                  actor_relation_chunk_size=2, combination_chunk_size=2)

    def test_cached_logits_values_and_all_parameter_gradients_match(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            for p in model.actor.parameters():
                p.add_(torch.randn_like(p) * .03)
        plain, cached = toy_state().state, toy_state().state
        expected = model(plain)
        loss = expected[0].logits.square().mean() + expected[1].square().mean() + expected[3].square().mean()
        loss.backward()
        gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.reuse_state_features = True
        actual = model(cached)
        for a, b in ((actual[0].logits, expected[0].logits), (actual[1], expected[1]), (actual[3], expected[3])):
            torch.testing.assert_close(a, b)
        (actual[0].logits.square().mean() + actual[1].square().mean() + actual[3].square().mean()).backward()
        for key, p in model.named_parameters():
            if key in gradients:
                torch.testing.assert_close(p.grad, gradients[key])
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in cached._prepared.tensors.values()))
        # Model parameters are live even when state geometry is reused.
        with torch.no_grad():
            model.actor.strategy_head.bias[0].add_(1.)
        self.assertFalse(torch.equal(actual[0].logits, model(cached)[0].logits))

    def test_environment_step_and_tensor_replacement_invalidate_cache(self):
        env = toy_state()
        prepare_state(env.state, self.config)
        env.step(torch.tensor([0, -1]))
        self.assertIsNone(cached_features(env.state, self.config))
        fresh = MKPEnv.from_state(env.state).state
        torch.testing.assert_close(env.state.feasible, fresh.feasible)
        prepare_state(env.state, self.config)
        env.state.profits = env.state.profits + 1
        self.assertIsNone(cached_features(env.state, self.config))

    def test_index_preserves_order_and_config_changes_recompute(self):
        state = prepare_state(toy_state().state, self.config)
        indices = torch.tensor([1, 0, 0])
        part = state.index(indices)
        independently = prepare_state(state.index(indices, with_features=False), self.config)
        for key, value in part._prepared.tensors.items():
            torch.testing.assert_close(value, independently._prepared.tensors[key])
        old = part._prepared
        # A chunk change reuses exact geometry; a bandwidth/scale change cannot.
        prepare_state(part, replace(self.config, combination_chunk_size=99))
        self.assertIs(part._prepared, old)
        prepare_state(part, replace(self.config, opportunity_bandwidth=.2))
        self.assertIsNot(part._prepared, old)

    def test_state_bank_budget_and_minibatch_reuse(self):
        state = toy_state().state
        data = state.profits, state.weights, state.capacities
        rollout = dict(remaining=state.remaining[None].repeat(2, 1, 1),
                       selected=state.selected[None].repeat(2, 1, 1))
        times, batches = torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])
        model = MKPModel(self.config)
        model.reuse_state_features = True
        config = TrainConfig(feature_cache_batch_size=2)
        bank, stats = build_rollout_cache(model, data, rollout, times, batches, config)
        self.assertGreater(stats["feature_cache_gib"], 0)
        part = bank.index(torch.tensor([2, 0]))
        torch.testing.assert_close(part.profits, data[0][torch.tensor([1, 0])])
        with patch("mkp_attention.candidate_relations", side_effect=AssertionError("recomputed")), \
             patch("mkp_Model.candidate_relations", side_effect=AssertionError("recomputed")):
            model(part)
        for budget in (0., 1e-9):
            bank, stats = build_rollout_cache(model, data, rollout, times, batches,
                                               replace(config, feature_cache_gib=budget))
            self.assertIsNone(bank)
            self.assertEqual(stats["feature_cache_gib"], 0)

    def test_cached_ppo_wiring_without_optimizer_update(self):
        state = toy_state().state.index(torch.tensor([0]))
        model = MKPModel(self.config)
        model.reuse_state_features = True
        with torch.no_grad():
            policy, value, _, _ = model(state)
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool), remaining=state.remaining[None],
                       selected=state.selected[None], actions=torch.tensor([[0]]),
                       log_probs=policy.log_prob(torch.tensor([0]))[None], values=value[None],
                       rewards=torch.tensor([[.4]]), dones=torch.ones(1, 1, dtype=torch.bool))
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config = TrainConfig(ppo_epochs=2, ranking_coefficient=0)
        trainer.device = torch.device("cpu")
        optimizer = Mock()
        optimizer.zero_grad.side_effect = lambda **kwargs: model.zero_grad(set_to_none=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        stats = trainer.ppo_update(model, optimizer, (state.profits, state.weights, state.capacities), rollout)
        self.assertEqual(stats["minibatches"], 2)
        self.assertEqual(optimizer.step.call_count, 2)  # Mock only.
        self.assertGreater(stats["feature_cache_gib"], 0)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_cache_allocation_failure_falls_back_but_computation_errors_do_not(self):
        state = toy_state().state
        model = MKPModel(self.config)
        model.reuse_state_features = True
        args = (model, (state.profits, state.weights, state.capacities),
                dict(remaining=state.remaining[None], selected=state.selected[None]),
                torch.tensor([0]), torch.tensor([0]), TrainConfig())
        with patch("mkp_performance.prepare_state", side_effect=torch.cuda.OutOfMemoryError("test only")), \
             patch("torch.cuda.empty_cache"):
            bank, stats = build_rollout_cache(*args)
        self.assertIsNone(bank)
        self.assertEqual(stats["feature_cache_gib"], 0)
        with patch("mkp_performance.prepare_state", side_effect=RuntimeError("geometry error")):
            with self.assertRaisesRegex(RuntimeError, "geometry error"):
                build_rollout_cache(*args)

    def test_ablation_cache_matches_uncached_policy(self):
        for changes in (dict(actor_postdecision_features=False, actor_pair_attention=False,
                             actor_contrast_pooling=False),
                        dict(opportunity_weight_max=0., reserve_weight_max=0.,
                             interaction_weight_max=0., companion_weight_max=0.)):
            model = MKPModel(replace(self.config, **changes))
            expected = model(toy_state().state)
            model.reuse_state_features = True
            actual = model(toy_state().state)
            torch.testing.assert_close(actual[0].logits, expected[0].logits)
            torch.testing.assert_close(actual[1], expected[1])

    def test_auto_cpu_and_cli_execution_settings(self):
        model = MKPModel(self.config)
        runtime = configure_training_runtime(model, TrainConfig(), "cpu")
        self.assertEqual(runtime["profile"], "reference")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            configure_training_runtime(model, TrainConfig(performance_profile="memory24"), "cpu")
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--performance-profile", "memory24", "--feature-cache-gib", "4",
                  "--cache-headroom-gib", "7", "--feature-cache-batch-size", "32"])
        config = trainer.call_args.args[0]
        self.assertEqual((config.performance_profile, config.feature_cache_gib,
                          config.cache_headroom_gib, config.feature_cache_batch_size), ("memory24", 4., 7., 32))
        for kwargs in (dict(performance_profile="typo"), dict(feature_cache_gib=-1),
                       dict(feature_cache_gib=float("nan")), dict(cache_headroom_gib=-1),
                       dict(feature_cache_batch_size=0)):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_profile_chunking_and_checkpoint_weights_remain_compatible(self):
        from mkp_utils import seed_everything
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(31, deterministic=True, cpu_threads=2)
            model = MKPModel(self.config).cuda()
            original = config_dict(model.config)
            weights = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                expected = model(toy_state("cuda").state)
            runtime = configure_training_runtime(model, TrainConfig(performance_profile="memory24"), "cuda")
            self.assertEqual(runtime["combination_chunk_size"], 4096)
            self.assertEqual(config_dict(model.config), original)
            actual = model(toy_state("cuda").state)
            torch.testing.assert_close(actual[0].logits, expected[0].logits)
            torch.testing.assert_close(actual[1], expected[1])
            (actual[0].logits.sum() + actual[1].sum() + actual[3].sum()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(torch.equal(v, weights[k]) for k, v in model.state_dict().items()))
            loaded = MKPModel(self.config).cuda()
            loaded.load_state_dict(model.state_dict(), strict=True)
            configure_training_runtime(model, TrainConfig(performance_profile="reference"), "cuda")
            self.assertFalse(model.reuse_state_features)
            self.assertIs(model.actor.encoder.config, model.config)
        finally:
            torch.use_deterministic_algorithms(previous)
