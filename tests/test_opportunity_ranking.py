"""Tiny synthetic interface fixtures; no optimizer updates or dataset evaluation."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import torch

from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv
from theta_fixtures import density_theta
from mkp_Model import MKPModel, WeakDecoder, CandidatePolicy, THETA_STATISTICS
from mkp_ranking import (RankingBatch, paired_terminal_returns, return_targets,
                         collect_ranking_batch, ranking_objective, ranking_kl)
from mkp_Trainer import MKPTrainer
from mkp_utils import atomic_checkpoint, load_checkpoint, construct_solution


def critical_state(device="cpu"):
    # A and B have equal total cost and near-identical profits. Only B leaves C feasible.
    return MKPEnv(torch.tensor([[10.05, 10., 5.]], dtype=torch.float64, device=device),
                  torch.tensor([[[5.2, 5.], [5., 5.2], [5., 1.]]], dtype=torch.float64, device=device),
                  torch.tensor([[10., 10.]], dtype=torch.float64, device=device)).state


def labeled_batch(model, state):
    with torch.no_grad():
        policy = model.policy(state)[0]
        old = torch.zeros_like(state.profits, dtype=torch.float32)
        old[policy.owners, policy.items] = policy.log_probs
    return RankingBatch(state, torch.tensor([0], device=state.profits.device),
                        torch.tensor([1], device=state.profits.device),
                        torch.tensor([-4.95 / 25.05], device=state.profits.device),
                        torch.zeros(1, device=state.profits.device), old)


class OpportunityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)
        self.train = TrainConfig(ranking_states=2, ranking_pool_size=4, ranking_repeats=2,
                                 ppo_epochs=1, minibatch_size=4, device="cpu")

    def test_similar_items_flip_towards_preserved_future_item(self):
        state = critical_state()
        decoder = WeakDecoder(self.config)
        theta = density_theta(state, decoder.config)
        self.assertEqual(decoder(state, theta).choose().item(), 0)
        theta.opportunity_weight.fill_(1.)
        self.assertEqual(decoder(state, theta).choose().item(), 1)
        owners, items = state.feasible.nonzero(as_tuple=True)
        blocked, reserve = decoder.opportunity_features(state, owners, items)
        self.assertAlmostEqual(blocked[0].item(), 15 / 25.05, places=6)
        self.assertAlmostEqual(blocked[1].item(), 10.05 / 25.05, places=6)
        self.assertTrue(torch.all((reserve >= 0) & (reserve <= 1)))

    def test_opportunity_formula_matches_independent_loop_and_chunks(self):
        state = critical_state()
        rows, cols = state.feasible.nonzero(as_tuple=True)
        expected_hard, expected_soft = [], []
        for i in range(3):
            hard, soft = 0., 0.
            for k in range(3):
                if k == i:
                    continue
                before = ((state.remaining[0] - state.weights[0, k]) / state.remaining[0]).min()
                after = ((state.remaining[0] - state.weights[0, i] - state.weights[0, k])
                         / state.remaining[0]).min()
                hard += float(after < 0) * state.profits[0, k]
                soft += (torch.sigmoid(before / .05) - torch.sigmoid(after / .05)) * state.profits[0, k]
            expected_hard.append(hard / state.profits.sum())
            expected_soft.append(soft / state.profits.sum())
        for chunk in (1, 2, 128):
            got = WeakDecoder(replace(self.config, opportunity_chunk_size=chunk)).opportunity_features(state, rows, cols)
            torch.testing.assert_close(got[0].double(), torch.stack(expected_hard), atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(got[1].double(), torch.stack(expected_soft), atol=1e-7, rtol=1e-6)

    def test_excludes_self_selected_and_infeasible(self):
        state = critical_state()
        state.selected[0, 1] = True
        state.remaining[0] = torch.tensor([5.2, 5.], dtype=torch.float64)
        rows, cols = state.feasible.nonzero(as_tuple=True)
        blocked, _ = WeakDecoder(self.config).opportunity_features(state, rows, cols)
        self.assertAlmostEqual(blocked[0].item(), 5 / 15.05, places=6)
        state.selected[0, 2] = True
        rows, cols = state.feasible.nonzero(as_tuple=True)
        hard, soft = WeakDecoder(self.config).opportunity_features(state, rows, cols)
        self.assertEqual(hard.item(), 0.)
        self.assertEqual(soft.item(), 0.)

    def test_capacity_boundary_uses_float64(self):
        state = MKPEnv(torch.ones(1, 2), torch.tensor([[[.5000000001], [.5]]], dtype=torch.float64),
                       torch.ones(1, 1)).state
        rows, cols = state.feasible.nonzero(as_tuple=True)
        hard, _ = WeakDecoder(self.config).opportunity_features(state, rows, cols)
        torch.testing.assert_close(hard, torch.tensor([.5, .5]))

    def test_ablation_is_exact_old_formula_and_critic_has_no_new_modules(self):
        config = replace(self.config, opportunity_weight_max=0, reserve_weight_max=0,
                         opportunity_tokens=0, actor_edge_values=False,
                         interaction_weight_max=0, companion_weight_max=0,
                         lp_primal_weight_max=0, lp_dual_weight_max=0)
        model = MKPModel(config)
        state = critical_state()
        theta = model.actor(state)[0]
        with patch.object(model.decoder, "opportunity_features", side_effect=AssertionError("disabled")):
            actual = model.decoder(state, theta)
        torch.testing.assert_close(actual.logits, model.decoder(state, density_theta(state, model.decoder.config)).logits)
        self.assertFalse(any("opportunity" in name or "edge_value" in name
                             for name, _ in model.critic_encoder.named_parameters()))

    def test_frozen_continuation_returns_true_terminal_profit(self):
        state = critical_state()
        before = state.remaining.clone(), state.selected.clone()
        model = MKPModel(self.config).eval()
        result = paired_terminal_returns(model, state, torch.tensor([0]), torch.tensor([1]), 1, False)
        torch.testing.assert_close(result, torch.tensor([[[10.05 / 25.05, 15 / 25.05]]]))
        torch.testing.assert_close(state.remaining, before[0])
        torch.testing.assert_close(state.selected, before[1])
        # A branch from an already-partial state retains its selected profit.
        env = MKPEnv.from_state(state)
        env.step(torch.tensor([2]))
        result = paired_terminal_returns(model, env.state, torch.tensor([1]), torch.tensor([1]), 1, False)
        torch.testing.assert_close(result, torch.tensor([[[15 / 25.05, 15 / 25.05]]]))

    def test_shared_item_noise_identical_actions_identical_returns(self):
        state = MKPEnv(torch.tensor([[3., 2., 4., 3., 2.]]),
                       torch.tensor([[[2., 2.], [1., 4.], [3., 1.], [2., 3.], [1., 2.]]]),
                       torch.tensor([[6., 6.]])).state
        model = MKPModel(self.config).eval()
        with torch.no_grad():
            model.actor.strategy_head.bias[3] = 10.  # maximal sampling temperature
        generator = torch.Generator().manual_seed(21)
        result = paired_terminal_returns(model, state, torch.tensor([0]), torch.tensor([0]), 24, True, generator)
        self.assertTrue(torch.equal(result[:, :, 0], result[:, :, 1]))
        self.assertGreater(result[:, :, 0].unique().numel(), 1)

    def test_target_filters_noise_small_gaps_and_conflicts(self):
        greedy = torch.tensor([[[.5, .5]], [[.8, .4]], [[.8, .4]]])
        sampled = torch.tensor([[[.5, .5], [.5, .5]], [[.8, .4], [.8, .4]], [[.4, .8], [.4, .8]]])
        delta, se, keep = return_targets(greedy, sampled, self.train)
        self.assertEqual(keep.tolist(), [False, True, False])
        self.assertEqual(se.tolist(), [0., 0., 0.])
        self.assertAlmostEqual(delta[1].item(), .4, places=6)
        noisy = torch.tensor([[[.8, .2], [.2, .8]]])
        _, _, keep = return_targets(torch.tensor([[[.51, .5]]]), noisy, self.train)
        self.assertFalse(keep.item())

    def test_collection_preserves_model_rng_and_mode(self):
        state = critical_state()
        model = MKPModel(self.config).train()
        snapshot = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        data = state.profits, state.weights, state.capacities
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool),
                       remaining=state.remaining[None], selected=state.selected[None])
        batch, metrics = collect_ranking_batch(model, data, rollout, self.train, seed=17)
        self.assertIsNotNone(batch)
        self.assertEqual(metrics["ranking_pairs"], 1)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, snapshot[k]) for k, v in model.state_dict().items()))
        self.assertFalse(batch.delta.requires_grad)
        with patch.object(model, "policy", side_effect=AssertionError("disabled")):
            batch, stats = collect_ranking_batch(model, data, rollout, replace(self.train, ranking_coefficient=0), 17)
        self.assertIsNone(batch)
        self.assertEqual(stats["ranking_proposed"], 0)

    def test_temperature_cannot_cheat_ranking_loss(self):
        model = MKPModel(self.config)
        batch = labeled_batch(model, critical_state())
        first = ranking_objective(model, batch, self.train)[0]
        first.backward()
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        with torch.no_grad():
            model.actor.strategy_head.bias[3].add_(2.)
        second = ranking_objective(model, batch, self.train)[0]
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_rank_gradient_reaches_controller_and_not_critic(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
            model.actor.strategy_head.weight.normal_(0, .1)
        loss, _, _ = ranking_objective(model, labeled_batch(model, critical_state()), self.train)
        loss.backward()
        for tensor in (model.actor.price_context[0].weight, model.actor.encoder.queries,
                       model.actor.encoder.blocks[0].item_cross.edge_value.weight,
                       model.actor.strategy_head.weight):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_kl_sees_competitor_outside_labeled_pair(self):
        model = MKPModel(self.config)
        batch = labeled_batch(model, critical_state())
        policy, theta, aux = model.policy(batch.state)
        altered = policy.logits.detach().clone()
        altered[policy.items == 2] = altered.max() + 1  # third item becomes competitive
        shifted = CandidatePolicy(policy.owners, policy.items, altered, 1)
        original_margin = policy.log_prob(batch.left) - policy.log_prob(batch.right)
        torch.testing.assert_close(original_margin, shifted.log_prob(batch.left) - shifted.log_prob(batch.right))
        with patch.object(model, "policy", return_value=(shifted, theta, aux)):
            self.assertGreater(ranking_kl(model, batch).item(), .03)

    def test_ppo_wiring_backward_without_optimizer_update(self):
        model = MKPModel(self.config)
        state = critical_state()
        batch = labeled_batch(model, state)
        with torch.no_grad():
            policy, value, _, _ = model(state)
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool), remaining=state.remaining[None],
                       selected=state.selected[None], actions=torch.tensor([[0]]),
                       log_probs=policy.log_prob(torch.tensor([0]))[None], values=value[None],
                       rewards=torch.tensor([[.4]]), dones=torch.ones(1, 1, dtype=torch.bool),
                       potentials=torch.tensor([[.45], [0.]]))
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config, trainer.device = self.train, torch.device("cpu")
        optimizer = Mock()
        optimizer.zero_grad.side_effect = lambda **kwargs: model.zero_grad(set_to_none=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with patch("mkp_Trainer.collect_ranking_batch", return_value=(batch, {})):
            stats = trainer.ppo_update(model, optimizer, (state.profits, state.weights, state.capacities), rollout)
        self.assertEqual(stats["ranking_minibatches"], 1)
        self.assertGreater(stats["ranking_loss"], 0)
        optimizer.step.assert_called_once()  # Mock only: no parameters changed.
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        optimizer.reset_mock()
        with patch("mkp_Trainer.collect_ranking_batch", return_value=(batch, {})), \
             patch("mkp_Trainer.ranking_kl", return_value=torch.tensor(1.)):
            stopped = trainer.ppo_update(model, optimizer, (state.profits, state.weights, state.capacities), rollout)
        optimizer.step.assert_not_called()
        self.assertEqual(stopped["kl_stopped"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_ranking_and_variable_resource_dimensions(self):
        from mkp_utils import seed_everything
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(41, deterministic=True, cpu_threads=2)
            model = MKPModel(self.config).cuda()
            state = critical_state("cuda")
            batch = labeled_batch(model, state)
            loss, kl, _ = ranking_objective(model, batch, self.train)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(kl))
            result = paired_terminal_returns(model, state, batch.left, batch.right, 2, True,
                                             torch.Generator(device="cuda").manual_seed(17))
            self.assertTrue(torch.isfinite(result).all())
            for m in (5, 10):
                env = MKPEnv(torch.rand(2, 7, device="cuda") + .1,
                             torch.rand(2, 7, m, device="cuda"), torch.ones(2, m, device="cuda") * 2)
                policy, _, theta, _ = model(env.state)
                self.assertEqual(theta.resource_prices.shape, (2, m))
                self.assertTrue(torch.isfinite(policy.logits).all())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_schema_rejects_old_checkpoint_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            for version in (1, 2, 3, 4, 5):
                atomic_checkpoint(path, dict(checkpoint_version=version))
                with self.assertRaisesRegex(ValueError, "fresh training"):
                    load_checkpoint(path, "cpu")

    def test_config_rejects_invalid_settings(self):
        for kwargs in (dict(ranking_repeats=1), dict(ranking_candidates=1),
                       dict(ranking_coefficient=-1), dict(ranking_min_gap=float("nan")),
                       dict(ranking_score_scale=0), dict(ranking_pool_size=1)):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)
        for kwargs in (dict(opportunity_tokens=-1), dict(opportunity_chunk_size=0),
                       dict(opportunity_bandwidth=0), dict(reserve_weight_max=-1)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_tiny_inference_stats_include_new_theta(self):
        state = critical_state()
        model = MKPModel(self.config)
        result = construct_solution((state.profits, state.weights, state.capacities), model)
        self.assertEqual(result["theta_mean"].shape, (1, len(THETA_STATISTICS)))
        self.assertEqual(result["decoder_mean"].shape, (1, 4))
        self.assertTrue(torch.isfinite(result["theta_mean"]).all())
        self.assertTrue((result["remaining"] >= 0).all())
        lp_weights = result["theta_mean"][0, [THETA_STATISTICS.index("lp_primal_weight"),
                                              THETA_STATISTICS.index("lp_dual_weight")]]
        torch.testing.assert_close(lp_weights, torch.full((2,), .5))

    def test_global_context_changes_relative_prices(self):
        model = MKPModel(self.config)
        d = self.config.embed_dim
        items, resources = torch.randn(1, 3, d), torch.randn(1, 2, d)
        global_state = torch.randn(1, d, requires_grad=True)
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .1)
        with patch.object(model.actor.encoder, "forward", return_value=(items, resources, global_state)):
            theta = model.actor(critical_state())[0]
        theta.resource_prices[0, 0].backward()
        self.assertGreater(global_state.grad.abs().sum().item(), 1e-6)

    def test_training_cli_passes_ablation_and_output_settings(self):
        from mkp_Trainer import main
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--resources", "5", "--ranking-coefficient", "0", "--ranking-states", "8",
                  "--ranking-repeats", "2", "--opportunity-weight-max", "0", "--reserve-weight-max", "0",
                  "--opportunity-tokens", "0", "--no-actor-edge-values", "--checkpoint-dir", "test-output"])
        train, model = trainer.call_args.args
        self.assertEqual(train.checkpoint_dir, Path("test-output"))
        self.assertEqual(train.ranking_coefficient, 0)
        self.assertEqual(train.ranking_states, 8)
        self.assertEqual(model.opportunity_weight_max, 0)
        self.assertFalse(model.actor_edge_values)
        trainer.return_value.train.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
