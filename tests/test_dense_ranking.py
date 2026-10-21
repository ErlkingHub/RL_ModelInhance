"""Training-label contracts on tiny synthetic states; no optimizer updates."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv
from mkp_Model import MKPModel
from mkp_ranking import (RankingBatch, candidate_terminal_returns, collect_ranking_batch,
                         dense_return_targets, ranking_objective, ranking_kl, return_targets)
from test_opportunity_ranking import critical_state, labeled_batch


class DenseRankingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.model = MKPModel(ModelConfig(embed_dim=16, num_heads=4, num_layers=1)).eval()
        self.config = TrainConfig(ranking_states=2, ranking_pool_size=4, ranking_repeats=2,
                                  ranking_branch_chunk_size=3, device="cpu")

    def test_small_greedy_gain_survives_stochastic_conflict(self):
        greedy = torch.tensor([[[.50008, .5, .499]]], dtype=torch.float64)
        sampled = torch.tensor([[[.49, .51, .52], [.49, .51, .52]]], dtype=torch.float64)
        best, delta, se, confidence, keep, conflicts = dense_return_targets(
            greedy, sampled, torch.tensor([.4]), torch.ones(1, 3, dtype=torch.bool), self.config)
        self.assertEqual(best.tolist(), [0])
        self.assertEqual(keep.tolist(), [[False, True, True]])
        self.assertEqual(conflicts.tolist(), [[False, True, True]])
        self.assertGreater(delta[0, 1].item(), 0)
        self.assertLess(delta[0, 1].item(), .001)
        self.assertAlmostEqual(confidence[0, 1].item(), self.config.ranking_confidence_floor)
        old_delta, _, old_keep = return_targets(greedy[:, :, :2], sampled[:, :, :2],
                                                replace(self.config, ranking_min_gap=.001))
        # The tiny greedy improvement misses the legacy conflict threshold, so
        # its stochastic mixture retains the opposite ordering in this fixture.
        self.assertTrue(old_keep.item())
        self.assertLess(old_delta.item(), 0)
        self.assertTrue(torch.equal(se, torch.zeros_like(se)))

    def test_residual_profit_normalization_is_prefix_invariant(self):
        greedy = torch.tensor([[[.15, .1499, .14]]], dtype=torch.float64)
        sampled = greedy.expand(-1, 2, -1)
        valid = torch.tensor([[True, True, False]])
        prefix = torch.tensor([.05], dtype=torch.float64)
        first = dense_return_targets(greedy, sampled, prefix, valid, self.config)
        shifted = dense_return_targets(greedy + .7, sampled + .7, prefix + .7, valid, self.config)
        torch.testing.assert_close(first[1], shifted[1])
        self.assertEqual(first[4].tolist(), [[False, True, False]])

    def test_candidate_returns_keep_small_difference_and_selected_prefix(self):
        env = MKPEnv(torch.tensor([[1e6, 1., 1.0001]], dtype=torch.float64),
                     torch.tensor([[[0.], [1.], [1.]]], dtype=torch.float64),
                     torch.tensor([[1.]], dtype=torch.float64))
        env.step(torch.tensor([0]))
        state = env.state
        before = state.remaining.clone(), state.selected.clone()
        actual = candidate_terminal_returns(self.model, state, torch.tensor([[1, 2]]),
                                             1, False, chunk_size=1)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertGreater(actual[0, 0, 1].item(), actual[0, 0, 0].item())
        expected = torch.tensor([[[1e6 + 1, 1e6 + 1.0001]]], dtype=torch.float64) / state.profits.sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state.remaining, before[0])
        torch.testing.assert_close(state.selected, before[1])

    def test_shared_noise_and_labels_do_not_depend_on_policy_batch_chunk(self):
        state = MKPEnv(torch.tensor([[3., 2., 4., 3., 2.]]),
                       torch.tensor([[[2., 2.], [1., 4.], [3., 1.], [2., 3.], [1., 2.]]]),
                       torch.tensor([[6., 6.]])).state
        candidates = torch.tensor([[0, 0, 1]])
        first = candidate_terminal_returns(self.model, state, candidates, 5, True,
                    torch.Generator().manual_seed(42), chunk_size=1)
        second = candidate_terminal_returns(self.model, state, candidates, 5, True,
                    torch.Generator().manual_seed(42), chunk_size=128)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(torch.equal(first[:, :, 0], first[:, :, 1]))

    def test_dense_collection_covers_third_candidate_and_keeps_frozen_state(self):
        state = critical_state()
        data = state.profits, state.weights, state.capacities
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool),
                       remaining=state.remaining[None], selected=state.selected[None])
        # Simulate a third candidate better than both first two. The previous
        # one-pair collector would never label it if its initial score was low.
        greedy = torch.tensor([[[.4, .5, .6, .4, .4, .4, .4, .4]]], dtype=torch.float64)
        sampled = greedy.expand(-1, 2, -1).clone()
        proposals = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]])
        valid = torch.tensor([[True, True, True, False, False, False, False, False]])
        snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.model.train()
        with patch("mkp_ranking._dense_candidates", return_value=(proposals, valid, torch.zeros(1))), \
             patch("mkp_ranking.candidate_terminal_returns", side_effect=[greedy, sampled]):
            batch, metrics = collect_ranking_batch(self.model, data, rollout, self.config, 17)
        self.assertEqual(metrics["ranking_proposed"], 2)
        self.assertEqual(metrics["ranking_pairs"], 2)
        self.assertEqual(metrics["ranking_candidates_labeled"], 3)
        self.assertEqual(batch.left.tolist(), [2, 2])
        self.assertEqual(batch.right.tolist(), [0, 1])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, snapshot[k]) for k, v in self.model.state_dict().items()))
        self.assertFalse(batch.delta.requires_grad)
        self.assertEqual(batch.index(torch.tensor([1])).confidence.shape, (1,))

    def test_confidence_reduces_gradient_scale_without_changing_target(self):
        batch = labeled_batch(self.model, critical_state())
        original_loss = ranking_objective(self.model, batch, self.config)[0]
        weighted = RankingBatch(batch.state, batch.left, batch.right, batch.delta,
                                batch.standard_error, batch.old_log_probs, torch.tensor([.25]))
        loss = ranking_objective(self.model, weighted, self.config)[0]
        torch.testing.assert_close(loss, original_loss * .25)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.critic_encoder.parameters()))
        self.assertIsNone(self.model.value_head.weight.grad)

    def test_loss_and_full_candidate_kl_balance_source_states_after_pair_expansion(self):
        state = critical_state().index(torch.tensor([0, 0]))
        state.profits[1] *= torch.tensor([1., 2., 3.])
        with torch.no_grad():
            policy = self.model.policy(state)[0]
            old = torch.zeros_like(state.profits, dtype=torch.float32)
            old[policy.owners, policy.items] = policy.log_probs
        base = RankingBatch(state, torch.tensor([0, 0]), torch.tensor([1, 1]),
                            torch.tensor([-.1, .2]), torch.zeros(2), old)
        expanded = base.index(torch.tensor([0, 0, 0, 1]))
        expanded.state_weights = torch.tensor([2 / 3, 2 / 3, 2 / 3, 2.])
        self.assertEqual(expanded.index(torch.tensor([3])).state_weights.item(), 2.)
        with torch.no_grad():
            self.model.actor.strategy_head.bias[0].add_(.7)
        torch.testing.assert_close(ranking_objective(self.model, base, self.config)[0],
                                   ranking_objective(self.model, expanded, self.config)[0])
        torch.testing.assert_close(ranking_kl(self.model, base), ranking_kl(self.model, expanded))

    def test_collector_assigns_equal_total_weight_to_states_with_different_pair_counts(self):
        state = critical_state().index(torch.tensor([0, 0]))
        data = state.profits, state.weights, state.capacities
        rollout = dict(active=torch.ones(1, 2, dtype=torch.bool),
                       remaining=state.remaining[None], selected=state.selected[None])
        candidates = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]]).repeat(2, 1)
        valid = torch.tensor([[True, True, True, False, False, False, False, False]]).repeat(2, 1)
        greedy = torch.tensor([[[.6, .5, .4]], [[.6, .6, .4]]], dtype=torch.float64)
        greedy = torch.cat((greedy, greedy[:, :, :1].expand(-1, -1, 5)), -1)
        sampled = greedy.expand(-1, 2, -1)
        with patch("mkp_ranking._dense_candidates", return_value=(candidates, valid, torch.zeros(2))), \
             patch("mkp_ranking.candidate_terminal_returns", side_effect=[greedy, sampled]):
            batch, stats = collect_ranking_batch(self.model, data, rollout, self.config, 17)
        self.assertEqual((stats["ranking_states_labeled"], stats["ranking_pairs"]), (2, 3))
        torch.testing.assert_close(batch.state_weights.sort().values, torch.tensor([.75, .75, 1.5], dtype=torch.float64))

    def test_legacy_mode_and_disabled_mode_remain_available(self):
        state = critical_state()
        data = state.profits, state.weights, state.capacities
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool),
                       remaining=state.remaining[None], selected=state.selected[None])
        batch, metrics = collect_ranking_batch(self.model, data, rollout,
                                  replace(self.config, ranking_target_mode="legacy_mixed"), 17)
        self.assertIsNotNone(batch)
        self.assertIsNone(batch.confidence)
        with patch.object(self.model, "policy", side_effect=AssertionError("disabled")):
            batch, metrics = collect_ranking_batch(self.model, data, rollout,
                                      replace(self.config, ranking_coefficient=0), 17)
        self.assertIsNone(batch)
        self.assertEqual(metrics["ranking_candidates_labeled"], 0)

    def test_new_config_controls_reject_invalid_settings(self):
        for fields in (dict(ranking_target_mode="oracle"), dict(ranking_confidence_floor=0),
                       dict(ranking_confidence_floor=float("nan")), dict(ranking_branch_chunk_size=0)):
            with self.assertRaises(ValueError):
                replace(self.config, **fields)


if __name__ == "__main__":
    unittest.main()
