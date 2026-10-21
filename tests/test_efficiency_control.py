"""Analytic gate/decoder contracts; no optimizer steps or benchmarks."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from mkp_config import ModelConfig
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, StrategyTheta, WeakDecoder
from mkp_performance import prepare_state


def counterexample(m=2):
    return MKPEnv(torch.tensor([[100., 101.]], dtype=torch.float64),
                  torch.tensor([[[1.] * m, [2.] * m]], dtype=torch.float64),
                  torch.full((1, m), 2., dtype=torch.float64)).state


def nonterminal_state():
    return MKPEnv(torch.tensor([[10., 11., 4.]], dtype=torch.float64),
                  torch.tensor([[[2., 1.], [1., 2.], [.5, .5]]], dtype=torch.float64),
                  torch.tensor([[4., 4.]], dtype=torch.float64)).state


class EfficiencyControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(193)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1,
                                  actor_relation_chunk_size=2)

    def test_old_positional_theta_keeps_full_cost_and_statistics_contract(self):
        theta = WeakDecoder(self.config).fixed_theta(nonterminal_state())
        legacy = StrategyTheta(*(getattr(theta, name) for name in
            ("resource_prices", "profit_exponent", "scarcity_exponent", "bottleneck_weight",
             "temperature", "opportunity_weight", "reserve_weight", "resource_interactions",
             "companion_weights")))
        self.assertEqual(legacy.efficiency_gate.tolist(), [1.])
        self.assertEqual(legacy.statistics().shape, (1, 9))
        torch.testing.assert_close(legacy.detach().efficiency_gate, theta.efficiency_gate)

    def test_analytic_bound_old_density_cannot_select_better_final_item(self):
        decoder = WeakDecoder(replace(self.config, terminal_profit_override=False))
        bound = 2 * math.log(101 / 100) - math.log(2) + 4 / 201
        self.assertLess(bound, -.65)
        for m in (1, 5, 10):
            state = counterexample(m)
            theta = decoder.fixed_theta(state)
            theta.profit_exponent.fill_(2)
            theta.scarcity_exponent.fill_(2)
            theta.bottleneck_weight.zero_()
            theta.reserve_weight.zero_()
            theta.opportunity_weight.fill_(4)
            theta.companion_weights.fill_(self.config.companion_weight_max / len(self.config.companion_scales))
            prices = torch.linspace(.1, 1., m)
            theta.resource_prices[0] = prices / prices.mean()
            policy = decoder(state, theta)
            self.assertEqual(policy.choose().item(), 0)
            self.assertAlmostEqual((policy.score_terms["score"][1] -
                                    policy.score_terms["score"][0]).item(), bound, places=6)
            _, reserve = decoder.opportunity_features(state, policy.owners, policy.items)
            self.assertGreater((reserve[1] - reserve[0]).item(), .24)
            self.assertEqual(policy.score_terms["companion_bonus"].sum().item(), 0)
            theta.efficiency_gate.zero_()
            self.assertEqual(decoder(state, theta).choose().item(), 1)

    def test_terminal_rule_overrides_all_cost_penalties(self):
        state = counterexample()
        decoder = WeakDecoder(self.config)
        theta = decoder.fixed_theta(state)
        theta.bottleneck_weight.fill_(2)
        theta.opportunity_weight.fill_(4)
        theta.reserve_weight.fill_(4)
        theta.resource_interactions.fill_(.1)
        policy = decoder(state, theta)
        self.assertEqual(policy.choose().item(), 1)
        self.assertEqual(policy.score_terms["effective_efficiency_gate"].tolist(), [0., 0.])
        self.assertTrue(policy.score_terms["terminal_single_pick"].all())
        theta.temperature.fill_(.05)
        self.assertEqual(decoder(state, theta).choose().item(), 1)

    def test_pair_detection_float64_boundary_and_ineligible_items(self):
        exact = MKPEnv(torch.ones(1, 2),
                       torch.tensor([[[.5], [.5]]], dtype=torch.float64), torch.ones(1, 1)).state
        outside = MKPEnv(torch.ones(1, 2),
                         torch.tensor([[[.5000000001], [.5]]], dtype=torch.float64), torch.ones(1, 1)).state
        self.assertFalse(WeakDecoder.terminal_single_pick(exact).item())
        self.assertTrue(WeakDecoder.terminal_single_pick(outside).item())
        exact.selected[0, 0] = True
        self.assertTrue(WeakDecoder.terminal_single_pick(exact).item())
        exact.selected.fill_(True)
        self.assertTrue(WeakDecoder.terminal_single_pick(exact).item())
        with self.assertRaises(ValueError):
            WeakDecoder.terminal_single_pick(exact, 0)

    def test_mixed_terminal_rows_zero_resources_and_chunk_invariance(self):
        state = MKPEnv(torch.tensor([[1., 2., 3.]]).expand(4, -1),
            torch.tensor([[[0., 0.], [0., 1.], [0., 2.]],
                          [[0., 0.], [0., 2.], [0., 3.]],
                          [[0., 2.], [0., 3.], [0., 4.]],
                          [[0., .6], [0., .7], [0., .8]]]),
            torch.tensor([[1., 1.]]).expand(4, -1)).state
        # Initial capacities must be positive; an exhausted resource may have
        # zero remaining capacity while zero-weight items are still feasible.
        state.remaining[:, 0] = 0
        for chunk in (1, 2, 32):
            self.assertEqual(WeakDecoder.terminal_single_pick(state, chunk).tolist(),
                             [False, True, True, True])
        model = MKPModel(self.config)
        policy, values, theta, aux = model(state)
        self.assertTrue(torch.isfinite(policy.logits).all())
        self.assertEqual(policy.choose()[2].item(), -1)
        (values.square().sum() + aux.square().sum() + theta.efficiency_gate.sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_gate_scales_resource_costs_but_preserves_companion_bonus(self):
        state = nonterminal_state()
        decoder = WeakDecoder(replace(self.config, terminal_profit_override=False))
        theta = decoder.fixed_theta(state)
        theta.opportunity_weight.fill_(1.2)
        theta.reserve_weight.fill_(.8)
        theta.resource_interactions.fill_(.2)
        theta.companion_weights.fill_(.3)
        full = decoder(state, theta)
        theta.efficiency_gate.zero_()
        zero = decoder(state, theta)
        profit = state.profits.float()[0] / state.profits.float().mean(-1)
        torch.testing.assert_close(zero.score_terms["score"], profit.log() + zero.score_terms["companion_bonus"])
        self.assertGreater(zero.score_terms["companion_bonus"].sum().item(), 0)
        self.assertEqual(zero.score_terms["interaction_penalty"].sum().item(), 0)
        theta.efficiency_gate.fill_(.37)
        partial = decoder(state, theta)
        torch.testing.assert_close(partial.score_terms["score"],
                                   .37 * full.score_terms["score"] + .63 * zero.score_terms["score"])
        torch.testing.assert_close(partial.score_terms["interaction_penalty"],
                                   .37 * full.score_terms["interaction_penalty"])
        self.assertFalse(WeakDecoder.terminal_single_pick(state).item())

    def test_gate_bounds_and_temperature_independent_actor_gradient(self):
        model = MKPModel(self.config)
        policy, theta, _ = model.policy(nonterminal_state())
        torch.testing.assert_close(theta.efficiency_gate,
                                   torch.tensor([self.config.efficiency_gate_initial]))
        self.assertTrue(((theta.efficiency_gate > 0) & (theta.efficiency_gate < 1)).all())
        raw = policy.logits * theta.temperature[policy.owners]
        (raw[0] - raw[2]).backward()
        self.assertGreater(model.actor.efficiency_head.bias.grad.abs().sum().item(), 0)
        self.assertGreater(model.actor.efficiency_head.weight.grad.abs().sum().item(), 0)
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))

    def test_item_resource_permutation_equivariance(self):
        state = nonterminal_state()
        model = MKPModel(self.config).eval()
        with torch.no_grad():
            model.actor.efficiency_head.weight.normal_(0, .2)
            model.actor.price_head.weight.normal_(0, .2)
            model.actor.strategy_head.weight.normal_(0, .2)
        items, resources = torch.tensor([2, 0, 1]), torch.tensor([1, 0])
        permuted = MKPState(state.profits[:, items], state.weights[:, items][:, :, resources],
                             state.capacities[:, resources], state.remaining[:, resources],
                             state.selected[:, items])
        first, theta, _ = model.policy(state)
        second, reordered, _ = model.policy(permuted)
        torch.testing.assert_close(theta.efficiency_gate, reordered.efficiency_gate)
        torch.testing.assert_close(first.logits[items], second.logits)

    def test_cached_terminal_flag_reused_without_recomputation(self):
        state = counterexample()
        decoder = WeakDecoder(self.config)
        theta = decoder.fixed_theta(state)
        expected = decoder(state, theta)
        prepare_state(state, self.config)
        self.assertIn("terminal_single_pick", state._prepared.tensors)
        with patch.object(decoder, "terminal_single_pick", side_effect=AssertionError("pair recomputed")):
            actual = decoder(state, theta)
        torch.testing.assert_close(expected.logits, actual.logits, rtol=0, atol=0)

    def test_fixed_baseline_retains_original_density_without_terminal_rule(self):
        from mkp_utils import construct_solution
        state = counterexample()
        with patch.object(WeakDecoder, "terminal_single_pick", side_effect=AssertionError("fixed baseline")):
            result = construct_solution((state.profits, state.weights, state.capacities),
                                        mode="fixed", model_config=self.config)
        self.assertEqual(result["objective"].item(), 100.)
        self.assertEqual(result["gate_mean"].tolist(), [[1., 1., 0.]])


if __name__ == "__main__":
    unittest.main()
