"""LP relaxation features, bound shaping and LP decoder terms. No training or benchmark runs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mkp_config import GeneratorConfig, ModelConfig, TrainConfig
from mkp_data import synthetic_batch
from mkp_env import MKPEnv, MKPState
import mkp_lp
from mkp_lp import lp_potential, lp_relaxation, solve_lp_relaxation
from mkp_Model import EDGE_DIM, GLOBAL_DIM, ITEM_DIM, RESOURCE_DIM, MKPModel, state_features
from mkp_performance import prepare_state
from mkp_Trainer import collect_rollout, main, shaped_rewards
from theta_fixtures import density_theta

try:
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover - optional reference solver
    linprog = None


def partial_states(n, m, rows, seed, fraction=.2):
    data = synthetic_batch(rows, n, m, GeneratorConfig(), np.random.default_rng(seed), "cpu")
    env = MKPEnv(*data)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(int(n * fraction)):
        feasible = env.state.feasible
        active = feasible.any(-1)
        if not active.any():
            break
        picks = torch.multinomial(feasible.double() + 1e-12, 1, generator=generator).squeeze(-1)
        env.step(torch.where(active, picks, -1))
    return env


class LPRelaxationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    @unittest.skipIf(linprog is None, "scipy unavailable")
    def test_bound_is_valid_and_tight_against_reference_lp(self):
        for n, m in ((40, 5), (120, 10)):
            state = partial_states(n, m, 6, seed=n + m).state
            lp = solve_lp_relaxation(state, self.config)
            for row in range(state.profits.shape[0]):
                feasible = state.feasible[row].numpy()
                reference = linprog(-state.profits[row].numpy()[feasible],
                                    A_ub=state.weights[row].numpy()[feasible].T,
                                    b_ub=state.remaining[row].numpy(), bounds=(0, 1), method="highs")
                optimum = -reference.fun
                bound = lp.bound[row].item()
                # Weak duality: any nonnegative dual gives an upper bound.
                self.assertGreaterEqual(bound, optimum * (1 - 1e-9))
                self.assertLess((bound - optimum) / optimum, 2e-4)
                # The smoothed primal is a feature, approximately (not exactly) capacity-feasible.
                usage = (lp.primal[row].double()[:, None] * state.weights[row]).sum(0)
                self.assertTrue(((usage - state.remaining[row]) / state.remaining[row] < 5e-2).all())

    def test_terminal_exhausted_and_zero_weight_rows(self):
        env = MKPEnv(torch.tensor([[3., 2., 1.], [1., 1., 1.]]),
                     torch.tensor([[[0., 1.], [0., 0.], [2., 5.]], [[1., 1.], [1., 1.], [1., 1.]]]),
                     torch.tensor([[1., 2.], [1., 1.]]))
        env.state.remaining[0, 0] = 0
        env.state.selected[1] = True
        lp = solve_lp_relaxation(env.state, self.config)
        for tensor in (lp.primal, lp.reduced, lp.dual, lp.slack, lp.dual_share, lp.bound):
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(lp.bound[1].item(), 0.)
        self.assertEqual(lp.primal[1].abs().sum().item(), 0.)
        # Both feasible items of row 0 fit together, so the bound is their profit.
        self.assertAlmostEqual(lp.bound[0].item(), 5., places=3)
        self.assertEqual(lp.primal[0, 2].item(), 0.)
        self.assertEqual(lp_potential(env.state, self.config)[1].item(), 0.)

    def test_units_and_permutations_do_not_change_relaxation(self):
        state = partial_states(30, 4, 3, seed=9).state
        base = solve_lp_relaxation(state, self.config)
        scale = torch.tensor([2., .5, 10., 3.], dtype=torch.float64)
        scaled = MKPState(state.profits * 7, state.weights * scale, state.capacities * scale,
                          state.remaining * scale, state.selected.clone())
        other = solve_lp_relaxation(scaled, self.config)
        torch.testing.assert_close(other.primal, base.primal, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(other.dual, base.dual, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(other.bound, base.bound * 7, rtol=1e-6, atol=1e-6)
        items, resources = torch.randperm(30), torch.tensor([2, 0, 3, 1])
        permuted = MKPState(state.profits[:, items], state.weights[:, items][:, :, resources],
                            state.capacities[:, resources], state.remaining[:, resources],
                            state.selected[:, items])
        other = solve_lp_relaxation(permuted, self.config)
        torch.testing.assert_close(other.primal, base.primal[:, items], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(other.dual, base.dual[:, resources], atol=1e-4, rtol=1e-4)

    def test_one_solve_per_state_shared_by_actor_critic_decoder(self):
        model = MKPModel(self.config)
        env = partial_states(12, 3, 2, seed=4)
        with patch("mkp_lp.solve_lp_relaxation", wraps=mkp_lp.solve_lp_relaxation) as solver:
            model(env.state)
            self.assertEqual(solver.call_count, 1)
            policy = model.decoder(env.state, model.actor(env.state)[0])
            self.assertEqual(solver.call_count, 1)
            env.step(policy.choose())
            model(env.state)
            self.assertEqual(solver.call_count, 2)

    def test_bipartite_inputs_include_lp_channels(self):
        state = partial_states(10, 3, 2, seed=5).state
        item, resource, edge, global_ = state_features(state, self.config)
        self.assertEqual((item.shape[-1], resource.shape[-1], edge.shape[-1], global_.shape[-1]),
                         (ITEM_DIM, RESOURCE_DIM, EDGE_DIM, GLOBAL_DIM))
        lp = lp_relaxation(state, self.config)
        torch.testing.assert_close(item[..., 8], lp.primal)
        torch.testing.assert_close(global_[:, 6], lp_potential(state, self.config))
        for tensor in (item, resource, edge, global_):
            self.assertTrue(torch.isfinite(tensor).all())
        model = MKPModel(self.config)
        self.assertEqual(model.critic_encoder.item_input.in_features, ITEM_DIM)
        self.assertEqual(model.critic_encoder.resource_input.in_features, RESOURCE_DIM)


class LPDecoderAndShapingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    def test_lp_terms_are_exact_theta_weighted_additions(self):
        state = partial_states(15, 3, 2, seed=11).state
        model = MKPModel(self.config)
        theta = density_theta(state, self.config)
        reference = model.decoder(state, theta).score_terms["score"]
        theta.lp_primal_weight[:] = torch.tensor([1.5, .3])
        theta.lp_dual_weight[:] = torch.tensor([.7, 2.])
        policy = model.decoder(state, theta)
        lp = lp_relaxation(state, self.config)
        rows, cols = policy.owners, policy.items
        expected = (reference + theta.lp_primal_weight[rows] * (lp.primal[rows, cols] + self.config.lp_primal_floor).log()
                    + theta.lp_dual_weight[rows] * lp.reduced[rows, cols])
        torch.testing.assert_close(policy.score_terms["score"], expected)

    def test_lp_terms_steer_choice_towards_relaxation(self):
        # Density prefers A (12 for one full resource); LP keeps B and C, which fit together for 18.
        state = MKPEnv(torch.tensor([[12., 9., 9.]]), torch.tensor([[[10., 1.], [5., 5.], [5., 5.]]]),
                       torch.tensor([[10., 10.]])).state
        model = MKPModel(self.config)
        theta = density_theta(state, self.config)
        self.assertEqual(model.decoder(state, theta).choose().item(), 0)
        theta.lp_primal_weight[:] = 1.
        self.assertIn(model.decoder(state, theta).choose().item(), (1, 2))

    def test_lp_controls_receive_policy_gradient(self):
        model = MKPModel(self.config)
        state = partial_states(12, 3, 3, seed=2).state
        policy, _, theta, _ = model(state)
        self.assertTrue(torch.allclose(theta.lp_primal_weight, torch.full((3,), .5)))
        self.assertTrue(torch.allclose(theta.lp_dual_weight, torch.full((3,), .5)))
        (-policy.log_prob(policy.choose(stochastic=True)).sum()).backward()
        self.assertGreater(model.actor.strategy_head.weight.grad[6:].abs().sum().item(), 0)

    def test_zero_bounds_skip_lp_scores_but_keep_features(self):
        config = replace(self.config, lp_primal_weight_max=0, lp_dual_weight_max=0)
        model = MKPModel(config)
        state = partial_states(12, 3, 2, seed=6).state
        theta = model.actor(state)[0]
        self.assertEqual(theta.lp_primal_weight.abs().sum().item(), 0)
        self.assertEqual(theta.lp_dual_weight.abs().sum().item(), 0)
        torch.testing.assert_close(model.decoder(state, theta).score_terms["score"],
                                   model.decoder(state, theta).score_terms["base"]
                                   - model.decoder(state, theta).score_terms["interaction_penalty"]
                                   + model.decoder(state, theta).score_terms["companion_bonus"])

    def test_cached_and_live_lp_scores_agree(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            model.actor.strategy_head.weight.normal_(0, .1)
        state = partial_states(14, 3, 3, seed=8).state
        live = MKPState(*(x.clone() for x in (state.profits, state.weights, state.capacities,
                                               state.remaining, state.selected)))
        prepare_state(state, model.config)
        with torch.no_grad():
            torch.testing.assert_close(model.policy(state)[0].logits, model.policy(live)[0].logits)

    def test_shaped_returns_telescope_to_objective_minus_root_bound(self):
        model = MKPModel(self.config)
        data = synthetic_batch(3, 12, 3, GeneratorConfig(), np.random.default_rng(1), "cpu")
        rollout = collect_rollout(model, data)
        self.assertEqual(rollout["potentials"].shape[0], rollout["rewards"].shape[0] + 1)
        self.assertTrue((rollout["potentials"][-1] == 0).all())
        shaped = shaped_rewards(rollout["rewards"], rollout["potentials"], rollout["active"].float())
        root = lp_potential(MKPEnv(*data).state, self.config)
        expected = (rollout["objective"] / data[0].sum(-1)).float() - root
        torch.testing.assert_close(shaped.sum(0), expected, atol=1e-6, rtol=1e-5)
        # The bound is an upper bound on the reachable remainder, so shaped returns are <= 0.
        self.assertTrue((shaped.sum(0) <= 1e-6).all())

    def test_config_and_cli(self):
        for kwargs in (dict(lp_temperatures=()), dict(lp_temperatures=(.1, .3)), dict(lp_newton_steps=0),
                       dict(lp_primal_weight_max=-1), dict(lp_primal_floor=0), dict(lp_dual_initial=float("nan"))):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        with self.assertRaises(ValueError):
            TrainConfig(gamma=.99)
        TrainConfig(gamma=.99, lp_reward_shaping=False)
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--no-lp-reward-shaping", "--lp-primal-weight-max", "2", "--lp-dual-weight-max", "0"])
        train, model = trainer.call_args.args
        self.assertFalse(train.lp_reward_shaping)
        self.assertEqual((model.lp_primal_weight_max, model.lp_dual_weight_max), (2., 0.))


if __name__ == "__main__":
    unittest.main()
