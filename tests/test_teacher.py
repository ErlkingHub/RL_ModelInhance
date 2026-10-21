"""Independent tiny solver checks; synthetic states only, no model training."""
from dataclasses import replace
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mkp_config import TrainConfig
from mkp_env import MKPEnv
from mkp_teacher import solve_teacher


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainConfig(teacher_exact_items=16)
        self.state = MKPEnv(torch.tensor([[10., 8., 7., 5.]]),
            torch.tensor([[[5., 3.], [3., 4.], [2., 3.], [4., 1.]]]), torch.tensor([[7., 6.]])).state

    def test_all_action_values_match_independent_brute_force(self):
        result = solve_teacher(self.state, self.config)
        best, q = 0., torch.full((4,), -torch.inf, dtype=torch.float64)
        for pattern in itertools.product((False, True), repeat=4):
            mask = torch.tensor(pattern)
            if (self.state.weights[0, mask].sum(0) <= self.state.remaining[0]).all():
                value = float(self.state.profits[0, mask].sum())
                best = max(best, value)
                q[mask] = torch.maximum(q[mask], torch.full_like(q[mask], value))
        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.objective, best)
        torch.testing.assert_close(result.action_values, q)
        torch.testing.assert_close(result.target_mask, q == best)
        self.assertEqual(float(self.state.profits[0, result.selected_mask].sum()), best)

    def test_optimal_action_union_is_not_mistaken_for_feasible_plan(self):
        state = MKPEnv(torch.tensor([[5., 5.]]), torch.tensor([[[2.], [2.]]]), torch.tensor([[2.]])).state
        result = solve_teacher(state, self.config)
        self.assertEqual(result.target_mask.tolist(), [True, True])
        self.assertEqual(result.selected_mask.sum().item(), 1)
        self.assertEqual(result.action_values.tolist(), [5., 5.])

    def test_prefix_profit_and_infeasible_items_do_not_enter_values(self):
        env = MKPEnv(torch.tensor([[10000., 6., 8., 999.]]),
                     torch.tensor([[[1.], [2.], [2.], [9.]]]), torch.tensor([[3.]]))
        env.step(torch.tensor([0]))
        baseline = torch.tensor([False, True, False, False])
        result = solve_teacher(env.state, self.config, baseline)
        self.assertEqual(result.objective, 8.)
        self.assertEqual(result.improvement_vs_baseline, 2.)
        self.assertTrue(torch.isnan(result.action_values[[0, 3]]).all())
        self.assertEqual(result.action_values[1:3].tolist(), [6., 8.])
        with self.assertRaisesRegex(ValueError, "prefix"):
            solve_teacher(env.state, self.config, torch.tensor([True, False, False, False]))
        with self.assertRaisesRegex(ValueError, "feasible"):
            solve_teacher(env.state, self.config, torch.tensor([False, True, True, False]))

    def test_original_float64_capacity_boundary_and_terminal(self):
        state = MKPEnv(torch.tensor([[5., 99.]], dtype=torch.float64),
            torch.tensor([[[1.], [1. + 1e-10]]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)).state
        result = solve_teacher(state, self.config)
        self.assertEqual(result.target_mask.tolist(), [True, False])
        state.selected.fill_(True)
        self.assertIsNone(solve_teacher(state, self.config))

    @staticmethod
    def mocked_result(**changes):
        values = dict(x=np.array([0., 1.]), fun=-8., mip_dual_bound=-8., mip_gap=0., status=0,
                      success=True, message="synthetic test")
        values.update(changes)
        return SimpleNamespace(**values)

    def small_milp_state(self):
        return MKPEnv(torch.tensor([[6., 8.]]), torch.tensor([[[2.], [2.]]]), torch.tensor([[2.]])).state

    def test_certified_milp_marks_only_known_action_values(self):
        config = replace(self.config, teacher_exact_items=0)
        with patch("mkp_teacher.milp", return_value=self.mocked_result()) as solver:
            result = solve_teacher(self.small_milp_state(), config)
        self.assertEqual(result.source, "milp_optimal")
        self.assertTrue(result.optimality_proven)
        self.assertTrue(torch.isnan(result.action_values[0]))
        self.assertEqual(result.action_values[1].item(), 8.)
        self.assertEqual(solver.call_args.kwargs["options"]["time_limit"], config.teacher_time_limit)
        self.assertEqual(solver.call_args.kwargs["options"]["node_limit"], config.teacher_node_limit)

    def test_timeout_incumbent_requires_witnessed_improvement(self):
        config, state = replace(self.config, teacher_exact_items=0), self.small_milp_state()
        with patch("mkp_teacher.milp", return_value=self.mocked_result(status=1, success=False,
                                                                     mip_dual_bound=-10., mip_gap=.25)):
            self.assertIsNone(solve_teacher(state, config))
            self.assertIsNone(solve_teacher(state, config, torch.tensor([False, True])))
            result = solve_teacher(state, config, torch.tensor([True, False]))
        self.assertEqual(result.source, "milp_incumbent")
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.upper_bound, 10.)
        self.assertTrue(torch.isnan(result.action_values).all())
        self.assertEqual(result.improvement_vs_baseline, 2.)

    def test_invalid_or_incomplete_solver_output_never_becomes_exact_label(self):
        config, state = replace(self.config, teacher_exact_items=0), self.small_milp_state()
        for changes in (dict(x=None), dict(x=np.array([.5, .5])), dict(x=np.array([1., 1.]), fun=-14.),
                        dict(fun=-9.), dict(x=np.array([0., float("nan")])), dict(mip_dual_bound=None),
                        dict(mip_dual_bound=-7.), dict(mip_gap=.01)):
            with self.subTest(changes=changes), patch("mkp_teacher.milp", return_value=self.mocked_result(**changes)):
                self.assertIsNone(solve_teacher(state, config))

    def test_actual_bounded_milp_tiny_synthetic_problem(self):
        result = solve_teacher(self.state, replace(self.config, teacher_exact_items=0, teacher_time_limit=2.))
        exact = solve_teacher(self.state, self.config)
        self.assertIsNotNone(result)
        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.objective, exact.objective)
        self.assertTrue((self.state.weights[0, result.selected_mask].sum(0) <= self.state.remaining[0]).all())

    def test_exact_budget_and_teacher_config_validation(self):
        for changes in (dict(teacher_exact_items=19), dict(teacher_steps=-1), dict(teacher_time_limit=0),
                        dict(teacher_node_limit=0), dict(teacher_mip_rel_gap=-1), dict(supervision_mode="bad")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)


if __name__ == "__main__":
    unittest.main()
