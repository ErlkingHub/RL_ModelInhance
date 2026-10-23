"""Tiny analytic theta contracts. No diagnostic search or model optimization."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mkp_config import ModelConfig
from mkp_env import MKPEnv
from mkp_Model import WeakDecoder
from mkp_audit_theta import LegalTheta, analytic_exclusion, reach_state, target_margin
from mkp_test_helpers import synthetic_theta


class ThetaAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=8, num_heads=2, num_layers=1)

    def state(self, m=2, profits=(100., 101.)):
        return MKPEnv(torch.tensor([profits]), torch.tensor([[[1.] * m, [2.] * m]]),
                       torch.full((1, m), 2.)).state

    def test_legal_parameterization_respects_all_mass_bounds(self):
        for m in (1, 5):
            state = self.state(m)
            module = LegalTheta(self.config, synthetic_theta(state, self.config))
            with torch.no_grad():
                for p in module.parameters():
                    p.add_(torch.linspace(-20, 20, p.numel()).reshape(p.shape))
            theta = module()
            torch.testing.assert_close(theta.resource_prices.mean(-1), torch.ones(1))
            self.assertTrue((theta.resource_prices >= self.config.price_floor).all())
            torch.testing.assert_close(theta.resource_interactions, theta.resource_interactions.transpose(1, 2))
            self.assertTrue((theta.resource_interactions >= 0).all())
            self.assertLessEqual(float(theta.resource_interactions.triu().sum()), 2.000001)
            self.assertLessEqual(float(theta.companion_weights.sum()), 2.000001)
            self.assertTrue(torch.isfinite(theta.profit_exponent).all())
            self.assertGreaterEqual(float(theta.profit_exponent), .5)
            self.assertLessEqual(float(theta.profit_exponent), 2.)

    def test_analytic_exclusion_certifies_known_counterexample_for_all_resource_counts(self):
        for m in (1, 2, 5):
            state = self.state(m)
            certificate = analytic_exclusion(state, self.config, torch.tensor([1]))
            self.assertIsNotNone(certificate)
            self.assertTrue(certificate["all_targets_excluded"])
            self.assertAlmostEqual(certificate["bounds"][0]["upper_bound"], -.653346, places=5)
            self.assertLess(certificate["bounds"][0]["guarded_upper_bound"], 0)

    def test_analytic_relaxation_does_not_exclude_obvious_high_profit_or_all_ties(self):
        self.assertIsNone(analytic_exclusion(self.state(profits=(100., 1000.)), self.config, torch.tensor([1])))
        self.assertIsNone(analytic_exclusion(self.state(), self.config, torch.tensor([0, 1])))

    def test_reachability_uses_actual_all_candidate_argmax_and_keeps_search_failure_unknown(self):
        state = self.state()
        theta = synthetic_theta(state, self.config)
        model = SimpleNamespace(config=self.config, decoder=WeakDecoder(self.config),
                                actor=Mock(return_value=(theta, None)))
        current = reach_state(model, state, torch.tensor([0]), steps=0)
        self.assertEqual((current["status"], current["witness_source"]), ("reachable", "current_controller"))
        blocked = reach_state(model, state, torch.tensor([1]), steps=0)
        self.assertEqual(blocked["status"], "analytic_unreachable")
        with patch("mkp_audit_theta.analytic_exclusion", return_value=None), \
             patch("mkp_audit_theta.torch.optim.Adam") as optimizer:
            unresolved = reach_state(model, state, torch.tensor([1]), restarts=1, steps=0)
        self.assertEqual((unresolved["status"], unresolved["optimizer_steps"]), ("unknown", 0))
        optimizer.return_value.step.assert_not_called()

    def test_target_margin_compares_best_target_to_all_nontargets(self):
        state = self.state()
        policy = WeakDecoder(self.config)(state, synthetic_theta(state, self.config))
        margin, trivial = target_margin(policy, torch.tensor([1]))
        self.assertFalse(trivial)
        self.assertLess(float(margin), 0)
        _, trivial = target_margin(policy, torch.tensor([0, 1]))
        self.assertTrue(trivial)


if __name__ == "__main__":
    unittest.main()
