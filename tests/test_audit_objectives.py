"""Tiny gradients/selection contracts; no fitting or optimizer updates."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mkp_Model import CandidatePolicy
from mkp_audit_objectives import action_fit_terms, fit_candidate_key
from mkp_audit_common import evaluate_records
from test_audit_bank import tiny_bank


def policy_for(scores):
    return CandidatePolicy(torch.zeros(len(scores), dtype=torch.long), torch.arange(len(scores)),
                           scores / .2, 1, dict(score=scores))


def record_for(values=(100., 90., float("nan"))):
    return dict(target_actions=torch.tensor([0]), future_values=torch.tensor(values, dtype=torch.float64),
                reference_future_value=100.)


class AuditObjectiveTests(unittest.TestCase):
    def test_unknown_action_is_not_assigned_an_invented_negative_value(self):
        scores = torch.tensor([0., .1, 4.], requires_grad=True)
        loss, _ = action_fit_terms(policy_for(scores), record_for())
        loss.backward()
        self.assertLess(scores.grad[0], 0)
        self.assertGreater(scores.grad[1], 0)
        self.assertEqual(scores.grad[2].item(), 0.)
        scores.grad = None
        old_loss, _ = action_fit_terms(policy_for(scores), record_for(), objective="set_ce")
        old_loss.backward()
        self.assertGreater(scores.grad[2], 0)

    def test_correct_sufficient_margin_has_zero_value_gradient(self):
        scores = torch.tensor([1., 0., 2.], requires_grad=True)
        loss, _ = action_fit_terms(policy_for(scores), record_for())
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        torch.testing.assert_close(scores.grad, torch.zeros(3))

    def test_smaller_known_gain_has_smaller_bounded_correction(self):
        scores = torch.zeros(3, requires_grad=True)
        high, _ = action_fit_terms(policy_for(scores), record_for())
        low, _ = action_fit_terms(policy_for(scores), record_for((100., 99.9, float("nan"))))
        self.assertGreater(high.item(), low.item())
        self.assertGreater(low.item(), 0.)
        self.assertLessEqual(high.item(), .5)

    def test_behavior_anchor_observes_all_actions_without_labeling_unknown_reward(self):
        source = torch.zeros(3, requires_grad=True)
        old_policy = policy_for(source)
        anchor = dict(items=old_policy.items, log_probs=old_policy.log_probs)
        _, same_kl = action_fit_terms(old_policy, record_for(), anchor=anchor)
        self.assertAlmostEqual(same_kl.item(), 0.)
        current = torch.tensor([0., 0., 2.], requires_grad=True)
        _, kl = action_fit_terms(policy_for(current), record_for(), anchor=anchor)
        kl.backward()
        self.assertGreater(kl.item(), .03)
        self.assertGreater(current.grad[2], 0)
        self.assertIsNone(source.grad)

    def test_all_known_optima_do_not_sharpen_without_an_ordering_violation(self):
        scores = torch.zeros(2, requires_grad=True)
        record = dict(target_actions=torch.tensor([0, 1]), future_values=torch.tensor([100., 100.]),
                      reference_future_value=100.)
        loss, _ = action_fit_terms(policy_for(scores), record)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros(2))

    def test_fit_selection_prioritizes_true_normalized_return_over_surrogate_loss(self):
        source = [dict(split="fit", greedy_delta_over_initial_profit=0., target_hit=False)]
        degraded = [dict(split="fit", greedy_delta_over_initial_profit=-.001, target_hit=True)]
        source_key = fit_candidate_key(source, 10., 0., .03)
        degraded_key = fit_candidate_key(degraded, .001, .01, .03)
        self.assertGreater(source_key, degraded_key)
        self.assertIsNone(fit_candidate_key(source, .001, .031, .03))
        with self.assertRaisesRegex(ValueError, "only nonempty fit"):
            fit_candidate_key([dict(split="heldout")], 0., 0., .03)

    def test_continuation_drift_is_only_measured_for_known_finite_labels(self):
        record = tiny_bank()["records"][0]
        record["label_kind"] = "finite_greedy"
        policy = policy_for(torch.tensor([1., 0., 0., 0.]))
        model = SimpleNamespace(eval=lambda: None, policy=lambda state: (policy,))
        with patch("mkp_audit_common.greedy_future", return_value=9.):
            row = evaluate_records(model, [record], "cpu")[0]
        self.assertEqual(row["frozen_continuation_gain"], 0.)
        self.assertEqual(row["continuation_policy_drift"], -2.)
        record["future_values"][0] = torch.nan
        record["target_actions"] = torch.tensor([1, 3])
        with patch("mkp_audit_common.greedy_future", return_value=9.):
            unknown = evaluate_records(model, [record], "cpu")[0]
        self.assertNotIn("continuation_policy_drift", unknown)
        record["future_values"][0] = 11.
        record["target_actions"] = torch.tensor([0, 1, 3])
        record["label_kind"] = "exact"
        with patch("mkp_audit_common.greedy_future", return_value=9.):
            exact = evaluate_records(model, [record], "cpu")[0]
        self.assertNotIn("continuation_policy_drift", exact)


if __name__ == "__main__":
    unittest.main()
