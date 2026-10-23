"""Diagnostic measurement contracts; no fitting or actual PPO update is run."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import torch

from mkp_audit_common import check_report, write_json
from mkp_audit_learning import GradientObserver, eligible_records, retention_metrics


class LearningAuditTests(unittest.TestCase):
    def test_observer_measures_gradient_direction_without_writing_parameter_gradients(self):
        model = SimpleNamespace(actor=torch.nn.Linear(2, 1, bias=False))
        observer = GradientObserver(model)
        loss = model.actor(torch.ones(1, 2)).sum()
        rng = torch.get_rng_state().clone()
        observer(dict(event="before_backward", losses=dict(policy=loss, ranking=-loss)))
        self.assertAlmostEqual(observer.rows[0]["policy_ranking_cosine"], -1., places=5)
        self.assertIsNone(model.actor.weight.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        loss.backward()
        torch.testing.assert_close(model.actor.weight.grad, torch.ones_like(model.actor.weight))
        observer(dict(event="kl_stop", ppo_kl=.04, ranking_kl=.01))
        self.assertEqual(len(observer.stops), 1)

    def test_retention_uses_only_initially_correct_denominator_and_same_state_ids(self):
        before = [dict(id="a", split="fit", target_hit=True, greedy_future=10.),
                  dict(id="b", split="fit", target_hit=False, greedy_future=5.),
                  dict(id="c", split="heldout", target_hit=False, greedy_future=7.)]
        after = [dict(id="a", split="fit", target_hit=False, greedy_future=8.),
                 dict(id="b", split="fit", target_hit=True, greedy_future=9.),
                 dict(id="c", split="heldout", target_hit=True, greedy_future=8.)]
        result = retention_metrics(before, after)
        self.assertEqual(result["fit"]["initially_correct"], 1)
        self.assertEqual(result["fit"]["target_retention_rate"], 0.)
        self.assertEqual(result["fit"]["mean_future_delta_from_fit"], 1.)
        self.assertIsNone(result["heldout"]["target_retention_rate"])
        with self.assertRaises(ValueError):
            retention_metrics(before, after[:2])

    def test_eligible_states_require_executed_witness_not_search_failure(self):
        bank = dict(records=[dict(id="a", label_kind="exact"), dict(id="b", label_kind="finite_greedy"),
                             dict(id="c", label_kind="exact")])
        reach = dict(records=[dict(id="a", status="reachable"), dict(id="b", status="reachable"),
                              dict(id="c", status="unknown")])
        self.assertEqual([r["id"] for r in eligible_records(bank, reach)], ["a", "b"])
        self.assertEqual([r["id"] for r in eligible_records(bank, reach, scope="exact")], ["a"])

    def test_provenance_rejects_different_bank_and_json_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.write_bytes(b"bank identity")
            with self.assertRaises(ValueError):
                check_report(bank, dict(bank_sha256="incorrect"))
            output = Path(directory) / "report.json"
            write_json(output, dict(unknown=float("nan"), infinite=float("inf"), tensor=torch.tensor([1.])))
            text = output.read_text()
            self.assertNotIn("NaN", text)
            self.assertIn("null", text)


if __name__ == "__main__":
    unittest.main()
