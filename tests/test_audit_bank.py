"""Small label/artifact contracts; no real model trajectory or optimizer runs."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mkp_env import MKPEnv
from mkp_audit_bank import (exact_action_values, _targets, _chosen_steps, _state_payload,
                            _finite_values, load_bank, save_bank, state_from_record)


def tiny_state():
    return MKPEnv(torch.tensor([[6., 5., 4., 0.]]),
                  torch.tensor([[[2., 1.], [1., 2.], [1., 1.], [0., 0.]]]),
                  torch.tensor([[3., 3.]])).state


def tiny_bank():
    state = tiny_state()
    values, status, nodes = exact_action_values(state)
    record = dict(id="a", instance_id="instance-a", split="fit", stage="tail", M=2, N=4,
                  capacity_ratio=.5, state=_state_payload(state), future_values=values,
                  target_actions=_targets(values), known=torch.isfinite(values), label_kind="exact",
                  exact_status=status, exact_nodes=nodes, baseline_action=0,
                  baseline_greedy_future=11., reference_future_value=11.)
    return dict(schema_version=1, data_source="synthetic_only", checkpoints={}, source_hashes={}, records=[record])


class StateBankTests(unittest.TestCase):
    def test_exact_values_cover_all_actions_and_all_optimal_ties(self):
        values, status, nodes = exact_action_values(tiny_state())
        self.assertEqual((status, nodes), ("complete", 16))
        torch.testing.assert_close(values, torch.tensor([11., 11., 10., 11.], dtype=torch.float64))
        self.assertEqual(_targets(values).tolist(), [0, 1, 3])

    def test_exact_values_exclude_prefix_and_already_selected_actions(self):
        env = MKPEnv(torch.tensor([[100., 6., 5., 4.]]),
                     torch.tensor([[[5., 5.], [2., 1.], [1., 2.], [1., 1.]]]),
                     torch.tensor([[8., 8.]]))
        env.step(torch.tensor([0]))
        values, status, _ = exact_action_values(env.state)
        self.assertEqual(status, "complete")
        self.assertTrue(torch.isnan(values[0]))
        torch.testing.assert_close(values[1:], torch.tensor([11., 11., 10.], dtype=torch.float64))

    def test_budget_or_arithmetic_failure_discards_every_partial_exact_label(self):
        values, status, nodes = exact_action_values(tiny_state(), node_budget=3)
        self.assertEqual((status, nodes), ("node_budget_exceeded", 3))
        self.assertTrue(torch.isnan(values).all())
        state = tiny_state()
        state.remaining -= .25
        values, status, nodes = exact_action_values(state)
        self.assertEqual((status, nodes), ("noninteger_capacity_arithmetic", 0))
        self.assertTrue(torch.isnan(values).all())

    def test_repeated_prefix_indices_are_not_duplicate_fit_examples(self):
        chosen = _chosen_steps(1, 0)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0][:2], ("tail", 0))
        self.assertEqual(set(chosen[0][2]), {"early", "middle", "late", "tail"})

    def test_finite_labels_leave_unqueried_actions_unknown_and_include_baseline(self):
        policy = SimpleNamespace(items=torch.arange(4), logits=torch.tensor([4., 3., 2., 1.]))
        model = SimpleNamespace(policy=lambda state: (policy,))
        with patch("mkp_audit_bank._forced_future", side_effect=lambda model, state, action: 10. + action) as forced:
            values = _finite_values(model, tiny_state(), 2, 0)
        self.assertEqual(forced.call_count, 2)
        self.assertTrue(torch.isfinite(values[0]))
        self.assertEqual(int(torch.isnan(values).sum()), 2)

    def test_bank_roundtrip_preserves_unknowns_and_returns_independent_state(self):
        bank = tiny_bank()
        record = bank["records"][0]
        record["label_kind"] = "finite_greedy"
        record["future_values"][2] = torch.nan
        record["known"] = torch.isfinite(record["future_values"])
        with tempfile.TemporaryDirectory() as directory:
            path = save_bank(Path(directory) / "bank.pt", bank)
            restored = load_bank(path)
        self.assertTrue(torch.isnan(restored["records"][0]["future_values"][2]))
        cloned = state_from_record(restored["records"][0])
        cloned.remaining.zero_()
        self.assertTrue((restored["records"][0]["state"]["remaining"] > 0).all())

    def test_artifact_rejects_fit_heldout_leakage_and_partial_exact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = tiny_bank()
            duplicate = deepcopy(bank["records"][0])
            duplicate.update(id="b", split="heldout")
            bank["records"].append(duplicate)
            with self.assertRaisesRegex(ValueError, "share a generated instance"):
                save_bank(directory, bank)
            bank = tiny_bank()
            bank["records"][0]["future_values"][2] = torch.nan
            with self.assertRaisesRegex(ValueError, "Partial enumeration"):
                save_bank(directory, bank)


if __name__ == "__main__":
    unittest.main()
