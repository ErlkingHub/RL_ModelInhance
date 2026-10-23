"""Tensor artifact comparison only; no model construction or training."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import torch

from compare_audit_banks import compare_banks, compare_fitted_actors, main
from mkp_utils import file_sha256


def fixture_bank():
    record = dict(id="instance/early/0", instance_id="instance", split="fit", stage="early",
                  M=2, N=3, capacity_ratio=.5, label_kind="finite_greedy", baseline_action=0,
                  baseline_greedy_future=7., reference_future_value=8.,
                  state=dict(profits=torch.tensor([[4., 3., 2.]], dtype=torch.float64),
                             weights=torch.tensor([[[1., 2.], [2., 1.], [3., 3.]]], dtype=torch.float64),
                             capacities=torch.tensor([[4., 4.]], dtype=torch.float64),
                             remaining=torch.tensor([[4., 4.]], dtype=torch.float64),
                             selected=torch.zeros(1, 3, dtype=torch.bool)),
                  future_values=torch.tensor([7., 8., torch.nan], dtype=torch.float64),
                  target_actions=torch.tensor([1]))
    return dict(schema_version=1, data_source="synthetic_only", source_hashes={"mkp_config.py": "old"},
                checkpoints={"2": dict(sha256="a" * 64, path="old/checkpoint.pt")}, records=[record])


class AuditBankComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.baseline, self.candidate = self.root / "old.pt", self.root / "new.pt"
        self.bank = fixture_bank()
        torch.save(self.bank, self.baseline)

    def compare(self, bank):
        torch.save(bank, self.candidate)
        return compare_banks(self.baseline, self.candidate)

    def test_metadata_changes_and_nan_values_still_match(self):
        bank = deepcopy(self.bank)
        bank["source_hashes"] = {"mkp_config.py": "new"}
        bank["checkpoints"]["2"]["path"] = "new/checkpoint.pt"
        bank["coverage"] = {"unused": "metadata"}
        self.assertTrue(self.compare(bank)["matched"])

    def test_each_physical_tensor_change_is_detected_exactly(self):
        for field in ("profits", "weights", "capacities", "remaining", "selected"):
            with self.subTest(field=field):
                bank = deepcopy(self.bank)
                tensor = bank["records"][0]["state"][field]
                if field == "selected":
                    tensor.flatten()[0] = True
                else:
                    tensor.flatten()[0] += 1e-10
                report = self.compare(bank)
                self.assertFalse(report["matched"])
                self.assertTrue(any(f"state/{field}" in issue["location"] for issue in report["differences"]))

    def test_label_changes_include_unknown_positions_and_source_weights(self):
        for mutation in ("unknown", "finite_value", "target", "checkpoint", "baseline_future", "reference"):
            with self.subTest(mutation=mutation):
                bank = deepcopy(self.bank)
                record = bank["records"][0]
                if mutation == "unknown":
                    record["future_values"][2] = 7.
                elif mutation == "finite_value":
                    record["future_values"][0] += 1e-10
                elif mutation == "target":
                    record["target_actions"] = torch.tensor([0, 1])
                elif mutation == "checkpoint":
                    bank["checkpoints"]["2"]["sha256"] = "b" * 64
                elif mutation == "baseline_future":
                    record["baseline_greedy_future"] += 1.
                else:
                    record["reference_future_value"] += 1.
                self.assertFalse(self.compare(bank)["matched"])

    def test_missing_extra_duplicate_and_changed_split_records_fail(self):
        for mutation in ("missing", "extra", "duplicate", "split", "stage", "capacity_ratio"):
            with self.subTest(mutation=mutation):
                bank = deepcopy(self.bank)
                if mutation == "missing":
                    bank["records"] = []
                elif mutation in ("extra", "duplicate"):
                    extra = deepcopy(bank["records"][0])
                    if mutation == "extra":
                        extra["id"] = "new/record"
                    bank["records"].append(extra)
                elif mutation == "split":
                    bank["records"][0]["split"] = "heldout"
                elif mutation == "stage":
                    bank["records"][0]["stage"] = "middle"
                else:
                    bank["records"][0]["capacity_ratio"] = .75
                self.assertFalse(self.compare(bank)["matched"])

    def test_invalid_schema_or_non_synthetic_source_fails(self):
        for key, value in (("schema_version", 2), ("data_source", "benchmark")):
            bank = deepcopy(self.bank)
            bank[key] = value
            self.assertFalse(self.compare(bank)["matched"])

    def test_cli_writes_json_and_exit_codes_are_zero_or_two(self):
        output = self.root / "comparison.json"
        torch.save(self.bank, self.candidate)
        arguments = ["--baseline", str(self.baseline), "--candidate", str(self.candidate), "--output", str(output)]
        with redirect_stdout(StringIO()):
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(["--before", str(self.baseline), "--after", str(self.candidate)]), 0)
        self.assertTrue(json.loads(output.read_text())["matched"])
        changed = deepcopy(self.bank)
        changed["records"][0]["baseline_action"] = 1
        torch.save(changed, self.candidate)
        with redirect_stdout(StringIO()):
            self.assertEqual(main(arguments), 2)
        self.assertFalse(json.loads(output.read_text())["matched"])

    def test_rebuilt_fit_requires_identical_actor_tensors_and_checked_provenance(self):
        torch.save(self.bank, self.candidate)
        reports = []
        for label, bank_path in (("old", self.baseline), ("new", self.candidate)):
            actor_path = self.root / f"{label}_actor.pt"
            actor = dict(artifact_type="diagnostic_actor_fit", bank_sha256=file_sha256(bank_path),
                         source_checkpoint_sha256="a" * 64, num_resources=2, fit_record_ids=["instance/early/0"],
                         label_scope="both", fit_objective="regret_margin", selected_step=2,
                         actor_state={"weight": torch.tensor([1., 2.])})
            torch.save(actor, actor_path)
            path = self.root / f"{label}_learning.json"
            path.write_text(json.dumps(dict(bank_sha256=file_sha256(bank_path), budget=dict(steps=100),
                models={"2": dict(status="complete", actor_artifact=str(actor_path), actor_sha256=file_sha256(actor_path))})), encoding="utf-8")
            reports.append(path)
        args = (*reports, self.baseline, self.candidate)
        self.assertTrue(compare_fitted_actors(*args)["matched"])
        actor_path = self.root / "new_actor.pt"
        changed = torch.load(actor_path, weights_only=True)
        changed["actor_state"]["weight"][0] += .1
        torch.save(changed, actor_path)
        # A changed actor is rejected even after the report's file SHA is updated.
        updated = json.loads(reports[1].read_text())
        updated["models"]["2"]["actor_sha256"] = file_sha256(actor_path)
        reports[1].write_text(json.dumps(updated), encoding="utf-8")
        result = compare_fitted_actors(*args)
        self.assertFalse(result["matched"])
        self.assertTrue(any("weight" in d["location"] for d in result["differences"]))


if __name__ == "__main__":
    unittest.main()
