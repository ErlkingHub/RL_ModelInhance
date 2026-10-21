"""Saved-evidence validation only; no network, training, or benchmark execution."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "compare_mkp_results.py"
SPEC = importlib.util.spec_from_file_location("saved_result_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class SavedResultComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline, self.candidate = self.root / "before", self.root / "after"
        self.rows = [dict(instance_id=str(i), M="5", N="100", method="learned_dynamic", rollouts="1",
                          objective="999.5", reference="1000", reference_type="dataset_best_known", gap_pct="0.05",
                          feasible="1", max_capacity_violation="0", checkpoint_sha256="a" * 64)
                     for i in (1, 2)]
        self.write(self.baseline, self.rows)
        self.write(self.candidate, self.rows)

    def write(self, directory, rows, **overrides):
        directory.mkdir(exist_ok=True)
        with (directory / "instances.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = dict(status="complete", completed_rows=len(rows), instance_count=len(rows),
                        dataset_sha256="b" * 64, checkpoints={"5": {"sha256": "a" * 64}},
                        config={"feasibility_tolerance": 1e-7})
        manifest.update(overrides)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_target_is_strict_and_uses_all_rows(self):
        self.assertTrue(comparison.compare_results(self.baseline, self.candidate)["target_met"])
        rows = [dict(row, objective="999", gap_pct="0.1") for row in self.rows]
        self.write(self.candidate, rows)
        report = comparison.compare_results(self.baseline, self.candidate)
        self.assertFalse(report["target_met"])
        self.assertEqual(report["overall"]["candidate_strict_target_hits"], 0)
        self.assertEqual(len(report["instances"]), 2)

    def test_infeasible_rows_fail_target_without_dropping_them(self):
        rows = [dict(self.rows[0], feasible="0"), self.rows[1]]
        self.write(self.candidate, rows)
        report = comparison.compare_results(self.baseline, self.candidate)
        self.assertFalse(report["target_met"])
        self.assertEqual(report["overall"]["instances"], 2)
        self.assertEqual(report["overall"]["candidate_feasible"], 1)

    def test_capacity_violation_overrides_feasible_flag(self):
        self.write(self.candidate, [dict(row, max_capacity_violation="0.001") for row in self.rows])
        self.assertFalse(comparison.compare_results(self.baseline, self.candidate)["target_met"])

    def test_rejects_missing_extra_and_duplicate_instances(self):
        cases = [self.rows[:1], self.rows + [dict(self.rows[0], instance_id="3")],
                 [self.rows[0], self.rows[0]]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write(self.candidate, rows)
                with self.assertRaises(ValueError):
                    comparison.compare_results(self.baseline, self.candidate)

    def test_rejects_null_nonfinite_and_forged_gap(self):
        for field, value in (("gap_pct", ""), ("gap_pct", "NaN"), ("gap_pct", "0.01"),
                             ("objective", "inf"), ("reference", "0"), ("max_capacity_violation", "nan")):
            with self.subTest(field=field, value=value):
                self.write(self.candidate, [dict(self.rows[0], **{field: value}), self.rows[1]])
                with self.assertRaises(ValueError):
                    comparison.compare_results(self.baseline, self.candidate)

    def test_rejects_unmatched_shape_reference_and_reference_type(self):
        for changes in (dict(N="250"), dict(reference="2000", objective="1999", gap_pct="0.05"),
                        dict(reference_type="optimal"), dict(rollouts="128")):
            with self.subTest(changes=changes):
                self.write(self.candidate, [dict(self.rows[0], **changes), self.rows[1]])
                with self.assertRaises(ValueError):
                    comparison.compare_results(self.baseline, self.candidate)

    def test_rejects_wrong_dataset_or_row_checkpoint_or_incomplete_manifest(self):
        for overrides in (dict(dataset_sha256="c" * 64), dict(status="failed"), dict(instance_count=3),
                          dict(completed_rows=3)):
            with self.subTest(overrides=overrides):
                self.write(self.candidate, self.rows, **overrides)
                with self.assertRaises(ValueError):
                    comparison.compare_results(self.baseline, self.candidate)
        self.write(self.candidate, [dict(self.rows[0], checkpoint_sha256="d" * 64), self.rows[1]])
        with self.assertRaises(ValueError):
            comparison.compare_results(self.baseline, self.candidate)

    def test_complete_geometry_groups_use_baseline_labels_without_changing_outcomes(self):
        analysis = self.baseline / "analysis"
        analysis.mkdir()
        labels = [dict(instance_id=row["instance_id"], M=5, N=100, reference=1000,
                       capacity_ratio=ratio) for row, ratio in zip(self.rows, (.25, .75))]
        (analysis / "instance_diagnosis.json").write_text(json.dumps(labels), encoding="utf-8")
        report = comparison.compare_results(self.baseline, self.candidate)
        self.assertTrue(report["capacity_groups_complete"])
        self.assertEqual(len(report["by_capacity_ratio"]), 2)
        self.assertEqual(sum(g["instances"] for g in report["by_capacity_ratio"]), 2)
        self.assertEqual(report["overall"]["candidate_mean_gap_pct"], .05)

    def test_partial_geometry_omits_groups_without_dropping_overall_rows(self):
        rows = [dict(row, capacity_ratio="0.25" if i == 0 else "") for i, row in enumerate(self.rows)]
        self.rows = rows
        self.write(self.candidate, rows)
        report = comparison.compare_results(self.baseline, self.candidate)
        self.assertFalse(report["capacity_groups_complete"])
        self.assertEqual(report["by_capacity_ratio"], [])
        self.assertEqual(report["overall"]["instances"], 2)


if __name__ == "__main__":
    unittest.main()
