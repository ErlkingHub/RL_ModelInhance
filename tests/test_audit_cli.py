"""CLI wiring only; every experiment entry point is mocked."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import diagnose_mkp
from mkp_utils import file_sha256


class AuditCliTests(unittest.TestCase):
    def test_all_passes_budgets_and_consistent_bank_paths_without_running_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("diagnose_mkp.resolve_device", return_value=torch.device("cpu")), \
                 patch("mkp_audit_bank.build_bank") as build, \
                 patch("mkp_audit_theta.run_reachability") as reach, \
                 patch("mkp_audit_learning.run_learning") as learn, \
                 patch("mkp_audit_learning.run_retention") as retention, \
                 patch("diagnose_mkp.make_report") as report:
                diagnose_mkp.main(["all", "--output", directory, "--theta-steps", "7", "--fit-steps", "9", "--ppo-updates", "2"])
            self.assertEqual(build.call_args.kwargs["resources"], (5, 10))
            self.assertEqual(reach.call_args.args[0], output / "bank/bank.pt")
            self.assertEqual(reach.call_args.kwargs["steps"], 7)
            self.assertEqual(learn.call_args.kwargs["steps"], 9)
            self.assertEqual(learn.call_args.kwargs["objective"], "regret_margin")
            self.assertEqual(learn.call_args.kwargs["max_kl"], .03)
            self.assertEqual(retention.call_args.kwargs["updates"], 2)
            self.assertEqual(retention.call_args.kwargs["arms"], ("ppo_only", "ppo_rank", "ppo_regret"))
            self.assertEqual(report.call_args.args[0], output / "bank/bank.pt")

    def test_report_phase_never_runs_experiment_phases(self):
        with patch("diagnose_mkp.make_report") as report:
            diagnose_mkp.main(["report", "--bank", "bank.pt", "--reach", "reach.json", "--output", "report"])
        report.assert_called_once_with(Path("bank.pt"), Path("reach.json"), Path("report"), None, None)

    def test_retention_forwards_independent_validation_and_matched_actor_controls(self):
        with patch("diagnose_mkp.resolve_device", return_value=torch.device("cpu")), \
             patch("mkp_audit_learning.run_retention") as run:
            diagnose_mkp.main(["retention", "--bank", "bank.pt", "--reach", "reach.json",
                "--learning", "learn.json", "--output", "new", "--ppo-updates", "50",
                "--ppo-seed", "5730", "--control-mode", "matched", "--actor-learning-rate", "1e-5",
                "--actor-continuation-guard", "--actor-guard-states", "8",
                "--validation-bank", "independent/bank.pt", "--arms", "ppo_only", "ppo_regret"])
        options = run.call_args.kwargs
        self.assertEqual(options["updates"], 50)
        self.assertEqual(options["seed"], 5730)
        self.assertEqual(options["control_mode"], "matched")
        self.assertEqual(options["actor_learning_rate"], 1e-5)
        self.assertTrue(options["actor_continuation_guard"])
        self.assertEqual(options["validation_bank_path"], Path("independent/bank.pt"))

    def test_complete_workflow_refuses_to_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "diagnosis.md").write_text("preserve this report")
            with patch("diagnose_mkp.resolve_device", return_value=torch.device("cpu")), \
                 self.assertRaises(FileExistsError):
                diagnose_mkp.main(["all", "--output", directory])

    def test_training_cli_exposes_matched_loss_ablation_and_separate_defaults(self):
        from mkp_Trainer import main as train_main
        from mkp_config import BenchmarkConfig
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            train_main(["--ranking-loss-type", "soft_bce", "--ranking-margin-cap", "2"])
        config = trainer.call_args.args[0]
        self.assertEqual(config.ranking_loss_type, "soft_bce")
        self.assertEqual(config.ranking_margin_cap, 2.)
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            train_main([])
        self.assertEqual(trainer.call_args.args[0].ranking_loss_type, "regret_margin")
        self.assertEqual(trainer.call_args.args[0].checkpoint_dir.name, "regret_margin_v2")
        self.assertEqual(BenchmarkConfig().checkpoint_dir.name, "regret_margin_v2")

    def test_build_does_not_accept_benchmark_dataset_input(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            diagnose_mkp.main(["build", "--output", "out", "--dataset", "MKP_dataset/Benchmarks_Chu.mat"])

    def test_report_separates_old_errors_from_existing_correct_witnesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path, reach_path = root / "bank.pt", root / "reach.json"
            bank_path.write_bytes(b"mock state-bank identity")
            record = dict(id="a", M=5, N=100, capacity_ratio=.25, split="fit", stage="early",
                          label_kind="finite_greedy", baseline_action=0, target_actions=torch.tensor([1]))
            reached = {k: v for k, v in record.items() if k not in ("baseline_action", "target_actions")}
            reached.update(status="reachable", witness_source="legal_theta_search")
            reach_path.write_text(json.dumps(dict(bank_sha256=file_sha256(bank_path), records=[reached],
                counts=dict(reachable=1, analytic_unreachable=0, unknown=0))), encoding="utf-8")
            with patch("mkp_audit_bank.load_bank", return_value=dict(records=[record])):
                result = diagnose_mkp.make_report(bank_path, reach_path, root / "report")
            group = next(iter(result["reachability_by_group"].values()))
            self.assertEqual(group["originally_wrong"], 1)
            self.assertEqual(group["recovered_by_theta_search"], 1)
            self.assertIn("原目标未命中", (root / "report/diagnosis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
