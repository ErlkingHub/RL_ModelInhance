"""Run current tracked contracts and routing tests; leave orphan v8/v9 tests alone."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import json
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = ("test_contracts", "test_combination_theta", "test_controller_attention",
           "test_diagnostics", "test_opportunity_ranking", "test_performance", "test_resource_routing")

if __name__ == "__main__":
    if "--audit-only" in sys.argv:
        from audit_routing_contracts import audit
        print(json.dumps(audit(), indent=2))
        raise SystemExit(0)
    out = Path(__file__).resolve().parents[1] / "outputs" / "resource_routing_v1"
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    # CUDA visibility alone is insufficient on some Windows installations.
    # Force the availability check before skip decorators are evaluated.
    with patch("torch.cuda.is_available", return_value=False):
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in MODULES)
        with (out / "cpu_tests.txt").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)
    report = dict(tests=result.testsRun, passed=result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
                  skipped=len(result.skipped), errors=len(result.errors), failures=len(result.failures),
                  seconds=time.perf_counter()-began, torch=str(torch.__version__),
                  scope="CPU contracts; no optimizer updates or real benchmark", modules=MODULES)
    (out / "cpu_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for test, error in result.errors + result.failures:
        print(str(test), error)
    if result.wasSuccessful():
        from audit_routing_contracts import audit
        print(json.dumps(audit(), indent=2))
    raise SystemExit(not result.wasSuccessful())
