"""Compare synthetic diagnostic banks without loading a model or checking live code.

Exit 0 means their source weights, records, physical states and labels match
exactly. Exit 2 means an invalid artifact or a difference. File/source-code
hashes may differ after a diagnostic implementation change and are not compared.
"""
import argparse
import json
import math
from pathlib import Path
import pickle
import re

import torch
from mkp_utils import file_sha256


STATE_FIELDS = ("profits", "weights", "capacities", "remaining", "selected")
RECORD_FIELDS = ("instance_id", "split", "stage", "M", "N", "capacity_ratio",
                 "label_kind", "baseline_action", "baseline_greedy_future",
                 "reference_future_value")


def _difference(differences, location, detail):
    differences.append({"location": location, "detail": detail})


def _validate(bank, label, differences):
    """Validate the subset of the bank contract required for this comparison."""
    start = len(differences)
    if not isinstance(bank, dict):
        _difference(differences, label, "Expected a dictionary bank")
        return None
    if bank.get("schema_version") != 1 or bank.get("data_source") != "synthetic_only":
        _difference(differences, label, "Expected schema_version=1 and synthetic_only")
    checkpoints = bank.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        _difference(differences, f"{label}/checkpoints", "Missing checkpoint metadata")
    else:
        for resource, checkpoint in checkpoints.items():
            digest = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                _difference(differences, f"{label}/checkpoints/{resource}", "Invalid source checkpoint SHA256")
    records = bank.get("records")
    if not isinstance(records, list) or not records:
        _difference(differences, f"{label}/records", "Expected a nonempty record list")
        return None
    indexed = {}
    for index, record in enumerate(records):
        location = f"{label}/records/{index}"
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            _difference(differences, location, "Missing nonempty record id")
            continue
        identity = record["id"]
        location = f"{label}/{identity}"
        if identity in indexed:
            _difference(differences, location, "Duplicate record id")
        indexed[identity] = record
        for key in RECORD_FIELDS:
            if key not in record:
                _difference(differences, f"{location}/{key}", "Missing required field")
        m, n = record.get("M"), record.get("N")
        if type(m) is not int or type(n) is not int or min(m, n) < 1:
            _difference(differences, location, "Invalid M/N")
            continue
        if isinstance(checkpoints, dict) and str(m) not in checkpoints:
            _difference(differences, location, "No source checkpoint for record M")
        if record.get("split") not in ("fit", "heldout"):
            _difference(differences, location, "Invalid split")
        if not isinstance(record.get("instance_id"), str) or not record["instance_id"]:
            _difference(differences, location, "Invalid instance id")
        if record.get("stage") not in ("early", "middle", "late", "tail"):
            _difference(differences, location, "Invalid stage")
        if record.get("label_kind") not in ("exact", "finite_greedy"):
            _difference(differences, location, "Invalid label kind")
        for key in ("capacity_ratio", "baseline_greedy_future", "reference_future_value"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                _difference(differences, f"{location}/{key}", "Expected a finite scalar")
        action = record.get("baseline_action")
        if type(action) is not int or not 0 <= action < n:
            _difference(differences, f"{location}/baseline_action", "Invalid baseline action")
        shapes = ((1, n), (1, n, m), (1, m), (1, m), (1, n))
        state = record.get("state")
        if not isinstance(state, dict):
            _difference(differences, f"{location}/state", "Missing state dictionary")
        else:
            for key, shape in zip(STATE_FIELDS, shapes):
                value = state.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    _difference(differences, f"{location}/state/{key}", "Invalid state tensor/shape")
                elif key == "selected" and value.dtype != torch.bool:
                    _difference(differences, f"{location}/state/{key}", "Expected boolean selected mask")
                elif key != "selected" and (value.dtype != torch.float64 or not torch.isfinite(value).all()):
                    _difference(differences, f"{location}/state/{key}", "Expected finite float64 physical tensor")
        values, targets = record.get("future_values"), record.get("target_actions")
        if not isinstance(values, torch.Tensor) or values.dtype != torch.float64 or tuple(values.shape) != (n,):
            _difference(differences, f"{location}/future_values", "Expected float64[N] values")
        elif torch.isinf(values).any():
            _difference(differences, f"{location}/future_values", "Infinite values are invalid; unknown values must be NaN")
        if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long or targets.ndim != 1 or not targets.numel():
            _difference(differences, f"{location}/target_actions", "Expected nonempty LongTensor targets")
        elif ((targets < 0) | (targets >= n)).any():
            _difference(differences, f"{location}/target_actions", "Target index out of range")
    return indexed if len(differences) == start else None


def _compare_tensor(first, second, location, differences, *, allow_nan=False):
    if first.shape != second.shape:
        _difference(differences, location, f"Tensor shape changed: {tuple(first.shape)} -> {tuple(second.shape)}")
        return
    if first.dtype != second.dtype:
        _difference(differences, location, f"Tensor dtype changed: {first.dtype} -> {second.dtype}")
        return
    if allow_nan:
        first_nan, second_nan = torch.isnan(first), torch.isnan(second)
        if not torch.equal(first_nan, second_nan):
            _difference(differences, location, "Unknown-action NaN positions changed")
        mask = ~first_nan & ~second_nan
        different = first[mask] != second[mask]
    else:
        different = first != second
    count = int(different.sum())
    if count:
        _difference(differences, location, f"{count} tensor values differ (zero tolerance)")


def compare_banks(baseline, candidate):
    """Read only weights_only-safe artifacts; never instantiate an MKP model."""
    paths = {"baseline": Path(baseline).resolve(), "candidate": Path(candidate).resolve()}
    differences, banks, records = [], {}, {}
    for label, path in paths.items():
        try:
            bank = torch.load(path, map_location="cpu", weights_only=True)
            banks[label] = bank
            records[label] = _validate(bank, label, differences)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            _difference(differences, label, f"Cannot read/validate bank: {type(error).__name__}: {error}")
            records[label] = None
    if all(records.get(label) is not None for label in paths):
        before, after = banks["baseline"], banks["candidate"]
        weights = lambda bank: {str(m): meta["sha256"].lower() for m, meta in bank["checkpoints"].items()}
        if weights(before) != weights(after):
            _difference(differences, "checkpoints", "Source checkpoint resource set or SHA256 changed")
        old, new = records["baseline"], records["candidate"]
        for identity in sorted(set(old) - set(new)):
            _difference(differences, identity, "Record missing from candidate")
        for identity in sorted(set(new) - set(old)):
            _difference(differences, identity, "Extra record in candidate")
        for identity in sorted(set(old) & set(new)):
            first, second = old[identity], new[identity]
            for key in RECORD_FIELDS:
                if first[key] != second[key]:
                    _difference(differences, f"{identity}/{key}", f"Value changed: {first[key]!r} -> {second[key]!r}")
            for key in STATE_FIELDS:
                _compare_tensor(first["state"][key], second["state"][key], f"{identity}/state/{key}", differences)
            _compare_tensor(first["future_values"], second["future_values"], f"{identity}/future_values", differences, allow_nan=True)
            _compare_tensor(first["target_actions"], second["target_actions"], f"{identity}/target_actions", differences)
    return dict(schema_version=1, matched=not differences, tolerance=0,
                scope="Synthetic bank source weights, record identities, physical states and labels only; no model execution",
                paths={key: str(value) for key, value in paths.items()},
                records={key: len(value) if value is not None else None for key, value in records.items()},
                ignored_metadata=["source_hashes", "physical .pt SHA256", "checkpoint paths", "generation/coverage metadata"],
                differences=differences)


def compare_fitted_actors(before_learning, after_learning, before_bank, after_bank):
    """Verify a rebuilt diagnostic fit has exactly the same Actor tensors."""
    differences, reports, artifacts = [], {}, {}
    for label, learning_path, bank_path in (("baseline", before_learning, before_bank),
                                            ("candidate", after_learning, after_bank)):
        try:
            report = json.loads(Path(learning_path).read_text(encoding="utf-8"))
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
            digest = file_sha256(bank_path)
            if report.get("bank_sha256") != digest:
                raise ValueError("Learning report has a different bank SHA")
            reports[label], artifacts[label] = report, {}
            for m, fit in report["models"].items():
                if fit.get("status") != "complete":
                    raise ValueError(f"M{m} fit is not complete")
                path = Path(fit["actor_artifact"])
                if file_sha256(path) != fit["actor_sha256"]:
                    raise ValueError(f"M{m} actor file changed")
                actor = torch.load(path, map_location="cpu", weights_only=True)
                if (actor.get("artifact_type") != "diagnostic_actor_fit"
                        or actor.get("bank_sha256") != digest
                        or actor.get("source_checkpoint_sha256") != bank["checkpoints"][m]["sha256"]
                        or actor.get("num_resources") != int(m)):
                    raise ValueError(f"M{m} actor provenance differs")
                if not actor.get("actor_state") or not all(
                        isinstance(value, torch.Tensor) and torch.isfinite(value).all()
                        for value in actor["actor_state"].values()):
                    raise ValueError(f"M{m} actor state is missing or nonfinite")
                artifacts[label][m] = actor
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            _difference(differences, f"actors/{label}", str(error))
    if len(reports) == 2 and len(artifacts) == 2 and not differences:
        if reports["baseline"]["budget"] != reports["candidate"]["budget"]:
            _difference(differences, "actors/budget", "Fit budgets differ")
        old, new = artifacts["baseline"], artifacts["candidate"]
        if set(old) != set(new):
            _difference(differences, "actors/resources", "Fitted resource sets differ")
        for m in sorted(set(old) & set(new)):
            a, b = old[m], new[m]
            for key in ("fit_record_ids", "label_scope", "fit_objective", "selected_step"):
                if a.get(key) != b.get(key):
                    _difference(differences, f"actors/M{m}/{key}", "Fit identity/selection changed")
            if a["source_checkpoint_sha256"] != b["source_checkpoint_sha256"]:
                _difference(differences, f"actors/M{m}/checkpoint", "Source weights changed")
            first, second = a["actor_state"], b["actor_state"]
            if set(first) != set(second):
                _difference(differences, f"actors/M{m}/state", "Actor state keys changed")
            for name in sorted(set(first) & set(second)):
                _compare_tensor(first[name], second[name], f"actors/M{m}/{name}", differences)
    return dict(matched=not differences, tolerance=0, differences=differences,
                scope="Diagnostic Actor tensors and fit identity; no model construction or inference")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", "--before", dest="baseline", type=Path, required=True)
    parser.add_argument("--candidate", "--after", dest="candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before-learning", type=Path)
    parser.add_argument("--after-learning", type=Path)
    args = parser.parse_args(argv)
    if (args.before_learning is None) != (args.after_learning is None):
        parser.error("--before-learning and --after-learning must be supplied together")
    report = compare_banks(args.baseline, args.candidate)
    if args.before_learning is not None:
        actors = compare_fitted_actors(args.before_learning, args.after_learning, args.baseline, args.candidate)
        report["actors"] = actors
        report["differences"].extend(actors["differences"])
        report["matched"] = report["matched"] and actors["matched"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("MATCH: source checkpoints, records, states and labels are identical" if report["matched"]
          else f"DIFFERENT/INVALID: {len(report['differences'])} issue(s)")
    for difference in report["differences"]:
        print(f"{difference['location']}: {difference['detail']}")
    return 0 if report["matched"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
