"""Audit saved single-trajectory results; never trains, infers, or selects weights.

Exit codes: 0 = strict target met, 1 = valid comparison but target not met,
2 = malformed/incomparable evidence. All learned_dynamic instances are retained.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parent
METHOD = "learned_dynamic"
TARGET = 0.1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label}: nonfinite number")
    return result


def digest_value(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label}: expected SHA-256")
    return value.lower()


def load_result(directory):
    directory = Path(directory).resolve()
    manifest_file, instance_file = directory / "manifest.json", directory / "instances.csv"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{directory}: benchmark manifest must be complete")
    dataset_digest = digest_value(manifest.get("dataset_sha256"), "dataset_sha256")
    tolerance = finite(manifest.get("config", {}).get("feasibility_tolerance", 1e-7),
                       "feasibility_tolerance")
    if tolerance < 0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    with instance_file.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    if manifest.get("completed_rows") != len(raw):
        raise ValueError(f"{directory}: completed_rows does not match instances.csv")
    rows = {}
    for row in raw:
        if row.get("method") != METHOD:
            continue
        identity = row.get("instance_id", "").strip()
        if not identity or identity in rows:
            raise ValueError(f"{directory}: empty or duplicate {METHOD} instance_id {identity!r}")
        try:
            m, n = int(row["M"]), int(row["N"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{identity}: invalid M/N") from error
        if m < 1 or n < 1 or row.get("rollouts") != "1":
            raise ValueError(f"{identity}: positive M/N and exactly one greedy rollout required")
        objective = finite(row.get("objective"), f"{identity}.objective")
        reference = finite(row.get("reference"), f"{identity}.reference")
        gap = finite(row.get("gap_pct"), f"{identity}.gap_pct")
        if objective < 0 or reference <= 0:
            raise ValueError(f"{identity}: objective must be nonnegative and reference positive")
        recomputed = 100 * (reference - objective) / reference
        if not math.isclose(gap, recomputed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{identity}: gap does not match objective/reference formula")
        if row.get("feasible") not in ("0", "1"):
            raise ValueError(f"{identity}: feasible must be 0 or 1")
        violation = finite(row.get("max_capacity_violation"), f"{identity}.max_capacity_violation")
        if violation < 0:
            raise ValueError(f"{identity}: max_capacity_violation must be nonnegative")
        checkpoint = manifest.get("checkpoints", {}).get(str(m), {})
        checkpoint_digest = digest_value(checkpoint.get("sha256"), f"manifest checkpoint M={m}")
        if digest_value(row.get("checkpoint_sha256"), f"{identity}.checkpoint_sha256") != checkpoint_digest:
            raise ValueError(f"{identity}: row checkpoint SHA differs from manifest")
        reference_type = row.get("reference_type", "").strip()
        if not reference_type:
            raise ValueError(f"{identity}: missing reference_type")
        capacity = row.get("capacity_ratio", row.get("alpha"))
        capacity = finite(capacity, f"{identity}.capacity_ratio") if capacity not in (None, "") else None
        if capacity is not None and not 0 < capacity < 1:
            raise ValueError(f"{identity}: capacity_ratio must lie in (0, 1)")
        rows[identity] = dict(instance_id=identity, M=m, N=n, objective=objective,
                              reference=reference, reference_type=reference_type, gap_pct=gap,
                              feasible=row["feasible"] == "1" and violation <= tolerance,
                              max_capacity_violation=violation, capacity_ratio=capacity)
    if not rows or manifest.get("instance_count") != len(rows):
        raise ValueError(f"{directory}: instance_count must match all {METHOD} rows")
    # Optional historic geometry labels supply groups only, never replace outcomes.
    diagnosis_file = directory / "analysis" / "instance_diagnosis.json"
    grouping_source = None
    if diagnosis_file.is_file() and any(r["capacity_ratio"] is None for r in rows.values()):
        diagnosis = json.loads(diagnosis_file.read_text(encoding="utf-8-sig"))
        seen = set()
        for item in diagnosis:
            identity = str(item.get("instance_id", ""))
            if identity in seen:
                raise ValueError(f"{diagnosis_file}: duplicate instance_id")
            seen.add(identity)
            if identity in rows and item.get("capacity_ratio") is not None:
                row = rows[identity]
                if (item.get("M"), item.get("N"), item.get("reference")) != (row["M"], row["N"], row["reference"]):
                    raise ValueError(f"{diagnosis_file}: geometry group identity mismatch for {identity}")
                value = finite(item["capacity_ratio"], f"{identity}.capacity_ratio")
                if not 0 < value < 1:
                    raise ValueError("Invalid geometry capacity_ratio")
                row["capacity_ratio"] = value
        grouping_source = dict(path=str(diagnosis_file), sha256=sha256(diagnosis_file))
    metadata = dict(directory=str(directory), dataset_sha256=dataset_digest,
                    manifest_sha256=sha256(manifest_file), instances_sha256=sha256(instance_file),
                    checkpoints=manifest.get("checkpoints", {}), capacity_group_source=grouping_source)
    return rows, metadata


def summarize(pairs):
    before, after = ([pair[side]["gap_pct"] for pair in pairs] for side in (0, 1))
    differences = [right - left for left, right in zip(before, after)]
    return dict(instances=len(pairs), baseline_mean_gap_pct=statistics.mean(before),
                candidate_mean_gap_pct=statistics.mean(after),
                paired_mean_delta_pp=statistics.mean(differences),
                candidate_median_gap_pct=statistics.median(after),
                candidate_worst_gap_pct=max(after),
                candidate_strict_target_hits=sum(gap < TARGET for gap in after),
                candidate_feasible=sum(pair[1]["feasible"] for pair in pairs),
                wins=sum(right["objective"] > left["objective"] for left, right in pairs),
                ties=sum(right["objective"] == left["objective"] for left, right in pairs),
                losses=sum(right["objective"] < left["objective"] for left, right in pairs))


def compare_results(baseline, candidate):
    old, old_meta = load_result(baseline)
    new, new_meta = load_result(candidate)
    if old_meta["dataset_sha256"] != new_meta["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs")
    if set(old) != set(new):
        raise ValueError(f"Instance IDs differ; missing={sorted(set(old) - set(new))}, extra={sorted(set(new) - set(old))}")
    pairs = []
    for identity in sorted(old):
        left, right = old[identity], new[identity]
        for field in ("M", "N", "reference", "reference_type"):
            if left[field] != right[field]:
                raise ValueError(f"{identity}: unmatched {field}")
        if left["capacity_ratio"] is not None and right["capacity_ratio"] is not None:
            if not math.isclose(left["capacity_ratio"], right["capacity_ratio"], abs_tol=1e-12):
                raise ValueError(f"{identity}: unmatched capacity_ratio")
        pairs.append((left, right))
    overall = summarize(pairs)
    by_shape = []
    for m, n in sorted({(p[0]["M"], p[0]["N"]) for p in pairs}):
        by_shape.append(dict(M=m, N=n, **summarize([p for p in pairs if (p[0]["M"], p[0]["N"]) == (m, n)])))
    ratios = [left["capacity_ratio"] if left["capacity_ratio"] is not None else right["capacity_ratio"]
              for left, right in pairs]
    by_ratio, by_shape_ratio = [], []
    capacity_complete = all(ratio is not None for ratio in ratios)
    if capacity_complete:
        for ratio in sorted(set(ratios)):
            group = [pair for pair, value in zip(pairs, ratios) if value == ratio]
            by_ratio.append(dict(capacity_ratio=ratio, **summarize(group)))
        for m, n, ratio in sorted({(p[0]["M"], p[0]["N"], ratio) for p, ratio in zip(pairs, ratios)}):
            group = [p for p, value in zip(pairs, ratios) if (p[0]["M"], p[0]["N"], value) == (m, n, ratio)]
            by_shape_ratio.append(dict(M=m, N=n, capacity_ratio=ratio, **summarize(group)))
    feasible_all = overall["candidate_feasible"] == overall["instances"]
    target_met = feasible_all and overall["candidate_mean_gap_pct"] < TARGET
    return dict(scope="saved single-greedy results only; no training or inference", method=METHOD,
                reference="dataset supplied reference; no optimality certificate implied",
                baseline=old_meta, candidate=new_meta, target_gap_pct=TARGET,
                criterion="all candidate rows feasible AND arithmetic mean signed gap_pct strictly below 0.1",
                status="target_met" if target_met else "target_not_met", target_met=target_met,
                overall=overall, by_shape=by_shape, capacity_groups_complete=capacity_complete,
                by_capacity_ratio=by_ratio, by_shape_capacity_ratio=by_shape_ratio,
                instances=[dict(instance_id=left["instance_id"], M=left["M"], N=left["N"],
                                baseline_gap_pct=left["gap_pct"], candidate_gap_pct=right["gap_pct"],
                                delta_pp=right["gap_pct"] - left["gap_pct"], candidate_feasible=right["feasible"])
                           for left, right in pairs])


def markdown(report):
    overall = report["overall"]
    lines = [f"# Saved result comparison: {report['status']}", "",
             f"Method: `{METHOD}`, one greedy trajectory per instance. No rows discarded.", "",
             f"Mean gap: **{overall['baseline_mean_gap_pct']:.10f}% → {overall['candidate_mean_gap_pct']:.10f}%**; "
             f"paired change **{overall['paired_mean_delta_pp']:+.10f} percentage points**.", "",
             f"Feasible: {overall['candidate_feasible']}/{overall['instances']}; "
             f"strict individual gap < 0.1%: {overall['candidate_strict_target_hits']}/{overall['instances']}; "
             f"wins/ties/losses: {overall['wins']}/{overall['ties']}/{overall['losses']}.", "",
             f"Criterion: {report['criterion']}.", "",
             "Feasibility is checked against saved flags and capacity violations; this script does not reconstruct solutions.",
             "Reference values are dataset best-known values, not certified optima.", ""]
    for title, groups, keys in (("By shape", report["by_shape"], ("M", "N")),
                                ("By capacity ratio", report["by_capacity_ratio"], ("capacity_ratio",)),
                                ("By shape and capacity ratio", report["by_shape_capacity_ratio"], ("M", "N", "capacity_ratio"))):
        if not groups:
            continue
        lines += [f"## {title}", "", "| " + " | ".join((*keys, "count", "baseline %", "candidate %", "delta pp")) + " |",
                  "| " + " | ".join("---" for _ in range(len(keys) + 4)) + " |"]
        for group in groups:
            values = [str(group[key]) for key in keys] + [str(group["instances"]),
                      f"{group['baseline_mean_gap_pct']:.7f}", f"{group['candidate_mean_gap_pct']:.7f}",
                      f"{group['paired_mean_delta_pp']:+.7f}"]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    if not report["capacity_groups_complete"]:
        lines += ["Capacity groups omitted because complete labels are unavailable; overall and shape reports retain all instances.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results" / "20260922_083444_537744")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_results(args.baseline, args.candidate)
        prefix = (args.output_prefix or ROOT / "outputs" / "resource_strategy_v2" / f"comparison_{args.candidate.name}").resolve()
        for source in (args.baseline.resolve(), args.candidate.resolve()):
            if prefix == source or source in prefix.parents:
                raise ValueError("Comparison outputs must be outside both input result directories")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        markdown_path.write_text(markdown(report), encoding="utf-8")
        print(f"{report['status']}: candidate mean gap {report['overall']['candidate_mean_gap_pct']:.10f}%")
        print(f"JSON: {json_path}\nReport: {markdown_path}")
        return 0 if report["target_met"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Comparison rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
