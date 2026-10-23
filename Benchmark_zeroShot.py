"""Frozen inference on read_mat-compatible datasets. Never trains or selects checkpoints."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from mkp_config import BenchmarkConfig, config_dict
from mkp_basis import COMBINATION_METRICS
from mkp_data import instance_batch, load_benchmark
from mkp_utils import (construct_solution, file_sha256, load_checkpoint, resolve_device,
                       seed_everything, write_json)


def verify_solution(instance, result, tolerance):
    """Independently recompute profit/capacity in raw float64 units, not decoder units."""
    chosen = result["selected"].detach().cpu().numpy().astype(bool).reshape(-1)
    sequence = result["sequence"].detach().cpu().numpy().reshape(-1)
    sequence = sequence[sequence >= 0].astype(int)
    if chosen.shape != (instance.n,):
        raise ValueError("Wrong solution length")
    if (sequence >= instance.n).any() or len(sequence) != len(set(sequence.tolist())):
        raise ValueError("Out-of-range or repeated items in sequence")
    if not np.array_equal(np.sort(sequence), np.flatnonzero(chosen)):
        raise ValueError("Item sequence does not match selected set")
    usage = instance.weights[chosen].sum(0)
    violation = float(np.maximum(usage - instance.capacities, 0).max())
    objective = float(instance.profits[chosen].sum())
    recorded = float(result["objective"].item())
    if not math.isclose(recorded, objective, rel_tol=1e-12, abs_tol=tolerance):
        raise ValueError("Reported objective differs from independently recomputed objective")
    if violation > tolerance:
        raise ValueError(f"Infeasible solution: max absolute capacity violation={violation}")
    remaining = instance.capacities - usage
    maximal = not np.any((~chosen) & np.all(instance.weights <= remaining[None, :], axis=1))
    if not maximal:
        raise ValueError("Construction stopped with a feasible unselected item")
    return objective, usage / instance.capacities, violation, sequence.tolist()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def take_result(result, index):
    return {key: value[index].detach().cpu() for key, value in result.items()}


@torch.no_grad()
def solve_instance(instance, model, device):
    """One deterministic state-conditioned construction, timed including transfer."""
    synchronize(device)
    began = time.perf_counter()
    result = construct_solution(instance_batch(instance, 1, device), model)
    result = take_result(result, 0)
    synchronize(device)
    elapsed = time.perf_counter() - began
    return result, elapsed


def summarize(rows):
    if not rows or any(row["method"] != "learned_dynamic" for row in rows):
        raise ValueError("Summary requires nonempty learned_dynamic results")
    summaries = []
    keys = sorted({(row["M"], row["N"]) for row in rows}) + [("all", "all")]
    for m, n in keys:
        group = [r for r in rows if m == "all" or (r["M"], r["N"]) == (m, n)]
        gaps = [r["gap_pct"] for r in group if r["gap_pct"] is not None]
        mean = lambda field: float(np.mean([r[field] for r in group]))
        summaries.append(dict(method="learned_dynamic", M=m, N=n, instances=len(group), references=len(gaps),
            mean_gap_pct=float(np.mean(gaps)) if gaps else None,
            median_gap_pct=float(np.median(gaps)) if gaps else None,
            worst_gap_pct=float(np.max(gaps)) if gaps else None,
            target_hit_rate=float(np.mean([r["target_hit"] for r in group if r["gap_pct"] is not None])) if gaps else None,
            feasible_rate=mean("feasible"), mean_objective=mean("objective"),
            mean_seconds=mean("solve_seconds"), total_seconds=float(sum(r["solve_seconds"] for r in group))))
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty report")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_diagnostics(payload, m):
    """Attach the originating training report without using Chu to tune/select models."""
    run_dir = payload.get("run_dir")
    info = {"run_dir": run_dir, "checkpoint_update": payload["update"],
            "report_status": "unavailable", "convergence": None}
    if run_dir:
        path = Path(run_dir) / f"convergence_M{m}.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("M") == m and report.get("training_complete")
                    and report.get("final_update", -1) >= payload["update"]):
                info.update(report_status="available", convergence=report)
    return info


def benchmark(config=None):
    c = config or BenchmarkConfig()
    device = resolve_device(c.device)
    seed_everything(c.seed, deterministic=True, cpu_threads=c.cpu_threads)
    instances = load_benchmark(c.dataset, c.num_resources, c.num_items)
    models, metadata = {}, {}
    # Fail before starting inference if any required checkpoint is missing/incompatible.
    for m in sorted({inst.m for inst in instances}):
        checkpoint = c.checkpoint_dir / f"mkp_M{m}_{c.checkpoint_tag}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {checkpoint}; run train_ConstrainedMKP.py first")
        models[m], payload = load_checkpoint(checkpoint, device, m)
        models[m].eval().requires_grad_(False)
        metadata[str(m)] = {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint),
                            "checkpoint_version": payload["checkpoint_version"],
                            "controller_version": payload["controller_version"],
                            "decoder_version": payload["decoder_version"],
                            "actor_implementation": payload.get("actor_implementation", "legacy_v5"),
                            "update": payload["update"], "model_config": payload["model_config"],
                            "train_num_items": payload["train_config"]["train_num_items"],
                            "ranking_target_mode": payload["train_config"].get("ranking_target_mode", "legacy_mixed"),
                            "training_data_source": payload["training_data_source"],
                            "training_diagnostics": training_diagnostics(payload, m)}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = c.results_dir / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config_dict(c), "dataset_sha256": file_sha256(c.dataset),
                "checkpoints": metadata, "torch": str(torch.__version__), "device": str(device),
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "reference_type": "dataset_best_known_not_verified_optimal",
                "gap_formula": "100 * (reference - objective) / reference; signed, not clipped",
                "inference": "learned_dynamic_single_greedy",
                "timing": "Full construction including transfer, synchronized CUDA, no warmup; excludes checkpoint loading",
                "instance_count": len(instances), "status": "running"}
    write_json(output / "manifest.json", manifest)
    fields = ["instance_id", "M", "N", "method", "rollouts", "objective", "reference", "reference_type",
              "gap_pct", "target_hit", "reference_exceeded", "feasible", "max_capacity_violation",
              "selected_count", "mean_utilization", "min_utilization", "max_utilization", "utilization_by_resource",
              "solve_seconds", "profit_exponent_mean",
              "scarcity_exponent_mean", "bottleneck_weight_mean", "temperature_mean", "price_std_mean",
              "opportunity_weight_mean", "reserve_weight_mean",
              "size_zero_shot", "checkpoint_sha256", "solution_file"]
    fields += list(COMBINATION_METRICS)
    rows = []
    with (output / "instances.csv").open("w", newline="", encoding="utf-8-sig") as stream, \
         (output / "solutions.jsonl").open("w", encoding="utf-8") as solution_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        try:
            for number, instance in enumerate(instances, start=1):
                result, elapsed = solve_instance(instance, models[instance.m], device)
                objective, utilization, violation, sequence = verify_solution(instance, result, c.feasibility_tolerance)
                gap = 100 * (instance.reference - objective) / instance.reference if instance.reference is not None else None
                stats = result["theta_mean"].tolist()
                combination = stats[-2:] + result["decoder_mean"].tolist()
                row = dict(instance_id=instance.instance_id, M=instance.m, N=instance.n, method="learned_dynamic",
                        rollouts=1, objective=objective, reference=instance.reference,
                        reference_type="dataset_best_known", gap_pct=gap,
                        target_hit=int(gap < c.target_gap_pct) if gap is not None else None,
                        reference_exceeded=int(gap < -1e-9) if gap is not None else None,
                        feasible=1, max_capacity_violation=violation, selected_count=len(sequence),
                        mean_utilization=float(utilization.mean()), min_utilization=float(utilization.min()),
                        max_utilization=float(utilization.max()), utilization_by_resource=json.dumps(utilization.tolist()),
                        solve_seconds=elapsed,
                        profit_exponent_mean=stats[0], scarcity_exponent_mean=stats[1],
                        bottleneck_weight_mean=stats[2], temperature_mean=stats[3], price_std_mean=stats[4],
                        opportunity_weight_mean=stats[5], reserve_weight_mean=stats[6],
                        size_zero_shot=int(instance.n not in metadata[str(instance.m)]["train_num_items"]),
                        checkpoint_sha256=metadata[str(instance.m)]["sha256"], solution_file="solutions.jsonl")
                row.update(zip(COMBINATION_METRICS, combination))
                solution_stream.write(json.dumps({"instance_id": instance.instance_id, "method": "learned_dynamic",
                                                  "item_indices_zero_based": sequence, "objective": objective}) + "\n")
                writer.writerow(row)
                rows.append(row)
                stream.flush()
                solution_stream.flush()
                gap_text = f"{gap:.4f}%" if gap is not None else "NA"
                print(f"[{number}/{len(instances)}] id={instance.instance_id} M={instance.m} N={instance.n} gap={gap_text}")
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_rows=len(rows))
            write_json(output / "manifest.json", manifest)
            raise
    write_csv(output / "summary.csv", summarize(rows))
    # Each gap is for this evaluated checkpoint, not the historical plateau-start model.
    training_links = []
    for m, meta in metadata.items():
        group = [r for r in rows if r["M"] == int(m) and r["method"] == "learned_dynamic"
                 and r["gap_pct"] is not None]
        training_links.append({"M": int(m), "checkpoint_update": meta["update"],
                               "checkpoint_sha256": meta["sha256"],
                               "evaluated_checkpoint_mean_gap_pct": float(np.mean([r["gap_pct"] for r in group])) if group else None,
                               **meta["training_diagnostics"]})
    write_json(output / "training_convergence.json", training_links)
    manifest.update(status="complete", completed_rows=len(rows))
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output}")
    return output


def main(argv=None):
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="Frozen zero-shot MKP benchmark; no training")
    parser.add_argument("--dataset", type=Path, default=defaults.dataset)
    parser.add_argument("--resources", type=int, nargs="+", default=defaults.num_resources)
    parser.add_argument("--items", type=int, nargs="+", default=defaults.num_items)
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    parser.add_argument("--checkpoint-tag", choices=("best", "last"), default=defaults.checkpoint_tag)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    benchmark(BenchmarkConfig(dataset=args.dataset, num_resources=tuple(args.resources),
                              num_items=tuple(args.items), checkpoint_dir=args.checkpoint_dir,
                              checkpoint_tag=args.checkpoint_tag, device=args.device, seed=args.seed))


if __name__ == "__main__":
    main()
