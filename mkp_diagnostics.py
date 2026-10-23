"""Post-hoc performance plateau diagnostics; never uses benchmark data or stops PPO."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from mkp_config import TrainConfig


def convergence_report(rows, config, *, complete=False):
    """Find the trailing validation band near the best observed greedy return.

    Every validation in the suffix must be within tolerance of the global best.
    A new improvement or regression can invalidate an earlier provisional plateau.
    This is a declared finite-horizon heuristic, not a statistical convergence proof.
    """
    observations = [(int(r["update"]), float(r["validation_return"])) for r in rows
                    if r.get("validation_return") not in (None, "")]
    if any(not math.isfinite(v) for _, v in observations):
        raise ValueError("Non-finite validation return in convergence history")
    if any(b[0] <= a[0] for a, b in zip(observations, observations[1:])):
        raise ValueError("Validation updates must be strictly increasing")
    report = {
        "status": "insufficient_history", "training_complete": complete,
        "scope": "synthetic_learned_dynamic_greedy_validation_only",
        "criterion": "All trailing validation returns within tolerance of global best; sufficient span and evaluations",
        "tolerance_pct": config.convergence_tolerance_pct,
        "patience_updates": config.convergence_patience_updates,
        "min_updates": config.convergence_min_updates,
        "min_evaluations": config.convergence_min_evaluations,
        "observed_from_update": observations[0][0] if observations else None,
        "observed_through_update": observations[-1][0] if observations else None,
        "best_update": None, "best_validation_return": None,
        "plateau_start_update": None, "confirmation_update": None,
        "tail_start_update": None, "tail_span_updates": 0, "tail_evaluations": 0,
    }
    if not observations:
        return report
    best_update, best = max(observations, key=lambda pair: pair[1])
    # Older resume checkpoints may retain a best score but no earlier curve.
    prior_best = max((float(r["best_validation_return"]) for r in rows
                      if r.get("best_validation_return") not in (None, "")), default=best)
    if not math.isfinite(prior_best):
        raise ValueError("Non-finite best validation return in convergence history")
    if prior_best > best:
        best, best_update = prior_best, None
    tolerance = max(abs(best), 1e-12) * config.convergence_tolerance_pct / 100
    start = len(observations)
    for i in range(len(observations) - 1, -1, -1):
        if best - observations[i][1] > tolerance + 1e-12:
            break
        start = i
    tail = observations[start:]
    report.update(best_update=best_update, best_validation_return=best,
                  lower_band=best - tolerance,
                  tail_start_update=tail[0][0] if tail else None,
                  tail_span_updates=tail[-1][0] - tail[0][0] if tail else 0,
                  tail_evaluations=len(tail))
    enough_history = (observations[-1][0] >= config.convergence_min_updates
                      and observations[-1][0] - observations[0][0] >= config.convergence_patience_updates
                      and len(observations) >= config.convergence_min_evaluations)
    report["status"] = "not_converged" if enough_history else "insufficient_history"
    confirmations = [u for i, (u, _) in enumerate(tail)
                     if u >= config.convergence_min_updates
                     and u - tail[0][0] >= config.convergence_patience_updates
                     and i + 1 >= config.convergence_min_evaluations]
    if confirmations:
        report.update(status="converged" if complete else "provisional",
                      plateau_start_update=tail[0][0], confirmation_update=confirmations[0])
    return report


def prepare_plotting():
    # Fail before expensive training if this required reporting dependency is missing.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training(rows, report, config, output_dir, m):
    """Raw curves plus trailing update-window means; marker positions are unsmoothed."""
    plt = prepare_plotting()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updates = np.array([int(row["update"]) for row in rows])
    panels = [
        ("Greedy learned_dynamic validation", ["validation_return", "best_validation_return"]),
        ("Training return", ["normalized_return"]),
        ("Policy entropy (normalized)", ["entropy"]),
        ("PPO policy loss", ["policy_loss"]),
        ("Critic value loss", ["value_loss"]),
        ("Constraint auxiliary loss", ["constraint_aux_loss"]),
        ("Approximate KL", ["approx_kl"]),
        ("PPO clip fraction", ["clip_fraction"]),
        ("Gradient norm (before clipping)", ["grad_norm"]),
        ("Critic explained variance", ["explained_variance"]),
        ("Capacity utilization", ["utilization"]),
        ("KL early-stop frequency", ["kl_stopped"]),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True)
    for ax, (title, fields) in zip(axes.flat, panels):
        plotted_values = []
        for field in fields:
            values = np.array([float(row[field]) if row.get(field) not in (None, "") else np.nan
                               for row in rows])
            valid = np.isfinite(values)
            if not valid.any():
                continue
            plotted_values.extend(values[valid])
            if field in ("validation_return", "best_validation_return"):
                ax.plot(updates[valid], values[valid], label=field, linewidth=1.3,
                        marker="." if field == "validation_return" else None, markersize=3)
            else:
                ax.plot(updates, values, color="#638cb5", alpha=0.25, linewidth=0.6, label="raw")
                # Vectorized, ignores NaNs, and uses update distance (including resumed runs).
                left = np.searchsorted(updates, updates - config.plot_smoothing_updates + 1)
                sums = np.r_[0., np.cumsum(np.where(valid, values, 0))]
                counts = np.r_[0, np.cumsum(valid)]
                count = counts[1:] - counts[left]
                smooth = np.divide(sums[1:] - sums[left], count,
                                   out=np.full(len(values), np.nan), where=count > 0)
                ax.plot(updates, smooth, color="#175a94", linewidth=1.4,
                        label=f"trailing {config.plot_smoothing_updates} updates")
        if title.startswith("Greedy") and report.get("lower_band") is not None:
            ax.axhspan(report["lower_band"], report["best_validation_return"], color="#289b64", alpha=0.12)
        if report["plateau_start_update"] is not None:
            ax.axvline(report["plateau_start_update"], color="#d22d39", linewidth=2.2,
                       label=f"plateau start: {report['plateau_start_update']}")
            ax.axvline(report["confirmation_update"], color="#b37800", linestyle="--", linewidth=1.6,
                       label=f"confirmed: {report['confirmation_update']}")
        if title.startswith("Greedy") and report["best_update"] is not None:
            ax.scatter([report["best_update"]], [report["best_validation_return"]],
                       marker="*", s=150, color="#7041b5", zorder=5, label=f"best: {report['best_update']}")
        ax.set(title=title, xlabel="PPO update")
        ax.ticklabel_format(axis="y", useOffset=False)
        if plotted_values:
            low, high = min(plotted_values), max(plotted_values)
            scale = max(abs(low), abs(high), 1e-8)
            if high - low < scale * 1e-6:
                # Do not magnify cumulative-sum roundoff on constant metric curves.
                margin = max(scale * .05, 1e-6)
                ax.set_ylim(low - margin, high + margin)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    status = report["status"].upper().replace("_", " ")
    points = (f" | plateau start={report['plateau_start_update']}, confirmation={report['confirmation_update']}"
              if report["plateau_start_update"] is not None else " | no confirmed convergence point")
    fig.suptitle(f"M={m} | {status}{points}\n"
                 f"Learned_dynamic synthetic validation tolerance={config.convergence_tolerance_pct}% relative to best; "
                 f"patience={config.convergence_patience_updates} updates (not Chu gap convergence)", fontsize=14)
    paths = [output_dir / f"training_M{m}.{ext}" for ext in ("png", "svg")]
    try:
        fig.savefig(paths[0], dpi=160)
        fig.savefig(paths[1])
    finally:
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate training plots/diagnostics without training or inference")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    saved = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    config = TrainConfig(**saved["train"])
    with (args.run_dir / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for m in sorted({int(row["M"]) for row in rows}):
        history_path = args.run_dir / f"diagnostics_history_M{m}.json"
        history = (json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists()
                   else [row for row in rows if int(row["M"]) == m])
        report = convergence_report(history, config, complete=int(history[-1]["update"]) == config.updates)
        print(json.dumps({"M": m, **report}, indent=2))
        print(*plot_training(history, report, config, args.run_dir / "plots", m), sep="\n")


if __name__ == "__main__":
    main()
