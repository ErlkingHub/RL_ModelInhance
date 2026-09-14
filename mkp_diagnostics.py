"""Training diagnostics only. Offline plotting never loads a model or runs an episode."""
import argparse
import csv
import json
import math
from pathlib import Path


def critic_diagnostics(rewards, values, dones, active, gamma):
    """Compare pre-update values with complete discounted Monte Carlo returns.

    Unlike GAE-derived explained variance, these targets contain no value bootstrap.
    Padded states are excluded. These numbers do not enter the PPO objective.
    """
    import torch
    with torch.no_grad():
        returns = torch.zeros_like(rewards)
        carry = torch.zeros_like(rewards[0])
        for t in reversed(range(len(rewards))):
            carry = (rewards[t] + gamma * carry * (~dones[t])) * active[t]
            returns[t] = carry
        target, prediction = returns[active], values[active]
        if target.numel() == 0:
            return {"critic_mc_rmse": float("nan"), "critic_mc_ev": float("nan")}
        error = target - prediction
        variance = target.var(unbiased=False)
        return {"critic_mc_rmse": error.square().mean().sqrt().item(),
                "critic_mc_ev": (1 - error.var(unbiased=False) / variance).item()
                if variance > 1e-12 else float("nan")}


def read_metrics(path):
    """Old CSV columns are optional; absent observations remain missing, never zero."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"M", "N", "update"}.issubset(reader.fieldnames or []):
            raise ValueError("metrics.csv must contain M, N, update")
        rows = []
        for record in reader:
            row = {}
            for key, value in record.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = float("nan")
            if all(math.isfinite(row[k]) for k in ("M", "N", "update")):
                rows.append(row)
    return rows


def create_training_diagnostics(run_dir, smoothing_window=25, dpi=140):
    """One 4x3 dashboard per M. Safe for partial runs and legacy metrics.csv files."""
    if smoothing_window < 1 or dpi < 1:
        raise ValueError("smoothing_window and dpi must be positive")
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    run_dir = Path(run_dir)
    rows = read_metrics(run_dir / "metrics.csv")
    if not rows:
        return []
    config_path = run_dir / "config.json"
    settings = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    train = settings.get("train", {})
    outputs = []

    def moving_average(values):
        # Trailing window of available observations (not a centered/future-dependent filter).
        count = np.arange(1, len(values) + 1)
        left = np.maximum(count - smoothing_window, 0)
        sums = np.concatenate(([0.0], np.cumsum(values)))
        return (sums[count] - sums[left]) / (count - left)

    for m in sorted({int(row["M"]) for row in rows}):
        selected = sorted((r for r in rows if int(r["M"]) == m), key=lambda r: r["update"])
        figure = Figure(figsize=(17, 15), facecolor="#f5f7fa", layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 3).ravel()

        def curve(ax, key, label, color, smooth=True, source=None):
            source = selected if source is None else source
            valid = [r for r in source if math.isfinite(r.get(key, float("nan")))]
            if not valid:
                return False
            x = np.array([r["update"] for r in valid])
            y = np.array([r[key] for r in valid])
            if smooth and len(y) > 1:
                ax.plot(x, y, color=color, alpha=0.16, linewidth=0.6)
                y = moving_average(y)
            ax.plot(x, y, label=label, color=color, linewidth=1.5,
                    marker="." if not smooth else None, markersize=3)
            return True

        titles = ["Sampled training return", "Fixed synthetic validation", "Validation vs fixed weak policy",
                  "PPO policy and total loss", "Value and constraint losses", "Policy entropy (normalized)",
                  "Approximate KL and early stopping", "PPO clipping and update budget", "Gradient norm before clipping",
                  "Critic prediction quality", "Terminal packing statistics", "High-level strategy (rollout means)"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("PPO update", fontsize=9)
            ax.grid(alpha=0.17)
            ax.tick_params(labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        colors = ["#146c94", "#d97706", "#7c3aed", "#059669", "#dc2626"]
        for i, n in enumerate(sorted({int(r["N"]) for r in selected})):
            curve(axes[0], "normalized_return", f"N={n}", colors[i % len(colors)],
                  source=[r for r in selected if int(r["N"]) == n])
        axes[0].set_ylabel("Selected profit / total item profit")
        curve(axes[1], "validation_return", "Greedy validation", colors[0], False)
        curve(axes[1], "best_validation_return", "Best so far", colors[3], False)
        axes[1].set_ylabel("Normalized return (independent fixed set)")
        if any(math.isfinite(r.get("validation_objective_mean", float("nan"))) for r in selected):
            axes[2].set_title("Dynamic-policy validation objective", loc="left", fontsize=11, fontweight="bold")
            curve(axes[2], "validation_objective_mean", "Mean objective (fixed validation mix)", colors[0], False)
            axes[2].set_ylabel("Raw profit - not optimality gap")
        else:
            curve(axes[2], "validation_improvement_pct", "Relative objective improvement", colors[0], False)
            axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
            axes[2].set_ylabel("Improvement (%) - not optimality gap")
        curve(axes[3], "policy_loss", "Policy loss", colors[0])
        curve(axes[3], "total_loss", "Weighted total loss", colors[1])
        curve(axes[4], "value_loss", "Value loss", colors[0])
        curve(axes[4], "constraint_aux_loss", "Constraint auxiliary loss", colors[1])
        axes[4].set_yscale("symlog", linthresh=1e-7)
        curve(axes[5], "entropy", "PPO entropy / log(candidate count)", colors[0])
        curve(axes[5], "rollout_entropy", "Collected-policy entropy (normalized)", colors[1])
        axes[5].set_ylim(-0.03, 1.03)
        curve(axes[6], "approx_kl", "Accepted minibatch mean KL", colors[0])
        curve(axes[6], "stop_kl", "Rejected minibatch KL", colors[4], False)
        if "target_kl" in train:
            axes[6].axhline(train["target_kl"], color=colors[1], linestyle="--", label="Stop threshold")
        curve(axes[7], "clip_fraction", "Clipped action ratios", colors[0])
        curve(axes[7], "ppo_update_fraction", "Applied transition updates / planned", colors[3])
        curve(axes[7], "kl_stopped", "Early-stop frequency (rolling)", colors[1])
        axes[7].set_ylim(-0.03, 1.03)
        curve(axes[8], "grad_norm", "Global parameter gradient norm", colors[0])
        if "max_grad_norm" in train:
            axes[8].axhline(train["max_grad_norm"], color=colors[1], linestyle="--", label="Clip threshold")
        curve(axes[9], "explained_variance", "EV against GAE targets", colors[0])
        curve(axes[9], "critic_mc_ev", "EV against MC returns", colors[3])
        axes[9].set_ylabel("Explained variance")
        axes[9].set_yscale("symlog", linthresh=0.1)
        extra = None
        if any(math.isfinite(r.get("critic_mc_rmse", float("nan"))) for r in selected):
            extra = axes[9].twinx()
            curve(extra, "critic_mc_rmse", "MC return RMSE", colors[1])
            extra.set_ylabel("MC RMSE (normalized return)", color=colors[1], fontsize=8)
            extra.tick_params(labelsize=8)
        curve(axes[10], "utilization", "Mean resource utilization", colors[0])
        curve(axes[10], "selected_fraction", "Selected item fraction", colors[1])
        axes[10].set_ylim(-0.03, 1.03)
        for key, label, color in zip(("theta_alpha", "theta_beta", "theta_kappa", "theta_temperature", "theta_price_std"),
                                     ("Profit alpha", "Scarcity beta", "Bottleneck kappa", "Temperature", "Resource-price std"), colors):
            curve(axes[11], key, label, color)
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if ax is axes[9] and extra is not None:
                eh, el = extra.get_legend_handles_labels()
                handles, labels = handles + eh, labels + el
            if handles:
                ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.85)
            if not ax.has_data():
                label = ("Not applicable: direct item head" if ax is axes[11] and
                         settings.get("model", {}).get("policy_head") == "direct_item" else "Not recorded in this CSV")
                ax.text(0.5, 0.5, label, transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
        model_settings = settings.get("model", {})
        model_type = (f"{model_settings.get('encoder_type', 'transformer')} / "
                      f"{model_settings.get('policy_head', 'weak')} / "
                      f"{model_settings.get('price_head_type', 'linear (legacy)')}")
        figure.suptitle(f"MKP training diagnostics | M={m} | model: {model_type}\n"
                        f"{run_dir.name} | {len(selected)} updates | trailing mean: {smoothing_window} observations\n"
                        "Faint lines: raw data. Missing legacy metrics are omitted. Synthetic validation is not Chu gap.",
                        fontsize=13)
        output = run_dir / f"training_curves_M{m}.png"
        temporary = output.with_suffix(".tmp.png")
        try:
            figure.savefig(temporary, dpi=dpi)
            temporary.replace(output)
        finally:
            figure.clear()
        outputs.append(output)
    return outputs


def main(argv=None):
    from mkp_config import TrainConfig
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Render existing training metrics only; no training/inference")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=defaults.plot_smoothing_window)
    parser.add_argument("--dpi", type=int, default=defaults.plot_dpi)
    args = parser.parse_args(argv)
    for output in create_training_diagnostics(args.run_dir, args.window, args.dpi):
        print(output.resolve())


if __name__ == "__main__":
    main()
