"""Shared diagnostic I/O and same-state measurements; never used by inference."""
from dataclasses import fields
import json
import math
from pathlib import Path

import torch

from mkp_Model import StrategyTheta
from mkp_env import MKPEnv
from mkp_utils import file_sha256, load_checkpoint


def diagnostic_provenance():
    root = Path(__file__).resolve().parent
    names = ("diagnose_mkp.py", "mkp_audit_bank.py", "mkp_audit_theta.py", "mkp_audit_learning.py", "mkp_audit_objectives.py",
             "mkp_audit_common.py", "mkp_Model.py", "mkp_basis.py", "mkp_env.py", "mkp_Trainer.py", "mkp_data.py",
             "mkp_attention.py", "mkp_resource.py", "mkp_controller.py", "mkp_performance.py", "mkp_config.py",
             "mkp_ranking.py", "mkp_utils.py", "mkp_actor_guard.py")
    return {name: file_sha256(root / name) for name in names if (root / name).is_file()}


def write_json(path, value):
    def clean(x):
        if isinstance(x, torch.Tensor):
            return clean(x.detach().cpu().tolist())
        if isinstance(x, Path):
            return str(x.resolve())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_model(bank, m, device):
    root = Path(__file__).resolve().parent
    for name, digest in bank.get("source_hashes", {}).items():
        if Path(name).name != name or file_sha256(root / name) != digest:
            raise ValueError(f"State-bank source code changed: {name}; rebuild the bank")
    metadata = bank["checkpoints"][str(m)]
    path = Path(metadata["path"])
    if file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"Source checkpoint changed since bank creation: {path}")
    model, payload = load_checkpoint(path, device, m)
    model.eval()
    return model, payload


def check_report(bank_path, report):
    if report.get("bank_sha256") != file_sha256(bank_path):
        raise ValueError("Diagnostic artifacts refer to different state banks")


def theta_payload(theta):
    return {f.name: getattr(theta, f.name).detach().cpu() for f in fields(theta)}


def theta_from_payload(payload, device):
    return StrategyTheta(**{f.name: torch.as_tensor(payload[f.name], device=device, dtype=torch.float32)
                            for f in fields(StrategyTheta)})


@torch.no_grad()
def greedy_future(model, state):
    """Complete the same true prefix under this model; no external teacher."""
    prefix = (state.profits * state.selected).sum(-1)
    env = MKPEnv.from_state(state)
    for _ in range(state.profits.shape[1]):
        if not env.state.active.any():
            return float((env.objective - prefix).item())
        env.step(model.policy(env.state)[0].choose())
    if env.state.active.any():
        raise RuntimeError("Diagnostic continuation did not terminate within N decisions")
    return float((env.objective - prefix).item())


@torch.no_grad()
def evaluate_records(model, records, device, *, returns=True):
    from mkp_audit_bank import state_from_record
    rows = []
    model.eval()
    for record in records:
        state = state_from_record(record, device)
        action = int(model.policy(state)[0].choose().item())
        if action < 0 or action >= state.profits.shape[1] or not state.feasible[0, action]:
            raise ValueError("Diagnostic bank must contain active states with a legal greedy action")
        targets = record["target_actions"].tolist()
        values = record["future_values"]
        known = bool(torch.isfinite(values[action]))
        reference = float(record["reference_future_value"])
        regret = (max(0., reference - float(values[action])) / max(reference, 1e-12)
                  if record["label_kind"] == "exact" and known else None)
        row = {key: record[key] for key in ("id", "instance_id", "split", "stage", "M", "N", "label_kind")}
        row["capacity_ratio"] = record["capacity_ratio"]
        row.update(action=action, target_hit=action in targets, action_value_known=known,
                   exact_action_regret=regret,
                   known_action_value=float(values[action]) if known else None,
                   baseline_greedy_future=float(record["baseline_greedy_future"]))
        if returns:
            future = greedy_future(model, state)
            delta = future - float(record["baseline_greedy_future"])
            row.update(greedy_future=future, greedy_future_delta=delta,
                       greedy_delta_over_initial_profit=delta / max(float(state.profits.sum()), 1e-12))
            if record["label_kind"] == "finite_greedy" and known:
                row.update(frozen_continuation_gain=float(values[action]) - float(record["baseline_greedy_future"]),
                           continuation_policy_drift=future - float(values[action]))
        rows.append(row)
    return rows


def summarize_rows(rows):
    def summary(part):
        exact = [r["exact_action_regret"] for r in part if r.get("exact_action_regret") is not None]
        differences = [r["greedy_future_delta"] for r in part if "greedy_future_delta" in r]
        frozen_gains = [r["frozen_continuation_gain"] for r in part if "frozen_continuation_gain" in r]
        drift = [r["continuation_policy_drift"] for r in part if "continuation_policy_drift" in r]
        return dict(states=len(part), target_hit_rate=sum(r["target_hit"] for r in part) / len(part),
            known_action_rate=sum(r["action_value_known"] for r in part) / len(part),
            mean_exact_action_regret=sum(exact) / len(exact) if exact else None,
            mean_greedy_future_delta=sum(differences) / len(differences) if differences else None,
            mean_frozen_continuation_gain=sum(frozen_gains) / len(frozen_gains) if frozen_gains else None,
            mean_continuation_policy_drift=sum(drift) / len(drift) if drift else None,
            greedy_improved_states=sum(d > 1e-8 for d in differences),
            greedy_worsened_states=sum(d < -1e-8 for d in differences))
    groups = {}
    for r in rows:
        key = f"M{r['M']}/N{r['N']}/r{r['capacity_ratio']:g}/{r['split']}/{r['label_kind']}/{r['stage']}"
        groups.setdefault(key, []).append(r)
    result = {key: summary(part) for key, part in groups.items()}
    for split in ("fit", "heldout"):
        part = [r for r in rows if r["split"] == split]
        if part:
            result[split] = summary(part)
    return result
