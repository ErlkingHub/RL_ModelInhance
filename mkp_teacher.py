"""Training-only residual MKP supervision from synthetic states supplied by callers.

Enumeration supplies exact all-action continuation values only for small feasible
sets. A bounded MILP run supplies either a certified optimal set or an independently
verified feasible improvement over a supplied frozen-policy suffix. An incumbent
is never presented as an exact action value. This module loads no datasets and
does not run a policy, optimizer, or decoder.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from mkp_env import MKPState


ENUMERATION_CHUNK_SIZE = 8192
MAX_EXACT_ITEMS = 18
INTEGRALITY_TOLERANCE = 1e-7


@dataclass
class TeacherResult:
    target_mask: torch.Tensor  # [N] bool, safe exact set or witnessed improvement set
    action_values: torch.Tensor  # [N] float64; NaN wherever exact Q is unknown
    objective: float  # Future profit only; never includes the selected prefix.
    upper_bound: float | None
    optimality_proven: bool
    source: str
    feasible_count: int
    improvement_vs_baseline: float | None
    solver_status: int | None = None
    solver_message: str = ""
    selected_mask: torch.Tensor | None = None  # One complete feasible future plan, not target union.
    baseline_objective: float | None = None


def _settings(config):
    exact = config.teacher_exact_items
    nodes = config.teacher_node_limit
    seconds = float(config.teacher_time_limit)
    gap = float(config.teacher_mip_rel_gap)
    if isinstance(exact, bool) or not isinstance(exact, int) or not 0 <= exact <= MAX_EXACT_ITEMS:
        raise ValueError("teacher_exact_items must be an integer in [0, 18]")
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError("teacher_node_limit must be a positive integer")
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(gap) or gap < 0:
        raise ValueError("Teacher time limit must be positive and relative gap nonnegative")
    return exact, dict(time_limit=seconds, node_limit=nodes, mip_rel_gap=gap, disp=False)


def _arrays(state):
    if state.profits.ndim != 2 or state.profits.shape[0] != 1 or state.profits.shape[1] < 1:
        raise ValueError("Teacher expects one nonempty MKPState row")
    n = state.profits.shape[1]
    if (state.weights.ndim != 3 or state.weights.shape[:2] != (1, n)
            or state.weights.shape[2] < 1 or state.remaining.shape != state.weights.shape[::2]
            or state.selected.shape != state.profits.shape or state.selected.dtype != torch.bool):
        raise ValueError("Invalid teacher state shapes or selection mask")
    p, w, rem = (x.detach().to(device="cpu", dtype=torch.float64).numpy()[0].copy()
                 for x in (state.profits, state.weights, state.remaining))
    selected = state.selected.detach().cpu().numpy()[0].copy()
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (p, w, rem)):
        raise ValueError("Teacher physical data must be finite and nonnegative")
    feasible = ~selected & (w <= rem[None]).all(axis=1)
    return p, w, rem, selected, feasible


def _baseline_objective(mask, p, w, remaining, selected, feasible):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (p.size,):
        raise ValueError("baseline_selected must be a boolean future-item mask [N]")
    chosen = mask.detach().cpu().numpy()
    if (chosen & selected).any():
        raise ValueError("Baseline future plan must exclude the selected prefix")
    if (chosen & ~feasible).any() or (w[chosen].sum(axis=0, dtype=np.float64) > remaining).any():
        raise ValueError("Baseline future plan must be feasible in original float64 capacities")
    return float(p[chosen].sum(dtype=np.float64))


def _value_tolerance(p):
    # Used for objective comparisons and optimal-action ties, never feasibility.
    return 1e-9 * max(1., float(p.sum(dtype=np.float64)))


def _enumerate(p, w, remaining):
    """All subsets, streamed in fixed chunks; every chosen action gets its Q."""
    k = p.size
    action_values = np.full(k, -np.inf, dtype=np.float64)
    optimum = 0.
    best_selected = np.zeros(k, dtype=bool)
    shifts = np.arange(k, dtype=np.uint64)
    for start in range(0, 1 << k, ENUMERATION_CHUNK_SIZE):
        numbers = np.arange(start, min(start + ENUMERATION_CHUNK_SIZE, 1 << k), dtype=np.uint64)
        chosen = ((numbers[:, None] >> shifts[None]) & 1).astype(np.float64)
        loads = chosen @ w
        valid = (loads <= remaining[None]).all(axis=1)
        if not valid.any():
            continue
        chosen = chosen[valid].astype(bool)
        values = chosen @ p
        best = int(values.argmax())
        if values[best] > optimum:
            optimum = float(values[best])
            best_selected = chosen[best].copy()
        for item in range(k):
            containing = chosen[:, item]
            if containing.any():
                action_values[item] = max(action_values[item], float(values[containing].max()))
    return optimum, action_values, best_selected


def _finite_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def solve_teacher(state: MKPState, config, baseline_selected: torch.Tensor | None = None) -> TeacherResult | None:
    """Solve one synthetic residual problem without using already selected profit.

    The caller is responsible for synthetic-only provenance. For non-exact MILP
    incumbents, a valid baseline suffix is mandatory, and only strictly improving
    feasible plans become targets. MILP feasibility is rechecked using the stored
    original float64 weights and capacities with no capacity tolerance or repair.
    """
    exact_limit, options = _settings(config)
    p, w, remaining, selected, feasible = _arrays(state)
    baseline = _baseline_objective(baseline_selected, p, w, remaining, selected, feasible)
    indices = np.flatnonzero(feasible)
    if not indices.size:
        return None
    future_p, future_w = p[indices], w[indices]
    tolerance = _value_tolerance(future_p)
    target = torch.zeros(p.size, dtype=torch.bool)
    action_values = torch.full((p.size,), float("nan"), dtype=torch.float64)
    solution = torch.zeros(p.size, dtype=torch.bool)
    if indices.size <= exact_limit:
        objective, exact_values, best_selected = _enumerate(future_p, future_w, remaining)
        action_values[torch.from_numpy(indices)] = torch.from_numpy(exact_values)
        target[torch.from_numpy(indices[exact_values >= objective - tolerance])] = True
        solution[torch.from_numpy(indices[best_selected])] = True
        return TeacherResult(target, action_values, objective, objective, True,
                             "exact_enumeration", int(indices.size),
                             None if baseline is None else objective - baseline,
                             selected_mask=solution, baseline_objective=baseline)

    result = milp(c=-future_p, integrality=np.ones(indices.size, dtype=np.uint8),
                  bounds=Bounds(np.zeros(indices.size), np.ones(indices.size)),
                  constraints=LinearConstraint(future_w.T, -np.inf, remaining), options=options)
    raw = getattr(result, "x", None)
    if raw is None:
        return None
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (indices.size,) or not np.isfinite(raw).all():
        return None
    rounded = np.rint(raw)
    if ((np.abs(raw - rounded) > INTEGRALITY_TOLERANCE).any()
            or (rounded < 0).any() or (rounded > 1).any()):
        return None
    chosen = rounded.astype(bool)
    if not chosen.any() or (future_w[chosen].sum(axis=0, dtype=np.float64) > remaining).any():
        return None
    objective = float(future_p[chosen].sum(dtype=np.float64))
    reported = _finite_number(getattr(result, "fun", None))
    if reported is None or abs(reported + objective) > tolerance:
        return None
    dual = _finite_number(getattr(result, "mip_dual_bound", None))
    upper = None if dual is None else -dual
    if upper is not None and upper < objective - tolerance:
        # An inconsistent certificate cannot make a feasible witness optimal.
        upper = None
    raw_status = getattr(result, "status", None)
    status = int(raw_status) if isinstance(raw_status, (int, np.integer)) else None
    gap = _finite_number(getattr(result, "mip_gap", None))
    optimal = (status == 0 and bool(getattr(result, "success", False))
               and upper is not None and abs(upper - objective) <= tolerance
               and gap is not None and 0 <= gap <= 1e-9)
    improvement = None if baseline is None else objective - baseline
    if not optimal and (improvement is None or improvement <= tolerance):
        return None
    target[torch.from_numpy(indices[chosen])] = True
    solution.copy_(target)
    if optimal:
        # Each member occurs in this certified optimal subset; other Qs remain unknown.
        action_values[target] = objective
    return TeacherResult(target, action_values, objective, upper, optimal,
                         "milp_optimal" if optimal else "milp_incumbent", int(indices.size),
                         improvement, status, str(getattr(result, "message", "")),
                         solution, baseline)
