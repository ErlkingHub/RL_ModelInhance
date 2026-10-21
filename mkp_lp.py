"""Batched LP relaxation of the residual MKP: structural features, not a solver.

For the current state, only feasible unselected items enter
    max sum_i p_i x_i  s.t.  sum_i w_ij x_i <= remaining_j,  0 <= x_i <= 1.
Profits are divided by the mean feasible profit and every resource by its
remaining capacity, so duals are in "mean profit per full remaining resource".
The entropy-smoothed dual
    D_tau(y) = sum_j y_j + tau * sum_i softplus((p_i - w_i . y) / tau),  y >= 0
is minimized by projected Newton in the M dual variables with a decreasing tau.
For any y >= 0, sum_j y_j + sum_i max(0, reduced_i) is a valid upper bound on
the residual LP, hence on every completion (weak duality). Nothing here selects,
fixes, rounds or repairs items; the weak decoder still picks one feasible item.
Idea source: LP solution/dual features on the variable-constraint bipartite
graph follow Gasse et al. (2019); pseudo-utility p_i / sum_j y_j w_ij is the
classic surrogate-dual ratio used by Chu & Beasley (1998).
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


LP_ITEM_DIM = 4
LP_RESOURCE_DIM = 3
LP_EDGE_DIM = 1
LP_GLOBAL_DIM = 2
# Geometric backtracking candidates evaluated in parallel; the final 0 keeps y.
_STEPS = torch.cat((0.5 ** torch.arange(0, 24, 2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)))


@dataclass
class LPRelaxation:
    primal: torch.Tensor      # [B,N] smoothed LP x in [0,1]; zero for infeasible items
    reduced: torch.Tensor     # [B,N] p_i - w_i . y in mean-feasible-profit units
    dual: torch.Tensor        # [B,M] normalized nonnegative resource prices
    slack: torch.Tensor       # [B,M] 1 - normalized LP usage, in [0,1]
    dual_share: torch.Tensor  # [B,N,M] y_j w_ij / p_i, resource share of the dual cost
    bound: torch.Tensor       # [B] float64 upper bound on remaining profit, raw units
    feasible_mass: torch.Tensor  # [B] float64 feasible profit mass, raw units


def lp_signature(config):
    return tuple(getattr(config, name) for name in ("eps", "lp_temperatures", "lp_newton_steps"))


def _smoothed_dual(y, profits, loads, feasible, tau):
    reduced = profits - (loads @ y[..., None]).squeeze(-1)
    return y.sum(-1) + tau * (F.softplus(reduced / tau) * feasible).sum(-1)


@torch.no_grad()
def solve_lp_relaxation(state, config):
    eps = config.eps
    feasible = state.feasible
    mask = feasible.double()
    p, w, rem = state.profits.double(), state.weights.double(), state.remaining.double()
    count = mask.sum(-1)
    mass = (p * mask).sum(-1)
    scale = (mass / count.clamp_min(1)).clamp_min(eps)
    profits = p / scale[:, None] * mask
    # Feasible items satisfy w_ij <= remaining_j, so loads lie in [0, 1].
    loads = w / rem[:, None].clamp_min(eps) * mask[..., None]
    b, m = rem.shape
    y = rem.new_zeros(b, m)
    steps = _STEPS.to(rem)
    eye = torch.eye(m, dtype=rem.dtype, device=rem.device)
    for tau in config.lp_temperatures:
        for _ in range(config.lp_newton_steps):
            reduced = profits - (loads @ y[..., None]).squeeze(-1)
            sigma = torch.sigmoid(reduced / tau) * mask
            gradient = 1 - torch.einsum("bn,bnm->bm", sigma, loads)
            curvature = sigma * (1 - sigma) / tau
            hessian = torch.einsum("bn,bnm,bnk->bmk", curvature, loads, loads)
            # Projected Newton: bound-active duals keep zero and leave the system.
            free = ~((y <= 0) & (gradient > 0))
            pair = free[:, :, None] & free[:, None, :]
            damping = 1e-9 + 1e-7 * hessian.diagonal(dim1=1, dim2=2).mean(-1)
            system = torch.where(pair, hessian, torch.zeros_like(hessian)) + eye * damping[:, None, None]
            system = system + torch.diag_embed((~free).to(rem.dtype))
            direction = torch.linalg.solve(system, (gradient * free)[..., None]).squeeze(-1)
            # Trust region: a flat Hessian (all items far from the boundary) cannot jump far.
            radius = y.amax(-1, keepdim=True).clamp_min(1.0)
            direction = direction * (radius / direction.abs().amax(-1, keepdim=True).clamp_min(1e-30)).clamp_max(1)
            candidates = (y[:, None] - steps[None, :, None] * direction[:, None]).clamp_min(0)
            values = _smoothed_dual(candidates, profits[:, None], loads[:, None], mask[:, None], tau)
            # Step 0 is the current point, so the smoothed dual never increases.
            y = candidates[torch.arange(b, device=y.device), values.argmin(-1)]
    tau = config.lp_temperatures[-1]
    reduced = profits - (loads @ y[..., None]).squeeze(-1)
    primal = torch.sigmoid(reduced / tau) * mask
    usage = torch.einsum("bn,bnm->bm", primal, loads)
    bound = (y.sum(-1) + (reduced.clamp_min(0) * mask).sum(-1)) * scale
    active = count > 0
    bound = torch.where(active, torch.minimum(bound, mass), torch.zeros_like(bound))
    share = loads * y[:, None] / profits.clamp_min(eps)[..., None] * mask[..., None]
    return LPRelaxation(primal.float(), (reduced * mask).float(), y.float(),
                        (1 - usage).clamp(0, 1).float(), share.float(), bound, mass)


def lp_relaxation(state, config):
    """Per-state memo so the actor, critic and decoder share one solve per step."""
    from mkp_performance import state_stamp
    key = (state_stamp(state), lp_signature(config))
    memo = state._lp
    if memo is not None and memo[0] == key:
        return memo[1]
    result = solve_lp_relaxation(state, config)
    state._lp = (key, result)
    return result


def lp_potential(state, config):
    """Remaining LP bound as a fraction of instance profit; exactly zero when terminal."""
    lp = lp_relaxation(state, config)
    return (lp.bound / state.profits.sum(-1)).float()
