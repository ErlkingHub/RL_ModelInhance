"""State-level second-order controls and fixed, bounded combination bases.

The controller emits resource/basis coefficients, never item coefficients.
Basis evaluation is static arithmetic: no environment branches or learned Q.
"""
import math
import torch
from torch import nn


COMBINATION_METRICS = ("interaction_weight_sum", "companion_weight_sum",
                       "interaction_penalty_mean", "companion_bonus_mean",
                       "combination_abs_mean", "combination_to_base_margin_ratio")
GATE_METRICS = ("efficiency_gate_mean", "effective_efficiency_gate_mean", "terminal_profit_override_rate")


class CombinationControlHead(nn.Module):
    """Permutation-equivariant symmetric resource coefficients and K basis weights."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, k = config.embed_dim, len(config.companion_scales)
        self.resource_pair = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1, bias=False))
        self.interaction_gate = nn.Linear(d, 1)
        self.companion_mix = nn.Linear(d, k)
        self.companion_gate = nn.Linear(d, 1)
        nn.init.zeros_(self.resource_pair[-1].weight)
        nn.init.zeros_(self.companion_mix.weight)
        nn.init.zeros_(self.companion_mix.bias)
        for layer, maximum in ((self.interaction_gate, config.interaction_weight_max),
                                (self.companion_gate, config.companion_weight_max)):
            nn.init.zeros_(layer.weight)
            fraction = min(config.combination_initial_weight / maximum, .5) if maximum else .5
            nn.init.constant_(layer.bias, math.log(fraction / (1 - fraction)))

    def forward(self, resources, pooled):
        b, m, d = resources.shape
        left, right = resources[:, :, None], resources[:, None, :]
        context = torch.cat((left + right, (left - right).abs(), left * right,
                             pooled[:, None, None].expand(b, m, m, d)), -1)
        logits = self.resource_pair(context).squeeze(-1)
        upper_mask = torch.ones(m, m, dtype=torch.bool, device=resources.device).triu()
        weights = logits.masked_fill(~upper_mask, -torch.inf).flatten(1).softmax(-1).view(b, m, m)
        weights = weights * (self.config.interaction_weight_max * self.interaction_gate(pooled).sigmoid())[:, :, None]
        # Store each unordered coefficient symmetrically, but count it once in scores.
        interaction = weights + weights.transpose(1, 2) - torch.diag_embed(weights.diagonal(dim1=1, dim2=2))
        companion = (self.companion_mix(pooled).softmax(-1)
                     * self.config.companion_weight_max * self.companion_gate(pooled).sigmoid())
        return interaction, companion


@torch.no_grad()
def feasible_means(state, eps):
    feasible = state.feasible
    count = feasible.sum(-1, keepdim=True).clamp_min(1)
    demand = (state.weights * feasible[..., None]).sum(1) / count
    profit = (state.profits * feasible).sum(-1, keepdim=True) / count
    return feasible, demand, profit.clamp_min(eps)


@torch.no_grad()
def interaction_loads(state, owners, items, eps):
    """Bounded demand-unit loads. No w/C factor shrinking quadratically with N."""
    _, demand, _ = feasible_means(state, eps)
    weight = state.weights[owners, items]
    return (weight / (weight + demand[owners]).clamp_min(eps)).float()


def _pair_geometry(state, owners, items, chunk_size):
    """Existing pair arithmetic in float64; distinct, currently feasible items only."""
    feasible = state.feasible
    ids = torch.arange(state.profits.shape[1], device=items.device)
    for row, col in zip(owners.split(chunk_size), items.split(chunk_size)):
        margin = (state.remaining[row] - state.weights[row, col])[:, None] - state.weights[row]
        valid = feasible[row] & (ids[None] != col[:, None]) & (margin >= 0).all(-1)
        yield row, col, margin, valid


def _state_has_pair(owners, candidate_has_pair, batch_size):
    # Integer counts keep this detached reduction deterministic on CUDA too.
    counts = owners.new_zeros(batch_size).scatter_add(0, owners, candidate_has_pair.long())
    return counts > 0


@torch.no_grad()
def has_feasible_pair(state, owners, items, chunk_size):
    """Whether a state can still accept two items; no rollout, partner choice or search."""
    parts = [valid.any(-1) for _, _, _, valid in _pair_geometry(state, owners, items, chunk_size)]
    candidates = torch.cat(parts) if parts else owners.new_empty(0, dtype=torch.bool)
    return _state_has_pair(owners, candidates, state.profits.shape[0])


@torch.no_grad()
def companion_basis(state, owners, items, config, *, return_has_pair=False):
    """K best static two-item matches; the partner is neither chosen nor rolled out.

    Pair utility combines bounded pair profit and leftover capacity in feasible
    mean-demand units. Max pooling keeps a rare good partner visible at larger N.
    Feasible-pair detection uses boolean geometry, never possibly underflowed bonuses.
    return_has_pair reuses that geometry for the terminal profit override.
    """
    _, demand, mean_profit = feasible_means(state, config.eps)
    scales = state.profits.new_tensor(config.companion_scales)
    parts, pair_parts = [], []
    for row, col, margin, valid in _pair_geometry(state, owners, items, config.combination_chunk_size):
        if return_has_pair:
            pair_parts.append(valid.any(-1))
        used_resource = demand[row] > 0
        residual = (margin.clamp_min(0) / demand[row, None].clamp_min(config.eps))
        residual = (residual * used_resource[:, None]).sum(-1) / used_resource.sum(-1)[:, None].clamp_min(1)
        pair_profit = state.profits[row, col, None] + state.profits[row]
        quality = pair_profit / (pair_profit + 2 * mean_profit[row]).clamp_min(config.eps)
        values = (quality[..., None] * torch.exp(-residual[..., None] / scales)
                  * valid[..., None])
        parts.append(values.amax(1).float())
    basis = torch.cat(parts) if parts else state.profits.new_empty((0, len(config.companion_scales)), dtype=torch.float32)
    if return_has_pair:
        candidates = torch.cat(pair_parts) if pair_parts else owners.new_empty(0, dtype=torch.bool)
        return basis, _state_has_pair(owners, candidates, state.profits.shape[0])
    return basis


def gate_statistics(policy):
    """Detached state-level controls, averaged by active transitions by the caller."""
    return torch.stack(tuple(policy.score_terms[key].detach().float() for key in
                             ("efficiency_gate", "effective_efficiency_gate", "terminal_profit_override")), -1)


def decoder_statistics(policy):
    """Per-state candidate score summaries, detached and temperature-independent.

    The final ratio compares *ranges*, so a common offset cannot change it.
    Singleton and terminal rows have ratio zero. A 1e-3 denominator floor avoids
    an unbounded ratio on an exact baseline tie; this is a diagnostic, not a loss.
    """
    terms = policy.score_terms
    base = terms["base"].detach()
    penalty, bonus = terms["interaction_penalty"].detach(), terms["companion_bonus"].detach()
    delta = bonus - penalty
    owners, b = policy.owners, policy.batch_size
    count = base.new_zeros(b).scatter_add(0, owners, torch.ones_like(base))
    means = [base.new_zeros(b).scatter_add(0, owners, x) / count.clamp_min(1)
             for x in (penalty, bonus, delta.abs())]
    def spread(values):
        high = values.new_full((b,), -torch.inf).scatter_reduce(0, owners, values, reduce="amax")
        low = values.new_full((b,), torch.inf).scatter_reduce(0, owners, values, reduce="amin")
        return torch.where(count > 1, high - low, torch.zeros_like(high))
    return torch.stack((*means, spread(delta) / spread(base).clamp_min(1e-3)), -1)
