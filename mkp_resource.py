"""Actor-only resource refinement from current feasible demand.

This path keeps the original pair-informed item embeddings and changes only
resource embeddings and the state summary. It has no item/action score head.
Capacity arithmetic is detached float64; learned inputs are compact float32.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from mkp_attention import masked_softmax


RESOURCE_GEOMETRY_DIM = 12
RESOURCE_RELATION_DIM = 6


@dataclass
class ResourceGeometry:
    resources: torch.Tensor   # [B,M,12], feasible-demand node statistics
    relations: torch.Tensor   # [B,M,M,6], co-demand/competition relations
    pressure: torch.Tensor    # [B,M], log1p current total feasible load


@torch.no_grad()
def resource_geometry(state, config):
    """Compute resource geometry without any completion/search or learned score.

    Node channels: remaining/used capacity fractions, feasible current-load
    mean/std/max, profit-weighted load, log1p(total load), log1p(remaining
    mean-item units), relative pressure, half-capacity-demand fraction,
    profit/load alignment, and feasible-item fraction. A resource with no
    feasible demand has zero node features and pressure.

    Pair channels: load cosine, profit-weighted cosine, centered correlation
    (negative values describe complementary demand), directed pressure
    difference, RMS load mismatch, and simultaneous half-capacity demand.
    The only item contraction is a [B,M,N] @ [B,N,M] matrix product; no new
    item-pair tensor is built. Independent resource-unit changes cancel.
    """
    feasible = state.feasible
    profits, weights, capacities, remaining = (x.to(torch.float64) for x in
        (state.profits, state.weights, state.capacities, state.remaining))
    valid = feasible.to(torch.float64)
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    # An exhausted resource permits only zero-weight candidates. Mask before
    # division and use an exact zero-safe denominator, preserving unit scaling.
    denominator = torch.where(remaining > 0, remaining, torch.ones_like(remaining))
    loads = torch.where(feasible[..., None], weights, 0.0) / denominator[:, None]
    total = loads.sum(1)
    demanding = total > 0
    mean = total / count
    maximum = loads.amax(1)
    minimum = loads.masked_fill(~feasible[..., None], torch.inf).amin(1)
    # Constant columns must have zero variance, including under roundoff.
    varies = maximum > minimum
    centered = (loads - mean[:, None]) * valid[..., None] * varies[:, None]
    variance = centered.square().sum(1) / count
    std = variance.clamp_min(0).sqrt()

    mass = profits * valid
    mass_total = mass.sum(-1, keepdim=True)
    mass = mass / torch.where(mass_total > 0, mass_total, torch.ones_like(mass_total))
    weighted_load = (loads * mass[..., None]).sum(1)
    pressure = total.clamp_max(config.feature_clip).log1p()
    slots = 1 / torch.where(mean > 0, mean, torch.ones_like(mean))
    slots = slots.clamp_max(config.feature_clip).log1p()
    alignment = weighted_load / torch.where(mean > 0, mean, torch.ones_like(mean))
    alignment = alignment.clamp_max(config.feature_clip).log1p()
    half = (loads >= 0.5).to(loads.dtype) * valid[..., None]
    slack = remaining / capacities
    resources = torch.stack((
        slack, 1 - slack, mean, std, maximum, weighted_load, pressure, slots,
        pressure - pressure.mean(-1, keepdim=True), half.sum(1) / count,
        alignment, valid.mean(-1, keepdim=True).expand_as(remaining)), -1)
    resources = resources * demanding[..., None]

    gram = loads.transpose(1, 2) @ loads
    squared_norm = loads.square().sum(1)

    def cosine(product, norms):
        scale = (norms[:, :, None] * norms[:, None, :]).clamp_min(0).sqrt()
        return product / torch.where(scale > 0, scale, torch.ones_like(scale))

    load_cosine = cosine(gram, squared_norm).clamp(0, 1)
    weighted_gram = (loads * mass[..., None]).transpose(1, 2) @ loads
    weighted_norm = (loads.square() * mass[..., None]).sum(1)
    profit_cosine = cosine(weighted_gram, weighted_norm).clamp(0, 1)
    centered_gram = centered.transpose(1, 2) @ centered
    correlation = cosine(centered_gram, centered.square().sum(1)).clamp(-1, 1)
    pressure_delta = pressure[:, :, None] - pressure[:, None, :]
    mismatch = ((squared_norm[:, :, None] + squared_norm[:, None, :] - 2 * gram)
                / count[..., None]).clamp_min(0).sqrt()
    shared_half = (half.transpose(1, 2) @ half) / count[..., None]
    relations = torch.stack((load_cosine, profit_cosine, correlation,
                             pressure_delta, mismatch, shared_half), -1)
    # Relations to unused resources carry no demand or attention keys.
    valid_pairs = demanding[:, :, None] & demanding[:, None, :]
    relations = relations * valid_pairs[..., None]
    return ResourceGeometry(resources.float(), relations.float(), pressure.float())


class ResourceRefinement(nn.Module):
    """Bounded residual candidate-detail and resource-competition summaries.

    Geometry-conditioned resource queries revisit the retained original item
    embeddings. Resource self-attention then passes relation bias AND values.
    All reductions respect feasibility; terminal/zero-demand rows produce the
    identity. Residual coordinates are bounded by actor_resource_gate_max.
    """
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.heads, self.head_dim = config.num_heads, d // config.num_heads
        self.gate_max = config.actor_resource_gate_max
        initial = config.actor_resource_gate_init / self.gate_max
        gate_logit = math.log(initial / (1 - initial))
        self.resource_gate = nn.Parameter(torch.full((d,), gate_logit))
        self.pooled_gate = nn.Parameter(torch.full((d,), gate_logit))
        self.geometry_input = nn.Sequential(nn.Linear(RESOURCE_GEOMETRY_DIM, d),
                                             nn.GELU(), nn.Linear(d, d))
        self.item_norm, self.resource_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.item_query, self.item_key, self.item_value = (nn.Linear(d, d) for _ in range(3))
        self.item_output = nn.Linear(d, d)
        self.relation_norm = nn.LayerNorm(d)
        self.resource_query, self.resource_key, self.resource_value = (nn.Linear(d, d) for _ in range(3))
        self.relation_bias = nn.Linear(RESOURCE_RELATION_DIM, self.heads, bias=False)
        self.relation_value = nn.Linear(RESOURCE_RELATION_DIM, d, bias=False)
        self.resource_output = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.pool_query, self.pool_key = nn.Linear(d, d), nn.Linear(d, d)
        self.pool_output = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))

    def _split(self, value):
        b, n, _ = value.shape
        return value.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, items, resources, pooled, geometry, feasible):
        b, m, d = resources.shape
        demanding = geometry.pressure > 0
        active = feasible.any(-1) & demanding.any(-1)
        source = self.resource_norm(resources) + self.geometry_input(geometry.resources)
        item_source = self.item_norm(items)
        query = self._split(self.item_query(source))
        key, value = (self._split(layer(item_source)) for layer in (self.item_key, self.item_value))
        item_logits = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        item_attention = masked_softmax(item_logits, feasible[:, None, None, :])
        item_summary = (item_attention @ value).transpose(1, 2).reshape(b, m, d)
        item_summary = self.item_output(item_summary) * feasible.any(-1)[:, None, None]

        refined_source = self.relation_norm(source + item_summary)
        query, key, value = (self._split(layer(refined_source)) for layer in
                             (self.resource_query, self.resource_key, self.resource_value))
        relation_logits = (query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
            + self.relation_bias(geometry.relations).permute(0, 3, 1, 2))
        attention = masked_softmax(relation_logits, demanding[:, None, None, :])
        # Contract relation features before projecting, avoiding [B,M,M,D].
        relation_mean = torch.einsum("bhij,bijf->bhif", attention, geometry.relations)
        relation_weight = self.relation_value.weight.view(self.heads, self.head_dim,
                                                           RESOURCE_RELATION_DIM)
        message = attention @ value + torch.einsum("bhif,hdf->bhid", relation_mean, relation_weight)
        message = message.transpose(1, 2).reshape(b, m, d)
        correction = self.resource_output(torch.cat((source, item_summary, message), -1)).tanh()
        resource_scale = self.gate_max * self.resource_gate.sigmoid()
        refined = resources + resource_scale * correction * demanding[..., None] * active[:, None, None]

        # A finite log1p(total demand) prior lets a cumulatively scarce resource
        # receive attention even when it is never any single item's argmax load.
        logits = ((self.pool_query(pooled)[:, None] * self.pool_key(refined)).sum(-1)
                   / math.sqrt(d) + geometry.pressure)
        weights = masked_softmax(logits, demanding)
        summary = (refined * weights[..., None]).sum(1)
        count = demanding.sum(-1, keepdim=True).clamp_min(1)
        mean = (refined * demanding[..., None]).sum(1) / count
        maximum = refined.masked_fill(~demanding[..., None], torch.finfo(refined.dtype).min).amax(1)
        maximum = torch.where(active[:, None], maximum, torch.zeros_like(maximum))
        pool_correction = self.pool_output(torch.cat((pooled, summary, mean, maximum - mean), -1)).tanh()
        pool_scale = self.gate_max * self.pooled_gate.sigmoid()
        return refined, pooled + pool_scale * pool_correction * active[:, None]
