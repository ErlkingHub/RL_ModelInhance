"""Feasible resource geometry and residual high-level control paths.

Every physical feature is computed from the current state. Learned outputs are
resource embeddings or a state summary; none are item/action scores.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from mkp_attention import masked_softmax


RESOURCE_NODE_DIM = 10
RESOURCE_PAIR_DIM = 5


@dataclass
class ResourceGeometry:
    nodes: torch.Tensor  # [B,M,10], feasible resource statistics
    pairs: torch.Tensor  # [B,M,M,5], symmetric co-demand geometry


@torch.no_grad()
def resource_geometry(state, config):
    """Unit-invariant resource statistics, with float64 physical reductions.

    Node channels are mean/max/std load, profit-weighted load, bounded demand
    pressure, remaining mean-demand slots, count/profit bottleneck attribution,
    and relative demand/profit pressure. Pair channels are load/profit cosine,
    load correlation, pressure imbalance, and RMS load mismatch. Empty feasible
    sets and unused resources have zero node features. A zero resource pair is
    still allowed to differ in pressure from an active resource.
    """
    from mkp_performance import cached_features

    cached = cached_features(state, config)
    if cached is not None and "resource_node_geometry" in cached:
        return ResourceGeometry(cached["resource_node_geometry"], cached["resource_pair_geometry"])
    feasible = state.feasible
    weights, remaining, profits = (x.to(torch.float64) for x in
                                   (state.weights, state.remaining, state.profits))
    mask = feasible.to(torch.float64)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    # Mask before dividing: exhausted resources cannot produce inf * 0.
    denominator = torch.where(remaining > 0, remaining, torch.ones_like(remaining))
    loads = weights * mask[..., None] / denominator[:, None]
    total = loads.sum(1)
    active = total > 0
    mean = total / count
    maximum = loads.amax(1)
    minimum = loads.masked_fill(~feasible[..., None], torch.inf).amin(1)
    varies = maximum > minimum
    # Explicitly suppress roundoff variance for constant resource columns.
    centered = (loads - mean[:, None]) * mask[..., None] * varies[:, None]
    std = (centered.square().sum(1) / count).sqrt()
    profit_mass = profits * mask
    mass_total = profit_mass.sum(-1, keepdim=True)
    profit_mass = profit_mass / torch.where(mass_total > 0, mass_total, torch.ones_like(mass_total))
    weighted = (loads * profit_mass[..., None]).sum(1)
    pressure = total / (1 + total)
    # r / (r + mean(w)) = slots / (1 + slots), without a large r / mean(w).
    slots = 1 / (1 + mean)
    critical = (loads == loads.amax(-1, keepdim=True)) & (loads > 0) & feasible[..., None]
    attribution = critical.to(loads.dtype) / critical.sum(-1, keepdim=True).clamp_min(1)
    count_attribution = attribution.sum(1) / count
    profit_attribution = (attribution * profit_mass[..., None]).sum(1)

    def relative(values):
        scale = values + values.mean(-1, keepdim=True)
        return values / torch.where(scale > 0, scale, torch.ones_like(scale))

    node = torch.stack((mean, maximum, std, weighted, pressure, slots,
                        count_attribution, profit_attribution,
                        relative(total), relative(weighted)), -1)
    node = node * active[..., None]
    gram = loads.transpose(1, 2) @ loads
    squared_norm = loads.square().sum(1)

    def cosine(product, norms):
        scale = (norms[:, :, None] * norms[:, None, :]).clamp_min(0).sqrt()
        return product / torch.where(scale > 0, scale, torch.ones_like(scale))

    load_cosine = cosine(gram, squared_norm).clamp(0, 1)
    weighted_gram = (loads * profit_mass[..., None]).transpose(1, 2) @ loads
    weighted_norm = (loads.square() * profit_mass[..., None]).sum(1)
    profit_cosine = cosine(weighted_gram, weighted_norm).clamp(0, 1)
    # Center only feasible observations; inactive/selected rows are absent.
    centered_gram = centered.transpose(1, 2) @ centered
    centered_norm = centered.square().sum(1)
    correlation = cosine(centered_gram, centered_norm).clamp(-1, 1)
    imbalance = (pressure[:, :, None] - pressure[:, None, :]).abs()
    mismatch = ((squared_norm[:, :, None] + squared_norm[:, None, :] - 2 * gram)
                / count[..., None]).clamp_min(0).sqrt()
    pair = torch.stack((load_cosine, profit_cosine, correlation, imbalance, mismatch), -1)
    return ResourceGeometry(node.float(), pair.float())


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ResourceRelationAttention(nn.Module):
    """Resource attention with physical co-demand bias and value messages."""
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.heads, self.head_dim = config.num_heads, d // config.num_heads
        self.norm = nn.LayerNorm(d)
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.relation_bias = nn.Linear(RESOURCE_PAIR_DIM, self.heads, bias=False)
        self.relation_value = nn.Linear(RESOURCE_PAIR_DIM, d, bias=False)
        self.output = zero_output(nn.Linear(d, d))

    def forward(self, resources, geometry):
        b, m, d = resources.shape
        source = self.norm(resources)
        split = lambda x: x.view(b, m, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = (split(layer(source)) for layer in (self.q, self.k, self.v))
        logits = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
                  + self.relation_bias(geometry.pairs).permute(0, 3, 1, 2))
        attention = logits.softmax(-1)
        relation_mean = torch.einsum("bhij,bijf->bhif", attention, geometry.pairs)
        relation_weight = self.relation_value.weight.view(self.heads, self.head_dim, RESOURCE_PAIR_DIM)
        message = attention @ v + torch.einsum("bhif,hdf->bhid", relation_mean, relation_weight)
        return self.output(message.transpose(1, 2).reshape(b, m, d))


class ResourceGeometryEncoder(nn.Module):
    """Zero-output residuals preserve an existing encoder on warm start."""
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.node_adapter = nn.Sequential(nn.Linear(RESOURCE_NODE_DIM, d), nn.GELU(),
                                          zero_output(nn.Linear(d, d)))
        self.relation_layers = nn.ModuleList(ResourceRelationAttention(config)
                                             for _ in range(config.num_layers))

    def forward(self, resources, geometry, layer_index=None):
        message = (self.node_adapter(geometry.nodes) if layer_index is None else
                   self.relation_layers[layer_index](resources, geometry))
        return resources + message


class ResourceControllerReadout(nn.Module):
    """Retain scarce-resource context and dispersion before theta prediction."""
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.node_input = nn.Sequential(nn.Linear(RESOURCE_NODE_DIM, d), nn.GELU())
        self.norm = nn.LayerNorm(d)
        self.query, self.key = nn.Linear(d, d), nn.Linear(d, d)
        self.pooled_output = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(),
                                           zero_output(nn.Linear(d, d)))
        self.resource_output = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(),
                                             zero_output(nn.Linear(d, d)))

    def forward(self, resources, pooled, geometry):
        nodes = self.node_input(geometry.nodes)
        source = self.norm(resources + nodes)
        active = geometry.nodes[..., 0] > 0
        count = active.sum(-1, keepdim=True).clamp_min(1)
        mean = (source * active[..., None]).sum(1) / count
        maximum = source.masked_fill(~active[..., None], torch.finfo(source.dtype).min).amax(1)
        maximum = torch.where(active.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        variance = ((source - mean[:, None]).square() * active[..., None]).sum(1) / count
        # Positive floor makes derivatives finite at equal resource embeddings.
        dispersion = (variance + 1e-8).sqrt() - 1e-4
        salience = (geometry.nodes[..., 4] + geometry.nodes[..., 7]).clamp_min(1e-4)
        logits = ((self.key(source) * self.query(pooled)[:, None]).sum(-1)
                  / math.sqrt(resources.shape[-1]) + salience.log())
        attention = masked_softmax(logits, active)
        summary = (source * attention[..., None]).sum(1)
        state_delta = self.pooled_output(torch.cat((pooled, summary, maximum - mean, dispersion), -1))
        context = summary[:, None].expand_as(source)
        resource_delta = self.resource_output(torch.cat((source, context, source - context, nodes), -1))
        return resources + resource_delta * active[..., None], pooled + state_delta
