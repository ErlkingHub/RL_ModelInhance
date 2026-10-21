"""Resource-centric Actor representation. No action scores or learned decoder.

Gather candidates into constraints, communicate between constraints, then
broadcast the updated constraint messages back to candidates in the same round.
Actor attention storage is O(B H (N M + M M)); no item-pair attention is built.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from mkp_attention import masked_softmax


@dataclass
class RoutingGeometry:
    items: torch.Tensor       # [B,N,8]
    resources: torch.Tensor   # [B,M,12]
    edges: torch.Tensor       # [B,N,M,4]
    pairs: torch.Tensor       # [B,M,M,5]
    global_features: torch.Tensor  # [B,6]


@torch.no_grad()
def routing_geometry(state, config):
    """Feasible-only demand/profit/competition statistics in dimensionless units.

    Geometry is not an optimal shadow price. Float64 accounting is retained
    until the final neural inputs are stored as float32.
    """
    from mkp_performance import cached_features
    cached = cached_features(state, config)
    names = ("items", "resources", "edges", "pairs", "global_features")
    if cached is not None and "routing_items" in cached:
        return RoutingGeometry(*(cached["routing_" + key] for key in names))
    valid = state.feasible
    mask = valid[..., None]
    p, w, rem, cap = state.profits, state.weights, state.remaining, state.capacities
    eps = config.eps
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    profit = torch.where(valid, p, 0.)
    profit_mass = profit / profit.sum(-1, keepdim=True).clamp_min(eps)
    mean_profit = profit.sum(-1, keepdim=True) / count
    u = torch.where(mask, w / rem[:, None].clamp_min(eps), 0.).clamp(0, 1)
    mean = u.sum(1) / count
    centered = torch.where(mask, u - mean[:, None], 0.)
    std = (centered.square().sum(1) / count).clamp_min(0).sqrt()
    weighted = (u * profit_mass[..., None]).sum(1)
    demand = u.sum(1)
    pressure = demand / (1 + demand)
    # Split ties equally across resources rather than assigning a resource index.
    tied = (u == u.amax(-1, keepdim=True)) & mask & (u > 0)
    attribution = tied / tied.sum(-1, keepdim=True).clamp_min(1)
    bottleneck_count = attribution.sum(1) / count
    bottleneck_profit = (attribution * profit_mass[..., None]).sum(1)
    slack = rem / cap.clamp_min(eps)
    resource = torch.stack((mean, u.amax(1), std, weighted, pressure,
                            1 / (1 + mean), bottleneck_count, bottleneck_profit,
                            slack, weighted / weighted.mean(-1, keepdim=True).clamp_min(eps),
                            (u.square() * profit_mass[..., None]).sum(1),
                            count.log1p().expand_as(rem)), -1)
    resource = resource * valid.any(-1)[:, None, None]
    gram = u.transpose(1, 2) @ u
    norm = u.square().sum(1).sqrt()
    cosine = gram / (norm[:, :, None] * norm[:, None]).clamp_min(eps)
    centered_gram = centered.transpose(1, 2) @ centered
    cnorm = centered.square().sum(1).sqrt()
    correlation = centered_gram / (cnorm[:, :, None] * cnorm[:, None]).clamp_min(eps)
    weighted_u = u * profit_mass.sqrt()[..., None]
    wnorm = weighted_u.square().sum(1).sqrt()
    profit_cosine = ((weighted_u.transpose(1, 2) @ weighted_u)
                     / (wnorm[:, :, None] * wnorm[:, None]).clamp_min(eps))
    pairs = torch.stack((cosine, correlation, profit_cosine,
                         (pressure[:, :, None] - pressure[:, None]).abs(),
                         ((u.square().sum(1)[:, :, None] + u.square().sum(1)[:, None]
                           - 2 * gram).clamp_min(0) / count[..., None]).sqrt()), -1)
    normalized_profit = profit / mean_profit.clamp_min(eps)
    relative_load = u / mean[:, None].clamp_min(eps)
    edges = torch.stack((u, relative_load.clamp_max(config.feature_clip).log1p(),
                         (u * normalized_profit[..., None]).clamp_max(config.feature_clip).log1p(),
                         1 - u), -1) * mask[..., None]
    items = torch.stack((normalized_profit.clamp_max(config.feature_clip).log1p(),
                         u.mean(-1), u.amax(-1), u.std(-1, unbiased=False),
                         relative_load.mean(-1).clamp_max(config.feature_clip).log1p(),
                         (u * bottleneck_profit[:, None]).sum(-1),
                         (1 - u).amin(-1), valid.to(p.dtype)), -1) * valid[..., None]
    global_f = torch.stack((valid.to(p.dtype).mean(-1), state.selected.to(p.dtype).mean(-1),
                            slack.mean(-1), slack.amin(-1), slack.amax(-1),
                            count.squeeze(-1).log1p()), -1)
    return RoutingGeometry(*(x.float() for x in (items, resource, edges, pairs, global_f)))


class GeometryAttention(nn.Module):
    """Edge bias and edge Values, aggregated before projection to avoid BNM D."""
    def __init__(self, config, edge_dim):
        super().__init__()
        d = config.embed_dim
        self.heads, self.width = config.num_heads, d // config.num_heads
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.bias = nn.Sequential(nn.Linear(edge_dim, d // 2), nn.GELU(),
                                  nn.Linear(d // 2, self.heads))
        self.edge_value = nn.Linear(edge_dim, d, bias=False)
        self.output = nn.Linear(d, d)

    def forward(self, queries, context, edges, valid=None):
        b, nq, d = queries.shape
        def split(x):
            return x.reshape(b, -1, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(queries)), split(self.k(context)), split(self.v(context))
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.width)
        logits = logits + self.bias(edges).permute(0, 3, 1, 2)
        attention = (logits.softmax(-1) if valid is None else
                     masked_softmax(logits, valid[:, None, None]))
        edge_mean = torch.einsum("bhij,bijf->bhif", attention, edges)
        edge_weight = self.edge_value.weight.reshape(self.heads, self.width, -1)
        result = attention @ v + torch.einsum("bhif,hdf->bhid", edge_mean, edge_weight)
        result = self.output(result.transpose(1, 2).reshape(b, nq, d))
        if valid is not None:
            result = result * valid.any(-1)[:, None, None]
        return result


class ResourceRoutingBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.gather = GeometryAttention(config, 4)
        self.compete = GeometryAttention(config, 5)
        self.broadcast = GeometryAttention(config, 4)
        self.item_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        self.resource_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        def ff():
            return nn.Sequential(nn.Linear(d, d * config.ff_multiplier), nn.GELU(),
                                 nn.Linear(d * config.ff_multiplier, d))
        self.item_ff, self.resource_ff = ff(), ff()

    def forward(self, items, resources, geometry, feasible):
        # The broadcast sees this round's candidate demand and resource competition.
        resources = resources + self.gather(self.resource_norms[0](resources),
            self.item_norms[0](items), geometry.edges.transpose(1, 2), feasible)
        r = self.resource_norms[1](resources)
        resources = resources + self.compete(r, r, geometry.pairs)
        resources = resources + self.resource_ff(self.resource_norms[2](resources))
        items = items + self.broadcast(self.item_norms[1](items),
            self.resource_norms[3](resources), geometry.edges)
        items = items + self.item_ff(self.item_norms[2](items))
        return items * feasible[..., None], resources


class ResourceRoutingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.embed_dim
        self.item_input, self.resource_input = nn.Linear(8, d), nn.Linear(12, d)
        self.blocks = nn.ModuleList(ResourceRoutingBlock(config) for _ in range(config.num_layers))
        self.item_norm, self.resource_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.resource_query = nn.Linear(d, 1, bias=False)
        self.pool = nn.Sequential(nn.Linear(4 * d + 6, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, state):
        geometry = routing_geometry(state, self.config)
        feasible = state.feasible
        items, resources = self.item_input(geometry.items), self.resource_input(geometry.resources)
        for block in self.blocks:
            items, resources = block(items, resources, geometry, feasible)
        items, resources = self.item_norm(items), self.resource_norm(resources)
        item_mean = (items * feasible[..., None]).sum(1) / feasible.sum(-1, keepdim=True).clamp_min(1)
        # Resource attention plus extremes prevent a single bottleneck vanishing in a mean.
        pressure = geometry.resources[..., 7]
        logits = self.resource_query(resources).squeeze(-1) + pressure.clamp_min(1e-8).log()
        resource_pool = (logits.softmax(-1)[..., None] * resources).sum(1)
        pooled = self.pool(torch.cat((item_mean, resource_pool, resources.amax(1),
                                      resources.std(1, unbiased=False), geometry.global_features), -1))
        return items, resources, pooled


class StrategyRoleReadout(nn.Module):
    """Nine fixed roles for six scalars, prices, interactions and companion mix.

    Item and resource populations normalize separately so N cannot dilute the
    resource population solely by having more tokens. Outputs are state-level.
    """
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.queries = nn.Parameter(torch.randn(9, d) / math.sqrt(d))
        self.item_k, self.item_v = nn.Linear(d, d), nn.Linear(d, d)
        self.resource_k, self.resource_v = nn.Linear(d, d), nn.Linear(d, d)
        self.gate = nn.Linear(3 * d, d)
        self.output = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, items, resources, pooled, feasible):
        query = pooled[:, None] + self.queries[None]
        scale = math.sqrt(query.shape[-1])
        iw = masked_softmax(query @ self.item_k(items).transpose(1, 2) / scale,
                            feasible[:, None])
        rw = (query @ self.resource_k(resources).transpose(1, 2) / scale).softmax(-1)
        i, r = iw @ self.item_v(items), rw @ self.resource_v(resources)
        gate = self.gate(torch.cat((query, i, r), -1)).sigmoid()
        return self.norm(query + self.output(torch.cat((query, gate * i, (1 - gate) * r), -1)))
