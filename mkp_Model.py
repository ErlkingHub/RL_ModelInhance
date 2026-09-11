"""Bipartite Transformer -> low-dimensional theta -> parameter-free weak decoder.

Ideas: variable/constraint graph (Gasse et al., 2019); attention for constructive
optimization (Kool et al., 2019). This is an MKP-specific implementation, not a
reproduction of either paper. The actor never produces learned per-item logits.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from mkp_config import ModelConfig
from mkp_env import MKPState


def state_features(state: MKPState, config: ModelConfig):
    p, w, c, rem = (x.float() for x in
                    (state.profits, state.weights, state.capacities, state.remaining))
    n = p.shape[1]
    eps, clip = config.eps, config.feature_clip
    selected, feasible = state.selected.float(), state.feasible.float()
    initial_cost = w / c[:, None, :].clamp_min(eps) * n
    current_cost = w / rem[:, None, :].clamp_min(eps)
    excess = ((w - rem[:, None, :]) / c[:, None, :].clamp_min(eps))
    selected_fraction = selected.mean(-1)
    feasible_fraction = feasible.mean(-1)
    log_feature = lambda x: x.clamp(0, clip).log1p()
    item_features = torch.stack((
        p / p.mean(-1, keepdim=True).clamp_min(eps),
        log_feature(initial_cost.mean(-1)), log_feature(initial_cost.amax(-1)),
        selected, feasible, log_feature(excess.clamp_min(0).amax(-1) * n),
        log_feature(current_cost.amax(-1)), feasible_fraction[:, None].expand_as(p)), dim=-1)
    unselected_demand = (w * (1 - selected[:, :, None])).sum(1)
    feasible_demand = (w * feasible[:, :, None]).sum(1)
    slack = rem / c.clamp_min(eps)
    resource_features = torch.stack((
        slack, 1 - slack, log_feature(unselected_demand / rem.clamp_min(eps)),
        log_feature(feasible_demand / rem.clamp_min(eps)),
        selected_fraction[:, None].expand_as(c), feasible_fraction[:, None].expand_as(c)), dim=-1)
    edge_features = torch.stack((log_feature(initial_cost), log_feature(current_cost),
                                 excess.clamp(-1, clip)), dim=-1)
    global_features = torch.stack((
        (p * selected).sum(-1) / p.sum(-1).clamp_min(eps), selected_fraction,
        feasible_fraction, slack.mean(-1), slack.amin(-1), slack.amax(-1)), dim=-1)
    return item_features, resource_features, edge_features, global_features


class EdgeCrossAttention(nn.Module):
    """Multihead cross-attention with resource-consumption edge biases, no masks."""
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.heads = config.num_heads
        self.head_dim = d // self.heads
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.edge_bias = nn.Sequential(nn.Linear(3, d // 2), nn.GELU(), nn.Linear(d // 2, self.heads))
        self.output = nn.Linear(d, d)

    def forward(self, queries, context, edges):
        b, nq, d = queries.shape
        nk = context.shape[1]
        q = self.q(queries).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        bias = self.edge_bias(edges).permute(0, 3, 1, 2)
        attention = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + bias).softmax(-1)
        result = (attention @ v).transpose(1, 2).reshape(b, nq, d)
        return self.output(result)


class BipartiteBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.embed_dim, config.num_heads
        self.item_cross, self.resource_cross = EdgeCrossAttention(config), EdgeCrossAttention(config)
        self.item_self = nn.MultiheadAttention(d, h, dropout=0.0, batch_first=True)
        self.resource_self = nn.MultiheadAttention(d, h, dropout=0.0, batch_first=True)
        self.item_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.resource_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        def ff():
            return nn.Sequential(nn.Linear(d, d * config.ff_multiplier), nn.GELU(),
                                 nn.Linear(d * config.ff_multiplier, d))
        self.item_ff, self.resource_ff = ff(), ff()

    def forward(self, items, resources, edges):
        i, r = self.item_norms[0](items), self.resource_norms[0](resources)
        items = items + self.item_cross(i, r, edges)
        resources = resources + self.resource_cross(r, i, edges.transpose(1, 2))
        i, r = self.item_norms[1](items), self.resource_norms[1](resources)
        # Item attention can compare profit/load tradeoffs; resource attention compares bottlenecks.
        items = items + self.item_self(i, i, i, need_weights=False)[0]
        resources = resources + self.resource_self(r, r, r, need_weights=False)[0]
        return (items + self.item_ff(self.item_norms[2](items)),
                resources + self.resource_ff(self.resource_norms[2](resources)))


class BipartiteEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.embed_dim
        self.item_input, self.resource_input = nn.Linear(8, d), nn.Linear(6, d)
        self.blocks = nn.ModuleList([BipartiteBlock(config) for _ in range(config.num_layers)])
        self.item_norm, self.resource_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(2 * d + 6, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, state):
        item_f, resource_f, edges, global_f = state_features(state, self.config)
        items, resources = self.item_input(item_f), self.resource_input(resource_f)
        for block in self.blocks:
            items, resources = block(items, resources, edges)
        items, resources = self.item_norm(items), self.resource_norm(resources)
        pooled = self.pool(torch.cat((items.mean(1), resources.mean(1), global_f), dim=-1))
        return items, resources, pooled


@dataclass
class StrategyTheta:
    resource_prices: torch.Tensor  # [B, M], strictly positive, mean 1
    profit_exponent: torch.Tensor # [B]
    scarcity_exponent: torch.Tensor
    bottleneck_weight: torch.Tensor
    temperature: torch.Tensor

    def detach(self):
        return StrategyTheta(*(x.detach() for x in (self.resource_prices, self.profit_exponent,
                             self.scarcity_exponent, self.bottleneck_weight, self.temperature)))


class HighLevelController(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = BipartiteEncoder(config)
        self.price_head = nn.Linear(2 * config.embed_dim, 1)
        self.strategy_head = nn.Linear(config.embed_dim, 4)
        # Auxiliary state supervision only. Its item outputs never enter decoder scores.
        self.feasibility_head = nn.Linear(config.embed_dim, 1)
        nn.init.zeros_(self.price_head.weight)
        nn.init.zeros_(self.price_head.bias)
        nn.init.zeros_(self.strategy_head.weight)
        defaults = (1.0, 1.0, 0.1, 0.2)
        bounds = ((config.profit_exponent_min, config.profit_exponent_max),
                  (0, config.scarcity_exponent_max), (0, config.bottleneck_weight_max),
                  (config.temperature_min, config.temperature_max))
        with torch.no_grad():
            for j, (v, (lo, hi)) in enumerate(zip(defaults, bounds)):
                fraction = min(0.999, max(0.001, (v - lo) / (hi - lo))) if hi > lo else 0.5
                self.strategy_head.bias[j] = math.log(fraction / (1 - fraction))

    def forward(self, state):
        items, resources, pooled = self.encoder(state)
        c = self.config
        m = resources.shape[1]
        price_logits = self.price_head(torch.cat((resources, pooled[:, None, :].expand_as(resources)), -1)).squeeze(-1)
        prices = c.price_floor + (1 - c.price_floor) * m * price_logits.softmax(-1)
        controls = self.strategy_head(pooled).sigmoid()
        theta = StrategyTheta(
            prices,
            c.profit_exponent_min + (c.profit_exponent_max - c.profit_exponent_min) * controls[:, 0],
            c.scarcity_exponent_max * controls[:, 1], c.bottleneck_weight_max * controls[:, 2],
            c.temperature_min + (c.temperature_max - c.temperature_min) * controls[:, 3])
        return theta, self.feasibility_head(items).squeeze(-1)


class CandidatePolicy:
    """Ragged categorical distribution. Only actual candidates are stored.

    No dense logits with -inf and no invalid-action rejection loop. Segment
    reductions support different candidate counts and already terminal rows.
    """
    def __init__(self, owners, items, logits, batch_size):
        self.owners, self.items, self.logits = owners, items, logits
        self.batch_size = batch_size
        maxima = logits.new_full((batch_size,), -torch.inf)
        maxima = maxima.scatter_reduce(0, owners, logits, reduce="amax", include_self=True)
        centered = logits - maxima[owners]
        total = logits.new_zeros(batch_size).scatter_add(0, owners, centered.exp())
        self.log_probs = centered - total[owners].log()

    def choose(self, stochastic=False):
        values = self.log_probs
        if stochastic:
            uniform = torch.rand_like(values).clamp(1e-7, 1 - 1e-7)
            values = values - (-uniform.log()).log()
        maxima = values.new_full((self.batch_size,), -torch.inf)
        maxima.scatter_reduce_(0, self.owners, values, reduce="amax", include_self=True)
        # Exact ties use the smallest original item index; empty rows receive -1.
        sentinel = torch.iinfo(torch.long).max
        chosen = torch.where(values == maxima[self.owners], self.items, sentinel)
        result = self.items.new_full((self.batch_size,), sentinel)
        result.scatter_reduce_(0, self.owners, chosen, reduce="amin", include_self=True)
        return torch.where(result == sentinel, -1, result)

    def log_prob(self, actions):
        selected = self.items == actions[self.owners]
        return self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners[selected], self.log_probs[selected])

    def entropy(self):
        return self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners, -self.log_probs.exp() * self.log_probs)

    def normalized_entropy(self):
        count = self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners, torch.ones_like(self.logits))
        return self.entropy() / count.clamp_min(2).log()


class WeakDecoder:
    """No nn.Module, trainable weight, pointer, search, local repair or solver.

    q_ij = N * w_ij / C_j ; s_j = C_j / max(c_j, eps)
    cost_i = mean_j(price_j * q_ij * s_j ** beta)
    score_i = alpha * log(p_i / mean(p)) - log(cost_i)
              - kappa * max_j(w_ij / max(c_j, eps))
    pi(i|s) = softmax(score_i / temperature) over feasible candidates.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    def fixed_theta(self, state):
        b, m = state.capacities.shape
        device = state.profits.device
        return StrategyTheta(torch.ones(b, m, device=device), torch.ones(b, device=device),
                             torch.ones(b, device=device), torch.full((b,), 0.1, device=device),
                             torch.full((b,), 0.2, device=device))

    def __call__(self, state, theta):
        owners, items = state.feasible.nonzero(as_tuple=True)
        eps = self.config.eps
        p, w, c, rem = (x.float() for x in
                        (state.profits, state.weights, state.capacities, state.remaining))
        # Calculate only feasible-item scores; infeasible nodes still entered the encoder.
        candidate_w = w[owners, items]
        scale = state.profits.shape[1]
        log_scarcity = (c[owners].clamp_min(eps).log() - rem[owners].clamp_min(eps).log())
        # Log-space cost avoids overflow when a resource is exhausted (zero loads are legal).
        zero_cost = (candidate_w == 0).all(-1)
        log_load = (scale * candidate_w / c[owners]).clamp_min(eps).log()
        # -inf here represents a zero summand in a resource cost, never an item logit.
        log_load = torch.where(candidate_w > 0, log_load, -torch.inf)
        log_components = (theta.resource_prices[owners].log() + log_load
                          + theta.scarcity_exponent[owners, None] * log_scarcity)
        log_components = torch.where(zero_cost[:, None], torch.zeros_like(log_components), log_components)
        log_cost = torch.logsumexp(log_components, -1) - math.log(c.shape[-1])
        log_cost = torch.where(zero_cost, log_cost.new_full(log_cost.shape, math.log(eps)), log_cost)
        profit = p[owners, items] / p.mean(-1)[owners].clamp_min(eps)
        bottleneck = (candidate_w / rem[owners].clamp_min(eps)).amax(-1)
        scores = theta.profit_exponent[owners] * profit.clamp_min(eps).log() - log_cost
        scores = scores - theta.bottleneck_weight[owners] * bottleneck
        return CandidatePolicy(owners, items, scores / theta.temperature[owners], p.shape[0])


class MKPModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.actor = HighLevelController(self.config)
        # Independent critic: actor changes cannot silently change the baseline representation.
        self.critic_encoder = BipartiteEncoder(self.config)
        self.value_head = nn.Linear(self.config.embed_dim, 1)
        self.decoder = WeakDecoder(self.config)

    def policy(self, state):
        theta, feasibility_logits = self.actor(state)
        return self.decoder(state, theta), theta, feasibility_logits

    def value(self, state):
        return self.value_head(self.critic_encoder(state)[2]).squeeze(-1)

    def forward(self, state):
        policy, theta, feasibility_logits = self.policy(state)
        return policy, self.value(state), theta, feasibility_logits
