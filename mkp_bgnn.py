"""Item-constraint bipartite GNN. No Transformer or item-item attention.

Variable/constraint graph inspiration: Gasse et al. (2019), arXiv:1906.01629.
Our dynamic features and PPO task differ from their imitation-learning task.
"""
import torch
from torch import nn


def dual_pressure_proxy(state, eps):
    """Relative profit-weighted demand pressure, NOT a solved LP dual multiplier.

    u_j = sum_{unselected i}(p_i/mean(p)) * w_ij / max(remaining_j, eps)
    proxy_j = u_j / mean_j(u_j); exhausted dimensions stay finite.
    """
    p, w, rem = state.profits.float(), state.weights.float(), state.remaining.float()
    profit = p / p.mean(-1, keepdim=True).clamp_min(eps)
    demand = (w * profit[:, :, None] * (~state.selected)[:, :, None]).sum(1)
    pressure = demand / rem.clamp_min(eps)
    return pressure / pressure.mean(-1, keepdim=True).clamp_min(eps)


class BipartiteMessage(nn.Module):
    """Each edge carries sender, receiver and explicit consumption information."""
    def __init__(self, d):
        super().__init__()
        self.sender = nn.Linear(d, d, bias=False)
        self.receiver = nn.Linear(d, d, bias=False)
        self.edge = nn.Linear(3, d)
        self.message = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU())

    def forward(self, items, constraints, edges, to_constraints, edge_present=None):
        sender, receiver = (items, constraints) if to_constraints else (constraints, items)
        if to_constraints:
            hidden = self.sender(sender)[:, :, None] + self.receiver(receiver)[:, None, :]
        else:
            hidden = self.sender(sender)[:, None, :] + self.receiver(receiver)[:, :, None]
        messages = self.message(hidden + self.edge(edges))  # [B,N,M,D]
        # Zero consumption means no physical edge. This is graph adjacency, not an action mask.
        dimension = 1 if to_constraints else 2
        if edge_present is None:
            return messages.mean(dimension)
        degree = edge_present.sum(dimension).clamp_min(1)[:, :, None]
        return (messages * edge_present[:, :, :, None]).sum(dimension) / degree


class BGNNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.item_to_constraint = BipartiteMessage(d)
        self.constraint_to_item = BipartiteMessage(d)
        self.item_norm, self.constraint_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        def update():
            return nn.Sequential(nn.Linear(2 * d, d * config.ff_multiplier), nn.GELU(),
                                 nn.Linear(d * config.ff_multiplier, d), nn.LayerNorm(d))
        self.constraint_update, self.item_update = update(), update()

    def forward(self, items, constraints, edges, edge_present):
        i, c = self.item_norm(items), self.constraint_norm(constraints)
        message = self.item_to_constraint(i, c, edges, True, edge_present)
        constraints = constraints + self.constraint_update(torch.cat((c, message), -1))
        # The second half-step sees the newly updated constraint nodes.
        c = self.constraint_norm(constraints)
        message = self.constraint_to_item(i, c, edges, False, edge_present)
        items = items + self.item_update(torch.cat((i, message), -1))
        return items, constraints


class BGNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.embed_dim
        self.item_input, self.constraint_input = nn.Linear(8, d), nn.Linear(7, d)
        self.blocks = nn.ModuleList([BGNNBlock(config) for _ in range(config.num_layers)])
        self.item_norm, self.constraint_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(2 * d + 6, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, state):
        # Lazy import avoids a module cycle while sharing the tested feature contract.
        from mkp_Model import state_features
        item_f, constraint_f, edges, global_f = state_features(state, self.config)
        proxy = dual_pressure_proxy(state, self.config.eps)
        constraint_f = torch.cat((constraint_f, proxy.clamp_max(self.config.feature_clip).log1p()[:, :, None]), -1)
        items, constraints = self.item_input(item_f), self.constraint_input(constraint_f)
        for block in self.blocks:
            items, constraints = block(items, constraints, edges, state.weights > 0)
        items, constraints = self.item_norm(items), self.constraint_norm(constraints)
        pooled = self.pool(torch.cat((items.mean(1), constraints.mean(1), global_f), -1))
        return items, constraints, pooled
