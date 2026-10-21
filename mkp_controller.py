"""Actor strategy readout. Its tokens control theta, never item action scores."""
import math

import torch
from torch import nn

from mkp_attention import safe_key_padding_mask


class TypeBalancedAttention(nn.Module):
    """Normalize candidate and resource attention separately before gated fusion.

    A resource competes with M resources rather than N+M keys. This keeps the
    meaning of the two summaries stable when transferring to larger N.
    """
    def __init__(self, config):
        super().__init__()
        d, h = config.embed_dim, config.num_heads
        self.item_attention = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.resource_attention = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.mix = nn.Linear(3 * d, d)

    def forward(self, queries, items, resources, feasible):
        item_summary = self.item_attention(queries, items, items,
            key_padding_mask=safe_key_padding_mask(feasible), need_weights=False)[0]
        item_summary = item_summary * feasible.any(-1)[:, None, None]
        resource_summary = self.resource_attention(queries, resources, resources,
                                                   need_weights=False)[0]
        joined = torch.cat((queries, item_summary, resource_summary), -1)
        gate = self.mix(joined).sigmoid()
        return self.fusion(joined) + gate * item_summary + (1 - gate) * resource_summary


class StrategyReadout(nn.Module):
    """Nine roles: six scalar controls, resource prices, interactions, companions.

    Role attention supplies distinct information to each control and then lets
    roles coordinate. The shared graph representation is retained by a bounded
    residual, with no change in the theta dimensions or fixed score formula.
    """
    roles = 9

    def __init__(self, config):
        super().__init__()
        d, h = config.embed_dim, config.num_heads
        self.tokens = nn.Parameter(torch.randn(self.roles, d) / math.sqrt(d))
        self.type_balanced = config.actor_type_balanced_pooling
        self.attention = (TypeBalancedAttention(config) if self.type_balanced else
                          nn.MultiheadAttention(d, h, dropout=0., batch_first=True))
        self.norm = nn.LayerNorm(d)
        self.coordination = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.correction = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d),
                                         nn.GELU(), nn.Linear(2 * d, d))
        initial = config.actor_readout_initial_scale
        self.scale_logits = nn.Parameter(torch.full((self.roles, 1), math.log(initial / (1 - initial))))

    def forward(self, items, resources, pooled, feasible):
        query = pooled[:, None] + self.tokens[None]
        if self.type_balanced:
            summary = self.attention(query, items, resources, feasible)
        else:
            context = torch.cat((items, resources), 1)
            mask = torch.cat((~feasible, torch.zeros(resources.shape[:2], dtype=torch.bool,
                                                    device=resources.device)), 1)
            summary = self.attention(query, context, context, key_padding_mask=mask, need_weights=False)[0]
        roles = query + summary
        normalized = self.norm(roles)
        roles = roles + self.coordination(normalized, normalized, normalized, need_weights=False)[0]
        return pooled[:, None] + self.scale_logits.sigmoid()[None] * self.correction(roles).tanh()
