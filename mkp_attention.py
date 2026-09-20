"""Actor-only candidate relations. No action scores, search, or return predictor.

All geometric features use the current state only. Pair compatibility describes
whether two items fit together, not whether an entire completion is optimal.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


PAIR_DIM = 7
POST_DIM = 4
RESOURCE_POST_DIM = 2


def masked_softmax(logits, valid):
    """A fully masked row returns zero, including in backward."""
    weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1)
    return weights * valid.to(weights.dtype)


def safe_key_padding_mask(valid):
    # MHA cannot softmax an all-masked row. Temporarily expose one key and
    # explicitly zero its entire output at the call site for empty rows.
    safe = valid.clone()
    safe[:, 0] |= ~valid.any(-1)
    return ~safe


@dataclass
class CandidateRelations:
    pairs: torch.Tensor       # [B,N,N,7], resource-permutation invariant
    valid: torch.Tensor       # [B,N,N], feasible distinct items only
    post: torch.Tensor        # [B,N,4], candidate-specific future availability
    resource_post: torch.Tensor  # [B,N,M,2], resource-specific lost opportunities


@torch.no_grad()
def candidate_relations(state, config):
    """Chunk float64 pair-capacity comparisons; store compact float32 features.

    The comparison is (remaining - w_i) - w_j, matching environment stepping.
    Excludes self, selected and currently infeasible items from every reduction.
    """
    feasible = state.feasible
    p, w, rem = state.profits, state.weights, state.remaining
    b, n = p.shape
    mass = p * feasible
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(config.eps)
    denom = rem[:, None].clamp_min(config.eps)
    baseline = (rem[:, None] - w) / denom
    bandwidth = config.actor_relation_bandwidth
    base_soft = torch.sigmoid(baseline / bandwidth)
    base_min = baseline.amin(-1)
    ids = torch.arange(n, device=p.device)
    pair_parts, valid_parts, post_parts, resource_parts = [], [], [], []
    for start in range(0, n, config.actor_relation_chunk_size):
        stop = min(n, start + config.actor_relation_chunk_size)
        valid = (feasible[:, start:stop, None] & feasible[:, None, :]
                 & (ids[start:stop, None] != ids[None, :]))
        margin = (rem[:, None, None] - w[:, start:stop, None]) - w[:, None]
        relative = margin / rem[:, None, None].clamp_min(config.eps)
        minimum = relative.amin(-1)
        blocked = (margin < 0).any(-1)
        soft = torch.sigmoid(minimum / bandwidth)
        erosion = (torch.sigmoid(base_min[:, None] / bandwidth) - soft).clamp_min(0)
        profit_delta = ((p[:, start:stop, None] - p[:, None])
                        / (p[:, start:stop, None] + p[:, None]).clamp_min(config.eps))
        load_distance = ((w[:, start:stop, None] - w[:, None]).abs()
                         / rem[:, None, None].clamp_min(config.eps)).mean(-1)
        other_mass = mass[:, None] * valid
        count = valid.sum(-1).clamp_min(1)
        post = torch.stack(((blocked * other_mass).sum(-1),
                            (soft * other_mass).sum(-1),
                            ((~blocked) & valid).sum(-1) / count,
                            (erosion * other_mass).sum(-1)), -1)
        # Preserve which resource caused blocking, rather than collapsing all
        # candidate consequences into a single scalar before the price head.
        resource_post = torch.stack((
            ((margin < 0) * other_mass[..., None]).sum(2),
            ((base_soft[:, None] - torch.sigmoid(relative / bandwidth)).clamp_min(0)
             * other_mass[..., None]).sum(2)), -1)
        pairs = torch.stack((profit_delta, load_distance.clamp_max(config.feature_clip),
                             minimum.clamp(-config.feature_clip, 1), blocked.to(p.dtype),
                             soft, erosion, mass[:, None].expand_as(minimum)), -1)
        pair_parts.append((pairs * valid[..., None]).float())
        valid_parts.append(valid)
        post_parts.append(post.float())
        resource_parts.append(resource_post.float())
    return CandidateRelations(torch.cat(pair_parts, 1), torch.cat(valid_parts, 1),
                              torch.cat(post_parts, 1), torch.cat(resource_parts, 1))


class PairRelationAttention(nn.Module):
    """Item-to-item attention with geometric relation bias AND value messages."""
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.heads, self.head_dim = config.num_heads, d // config.num_heads
        self.chunk_size = config.actor_relation_chunk_size
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.relation_bias = nn.Linear(PAIR_DIM, self.heads)
        self.relation_value = nn.Linear(PAIR_DIM, d, bias=False)
        self.output = nn.Linear(d, d)

    def forward(self, items, relations):
        b, n, d = items.shape
        split = lambda x: x.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = (split(layer(items)) for layer in (self.q, self.k, self.v))
        parts = []
        edge_weight = self.relation_value.weight.view(self.heads, self.head_dim, PAIR_DIM)
        for start in range(0, n, self.chunk_size):
            stop = min(n, start + self.chunk_size)
            edges = relations.pairs[:, start:stop]
            logits = (q[:, :, start:stop] @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
                      + self.relation_bias(edges).permute(0, 3, 1, 2))
            attention = masked_softmax(logits, relations.valid[:, None, start:stop])
            # Project after weighted aggregation: exactly equivalent to linear
            # edge Values, without a [B,N,N,D] tensor.
            edge_mean = torch.einsum("bhij,bijf->bhif", attention, edges)
            message = attention @ v + torch.einsum("bhif,hdf->bhid", edge_mean, edge_weight)
            message = self.output(message.transpose(1, 2).reshape(b, stop - start, d))
            parts.append(message * relations.valid[:, start:stop].any(-1, keepdim=True))
        return torch.cat(parts, 1)


class CriticalContrastPool(nn.Module):
    """Compare similar peers, then retain signed candidate differences in theta.

    These internal attention weights never become decoder/action logits.
    """
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.bandwidth = config.actor_similarity_bandwidth
        self.message = nn.Sequential(nn.Linear(2 * d + POST_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Linear(d, d)
        self.output = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, items, pooled, relations):
        distance = relations.pairs[..., 0].abs() + relations.pairs[..., 1]
        peer_weights = masked_softmax(-distance / self.bandwidth, relations.valid)
        peer_items = peer_weights @ items
        post_difference = relations.post - peer_weights @ relations.post
        messages = self.message(torch.cat((items, items - peer_items, post_difference), -1))
        # A bounded physical prior keeps rare, consequential alternatives from
        # disappearing in an N-way mean. The learned query chooses orientation.
        salience = post_difference.abs().mean(-1).clamp_min(1e-3)
        logits = ((messages * self.query(pooled)[:, None]).sum(-1) / math.sqrt(items.shape[-1])
                  + salience.log())
        weights = masked_softmax(logits, relations.valid.any(-1))
        return self.norm(pooled + self.output((weights[..., None] * messages).sum(1)))
