"""Training-only proposals in the existing StrategyTheta domain.

No fitted proposal model, oracle, per-item parameter or theta regression target.
Temperature is held fixed because it cannot change a greedy decoder action.
"""
from dataclasses import dataclass, fields
import torch
from mkp_Model import StrategyTheta
from mkp_performance import prepare_state


@torch.no_grad()
def propose_thetas(theta, model_config, count, local_scale, generator):
    """Current theta, then alternating local logit perturbations / broad draws.

    Normalized prices and coefficient budgets use simplexes. Upper-triangle
    interactions are sampled once per unordered resource pair and mirrored.
    Proposal distributions are exchangeable over resources and independent of N.
    """
    if count < 1 or local_scale < 0:
        raise ValueError("Need positive candidate count and nonnegative perturbation scale")
    c = model_config
    base = theta.detach()
    proposals = [base]
    b, m = base.resource_prices.shape
    upper = torch.triu_indices(m, m, device=base.resource_prices.device)

    def uniform_like(value):
        return torch.rand(value.shape, dtype=value.dtype, device=value.device, generator=generator).clamp(1e-6, 1 - 1e-6)

    def simplex(value, local):
        if local:
            logits = value.clamp_min(1e-8).log()
            noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
            return (logits + local_scale * noise).softmax(-1)
        # Independent exponential draws give a uniform simplex distribution.
        mass = -uniform_like(value).log()
        return mass / mass.sum(-1, keepdim=True)

    def bounded(value, lo, hi, local):
        if hi == lo:
            return torch.full_like(value, lo)
        if local:
            fraction = ((value - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
            noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
            fraction = (torch.logit(fraction) + local_scale * noise).sigmoid()
        else:
            fraction = uniform_like(value)
        return lo + (hi - lo) * fraction

    def budgeted(value, maximum, local):
        mass = bounded(value.sum(-1), 0., maximum, local)
        return simplex(value, local) * mass[..., None]

    for index in range(1, count):
        local = index % 2 == 1
        prices = c.price_floor + (1 - c.price_floor) * m * simplex(
            (base.resource_prices - c.price_floor) / ((1 - c.price_floor) * m), local)
        entries = budgeted(base.resource_interactions[:, upper[0], upper[1]], c.interaction_weight_max, local)
        interactions = entries.new_zeros(b, m, m)
        interactions[:, upper[0], upper[1]] = entries
        interactions[:, upper[1], upper[0]] = entries
        proposals.append(StrategyTheta(
            prices,
            bounded(base.profit_exponent, c.profit_exponent_min, c.profit_exponent_max, local),
            bounded(base.scarcity_exponent, 0., c.scarcity_exponent_max, local),
            bounded(base.bottleneck_weight, 0., c.bottleneck_weight_max, local),
            base.temperature.clone(),
            bounded(base.opportunity_weight, 0., c.opportunity_weight_max, local),
            bounded(base.reserve_weight, 0., c.reserve_weight_max, local),
            interactions, budgeted(base.companion_weights, c.companion_weight_max, local)))
    return proposals


@dataclass
class InducedActions:
    owners: torch.Tensor  # [U], grouped by state; current action first
    items: torch.Tensor   # [U], unique within each state
    witnesses: torch.Tensor  # [U], index into the proposed theta list
    counts: torch.Tensor  # [B]


@torch.no_grad()
def deduplicate_actions(actions, max_actions, generator):
    """Keep the current action and uniformly subsample distinct alternatives.

    Subsampling occurs before any return is observed. More theta witnesses for
    one action do not increase that action's sampling probability or loss weight.
    """
    if actions.ndim != 2 or max_actions < 2:
        raise ValueError("Expected actions[B,T] and max_actions >= 2")
    owners, items, witnesses, counts = [], [], [], []
    # One small transfer, instead of a device synchronization per proposal.
    for row, row_actions in enumerate(actions.tolist()):
        first = {}
        for proposal, item in enumerate(row_actions):
            if item >= 0:
                first.setdefault(item, proposal)
        unique = list(first)
        if len(unique) > max_actions:
            chosen = torch.randperm(len(unique) - 1, device=actions.device,
                                   generator=generator)[:max_actions - 1].tolist()
            unique = unique[:1] + [unique[j + 1] for j in chosen]
        counts.append(len(unique))
        owners.extend([row] * len(unique))
        items.extend(unique)
        witnesses.extend(first[item] for item in unique)
    return InducedActions(*(torch.tensor(x, dtype=torch.long, device=actions.device)
                            for x in (owners, items, witnesses, counts)))


@torch.no_grad()
def decode_proposals(model, state, proposals):
    # Only parameter-free inputs are shared. Each theta goes through the real decoder.
    prepare_state(state, model.feature_config)
    return torch.stack([model.decoder(state, proposal).choose() for proposal in proposals], -1)


def index_theta(theta, indices):
    return StrategyTheta(*(getattr(theta, field.name)[indices] for field in fields(theta)))
