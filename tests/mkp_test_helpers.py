"""Literal legal controls for analytic unit fixtures, outside production inference."""
import torch
from mkp_Model import StrategyTheta


def synthetic_theta(state, config):
    b, m = state.capacities.shape
    device = state.profits.device
    return StrategyTheta(torch.ones(b, m, device=device), torch.ones(b, device=device),
        torch.ones(b, device=device), torch.full((b,), .1, device=device),
        torch.full((b,), .2, device=device), torch.zeros(b, device=device),
        torch.zeros(b, device=device), torch.zeros(b, m, m, device=device),
        torch.zeros(b, len(config.companion_scales), device=device))
