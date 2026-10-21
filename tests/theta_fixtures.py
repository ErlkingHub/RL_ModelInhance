"""Hand-set strategy parameters for analytic decoder tests; not an evaluated method."""
import torch

from mkp_Model import StrategyTheta


def density_theta(state, config):
    """Plain density coefficients with every optional score term switched off."""
    b, m = state.capacities.shape
    device = state.profits.device
    ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
    return StrategyTheta(torch.ones(b, m, device=device), ones.clone(), ones.clone(),
                         torch.full((b,), 0.1, device=device), torch.full((b,), 0.2, device=device),
                         zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(),
                         torch.zeros(b, m, m, device=device),
                         torch.zeros(b, len(config.companion_scales), device=device))
