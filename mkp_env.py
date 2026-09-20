"""Batched dynamic MKP. Float64 accounting; learned features use float32."""
from dataclasses import dataclass
import torch


@dataclass
class MKPState:
    profits: torch.Tensor          # [B, N], float64
    weights: torch.Tensor          # [B, N, M], float64
    capacities: torch.Tensor       # [B, M], float64
    remaining: torch.Tensor        # [B, M], float64
    selected: torch.Tensor         # [B, N], bool

    @property
    def feasible(self):
        # Physical action set; also available to Actor attention for candidate aggregation.
        return (~self.selected) & (self.weights <= self.remaining[:, None, :]).all(dim=-1)

    @property
    def active(self):
        return self.feasible.any(dim=-1)

    def index(self, indices):
        return MKPState(*(x[indices] for x in
                         (self.profits, self.weights, self.capacities, self.remaining, self.selected)))


class MKPEnv:
    @classmethod
    def from_state(cls, state):
        """Independent branch with exact remaining capacity and selected history."""
        env = cls.__new__(cls)
        env.state = MKPState(*(x.clone() for x in
            (state.profits, state.weights, state.capacities, state.remaining, state.selected)))
        return env

    def __init__(self, profits, weights, capacities):
        if profits.ndim != 2 or weights.ndim != 3 or capacities.ndim != 2:
            raise ValueError("Expected P[B,N], R[B,N,M], B[B,M]")
        if weights.shape[:2] != profits.shape or weights.shape[::2] != capacities.shape:
            raise ValueError("Inconsistent batch shapes")
        if profits.shape[0] < 1 or profits.shape[1] < 1 or capacities.shape[1] < 1:
            raise ValueError("Empty batch or instance")
        if any(not torch.isfinite(x).all() for x in (profits, weights, capacities)):
            raise ValueError("Non-finite instance")
        if (profits < 0).any() or (weights < 0).any() or (capacities <= 0).any():
            raise ValueError("Invalid signs in instance data")
        if (profits.sum(-1) <= 0).any():
            raise ValueError("Each instance must have positive total profit")
        self.state = MKPState(profits.double(), weights.double(), capacities.double(),
                              capacities.double().clone(), torch.zeros_like(profits, dtype=torch.bool))

    def step(self, actions):
        """One item per active row. -1 is only legal for an already terminal row."""
        s = self.state
        if actions.shape != (s.profits.shape[0],) or actions.dtype != torch.long:
            raise ValueError("actions must be a LongTensor of shape [B]")
        feasible = s.feasible
        active = feasible.any(-1)
        if ((actions < -1) | (actions >= s.profits.shape[1])).any():
            raise ValueError("Action out of range")
        if ((actions == -1) != ~active).any():
            raise ValueError("-1 is required exactly for terminal rows")
        rows = active.nonzero(as_tuple=True)[0]
        cols = actions[rows]
        if not feasible[rows, cols].all():
            raise ValueError("Attempted repeated or infeasible item")
        reward = torch.zeros(s.profits.shape[0], dtype=torch.float32, device=s.profits.device)
        # Positive affine normalization independent of policy; gamma=1 preserves the objective.
        reward[rows] = (s.profits[rows, cols] / s.profits[rows].sum(-1)).float()
        s.remaining[rows] -= s.weights[rows, cols]
        s.selected[rows, cols] = True
        return reward, ~s.active

    @property
    def objective(self):
        return (self.state.profits * self.state.selected).sum(-1)
