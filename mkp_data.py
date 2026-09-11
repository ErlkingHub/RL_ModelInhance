"""Dataset adapter and an independent synthetic generator; no benchmark training."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from mkp_config import GeneratorConfig
from read_mat import read_mat


@dataclass
class MKPInstance:
    instance_id: str
    profits: np.ndarray             # [N]
    weights: np.ndarray             # [N, M]
    capacities: np.ndarray          # [M]
    reference: float | None = None  # supplied best-known value, not assumed proven optimal

    def __post_init__(self):
        self.profits = np.asarray(self.profits, dtype=np.float64).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.capacities = np.asarray(self.capacities, dtype=np.float64).reshape(-1)
        if self.weights.shape != (self.profits.size, self.capacities.size):
            raise ValueError(f"{self.instance_id}: inconsistent P/R/B shapes")
        if not self.profits.size or not self.capacities.size:
            raise ValueError("Empty MKP instance")
        if any(not np.isfinite(a).all() for a in (self.profits, self.weights, self.capacities)):
            raise ValueError("Non-finite instance data")
        if np.any(self.profits < 0) or np.any(self.weights < 0) or np.any(self.capacities <= 0):
            raise ValueError("Expected nonnegative profits/weights and positive capacities")
        if self.profits.sum() <= 0:
            raise ValueError("At least one positive profit is required")
        if self.reference is not None and (not np.isfinite(self.reference) or self.reference <= 0):
            raise ValueError("Reference objective must be finite and positive")

    @property
    def n(self):
        return self.profits.size

    @property
    def m(self):
        return self.capacities.size


def load_benchmark(path: str | Path, resources=(5, 10), items=(100, 250, 500)):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_mat(str(path))
    if not rows:
        raise ValueError(f"read_mat returned no instances: {path}")
    result = []
    for row in rows:
        m, n = row["M"], row["N"]
        if m not in resources or n not in items:
            continue
        weights = np.asarray(row["R"], dtype=np.float64)
        # read_mat's Chu layout is resources x items. Reject ambiguous square layouts.
        if m == n and weights.shape == (m, n):
            raise ValueError("Square R layout is ambiguous; specify orientation in a dataset adapter")
        if weights.shape == (m, n):
            weights = weights.T
        elif weights.shape != (n, m):
            raise ValueError(f"Instance {row['id']}: unexpected R shape {weights.shape}")
        inst = MKPInstance(str(row["id"]), row["P"], weights, row["B"], row.get("best"))
        if (inst.m, inst.n) != (m, n):
            raise ValueError("Declared M/N differ from actual arrays")
        result.append(inst)
    if not result:
        raise ValueError("No instances match the requested M/N filters")
    return result


def synthetic_batch(batch_size, n, m, config: GeneratorConfig, rng: np.random.Generator, device):
    """Chu-inspired distribution, not a claim to reproduce the original generator exactly."""
    w = rng.integers(config.weight_low, config.weight_high + 1, size=(batch_size, n, m))
    noise = rng.uniform(0.0, config.profit_noise, size=(batch_size, n))
    correlated = w.mean(axis=-1) + noise
    independent = rng.uniform(1.0, config.weight_high + config.profit_noise, size=(batch_size, n))
    choose = rng.random((batch_size, 1)) < config.correlated_probability
    p = np.maximum(1.0, np.floor(np.where(choose, correlated, independent)))
    ratios = rng.choice(config.capacity_ratios, size=(batch_size, 1))
    c = np.maximum(1.0, np.floor(w.sum(axis=1) * ratios))
    return tuple(torch.as_tensor(x, dtype=torch.float64, device=device) for x in (p, w, c))


def instance_batch(instance: MKPInstance, repeats, device):
    return tuple(torch.as_tensor(x, dtype=torch.float64, device=device).unsqueeze(0)
                 .expand(repeats, *x.shape).clone()
                 for x in (instance.profits, instance.weights, instance.capacities))
