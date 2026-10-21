"""All experiment defaults. Paths are relative to this project, not the shell cwd."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 5
CONTROLLER_VERSION = "combination_theta_v5"
DECODER_VERSION = "second_order_density_v5"

@dataclass
class TrainConfig:
    num_resources: tuple[int, ...] = (5, 10)
    train_num_items: tuple[int, ...] = (100,)
    updates: int = 2000
    batch_size: int = 64
    minibatch_size: int = 128
    ppo_epochs: int = 3
    learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    constraint_aux_coefficient: float = 0.05
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    # Training-only paired continuations under the frozen pre-update policy.
    ranking_coefficient: float = 0.1
    ranking_states: int = 16
    ranking_pool_size: int = 64
    ranking_candidates: int = 8
    ranking_repeats: int = 4
    ranking_minibatch_size: int = 8
    ranking_min_gap: float = 0.001
    ranking_se_multiplier: float = 1.0
    ranking_return_scale: float = 0.02
    ranking_score_scale: float = 0.2
    ranking_greedy_weight: float = 0.5
    validation_every: int = 25
    validation_instances_per_shape: int = 64
    checkpoint_every: int = 100
    # Operational performance plateau on fixed synthetic validation (not Chu gap).
    convergence_min_updates: int = 1000
    convergence_patience_updates: int = 1000
    convergence_min_evaluations: int = 20
    convergence_tolerance_pct: float = 0.05
    plot_smoothing_updates: int = 100
    seed: int = 20260910
    validation_seed: int = 1729
    device: str = "auto"
    deterministic: bool = True
    cpu_threads: int = 4
    # Execution only: preserve PPO batch sizes, precision and checkpoint schema.
    performance_profile: str = "auto"  # >=20 GiB CUDA -> memory24; otherwise reference
    feature_cache_gib: float = 6.0
    cache_headroom_gib: float = 6.0
    feature_cache_batch_size: int = 64
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints" / "resource_routing_v1")

    def __post_init__(self):
        counts = (self.updates, self.batch_size, self.minibatch_size, self.ppo_epochs,
                  self.validation_every, self.validation_instances_per_shape,
                  self.checkpoint_every, self.cpu_threads, self.convergence_min_updates,
                  self.convergence_patience_updates, self.convergence_min_evaluations,
                  self.plot_smoothing_updates, self.feature_cache_batch_size)
        if any(x < 1 for x in counts):
            raise ValueError("Training counts/intervals must be positive")
        if self.performance_profile not in ("auto", "reference", "memory24"):
            raise ValueError("performance_profile must be auto, reference or memory24")
        if not all(math.isfinite(x) and x >= 0 for x in (self.feature_cache_gib, self.cache_headroom_gib)):
            raise ValueError("Cache budget and headroom must be finite and nonnegative")
        if not self.num_resources or any(x < 1 for x in self.num_resources):
            raise ValueError("num_resources must be positive")
        if not self.train_num_items or any(x < 1 for x in self.train_num_items):
            raise ValueError("train_num_items must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid GAE parameters")
        if self.learning_rate <= 0 or not 0 < self.clip_ratio < 1 or self.value_clip <= 0:
            raise ValueError("Invalid optimizer or PPO parameters")
        if min(self.value_coefficient, self.entropy_coefficient,
               self.constraint_aux_coefficient) < 0:
            raise ValueError("Loss coefficients must be nonnegative")
        if self.max_grad_norm <= 0 or self.target_kl <= 0:
            raise ValueError("Gradient norm and target KL must be positive")
        if not 0 < self.convergence_tolerance_pct < 100:
            raise ValueError("convergence_tolerance_pct must lie in (0, 100)")
        if self.convergence_min_evaluations < 2:
            raise ValueError("convergence_min_evaluations must be at least 2")
        if min(self.ranking_states, self.ranking_pool_size, self.ranking_minibatch_size) < 1:
            raise ValueError("Ranking batch sizes must be positive")
        if self.ranking_pool_size < self.ranking_states or self.ranking_candidates < 2 or self.ranking_repeats < 2:
            raise ValueError("Ranking needs pool >= states, candidates >= 2 and repeats >= 2")
        if not (self.ranking_coefficient >= 0 and self.ranking_min_gap >= 0
                and self.ranking_se_multiplier >= 0 and self.ranking_return_scale > 0
                and self.ranking_score_scale > 0 and 0 <= self.ranking_greedy_weight <= 1):
            raise ValueError("Invalid ranking loss settings")
        if not all(math.isfinite(x) for x in (self.ranking_coefficient, self.ranking_min_gap,
                self.ranking_se_multiplier, self.ranking_return_scale, self.ranking_score_scale)):
            raise ValueError("Ranking loss settings must be finite")
        self.runs_dir, self.checkpoint_dir = Path(self.runs_dir), Path(self.checkpoint_dir)


@dataclass
class ModelConfig:
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    ff_multiplier: int = 4
    # Actor architecture is checkpointed independently of the unchanged v5 theta.
    actor_encoder: str = "resource_routing_v1"
    actor_strategy_readout: bool = True
    # No positional embeddings: item/resource order must not encode identity.
    price_floor: float = 0.05
    profit_exponent_min: float = 0.5
    profit_exponent_max: float = 2.0
    scarcity_exponent_max: float = 2.0
    bottleneck_weight_max: float = 2.0
    temperature_min: float = 0.05
    temperature_max: float = 1.0
    feature_clip: float = 100.0
    # Actor-only enhancements; critic retains its original independent encoder.
    actor_edge_values: bool = True
    actor_postdecision_features: bool = True
    actor_pair_attention: bool = True
    actor_contrast_pooling: bool = True
    actor_feasible_attention: bool = True
    actor_relation_chunk_size: int = 32
    actor_relation_bandwidth: float = 0.05
    actor_similarity_bandwidth: float = 0.1
    opportunity_tokens: int = 4
    opportunity_weight_max: float = 4.0
    reserve_weight_max: float = 4.0
    opportunity_bandwidth: float = 0.05
    opportunity_chunk_size: int = 128
    # State-level basis coefficients; their dimensions do not grow with N.
    interaction_weight_max: float = 2.0
    companion_weight_max: float = 2.0
    combination_initial_weight: float = 0.02
    companion_scales: tuple[float, ...] = (0.5, 2.0, 8.0)
    combination_chunk_size: int = 64
    eps: float = 1e-8

    def __post_init__(self):
        if self.actor_encoder not in ("legacy", "resource_routing_v1"):
            raise ValueError("actor_encoder must be legacy or resource_routing_v1")
        if self.embed_dim < 1 or self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if self.num_layers < 1 or self.ff_multiplier < 1:
            raise ValueError("num_layers and ff_multiplier must be positive")
        if not 0 < self.price_floor < 1:
            raise ValueError("price_floor must be in (0, 1)")
        if not 0 < self.temperature_min <= self.temperature_max:
            raise ValueError("Invalid temperature range")
        if not 0 < self.profit_exponent_min <= self.profit_exponent_max:
            raise ValueError("Invalid profit exponent range")
        if self.scarcity_exponent_max < 0 or self.bottleneck_weight_max < 0:
            raise ValueError("Strategy ranges must be nonnegative")
        if self.eps <= 0 or self.feature_clip <= 0:
            raise ValueError("eps and feature_clip must be positive")
        if self.opportunity_tokens < 0 or self.opportunity_chunk_size < 1:
            raise ValueError("Invalid opportunity attention/chunk sizes")
        if self.actor_relation_chunk_size < 1:
            raise ValueError("actor_relation_chunk_size must be positive")
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.actor_relation_bandwidth, self.actor_similarity_bandwidth)):
            raise ValueError("Actor relation bandwidths must be finite and positive")
        if not (self.opportunity_weight_max >= 0 and self.reserve_weight_max >= 0
                and self.opportunity_bandwidth > 0):
            raise ValueError("Invalid opportunity score settings")
        if not all(math.isfinite(x) for x in (self.opportunity_weight_max, self.reserve_weight_max,
                                             self.opportunity_bandwidth)):
            raise ValueError("Opportunity score settings must be finite")
        self.companion_scales = tuple(self.companion_scales)
        if not self.companion_scales or any(not math.isfinite(x) or x <= 0 for x in self.companion_scales):
            raise ValueError("companion_scales must be finite and positive")
        if tuple(sorted(set(self.companion_scales))) != self.companion_scales:
            raise ValueError("companion_scales must be strictly increasing")
        if not all(math.isfinite(x) and x >= 0 for x in (self.interaction_weight_max, self.companion_weight_max)):
            raise ValueError("Combination weight bounds must be finite and nonnegative")
        if not math.isfinite(self.combination_initial_weight) or self.combination_initial_weight <= 0:
            raise ValueError("combination_initial_weight must be finite and positive")
        if self.combination_chunk_size < 1:
            raise ValueError("combination_chunk_size must be positive")

    @classmethod
    def from_checkpoint(cls, values):
        # Missing architecture fields identify the original v5 model, never new defaults.
        values = dict(values)
        values.setdefault("actor_encoder", "legacy")
        values.setdefault("actor_strategy_readout", False)
        return cls(**values)


@dataclass
class GeneratorConfig:
    # Synthetic training data only; these settings are never fitted on Chu.
    capacity_ratios: tuple[float, ...] = (0.25, 0.5, 0.75)
    weight_low: int = 1
    weight_high: int = 1000
    profit_noise: float = 500.0
    correlated_probability: float = 0.8

    def __post_init__(self):
        if not self.capacity_ratios or any(not 0 < x < 1 for x in self.capacity_ratios):
            raise ValueError("capacity_ratios must lie in (0, 1)")
        if not 1 <= self.weight_low <= self.weight_high:
            raise ValueError("Invalid positive integer weight range")
        if not 0 <= self.correlated_probability <= 1 or self.profit_noise < 0:
            raise ValueError("Invalid profit distribution")



@dataclass
class BenchmarkConfig:
    dataset: Path = field(default_factory=lambda: ROOT / "MKP_dataset" / "Benchmarks_Chu.mat")
    num_resources: tuple[int, ...] = (5, 10)
    num_items: tuple[int, ...] = (100, 250, 500)
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints" / "resource_routing_v1")
    results_dir: Path = field(default_factory=lambda: ROOT / "results")
    checkpoint_tag: str = "best"
    device: str = "auto"
    seed: int = 20260910
    # Greedy single trajectory is the primary result. Sampling is opt-in and separate.
    sample_rollouts: int = 0
    sample_batch_size: int = 32
    target_gap_pct: float = 0.1
    feasibility_tolerance: float = 1e-7
    cpu_threads: int = 4

    def __post_init__(self):
        if self.sample_rollouts < 0 or self.sample_batch_size < 1 or self.cpu_threads < 1:
            raise ValueError("Invalid benchmark budget")
        if self.target_gap_pct < 0 or self.feasibility_tolerance < 0:
            raise ValueError("Invalid benchmark tolerances")
        if not self.num_resources or not self.num_items or min(*self.num_resources, *self.num_items) < 1:
            raise ValueError("Benchmark shapes must be positive")
        if self.checkpoint_tag not in ("best", "last"):
            raise ValueError("checkpoint_tag must be best or last")
        self.dataset = Path(self.dataset)
        self.checkpoint_dir, self.results_dir = Path(self.checkpoint_dir), Path(self.results_dir)


def config_dict(config):
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
