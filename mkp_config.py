"""All experiment defaults. Paths are relative to this project, not the shell cwd."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 1
DECODER_VERSION = "resource_price_density_v1"

@dataclass
class TrainConfig:
    num_resources: tuple[int, ...] = (5, 10)
    train_num_items: tuple[int, ...] = (100,)
    updates: int = 800
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
    validation_every: int = 25
    validation_instances_per_shape: int = 64
    checkpoint_every: int = 100
    plot_every: int = 100
    plot_smoothing_window: int = 25
    plot_dpi: int = 140
    seed: int = 20260910
    validation_seed: int = 1729
    device: str = "auto"
    deterministic: bool = True
    cpu_threads: int = 4
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")

    def __post_init__(self):
        counts = (self.updates, self.batch_size, self.minibatch_size, self.ppo_epochs,
                  self.validation_every, self.validation_instances_per_shape,
                  self.checkpoint_every, self.cpu_threads, self.plot_every,
                  self.plot_smoothing_window, self.plot_dpi)
        if any(x < 1 for x in counts):
            raise ValueError("Training counts/intervals must be positive")
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
        self.runs_dir, self.checkpoint_dir = Path(self.runs_dir), Path(self.checkpoint_dir)


@dataclass
class ModelConfig:
    encoder_type: str = "bgnn"
    policy_head: str = "weak"
    direct_logit_clip: float = 10.0
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    ff_multiplier: int = 4
    # No positional embeddings: item/resource order must not encode identity.
    price_floor: float = 0.05
    price_head_type: str = "conditional"
    price_head_init_std: float = 0.01
    profit_exponent_min: float = 0.5
    profit_exponent_max: float = 2.0
    scarcity_exponent_max: float = 2.0
    bottleneck_weight_max: float = 2.0
    temperature_min: float = 0.05
    temperature_max: float = 1.0
    feature_clip: float = 100.0
    eps: float = 1e-8

    def __post_init__(self):
        if self.encoder_type not in ("bgnn", "transformer"):
            raise ValueError("encoder_type must be bgnn or transformer")
        if self.policy_head not in ("weak", "direct_item") or self.direct_logit_clip <= 0:
            raise ValueError("Invalid policy_head or direct_logit_clip")
        if self.price_head_type not in ("conditional", "linear"):
            raise ValueError("price_head_type must be conditional or linear")
        if not 0 < self.price_head_init_std < 1:
            raise ValueError("price_head_init_std must be in (0, 1)")
        if self.embed_dim < 1 or self.num_heads < 1 or (self.encoder_type == "transformer" and self.embed_dim % self.num_heads):
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
    encoder_type: str = "bgnn"
    policy_head: str = "weak"
    compare_heads: bool = False
    dataset: Path = field(default_factory=lambda: ROOT / "MKP_dataset" / "Benchmarks_Chu.mat")
    num_resources: tuple[int, ...] = (5, 10)
    num_items: tuple[int, ...] = (100, 250, 500)
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
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
        if self.encoder_type not in ("bgnn", "transformer") or self.policy_head not in ("weak", "direct_item"):
            raise ValueError("Invalid encoder_type or policy_head")
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


def variant_name(config):
    return f"{config.encoder_type}_{config.policy_head}"


def decoder_version(config):
    return DECODER_VERSION if config.policy_head == "weak" else "direct_item_masked_v1"
