"""Reproducibility, checkpoint contracts and shared constructive inference."""
import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from mkp_config import CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION, ModelConfig, config_dict
from mkp_env import MKPEnv
from mkp_Model import MKPModel, WeakDecoder
from mkp_basis import decoder_statistics, gate_statistics


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def seed_everything(seed, deterministic=True, cpu_threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    # Unsupported deterministic kernels fail visibly rather than silently changing semantics.
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, device, expected_m=None):
    # Checkpoints are local trusted training artifacts; loading still uses restricted unpickling.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Incompatible checkpoint schema: decision teacher v9 requires a fresh training run "
                         "or explicit --warm-start from original v5/v8; --resume only accepts v9")
    if payload.get("controller_version") != CONTROLLER_VERSION:
        raise ValueError("Checkpoint's high-level controller differs from this implementation")
    if payload.get("decoder_version") != DECODER_VERSION:
        raise ValueError("Checkpoint's weak decoder differs from this implementation")
    if expected_m is not None and payload["num_resources"] != expected_m:
        raise ValueError(f"Checkpoint M={payload['num_resources']} differs from requested M={expected_m}")
    if payload.get("training_data_source") != "synthetic_only":
        raise ValueError("Strict zero-shot pipeline requires a synthetic-only training checkpoint")
    model = MKPModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload


def warm_start_checkpoint(path, config, device, expected_m=None):
    """Transfer v5/v8 representations into v9, starting a distinct experiment.

    All old parameters load strictly. The new efficiency head uses its declared
    initialization, and the new terminal rule changes scoring. Unlike v5->v8,
    this migration intentionally does NOT promise identical policy outputs.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload.get("checkpoint_version")
    controllers = {5: "combination_theta_v5", 8: "resource_aware_theta_v8"}
    if (version not in controllers or payload.get("controller_version") != controllers[version]
            or payload.get("decoder_version") != "second_order_density_v5"):
        raise ValueError("Warm start requires an original combination_theta_v5 or resource_aware_theta_v8 checkpoint")
    if payload.get("training_data_source") != "synthetic_only":
        raise ValueError("Warm start requires a synthetic-only training checkpoint")
    if expected_m is not None and payload.get("num_resources") != expected_m:
        raise ValueError("Warm-start checkpoint resource count differs from requested M")
    new_fields = {"efficiency_gate_initial", "terminal_profit_override"}
    if version == 5:
        new_fields |= {"actor_resource_geometry", "controller_resource_readout"}
    source_config = payload.get("model_config", {})
    legacy_fields = set(config_dict(config)) - new_fields
    if set(source_config) != legacy_fields:
        raise ValueError(f"Warm-start model config is not the original v{version} architecture")
    normalized = config_dict(ModelConfig(**source_config))
    expected = config_dict(config)
    differing = [key for key in legacy_fields if normalized[key] != expected[key]]
    if differing:
        raise ValueError(f"Warm start requires unchanged v{version} model settings: " + ", ".join(sorted(differing)))
    model = MKPModel(config)
    target = model.state_dict()
    new_modules = {"efficiency_head"}
    if version == 5:
        new_modules |= {"resource_geometry_encoder", "resource_relation", "resource_readout"}
    legacy_keys = {name for name in target if not new_modules.intersection(name.split("."))}
    source = payload["model_state"]
    if set(source) != legacy_keys:
        raise ValueError(f"Warm-start state does not match all original v{version} parameter keys")
    for name in legacy_keys:
        if source[name].shape != target[name].shape:
            raise ValueError("Warm-start parameter shape differs: " + name)
    target.update(source)
    model.load_state_dict(target, strict=True)
    provenance = dict(mode=f"v{version}_warm_start", source_path=str(Path(path).resolve()),
                      source_sha256=file_sha256(path), source_update=payload.get("update"),
                      source_controller_version=payload["controller_version"],
                      optimizer_reset=True, policy_outputs_preserved=False,
                      efficiency_gate_initial=config.efficiency_gate_initial,
                      terminal_profit_override=config.terminal_profit_override)
    return model.to(device), provenance


@torch.no_grad()
def construct_solution(data, model=None, mode="learned_dynamic", stochastic=False, model_config=None):
    """Full constructive episode, no repair/search. Used only by explicit train/benchmark runs."""
    if mode not in ("fixed", "learned_static", "learned_dynamic"):
        raise ValueError(mode)
    if mode != "fixed" and model is None:
        raise ValueError("A model is required for learned inference")
    env = MKPEnv(*data)
    decoder = model.decoder if model is not None else WeakDecoder(model_config or ModelConfig())
    if mode == "fixed":
        # The original density baseline has zero opportunity weights: avoid
        # charging its runtime for pairwise features it never uses.
        decoder = WeakDecoder(replace(decoder.config, opportunity_weight_max=0, reserve_weight_max=0,
                                      interaction_weight_max=0, companion_weight_max=0,
                                      terminal_profit_override=False))
    if model is not None:
        model.eval()
    theta = decoder.fixed_theta(env.state) if mode == "fixed" else None
    if mode == "learned_static":
        theta = model.actor(env.state)[0]
    sequences = []
    theta_sum = torch.zeros(data[0].shape[0], 9, device=data[0].device)
    decoder_sum = torch.zeros(data[0].shape[0], 4, device=data[0].device)
    gate_sum = torch.zeros(data[0].shape[0], 3, device=data[0].device)
    theta_count = torch.zeros(data[0].shape[0], device=data[0].device)
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        if mode == "learned_dynamic":
            theta = model.actor(env.state)[0]
        policy = decoder(env.state, theta)
        actions = policy.choose(stochastic)
        stats = theta.statistics()
        theta_sum += stats * active[:, None]
        decoder_sum += decoder_statistics(policy) * active[:, None]
        gate_sum += gate_statistics(policy) * active[:, None]
        theta_count += active
        sequences.append(actions)
        env.step(actions)
    if env.state.active.any():
        raise RuntimeError("Construction exceeded N steps")
    sequence = (torch.stack(sequences, dim=1) if sequences else
                torch.empty(data[0].shape[0], 0, dtype=torch.long, device=data[0].device))
    return {"objective": env.objective, "selected": env.state.selected,
            "remaining": env.state.remaining, "sequence": sequence,
            "theta_mean": theta_sum / theta_count.clamp_min(1)[:, None],
            "decoder_mean": decoder_sum / theta_count.clamp_min(1)[:, None],
            "gate_mean": gate_sum / theta_count.clamp_min(1)[:, None]}
