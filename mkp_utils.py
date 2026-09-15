"""Reproducibility, checkpoint contracts and shared constructive inference."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from mkp_config import CHECKPOINT_VERSION, DECODER_VERSION, ModelConfig
from mkp_env import MKPEnv
from mkp_Model import MKPModel, WeakDecoder


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
        raise ValueError("Incompatible checkpoint schema")
    if payload.get("decoder_version") != DECODER_VERSION:
        raise ValueError("Checkpoint's weak decoder differs from this implementation")
    if expected_m is not None and payload["num_resources"] != expected_m:
        raise ValueError(f"Checkpoint M={payload['num_resources']} differs from requested M={expected_m}")
    if payload.get("training_data_source") != "synthetic_only":
        raise ValueError("Strict zero-shot pipeline requires a synthetic-only training checkpoint")
    model = MKPModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload


@torch.no_grad()
def construct_solution(data, model=None, mode="learned_dynamic", stochastic=False, model_config=None):
    """Full constructive episode, no repair/search. Used only by explicit train/benchmark runs."""
    if mode not in ("fixed", "learned_static", "learned_dynamic"):
        raise ValueError(mode)
    if mode != "fixed" and model is None:
        raise ValueError("A model is required for learned inference")
    env = MKPEnv(*data)
    decoder = model.decoder if model is not None else WeakDecoder(model_config or ModelConfig())
    if model is not None:
        model.eval()
    theta = decoder.fixed_theta(env.state) if mode == "fixed" else None
    if mode == "learned_static":
        theta = model.actor(env.state)[0]
    sequences = []
    theta_sum = torch.zeros(data[0].shape[0], 3, device=data[0].device)
    theta_count = torch.zeros(data[0].shape[0], device=data[0].device)
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        if mode == "learned_dynamic":
            theta = model.actor(env.state)[0]
        policy = decoder(env.state, theta)
        actions = policy.choose(stochastic)
        stats = torch.stack((theta.profit_exponent, theta.bottleneck_weight,
                             theta.resource_log_weights.exp().std(-1, unbiased=False)), dim=-1)
        theta_sum += stats * active[:, None]
        theta_count += active
        sequences.append(actions)
        env.step(actions)
    if env.state.active.any():
        raise RuntimeError("Construction exceeded N steps")
    sequence = (torch.stack(sequences, dim=1) if sequences else
                torch.empty(data[0].shape[0], 0, dtype=torch.long, device=data[0].device))
    return {"objective": env.objective, "selected": env.state.selected,
            "remaining": env.state.remaining, "sequence": sequence,
            "theta_mean": theta_sum / theta_count.clamp_min(1)[:, None]}
