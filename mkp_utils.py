"""Reproducibility, checkpoint contracts and shared constructive inference."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from mkp_config import CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION, ModelConfig
from mkp_env import MKPEnv
from mkp_Model import MKPModel, THETA_STATISTICS
from mkp_basis import decoder_statistics


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
        raise ValueError("Incompatible checkpoint schema: LP-dual theta v6 requires a fresh training run; "
                         "v1-v5 weights cannot be resumed as the new encoder/controller/decoder")
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


@torch.no_grad()
def construct_solution(data, model, stochastic=False):
    """Full learned_dynamic episode: theta is recomputed at every step; no repair/search."""
    env = MKPEnv(*data)
    model.eval()
    sequences = []
    rows = data[0].shape[0]
    theta_sum = torch.zeros(rows, len(THETA_STATISTICS), device=data[0].device)
    decoder_sum = torch.zeros(rows, 4, device=data[0].device)
    theta_count = torch.zeros(rows, device=data[0].device)
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        policy, theta, _ = model.policy(env.state)
        actions = policy.choose(stochastic)
        theta_sum += theta.statistics() * active[:, None]
        decoder_sum += decoder_statistics(policy) * active[:, None]
        theta_count += active
        sequences.append(actions)
        env.step(actions)
    if env.state.active.any():
        raise RuntimeError("Construction exceeded N steps")
    sequence = (torch.stack(sequences, dim=1) if sequences else
                torch.empty(rows, 0, dtype=torch.long, device=data[0].device))
    return {"objective": env.objective, "selected": env.state.selected,
            "remaining": env.state.remaining, "sequence": sequence,
            "theta_mean": theta_sum / theta_count.clamp_min(1)[:, None],
            "decoder_mean": decoder_sum / theta_count.clamp_min(1)[:, None]}
