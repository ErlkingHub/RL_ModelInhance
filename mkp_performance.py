"""Training execution settings and detached geometry caches; no learned state."""
from dataclasses import dataclass, replace
import time
import torch

GIB = 1024 ** 3
PERFORMANCE_FIELDS = ("feature_cache_gib", "feature_cache_seconds", "ppo_seconds",
                      "rollout_seconds", "gpu_peak_allocated_gib", "gpu_peak_reserved_gib")


def state_stamp(state):
    # Detect both tensor replacement and in-place environment steps.
    return tuple((id(x), x._version) for x in
                 (state.profits, state.weights, state.capacities, state.remaining, state.selected))


def feature_signature(config):
    # Chunk sizes affect execution only. All feature-defining settings belong here.
    return tuple(getattr(config, name) for name in (
        "eps", "feature_clip", "actor_postdecision_features", "actor_pair_attention",
        "actor_contrast_pooling", "actor_relation_bandwidth", "opportunity_weight_max",
        "reserve_weight_max", "opportunity_bandwidth", "interaction_weight_max",
        "companion_weight_max", "companion_scales"))


@dataclass
class PreparedFeatures:
    tensors: dict
    signature: tuple
    stamp: tuple

    def valid_for(self, state):
        return self.stamp == state_stamp(state)

    def index(self, indices, state):
        return PreparedFeatures({k: v[indices] for k, v in self.tensors.items()},
                                self.signature, state_stamp(state))

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors.values())


def cached_features(state, config):
    prepared = state._prepared
    if (prepared is not None and prepared.valid_for(state)
            and prepared.signature == feature_signature(config)):
        return prepared.tensors
    return None


@torch.no_grad()
def prepare_state(state, config):
    """Cache only exact, parameter-free inputs. Embeddings and theta stay live."""
    if cached_features(state, config) is not None:
        return state
    from mkp_Model import state_features, WeakDecoder
    from mkp_attention import candidate_relations
    from mkp_basis import interaction_loads, companion_basis

    state._prepared = None
    payload = dict(zip(("item", "resource", "edge", "global"), state_features(state, config)))
    payload["feasible"] = state.feasible
    if config.actor_postdecision_features or config.actor_pair_attention or config.actor_contrast_pooling:
        relations = candidate_relations(state, config)
        for name in ("pairs", "valid", "post", "resource_post"):
            payload["relation_" + name] = getattr(relations, name)
    owners, items = payload["feasible"].nonzero(as_tuple=True)

    def dense(values):
        result = values.new_zeros((*state.profits.shape, *values.shape[1:]))
        result[owners, items] = values
        return result

    if config.opportunity_weight_max or config.reserve_weight_max:
        blocked, reserve = WeakDecoder(config).opportunity_features(state, owners, items)
        payload.update(blocked=dense(blocked), reserve=dense(reserve))
    if config.interaction_weight_max:
        payload["interaction"] = dense(interaction_loads(state, owners, items, config.eps))
    if config.companion_weight_max:
        payload["companion"] = dense(companion_basis(state, owners, items, config))
    state._prepared = PreparedFeatures(payload, feature_signature(config), state_stamp(state))
    return state


def configure_training_runtime(model, train, device):
    """Keep model.config/state_dict unchanged so v5 resume contracts still hold."""
    from mkp_attention import PairRelationAttention
    device = torch.device(device)
    profile = train.performance_profile
    if profile == "auto":
        profile = ("memory24" if device.type == "cuda" and
                   torch.cuda.get_device_properties(device).total_memory >= 20 * GIB else "reference")
    if profile == "memory24" and device.type != "cuda":
        raise ValueError("memory24 requires a CUDA device; use auto or reference on CPU")
    effective = model.config
    if profile == "memory24":
        effective = replace(effective,
            actor_relation_chunk_size=max(128, effective.actor_relation_chunk_size),
            opportunity_chunk_size=max(4096, effective.opportunity_chunk_size),
            combination_chunk_size=max(4096, effective.combination_chunk_size))
    model.reuse_state_features = profile == "memory24"
    model.feature_config = effective
    model.actor.encoder.config = effective
    model.decoder.config = effective
    for module in model.actor.modules():
        if isinstance(module, PairRelationAttention):
            module.chunk_size = effective.actor_relation_chunk_size
    return dict(profile=profile, reuse_state_features=model.reuse_state_features,
                actor_relation_chunk_size=effective.actor_relation_chunk_size,
                opportunity_chunk_size=effective.opportunity_chunk_size,
                combination_chunk_size=effective.combination_chunk_size,
                feature_cache_gib=train.feature_cache_gib,
                cache_headroom_gib=train.cache_headroom_gib,
                feature_cache_batch_size=train.feature_cache_batch_size,
                arithmetic="float32 model; float64 capacity geometry")


@torch.no_grad()
def build_rollout_cache(model, data, rollout, times, batches, train):
    """Immutable PPO state bank shared across epochs, bounded by a memory budget.

    Reserve headroom for gradients/activations. Budget rejection falls back to
    per-minibatch preparation; it does not change samples or optimizer updates.
    """
    from mkp_env import MKPState
    metrics = dict(feature_cache_gib=0., feature_cache_seconds=0.)
    if not model.reuse_state_features or train.feature_cache_gib == 0 or times.numel() == 0:
        return None, metrics
    device = data[0].device
    budget = int(train.feature_cache_gib * GIB)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, max(0, free - int(train.cache_headroom_gib * GIB)))
    if budget <= 0:
        return None, metrics
    began = time.perf_counter()
    storage, state, sample = {}, None, None

    def gather(ids):
        b, t = batches[ids], times[ids]
        return MKPState(data[0][b], data[1][b], data[2][b],
                        rollout["remaining"][t, b], rollout["selected"][t, b])

    try:
        count = times.numel()
        size = min(train.feature_cache_batch_size, count)
        sample = prepare_state(gather(slice(0, size)), model.feature_config)
        state_bytes = sum(x.numel() * x.element_size() for x in (
            sample.profits, sample.weights, sample.capacities, sample.remaining, sample.selected))
        required = (sample._prepared.nbytes + state_bytes) // size * count
        if required <= budget:
            state = gather(slice(None))
            for key, value in sample._prepared.tensors.items():
                storage[key] = value.new_empty((count, *value.shape[1:]))
                storage[key][:size].copy_(value)
            sample = None
            for start in range(size, count, size):
                sample = prepare_state(state.index(slice(start, start + size), with_features=False),
                                       model.feature_config)
                for key, value in sample._prepared.tensors.items():
                    storage[key][start:start + size].copy_(value)
                sample = None
            state._prepared = PreparedFeatures(storage, feature_signature(model.feature_config), state_stamp(state))
            metrics["feature_cache_gib"] = required / GIB
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        # Only cache allocation may fall back. Never silently retry an optimizer step.
        storage.clear()
        state, sample = None, None
        metrics["feature_cache_gib"] = 0.
        torch.cuda.empty_cache()
    metrics["feature_cache_seconds"] = time.perf_counter() - began
    return state, metrics
