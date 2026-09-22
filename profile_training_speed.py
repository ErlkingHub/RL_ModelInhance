"""Short synthetic forward/backward timing. No optimizer, episodes or evaluation.

Run when the GPU is otherwise idle for interpretable timings. This measures the
PPO compute path, not full-update throughput or learned solution quality.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv
from mkp_Model import MKPModel
from mkp_performance import GIB, build_rollout_cache, configure_training_runtime
from mkp_utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--resources", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bank-batches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.items, args.resources, args.batch_size, args.bank_batches, args.repeats, args.warmup) < 1:
        parser.error("All counts must be positive")
    if not torch.cuda.is_available():
        parser.error("This timing probe requires CUDA")
    seed_everything(97, deterministic=True)
    device = torch.device("cuda")
    count = args.batch_size * args.bank_batches
    weights = torch.randint(1, 100, (count, args.items, args.resources), device=device).double()
    profits = torch.randint(1, 100, (count, args.items), device=device).double()
    env = MKPEnv(profits, weights, weights.sum(1) * .5)
    # Include mid-episode states without constructing any solutions.
    chosen = torch.arange(args.items, device=device)[None] < (torch.arange(count, device=device) % max(1, args.items // 4))[:, None]
    env.state.selected.copy_(chosen)
    env.state.remaining.sub_((weights * chosen[..., None]).sum(1))
    state = env.state
    data = state.profits, state.weights, state.capacities
    rollout = dict(remaining=state.remaining[None], selected=state.selected[None])
    times = torch.zeros(count, dtype=torch.long, device=device)
    batches = torch.arange(count, device=device)
    minibatches = batches.split(args.batch_size)
    model = MKPModel(ModelConfig()).to(device)
    with torch.no_grad():
        model.actor.price_head.weight.normal_(0, .02)
        model.actor.strategy_head.weight.normal_(0, .02)

    results, outputs = {}, {}
    for label, profile, use_bank in (("reference", "reference", False),
                                      ("larger_chunks", "memory24", False),
                                      ("larger_chunks_and_cache", "memory24", True)):
        train = TrainConfig(performance_profile=profile)
        configure_training_runtime(model, train, device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        bank, cache_stats = (build_rollout_cache(model, data, rollout, times, batches, train)
                             if use_bank else (None, dict(feature_cache_gib=0., feature_cache_seconds=0.)))
        if use_bank and bank is None:
            raise RuntimeError("Feature bank skipped: insufficient free memory for this timing probe")

        def step(index):
            model.zero_grad(set_to_none=True)
            inputs = (bank if bank is not None else state).index(minibatches[index % len(minibatches)])
            policy, values, _, auxiliary = model(inputs)
            loss = policy.logits.square().mean() + values.square().mean() + auxiliary.square().mean()
            loss.backward()

        for i in range(args.warmup):
            step(i)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(args.repeats):
            step(i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        with torch.no_grad():
            outputs[label] = model((bank if bank is not None else state).index(minibatches[0]))[0].logits.cpu()
        results[label] = dict(forward_backward_ms=elapsed * 1000 / args.repeats,
                             cache_build_ms=cache_stats["feature_cache_seconds"] * 1000,
                             cache_gib=cache_stats["feature_cache_gib"],
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                             peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
        # Cost of one bank build amortized over the default three PPO epochs.
        results[label]["ms_including_cache_amortization"] = (results[label]["forward_backward_ms"]
            + results[label]["cache_build_ms"] / (len(minibatches) * train.ppo_epochs))
        bank = None
    for label, output in outputs.items():
        torch.testing.assert_close(output, outputs["reference"], rtol=2e-5, atol=2e-5)
        results[label]["max_logit_difference"] = (output - outputs["reference"]).abs().max().item()
    report = dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  shape=dict(B=args.batch_size, N=args.items, M=args.resources),
                  repeats=args.repeats, warmup=args.warmup, bank_batches=args.bank_batches,
                  note="Synthetic forward/backward only; no optimizer updates or full-training speed claim",
                  results=results)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
