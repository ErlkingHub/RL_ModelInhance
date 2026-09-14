"""On-policy PPO for the item distribution induced by high-level strategy theta.

PPO: Schulman et al. (2017), https://arxiv.org/abs/1707.06347.
All complete training episodes and checkpoint selection use synthetic data only.
"""
import argparse
import csv
from datetime import datetime
import logging
import time
import numpy as np
import torch
from torch.nn import functional as F
from mkp_config import (CHECKPOINT_VERSION, ModelConfig, GeneratorConfig,
                        TrainConfig, config_dict, variant_name, decoder_version)
from mkp_data import synthetic_batch
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, strategy_statistics
from mkp_diagnostics import create_training_diagnostics, critic_diagnostics
from mkp_utils import (atomic_checkpoint, construct_solution, load_checkpoint, resolve_device,
                       seed_everything, write_json)


def generalized_advantage(rewards, values, dones, active, gamma, gae_lambda):
    """[T,B] complete episodes; padding and true terminations never bootstrap."""
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros_like(values[0])
    gae = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        continuation = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * continuation - values[t]
        gae = (delta + gamma * gae_lambda * continuation * gae) * active[t]
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


@torch.no_grad()
def collect_rollout(model, data):
    env = MKPEnv(*data)
    records = {key: [] for key in ("remaining", "selected", "actions", "log_probs",
                                  "values", "rewards", "dones", "active")}
    diagnostic_sum = torch.zeros(7, device=data[0].device)
    diagnostic_count = torch.zeros((), device=data[0].device)
    model.eval()  # no dropout anywhere: collection and PPO recomputation agree
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        policy, value, theta, _ = model(env.state)
        diagnostics = torch.cat((policy.entropy()[:, None], policy.normalized_entropy()[:, None],
            strategy_statistics(theta, data[0].shape[0], data[0].device)), dim=-1)
        diagnostic_sum += (diagnostics * active[:, None]).sum(0)
        diagnostic_count += active.sum()
        actions = policy.choose(stochastic=True)
        records["remaining"].append(env.state.remaining.clone())
        records["selected"].append(env.state.selected.clone())
        records["actions"].append(actions)
        records["log_probs"].append(policy.log_prob(actions))
        records["values"].append(value)
        records["active"].append(active)
        reward, done = env.step(actions)
        records["rewards"].append(reward)
        records["dones"].append(done)
    if not records["actions"]:
        raise ValueError("Synthetic batch has no feasible item; adjust the generator")
    if env.state.active.any():
        raise RuntimeError("Rollout exceeded N steps")
    result = {k: torch.stack(v) for k, v in records.items()}
    result["objective"] = env.objective
    result["utilization"] = (1 - env.state.remaining / env.state.capacities).mean()
    result["diagnostics"] = dict(zip(
        ("rollout_entropy_raw", "rollout_entropy", "theta_alpha", "theta_beta",
         "theta_kappa", "theta_temperature", "theta_price_std"),
        (diagnostic_sum / diagnostic_count.clamp_min(1)).tolist()))
    result["diagnostics"]["selected_fraction"] = env.state.selected.float().mean().item()
    return result


class MKPTrainer:
    def __init__(self, train=None, model=None, generator=None):
        self.config = train or TrainConfig()
        self.model_config = model or ModelConfig()
        self.generator_config = generator or GeneratorConfig()
        self.device = resolve_device(self.config.device)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.config.runs_dir / variant_name(self.model_config) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.config.checkpoint_dir / variant_name(self.model_config)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"mkp.{stamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(self.run_dir / "train.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.fields = ["M", "N", "update", "transitions", "objective_mean", "normalized_return",
                       "utilization", "policy_loss", "value_loss", "constraint_aux_loss",
                       "entropy", "approx_kl", "clip_fraction", "grad_norm", "explained_variance",
                       "minibatches", "kl_stopped", "validation_return", "validation_objective_mean",
                       "best_validation_return", "elapsed_seconds"]
        self.fields += ["total_loss", "entropy_raw", "rollout_entropy", "rollout_entropy_raw",
                        "stop_kl", "ppo_update_fraction", "critic_mc_rmse", "critic_mc_ev",
                        "selected_fraction", "theta_alpha", "theta_beta", "theta_kappa",
                        "theta_temperature", "theta_price_std"]
        with (self.run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()
        write_json(self.run_dir / "config.json", {
            "train": config_dict(self.config), "model": config_dict(self.model_config),
            "generator": config_dict(self.generator_config), "torch": str(torch.__version__),
            "device": str(self.device), "training_data_source": "synthetic_only"})

    def ppo_update(self, model, optimizer, data, rollout):
        c = self.config
        advantages, returns = generalized_advantage(rollout["rewards"], rollout["values"],
            rollout["dones"], rollout["active"], c.gamma, c.gae_lambda)
        times, batches = rollout["active"].nonzero(as_tuple=True)
        adv = advantages[times, batches]
        normalized_adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        targets = returns[times, batches]
        old_values = rollout["values"][times, batches]
        old_log_probs = rollout["log_probs"][times, batches]
        actions = rollout["actions"][times, batches]
        model.train()
        sums = {k: 0.0 for k in ("policy_loss", "value_loss", "constraint_aux_loss", "entropy",
                                "approx_kl", "clip_fraction", "grad_norm", "total_loss", "entropy_raw")}
        minibatches, observations, stopped = 0, 0, False
        stop_kl = None
        for _ in range(c.ppo_epochs):
            order = torch.randperm(times.numel(), device=self.device)
            for ids in order.split(c.minibatch_size):
                b, t = batches[ids], times[ids]
                state = MKPState(data[0][b], data[1][b], data[2][b],
                                 rollout["remaining"][t, b], rollout["selected"][t, b])
                policy, values, _, constraint_logits = model(state)
                log_ratio = policy.log_prob(actions[ids]) - old_log_probs[ids]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approximate_kl):
                    raise FloatingPointError("Non-finite PPO probability ratio")
                if approximate_kl.item() > c.target_kl:
                    stopped = True
                    stop_kl = approximate_kl.item()
                    break
                surrogate = ratio * normalized_adv[ids]
                clipped = ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio) * normalized_adv[ids]
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                clipped_values = old_values[ids] + (values - old_values[ids]).clamp(-c.value_clip, c.value_clip)
                value_loss = 0.5 * torch.maximum((values - targets[ids]).square(),
                                                (clipped_values - targets[ids]).square()).mean()
                constraint_loss = F.binary_cross_entropy_with_logits(constraint_logits, state.feasible.float())
                entropy = policy.normalized_entropy().mean()
                loss = (policy_loss + c.value_coefficient * value_loss
                        + c.constraint_aux_coefficient * constraint_loss - c.entropy_coefficient * entropy)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                          error_if_nonfinite=True)
                optimizer.step()
                metrics = (policy_loss, value_loss, constraint_loss, entropy, approximate_kl,
                           ((ratio - 1).abs() > c.clip_ratio).float().mean(), grad_norm,
                           loss, policy.entropy().mean())
                for key, val in zip(sums, metrics):
                    sums[key] += val.detach().item() * ids.numel()
                observations += ids.numel()
                minibatches += 1
            if stopped:
                break
        variance = targets.var(unbiased=False)
        explained = (1 - (targets - old_values).var(unbiased=False) / variance).item() if variance > 1e-12 else float("nan")
        result = {k: v / max(1, observations) for k, v in sums.items()}
        result.update(minibatches=minibatches, kl_stopped=int(stopped), explained_variance=explained)
        result.update(stop_kl=stop_kl, ppo_update_fraction=observations / (c.ppo_epochs * times.numel()))
        result.update(critic_diagnostics(rollout["rewards"], rollout["values"],
                                         rollout["dones"], rollout["active"], c.gamma))
        return result

    def _plot(self):
        """Plot persisted CSV only; failures do not discard trained weights or log rows."""
        try:
            create_training_diagnostics(self.run_dir, self.config.plot_smoothing_window,
                                        self.config.plot_dpi)
        except Exception:
            self.logger.exception("Training plots failed; metrics.csv remains available for offline plotting")

    def train(self, resume=None):
        if resume is not None and len(self.config.num_resources) != 1:
            raise ValueError("--resume requires exactly one --resources value")
        try:
            for m in self.config.num_resources:
                self._train_shape(m, resume)
        finally:
            self._plot()
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

    def _train_shape(self, m, resume):
        c = self.config
        seed_everything(c.seed + m, c.deterministic, c.cpu_threads)
        rng = np.random.default_rng(c.seed + m)
        model = MKPModel(self.model_config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.0)
        start, best = 1, -float("inf")
        saved = None
        if resume:
            model, saved = load_checkpoint(resume, self.device, m)
            if config_dict(model.config) != config_dict(self.model_config):
                raise ValueError("Resume model config differs (encoder/head included); start a fresh "
                                 "BGNN experiment or select the original architecture explicitly.")
            if saved["generator_config"] != config_dict(self.generator_config):
                raise ValueError("Resume generator config differs")
            old = saved["train_config"]
            for key in ("train_num_items", "validation_seed", "validation_instances_per_shape", "seed"):
                if old[key] != config_dict(c)[key]:
                    raise ValueError(f"Resume requires unchanged {key}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.0)
            optimizer.load_state_dict(saved["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = c.learning_rate
            start, best = saved["update"] + 1, saved["best_validation_return"]
            if start > c.updates:
                raise ValueError("--updates is the total target and must exceed the resumed update")
        # Fixed held-out synthetic validation has an independent RNG and only training N shapes.
        vrng = np.random.default_rng(c.validation_seed + m)
        validation = [synthetic_batch(c.validation_instances_per_shape, n, m, self.generator_config,
                                     vrng, self.device) for n in c.train_num_items]
        if saved:
            rng.bit_generator.state = saved["numpy_generator_state"]
            torch.set_rng_state(saved["torch_rng_state"])
            if self.device.type == "cuda" and saved["cuda_rng_state"]:
                if len(saved["cuda_rng_state"]) != torch.cuda.device_count():
                    raise ValueError("Resume CUDA device count differs from the saved RNG state")
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        self.logger.info("M=%s | device=%s | parameters=%s | synthetic N=%s | updates=%s..%s",
                         m, self.device, sum(p.numel() for p in model.parameters()), c.train_num_items, start, c.updates)
        began = time.perf_counter()
        for update in range(start, c.updates + 1):
            n = int(rng.choice(c.train_num_items))
            data = synthetic_batch(c.batch_size, n, m, self.generator_config, rng, self.device)
            rollout = collect_rollout(model, data)
            stats = self.ppo_update(model, optimizer, data, rollout)
            validation_return, validation_objective = None, None
            improved = False
            if update == start or update % c.validation_every == 0 or update == c.updates:
                objectives = [construct_solution(d, model)["objective"] for d in validation]
                validation_return = torch.cat([obj / d[0].sum(-1) for obj, d in zip(objectives, validation)]).mean().item()
                validation_objective = torch.cat(objectives).mean().item()
                improved = validation_return > best
                best = max(best, validation_return)
            row = dict(M=m, N=n, update=update, transitions=int(rollout["active"].sum()),
                       objective_mean=rollout["objective"].mean().item(),
                       normalized_return=(rollout["objective"] / data[0].sum(-1)).mean().item(),
                       utilization=rollout["utilization"].item(), **stats, **rollout["diagnostics"],
                       validation_return=validation_return, validation_objective_mean=validation_objective,
                       best_validation_return=best, elapsed_seconds=time.perf_counter() - began)
            with (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow(row)
            self.logger.info("M=%d update=%d/%d N=%d reward=%.5f policy=%.5f value=%.5f KL=%.5f val=%s",
                             m, update, c.updates, n, row["normalized_return"], stats["policy_loss"],
                             stats["value_loss"], stats["approx_kl"], validation_return)
            if improved or update % c.checkpoint_every == 0 or update == c.updates:
                payload = {"checkpoint_version": CHECKPOINT_VERSION, "decoder_version": decoder_version(self.model_config),
                           "model_config": config_dict(self.model_config), "train_config": config_dict(c),
                           "generator_config": config_dict(self.generator_config), "num_resources": m,
                           "training_data_source": "synthetic_only", "update": update,
                           "best_validation_return": best, "model_state": model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "numpy_generator_state": rng.bit_generator.state,
                           "torch_rng_state": torch.get_rng_state(),
                           "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                           "run_dir": str(self.run_dir), "torch_version": str(torch.__version__)}
                atomic_checkpoint(self.checkpoint_dir / f"mkp_M{m}_last.pt", payload)
                if improved:
                    atomic_checkpoint(self.checkpoint_dir / f"mkp_M{m}_best.pt", payload)
            if update == start or update % c.plot_every == 0 or update == c.updates:
                self._plot()
        self.logger.info("M=%s complete | best synthetic validation=%.6f", m, best)


def main(argv=None):
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train MKP high-level strategies on synthetic data only")
    parser.add_argument("--resources", type=int, nargs="+", default=defaults.num_resources)
    parser.add_argument("--train-items", type=int, nargs="+", default=defaults.train_num_items)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--encoder", choices=("bgnn", "transformer"), default=ModelConfig().encoder_type)
    parser.add_argument("--policy-head", choices=("weak", "direct_item"), default=ModelConfig().policy_head)
    parser.add_argument("--compare-heads", action="store_true", help="Train both heads independently with matched settings")
    args = parser.parse_args(argv)
    config = TrainConfig(num_resources=tuple(args.resources), train_num_items=tuple(args.train_items),
                         updates=args.updates, batch_size=args.batch_size, minibatch_size=args.minibatch_size,
                         device=args.device, seed=args.seed)
    if args.resume and args.compare_heads:
        parser.error("--resume cannot be combined with --compare-heads")
    for head in (("weak", "direct_item") if args.compare_heads else (args.policy_head,)):
        MKPTrainer(config, ModelConfig(encoder_type=args.encoder, policy_head=head)).train(args.resume)
