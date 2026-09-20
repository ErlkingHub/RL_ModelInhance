"""On-policy PPO for the item distribution induced by high-level strategy theta.

PPO: Schulman et al. (2017), https://arxiv.org/abs/1707.06347.
All complete training episodes and checkpoint selection use synthetic data only.
"""
import argparse
import csv
from datetime import datetime
import logging
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mkp_config import (CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION, ModelConfig, GeneratorConfig,
                        TrainConfig, config_dict)
from mkp_data import synthetic_batch
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel
from mkp_ranking import collect_ranking_batch, ranking_objective, ranking_kl
from mkp_diagnostics import convergence_report, plot_training, prepare_plotting
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
    model.eval()  # no dropout anywhere: collection and PPO recomputation agree
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        policy, value, _, _ = model(env.state)
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
    return result


class MKPTrainer:
    def __init__(self, train=None, model=None, generator=None):
        self.config = train or TrainConfig()
        self.model_config = model or ModelConfig()
        self.generator_config = generator or GeneratorConfig()
        self.device = resolve_device(self.config.device)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.config.runs_dir / stamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
                       "minibatches", "kl_stopped", "validation_return", "validation_improvement_pct",
                       "best_validation_return", "elapsed_seconds"]
        self.fields += ["ranking_proposed", "ranking_pairs", "ranking_abs_return_gap",
                        "ranking_standard_error", "ranking_label_seconds", "ranking_loss",
                        "ranking_accuracy", "ranking_kl", "ranking_minibatches"]
        with (self.run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()
        write_json(self.run_dir / "config.json", {
            "train": config_dict(self.config), "model": config_dict(self.model_config),
            "generator": config_dict(self.generator_config), "torch": str(torch.__version__),
            "device": str(self.device), "training_data_source": "synthetic_only"})

    def ppo_update(self, model, optimizer, data, rollout, update=0):
        c = self.config
        # No parameter update has occurred since collection: this is the frozen
        # behavior policy. Branches never enter the PPO buffer or critic targets.
        ranking, ranking_stats = collect_ranking_batch(model, data, rollout, c,
            seed=c.seed + 1000003 * update + data[2].shape[-1])
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
                                "approx_kl", "clip_fraction", "grad_norm")}
        minibatches, observations, stopped = 0, 0, False
        ranking_sums = dict(ranking_loss=0., ranking_accuracy=0.)
        ranking_updates, max_ranking_kl = 0, 0.
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
                full_ranking_kl = ranking_kl(model, ranking).item() if ranking is not None else 0.
                if not math.isfinite(full_ranking_kl):
                    raise FloatingPointError("Non-finite ranking KL")
                max_ranking_kl = max(max_ranking_kl, full_ranking_kl)
                if max(approximate_kl.item(), full_ranking_kl) > c.target_kl:
                    stopped = True
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
                if ranking is not None:
                    rank_ids = torch.randperm(ranking.left.numel(), device=self.device)[:c.ranking_minibatch_size]
                    rank_loss, _, rank_accuracy = ranking_objective(model, ranking.index(rank_ids), c)
                    loss = loss + c.ranking_coefficient * rank_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                          error_if_nonfinite=True)
                optimizer.step()
                if ranking is not None:
                    ranking_sums["ranking_loss"] += rank_loss.detach().item()
                    ranking_sums["ranking_accuracy"] += rank_accuracy.item()
                    ranking_updates += 1
                metrics = (policy_loss, value_loss, constraint_loss, entropy, approximate_kl,
                           ((ratio - 1).abs() > c.clip_ratio).float().mean(), grad_norm)
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
        if ranking is not None:
            final_kl = ranking_kl(model, ranking).item()
            if not math.isfinite(final_kl):
                raise FloatingPointError("Non-finite final ranking KL")
            max_ranking_kl = max(max_ranking_kl, final_kl)
        result.update(ranking_stats, **{k: v / max(1, ranking_updates) for k, v in ranking_sums.items()},
                      ranking_kl=max_ranking_kl, ranking_minibatches=ranking_updates)
        return result

    def train(self, resume=None):
        if resume is not None and len(self.config.num_resources) != 1:
            raise ValueError("--resume requires exactly one --resources value")
        try:
            prepare_plotting()
            for m in self.config.num_resources:
                self._train_shape(m, resume)
        finally:
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
        history = []
        saved = None
        if resume:
            model, saved = load_checkpoint(resume, self.device, m)
            if saved["model_config"] != config_dict(self.model_config):
                raise ValueError("Resume model config differs; supply the original model settings")
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
            history = saved.get("diagnostics_history", [])
            if not history:
                self.logger.warning("M=%d resume checkpoint has no diagnostics history; convergence evidence starts at update=%d",
                                    m, start)
            if start > c.updates:
                raise ValueError("--updates is the total target and must exceed the resumed update")
        # Fixed held-out synthetic validation has an independent RNG and only training N shapes.
        vrng = np.random.default_rng(c.validation_seed + m)
        validation = [synthetic_batch(c.validation_instances_per_shape, n, m, self.generator_config,
                                     vrng, self.device) for n in c.train_num_items]
        fixed_objectives = [construct_solution(data, mode="fixed", model_config=self.model_config)["objective"]
                            for data in validation]
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
        previous_plateau = None
        self.logger.info("M=%d CONVERGENCE_RULE | synthetic greedy validation within %.4f%% of best | "
                         "span>=%d updates | evaluations>=%d | confirmation update>=%d | training continues to %d",
                         m, c.convergence_tolerance_pct, c.convergence_patience_updates,
                         c.convergence_min_evaluations, c.convergence_min_updates, c.updates)
        for update in range(start, c.updates + 1):
            n = int(rng.choice(c.train_num_items))
            data = synthetic_batch(c.batch_size, n, m, self.generator_config, rng, self.device)
            rollout = collect_rollout(model, data)
            stats = self.ppo_update(model, optimizer, data, rollout, update)
            validation_return, improvement = None, None
            improved = False
            if update == start or update % c.validation_every == 0 or update == c.updates:
                objectives = [construct_solution(d, model)["objective"] for d in validation]
                validation_return = torch.cat([obj / d[0].sum(-1) for obj, d in zip(objectives, validation)]).mean().item()
                improvement = torch.cat([100 * (obj - ref) / ref.clamp_min(1)
                                         for obj, ref in zip(objectives, fixed_objectives)]).mean().item()
                improved = validation_return > best
                best = max(best, validation_return)
            row = dict(M=m, N=n, update=update, transitions=int(rollout["active"].sum()),
                       objective_mean=rollout["objective"].mean().item(),
                       normalized_return=(rollout["objective"] / data[0].sum(-1)).mean().item(),
                       utilization=rollout["utilization"].item(), **stats,
                       validation_return=validation_return, validation_improvement_pct=improvement,
                       best_validation_return=best, elapsed_seconds=time.perf_counter() - began)
            history.append(row)
            with (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writerow(row)
            self.logger.info("M=%d update=%d/%d N=%d reward=%.5f policy=%.5f value=%.5f "
                             "entropy=%.5f KL=%.5f clip=%.5f grad=%.5f explained_variance=%.5f val=%s",
                             m, update, c.updates, n, row["normalized_return"], stats["policy_loss"],
                             stats["value_loss"], stats["entropy"], stats["approx_kl"], stats["clip_fraction"],
                             stats["grad_norm"], stats["explained_variance"], validation_return)
            self.logger.info("M=%d update=%d RANKING | accepted=%d/%d loss=%.5f accuracy=%.4f "
                             "return_gap=%.6f SE=%.6f KL=%.5f label_seconds=%.2f",
                             m, update, stats.get("ranking_pairs", 0), stats.get("ranking_proposed", 0),
                             stats.get("ranking_loss", 0.), stats.get("ranking_accuracy", 0.),
                             stats.get("ranking_abs_return_gap", 0.), stats.get("ranking_standard_error", 0.),
                             stats.get("ranking_kl", 0.), stats.get("ranking_label_seconds", 0.))
            if validation_return is not None:
                report = convergence_report(history, c)
                plateau = (report["plateau_start_update"], report["confirmation_update"])
                if plateau != previous_plateau and (plateau[0] is not None or
                                                     (previous_plateau and previous_plateau[0] is not None)):
                    self.logger.info("M=%d CONVERGENCE_CANDIDATE | at_update=%d status=%s "
                                     "plateau_start_update=%s confirmation_update=%s | provisional; final report supersedes this",
                                     m, update, report["status"], *plateau)
                previous_plateau = plateau
            if improved or update % c.checkpoint_every == 0 or update == c.updates:
                payload = {"checkpoint_version": CHECKPOINT_VERSION, "decoder_version": DECODER_VERSION,
                           "controller_version": CONTROLLER_VERSION,
                           "model_config": config_dict(self.model_config), "train_config": config_dict(c),
                           "generator_config": config_dict(self.generator_config), "num_resources": m,
                           "training_data_source": "synthetic_only", "update": update,
                           "diagnostics_history": history,
                           "best_validation_return": best, "model_state": model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "numpy_generator_state": rng.bit_generator.state,
                           "torch_rng_state": torch.get_rng_state(),
                           "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                           "run_dir": str(self.run_dir), "torch_version": str(torch.__version__)}
                atomic_checkpoint(c.checkpoint_dir / f"mkp_M{m}_last.pt", payload)
                if improved:
                    atomic_checkpoint(c.checkpoint_dir / f"mkp_M{m}_best.pt", payload)
        self.logger.info("M=%s complete | best synthetic validation=%.6f", m, best)
        report = convergence_report(history, c, complete=True)
        report.update(M=m, run_dir=str(self.run_dir), final_update=c.updates)
        write_json(self.run_dir / f"convergence_M{m}.json", report)
        write_json(self.run_dir / f"diagnostics_history_M{m}.json", history)
        self.logger.info("M=%d CONVERGENCE_FINAL | status=%s plateau_start_update=%s confirmation_update=%s "
                         "best_update=%s best_validation_return=%.8f observed_through_update=%s | "
                         "fixed synthetic validation; not a claim about Chu gap convergence",
                         m, report["status"], report["plateau_start_update"], report["confirmation_update"],
                         report["best_update"], report["best_validation_return"], report["observed_through_update"])
        paths = plot_training(history, report, c, self.run_dir / "plots", m)
        self.logger.info("M=%d TRAINING_PLOTS | %s", m, " | ".join(map(str, paths)))


def main(argv=None):
    defaults = TrainConfig()
    model_defaults = ModelConfig()
    parser = argparse.ArgumentParser(description="Train MKP high-level strategies on synthetic data only")
    parser.add_argument("--resources", type=int, nargs="+", default=defaults.num_resources)
    parser.add_argument("--train-items", type=int, nargs="+", default=defaults.train_num_items)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    parser.add_argument("--runs-dir", type=Path, default=defaults.runs_dir)
    parser.add_argument("--ranking-coefficient", type=float, default=defaults.ranking_coefficient)
    parser.add_argument("--ranking-states", type=int, default=defaults.ranking_states)
    parser.add_argument("--ranking-pool-size", type=int, default=defaults.ranking_pool_size)
    parser.add_argument("--ranking-repeats", type=int, default=defaults.ranking_repeats)
    parser.add_argument("--opportunity-weight-max", type=float, default=model_defaults.opportunity_weight_max)
    parser.add_argument("--reserve-weight-max", type=float, default=model_defaults.reserve_weight_max)
    parser.add_argument("--opportunity-tokens", type=int, default=model_defaults.opportunity_tokens)
    parser.add_argument("--actor-edge-values", action=argparse.BooleanOptionalAction,
                        default=model_defaults.actor_edge_values)
    for field in ("actor_postdecision_features", "actor_pair_attention", "actor_contrast_pooling",
                  "actor_feasible_attention"):
        parser.add_argument("--" + field.replace("_", "-"), action=argparse.BooleanOptionalAction,
                            default=getattr(model_defaults, field))
    parser.add_argument("--actor-relation-chunk-size", type=int, default=model_defaults.actor_relation_chunk_size)
    parser.add_argument("--actor-relation-bandwidth", type=float, default=model_defaults.actor_relation_bandwidth)
    parser.add_argument("--actor-similarity-bandwidth", type=float, default=model_defaults.actor_similarity_bandwidth)
    args = parser.parse_args(argv)
    config = TrainConfig(num_resources=tuple(args.resources), train_num_items=tuple(args.train_items),
                         updates=args.updates, batch_size=args.batch_size, minibatch_size=args.minibatch_size,
                         device=args.device, seed=args.seed, checkpoint_dir=args.checkpoint_dir,
                         runs_dir=args.runs_dir, ranking_coefficient=args.ranking_coefficient,
                         ranking_states=args.ranking_states, ranking_pool_size=args.ranking_pool_size,
                         ranking_repeats=args.ranking_repeats)
    model = ModelConfig(opportunity_weight_max=args.opportunity_weight_max,
                        reserve_weight_max=args.reserve_weight_max, opportunity_tokens=args.opportunity_tokens,
                        actor_edge_values=args.actor_edge_values,
                        actor_postdecision_features=args.actor_postdecision_features,
                        actor_pair_attention=args.actor_pair_attention,
                        actor_contrast_pooling=args.actor_contrast_pooling,
                        actor_feasible_attention=args.actor_feasible_attention,
                        actor_relation_chunk_size=args.actor_relation_chunk_size,
                        actor_relation_bandwidth=args.actor_relation_bandwidth,
                        actor_similarity_bandwidth=args.actor_similarity_bandwidth)
    MKPTrainer(config, model).train(args.resume)
