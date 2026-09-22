"""Tiny counterfactual/gradient checks. Optimizer steps are replaced by a no-op."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_config import ModelConfig, TrainConfig, BenchmarkConfig, TRAINING_VERSION
from mkp_env import MKPEnv
from mkp_Model import MKPModel
from mkp_ranking import (THETA_FIELDS, actor_hash, collect_ranking_batch, deduplicate_actions,
                         paired_terminal_returns, preference_pairs, ranking_objective,
                         theta_proposals, verify_ranking_record, write_ranking_audit)
from mkp_Trainer import MKPTrainer, main
from mkp_utils import seed_everything


def critical_env(device="cpu"):
    # D was already selected; Q must exclude its 7 profit. A yields 100, B+C yields 160.
    env = MKPEnv(torch.tensor([[100., 100., 60., 7.]], device=device),
                 torch.tensor([[[52., 50.], [50., 52.], [50., 48.], [1., 1.]]], device=device),
                 torch.tensor([[101., 101.]], device=device))
    env.step(torch.tensor([3], device=device))
    return env


def visited_rollout(model, env):
    state = env.state
    with torch.no_grad():
        policy, value, _, _ = model(state)
        actions = policy.choose()
        branch = MKPEnv.from_state(state)
        rewards, done = branch.step(actions)
    return {"active": state.active[None], "remaining": state.remaining[None].clone(),
            "selected": state.selected[None].clone(), "actions": actions[None],
            "log_probs": policy.log_prob(actions)[None], "values": value[None],
            "rewards": rewards[None], "dones": done[None],
            "objective": branch.objective, "utilization": torch.tensor(.8, device=state.profits.device)}


class NoStepOptimizer:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.calls += 1


class ThetaRankingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)
        self.mc = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)
        self.tc = TrainConfig(ranking_states=1, ranking_theta_candidates=12, ranking_max_actions=4,
                              ranking_rollout_batch_size=2, ppo_epochs=1, minibatch_size=4, device="cpu")
        self.model = MKPModel(self.mc)

    def batch(self, model=None, config=None, device="cpu"):
        model = model or self.model
        env = critical_env(device)
        data = (env.state.profits, env.state.weights, env.state.capacities)
        rollout = visited_rollout(model, env)
        return collect_ranking_batch(model, data, rollout, config or self.tc, seed=719)

    def test_theta_proposals_legal_at_m5_m10_and_degenerate_ranges(self):
        for m in (5, 10):
            for config in (self.mc, replace(self.mc, scarcity_exponent_max=0, bottleneck_weight_max=0,
                                            profit_exponent_min=1., profit_exponent_max=1.)):
                model = MKPModel(config)
                state = MKPEnv(torch.ones(2, 3), torch.ones(2, 3, m), torch.ones(2, m) * 2).state
                current = model.actor(state)[0].detach()
                proposals = theta_proposals(current, config, 12, 10., torch.Generator().manual_seed(4))
                prices = proposals.resource_prices
                self.assertTrue((prices >= config.price_floor - 1e-6).all())
                self.assertTrue(torch.allclose(prices.mean(-1), torch.ones(24), atol=1e-6))
                for name, lo, hi in (("profit_exponent", config.profit_exponent_min, config.profit_exponent_max),
                                    ("scarcity_exponent", 0, config.scarcity_exponent_max),
                                    ("bottleneck_weight", 0, config.bottleneck_weight_max)):
                    value = getattr(proposals, name)
                    self.assertTrue(((value >= lo) & (value <= hi)).all())
                for name in THETA_FIELDS:
                    self.assertTrue(torch.equal(getattr(proposals, name)[::12], getattr(current, name)))
                self.assertTrue(torch.equal(proposals.temperature, current.temperature.repeat_interleave(12)))

    def test_action_deduplication_preserves_current_and_first_witness(self):
        actions, witnesses = deduplicate_actions(torch.tensor([[2, 2, 1, 3, 1, 0], [1, 1, 1, 1, 1, 1]]), 3)
        self.assertEqual(actions.tolist(), [[2, 1, 3], [1, -1, -1]])
        self.assertEqual(witnesses.tolist(), [[0, 2, 3], [0, -1, -1]])

    def test_true_successors_additional_profit_and_source_isolation(self):
        env = critical_env()
        before = deepcopy(env.state)
        teacher = deepcopy(self.model).eval().requires_grad_(False)
        visited = []
        policy = teacher.policy

        def inspect(state):
            visited.append(deepcopy(state))
            return policy(state)

        with patch.object(teacher, "policy", side_effect=inspect):
            returns, sequences = paired_terminal_returns(teacher, env.state, torch.tensor([[0, 1]]), self.tc, 17)
        self.assertEqual(returns.flatten().tolist(), [100., 160.])
        self.assertEqual(sequences[0, 1, 0, :2].tolist(), [1, 2])
        after_b = next(state for state in visited if state.selected[:, 1].any())
        index = after_b.selected[:, 1].nonzero(as_tuple=True)[0][0]
        self.assertEqual(after_b.remaining[index].tolist(), [50., 48.])
        self.assertTrue(after_b.selected[index, 3])
        self.assertTrue(after_b.feasible[index, 2])
        self.assertFalse(after_b.feasible[index, 0])
        for field in ("profits", "weights", "capacities", "remaining", "selected"):
            self.assertTrue(torch.equal(getattr(env.state, field), getattr(before, field)))

    def test_collection_freezes_teacher_and_does_not_consume_ppo_rng(self):
        before_rng = torch.get_rng_state().clone()
        before_model = deepcopy(self.model.state_dict())
        batch = self.batch()
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in before_model.items()))
        self.assertEqual(batch.teacher_sha256, actor_hash(self.model))
        self.assertEqual(batch.source_times.tolist(), [0])
        self.assertEqual(batch.source_batches.tolist(), [0])
        self.assertGreater(batch.pair_states.numel(), 0)
        self.assertEqual(batch.metrics["ranking_gain_raw"], 60.)
        self.assertTrue(all(not getattr(batch.proposal_theta, name).requires_grad for name in THETA_FIELDS))
        self.assertFalse(batch.returns.requires_grad)
        actual_rollout = paired_terminal_returns

        def inspect(teacher, *args):
            self.assertIsNot(teacher, self.model)
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            old_hash = actor_hash(teacher)
            with torch.no_grad():
                self.model.actor.strategy_head.bias.add_(.3)
            self.assertEqual(actor_hash(teacher), old_hash)
            return actual_rollout(teacher, *args)

        with patch("mkp_ranking.paired_terminal_returns", side_effect=inspect):
            self.batch()

    def test_sampled_noise_is_shared_across_actions_and_chunk_sizes(self):
        state = MKPEnv(torch.tensor([[1., 1.1, 1.2, 1.3, 1.4]]),
                       torch.ones(1, 5, 2), torch.ones(1, 2) * 3).state
        teacher = MKPModel(replace(self.mc, temperature_min=1., temperature_max=1.)).eval().requires_grad_(False)
        actions = torch.tensor([[0, 0, 1]])  # repeated action deliberately checks shared noise
        c = replace(self.tc, ranking_continuation="sampled", ranking_repeats=8)
        a, a_seq = paired_terminal_returns(teacher, state, actions, replace(c, ranking_rollout_batch_size=1), 91)
        b, b_seq = paired_terminal_returns(teacher, state, actions, replace(c, ranking_rollout_batch_size=64), 91)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a_seq, b_seq))
        self.assertTrue(torch.equal(a[:, 0], a[:, 1]))
        self.assertTrue(torch.equal(a_seq[:, 0], a_seq[:, 1]))
        self.assertGreater(a_seq[0, 0].unique(dim=0).shape[0], 1)

    def test_pair_filter_uses_paired_se_and_excludes_ties(self):
        actions = torch.tensor([[0, 1], [0, 1], [0, 1]])
        q = torch.tensor([[[110., 210.], [100., 200.]],
                          [[110., 90.], [100., 100.]], [[100., 100.], [100., 100.]]])
        rows, winner, loser, gap, se = preference_pairs(actions, q, torch.ones(3) * 1000, self.tc)
        self.assertEqual(rows.tolist(), [0])
        self.assertEqual(winner.tolist(), [0])
        self.assertEqual(loser.tolist(), [1])
        self.assertAlmostEqual(gap.item(), .01)
        self.assertEqual(se.item(), 0.)

    def test_ranking_is_temperature_and_theta_witness_independent(self):
        batch = self.batch()
        loss, _ = ranking_objective(self.model, batch, self.tc)
        with torch.no_grad():
            self.model.actor.strategy_head.bias[3] += 4
        for name in THETA_FIELDS:
            getattr(batch.proposal_theta, name).fill_(1234.)
        changed, _ = ranking_objective(self.model, batch, self.tc)
        self.assertEqual(loss.item(), changed.item())
        changed.backward()
        self.assertEqual(self.model.actor.strategy_head.bias.grad[3].item(), 0.)
        self.assertEqual(self.model.actor.strategy_head.weight.grad[3].abs().sum().item(), 0.)

    def test_ranking_backpropagates_through_scores_to_encoder_only(self):
        batch = self.batch()
        # v1 initializes output matrices to zero; a nonzero head exercises later-step encoder gradients.
        with torch.no_grad():
            self.model.actor.price_head.weight.normal_(0, .01)
            self.model.actor.strategy_head.weight.normal_(0, .01)
        loss, stats = ranking_objective(self.model, batch, self.tc)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.model.actor.price_head.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(self.model.actor.encoder.item_input.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.critic_encoder.parameters()))
        self.assertIsNone(self.model.value_head.weight.grad)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertGreaterEqual(stats["ranking_kl"], 0.)

    def test_full_candidate_kl_detects_unlabelled_competitor_probability_shift(self):
        batch = self.batch()
        initial = self.model.policy(batch.state)[0]
        with torch.no_grad():
            self.model.actor.strategy_head.bias[0] = 6.  # changes C mass; A/B ratio stays identical
        changed = self.model.policy(batch.state)[0]
        self.assertAlmostEqual((initial.log_probs[0] - initial.log_probs[1]).item(),
                               (changed.log_probs[0] - changed.log_probs[1]).item(), places=6)
        self.assertGreater(ranking_objective(self.model, batch, self.tc)[1]["ranking_kl"], 1e-4)

    def test_no_pairs_and_disabled_collection(self):
        batch = self.batch()
        no_pairs = replace(batch, pair_states=batch.pair_states[:0], winners=batch.winners[:0],
                           losers=batch.losers[:0], gaps=batch.gaps[:0], standard_errors=batch.standard_errors[:0])
        loss, _ = ranking_objective(self.model, no_pairs, self.tc)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        with patch("mkp_ranking.deepcopy", side_effect=AssertionError("Disabled ranking must not freeze/roll out")):
            self.assertIsNone(collect_ranking_batch(None, None, None, replace(self.tc, ranking_coefficient=0), 1))

    def test_audit_replay_checks_witness_feasibility_and_raw_profit(self):
        batch = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            write_ranking_audit(path, batch, 1, 2, self.tc)
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(verify_ranking_record(row), int((batch.actions >= 0).sum()))
            row["candidates"][0]["additional_profit"][0] += 1
            with self.assertRaisesRegex(ValueError, "profit"):
                verify_ranking_record(row)

    def test_ppo_combines_ranking_gradients_without_any_optimizer_update(self):
        batch = self.batch()
        env = critical_env()
        data = (env.state.profits, env.state.weights, env.state.capacities)
        rollout = visited_rollout(self.model, env)
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config = replace(self.tc, entropy_coefficient=0, value_coefficient=0, constraint_aux_coefficient=0)
        trainer.device = torch.device("cpu")
        optimizer = NoStepOptimizer(self.model)
        before = deepcopy(self.model.state_dict())
        result = trainer.ppo_update(self.model, optimizer, data, rollout, batch)
        self.assertEqual(optimizer.calls, 1)
        self.assertGreater(result["ranking_loss"], 0.)
        self.assertGreater(self.model.actor.price_head.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(torch.equal(rollout["actions"], torch.tensor([[0]])))

    def test_ranking_full_kl_stops_before_optimizer_step(self):
        batch = self.batch()
        env = critical_env()
        with torch.no_grad():
            self.model.actor.strategy_head.bias[0] = 6.
        # PPO old probabilities match student, isolating the ranking KL trigger.
        rollout = visited_rollout(self.model, env)
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config = replace(self.tc, ranking_target_kl=1e-6)
        trainer.device = torch.device("cpu")
        optimizer = NoStepOptimizer(self.model)
        result = trainer.ppo_update(self.model, optimizer,
                                    (env.state.profits, env.state.weights, env.state.capacities), rollout, batch)
        self.assertEqual(optimizer.calls, 0)
        self.assertEqual(result["ranking_kl_stopped"], 1)
        self.assertEqual(result["minibatches"], 0)

    def test_cli_configuration_and_legacy_checkpoint_defaults(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--ranking-coefficient", "0", "--ranking-states", "3", "--ranking-theta-candidates", "7",
                  "--ranking-max-actions", "2", "--ranking-continuation", "sampled", "--ranking-repeats", "5",
                  "--no-ranking-audit", "--checkpoint-dir", "checkpoints/test"])
            config = trainer.call_args.args[0]
            self.assertEqual(config.ranking_coefficient, 0.)
            self.assertEqual(config.ranking_states, 3)
            self.assertEqual(config.ranking_theta_candidates, 7)
            self.assertEqual(config.ranking_repeats, 5)
            self.assertFalse(config.ranking_audit)
            self.assertEqual(config.checkpoint_dir, Path("checkpoints/test"))
        self.assertEqual(TrainConfig().checkpoint_dir, BenchmarkConfig().checkpoint_dir)
        for kwargs in (dict(ranking_states=0), dict(ranking_theta_candidates=1), dict(ranking_max_actions=1),
                       dict(ranking_continuation="sampled", ranking_repeats=1), dict(ranking_score_scale=0),
                       dict(ranking_coefficient=float("nan")), dict(gamma=.9)):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    def test_training_schedule_audit_metadata_and_resume_with_mocked_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(self.tc, num_resources=(2,), train_num_items=(4,), updates=4, ranking_every=2,
                             batch_size=1, validation_every=1, validation_instances_per_shape=1,
                             checkpoint_every=1, runs_dir=root / "runs", checkpoint_dir=root / "checkpoints")
            env = critical_env()
            data = (env.state.profits, env.state.weights, env.state.capacities)
            fake_metrics = {key: .1 for key in ("policy_loss", "value_loss", "constraint_aux_loss", "entropy",
                                               "approx_kl", "clip_fraction", "grad_norm", "explained_variance")}
            fake_metrics.update(minibatches=0, kl_stopped=0)
            with patch("mkp_Trainer.synthetic_batch", return_value=data), \
                 patch("mkp_Trainer.collect_rollout", side_effect=lambda model, data: visited_rollout(model, env)), \
                 patch("mkp_Trainer.construct_solution", return_value={"objective": torch.tensor([160.])}), \
                 patch.object(MKPTrainer, "ppo_update", return_value=fake_metrics), \
                 patch("mkp_Trainer.plot_training", return_value=[]):
                trainer = MKPTrainer(config, self.mc)
                trainer.train()
                records = [json.loads(line) for line in (trainer.run_dir / "counterfactual_M2.jsonl").read_text().splitlines()]
                self.assertEqual([r["update"] for r in records], [1, 2, 4])
                for record in records:
                    verify_ranking_record(record)
                checkpoint = config.checkpoint_dir / "mkp_M2_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["training_version"], TRAINING_VERSION)
                self.assertEqual(saved["train_config"]["ranking_theta_candidates"], config.ranking_theta_candidates)
                resumed = MKPTrainer(replace(config, updates=6), self.mc)
                resumed.train(checkpoint)
                records = [json.loads(line) for line in (resumed.run_dir / "counterfactual_M2.jsonl").read_text().splitlines()]
                self.assertEqual([r["update"] for r in records], [6])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_counterfactual_and_deterministic_ranking_gradient(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(3, deterministic=True, cpu_threads=2)
            model = MKPModel(self.mc).cuda()
            batch = self.batch(model, device="cuda")
            self.assertGreater(batch.pair_states.numel(), 0)
            loss, _ = ranking_objective(model, batch, self.tc)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
