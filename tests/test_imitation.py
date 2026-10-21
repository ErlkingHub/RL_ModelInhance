"""Teacher loss/collection/rollback contracts; every optimizer step is mocked."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv
from mkp_Model import MKPModel, CandidatePolicy
from mkp_imitation import (TeacherExample, TeacherTrainer, teacher_objective, acceptable_step,
                           _sample_states, empty_teacher_metrics, copy_state)
from mkp_teacher import solve_teacher


class ImitationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(93)
        self.config = TrainConfig(teacher_every=1, teacher_states=4, teacher_steps=2,
                                  teacher_minibatch_size=4, teacher_plan_prefixes=2)
        self.model = MKPModel(ModelConfig(embed_dim=16, num_heads=4, num_layers=1))
        self.state = MKPEnv(torch.tensor([[12., 8., 8.]]),
            torch.tensor([[[6., 6.], [10., 0.], [0., 10.]]]), torch.tensor([[10., 10.]])).state
        result = solve_teacher(self.state, self.config)
        self.example = TeacherExample(copy_state(self.state), result.target_mask[None],
            result.action_values[None], result.objective, True, result.source, "ppo", 1)

    def mocked_policy(self, scores):
        policy, theta, aux = self.model.policy(self.state)
        return CandidatePolicy(policy.owners, policy.items, scores / theta.temperature[policy.owners],
                               1, dict(score=scores)), theta, aux

    def test_loss_includes_competitors_outside_a_target_pair(self):
        # Optimal targets are B/C; changing only the unlabelled A competitor must
        # change loss, even though the B-C score difference stays identical.
        low = self.mocked_policy(torch.tensor([-2., 1., 1.], requires_grad=True))
        high = self.mocked_policy(torch.tensor([2., 1., 1.], requires_grad=True))
        with patch.object(self.model, "policy", return_value=low):
            first = teacher_objective(self.model, [self.example], self.config)
        with patch.object(self.model, "policy", return_value=high):
            second = teacher_objective(self.model, [self.example], self.config)
        self.assertGreater(second["loss"].item(), first["loss"].item() + 1.)
        self.assertFalse(second["safe"].item())
        self.assertAlmostEqual(second["regrets"].item(), .25)

    def test_safe_set_accepts_either_optimal_action_without_uniform_imitation(self):
        for scores in ([-2., 2., -2.], [-2., -2., 2.]):
            with patch.object(self.model, "policy", return_value=self.mocked_policy(torch.tensor(scores))):
                stats = teacher_objective(self.model, [self.example], self.config)
            self.assertTrue(stats["safe"].item())
            self.assertLess(stats["loss"].item(), 1e-6)

    def test_raw_score_gradients_reach_actor_not_temperature_or_critic(self):
        stats = teacher_objective(self.model, [self.example], self.config)
        stats["loss"].backward()
        self.assertGreater(self.model.actor.strategy_head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.model.actor.strategy_head.bias.grad[3].item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.critic_encoder.parameters()))
        self.assertIsNone(self.model.value_head.weight.grad)
        before = stats["loss"].detach()
        with torch.no_grad():
            self.model.actor.strategy_head.bias[3].add_(3.)
        torch.testing.assert_close(teacher_objective(self.model, [self.example], self.config)["loss"], before)

    def test_unknown_action_values_are_not_filled_with_zero_regret(self):
        example = copy.deepcopy(self.example)
        example.values.fill_(float("nan"))
        example.certified = False
        stats = teacher_objective(self.model, [example], self.config)
        self.assertTrue(torch.isnan(stats["regrets"]).all())
        self.assertTrue(torch.isfinite(stats["loss"]))

    def test_progress_sampling_covers_early_middle_late_without_global_rng_changes(self):
        state = self.state
        rollout = dict(active=torch.ones(9, 1, dtype=torch.bool),
            remaining=state.remaining[None].repeat(9, 1, 1),
            selected=state.selected[None].repeat(9, 1, 1))
        rng = torch.get_rng_state().clone()
        selected = _sample_states((state.profits, state.weights, state.capacities), rollout, 6,
                                  torch.Generator().manual_seed(17), "ppo")
        self.assertEqual(sorted(x[2] for x in selected), [0, 0, 1, 1, 2, 2])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_collection_keeps_targets_detached_and_feasible_plans_distinct(self):
        trainer = TeacherTrainer(self.model, self.config)
        state = self.state
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool), remaining=state.remaining[None],
                       selected=state.selected[None])
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        metrics = empty_teacher_metrics()
        examples = trainer.collect((state.profits, state.weights, state.capacities), rollout, 1, metrics)
        self.assertGreater(len(examples), 0)
        self.assertGreater(metrics["teacher_greedy_states"], 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, weights[k]) for k, v in self.model.state_dict().items()))
        for example in examples:
            self.assertFalse(example.values.requires_grad)
            self.assertTrue((example.target <= example.state.feasible).all())
            if example.origin == "plan":
                self.assertTrue((example.state.weights[0, example.target[0]].sum(0)
                                 <= example.state.remaining[0]).all())

    @staticmethod
    def stats(loss=1., regret=.2, safe=False):
        return dict(loss=torch.tensor(loss), regrets=torch.tensor([regret]), safe=torch.tensor([safe]),
                    log_probs=[torch.tensor([-.693147, -.693147])],
                    raw_log_probs=[torch.tensor([-.693147, -.693147])])

    def test_guard_requires_retention_but_accepts_progress_before_argmax_changes(self):
        before = self.stats()
        self.assertTrue(acceptable_step(before, self.stats(loss=.9), before, .05)[0])
        self.assertFalse(acceptable_step(before, self.stats(loss=.9, regret=.3), before, .05)[0])
        self.assertFalse(acceptable_step(before, self.stats(loss=1.1), before, .05)[0])
        safe = self.stats(regret=0., safe=True)
        self.assertFalse(acceptable_step(safe, self.stats(loss=.9, regret=0.), safe, .05)[0])
        extreme = self.stats(loss=.9)
        extreme["raw_log_probs"] = [torch.tensor([0., -20.])]
        self.assertFalse(acceptable_step(before, extreme, before, .05)[0])

    def test_rejected_step_restores_actor_and_adam_state(self):
        trainer = TeacherTrainer(self.model, self.config)
        parameter = self.model.actor.price_head.weight
        trainer.optimizer.state[parameter]["fixture"] = torch.tensor(2.)
        trainer.optimizer.state[parameter]["step"] = torch.tensor(1.)
        trainer.optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
        trainer.optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        def fake_step():
            with torch.no_grad():
                parameter.add_(.01)
            trainer.optimizer.state[parameter]["fixture"].fill_(99.)
        with patch.object(trainer, "collect", return_value=[self.example]), \
             patch.object(trainer.optimizer, "step", side_effect=fake_step) as step, \
             patch("mkp_imitation.acceptable_step", return_value=(False, 1.)):
            stats = trainer.update((self.state.profits, self.state.weights, self.state.capacities), {}, 1)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(stats["teacher_rejected"], 1)
        self.assertEqual(stats["teacher_updates"], 0)
        self.assertEqual(trainer.optimizer.state[parameter]["fixture"].item(), 2.)
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in self.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.critic_encoder.parameters()))

    def test_only_certified_replay_survives_checkpoint_and_disabled_mode_is_idle(self):
        trainer = TeacherTrainer(self.model, self.config)
        trainer.replay = [self.example]
        payload = trainer.state_dict()
        restored = TeacherTrainer(self.model, self.config)
        restored.load_state_dict(payload)
        torch.testing.assert_close(restored.replay[0].values, self.example.values, equal_nan=True)
        payload["replay"][0]["certified"] = False
        with self.assertRaisesRegex(ValueError, "Uncertified"):
            restored.load_state_dict(payload)
        idle = TeacherTrainer(self.model, replace(self.config, supervision_mode="legacy"))
        with patch.object(idle, "collect", side_effect=AssertionError("disabled")), \
             patch.object(idle.optimizer, "step", side_effect=AssertionError("disabled")):
            self.assertEqual(idle.update(None, None, 1)["teacher_updates"], 0)


if __name__ == "__main__":
    unittest.main()
