"""Actor-control CLI/resume/PPO wiring; no actual PPO optimizer step."""
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import torch

from mkp_actor_guard import ACTOR_GUARD_FIELDS
from mkp_config import ModelConfig, TrainConfig, config_dict
from mkp_Model import MKPModel
from mkp_Trainer import MKPTrainer, main, validate_actor_resume
from test_contracts import toy_state


class ActorUpdateWiringTests(unittest.TestCase):
    def test_training_cli_uses_explicit_actor_controls_and_keeps_critic_rate(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--actor-learning-rate", "1e-5", "--actor-continuation-guard",
                  "--actor-guard-states", "6", "--actor-guard-tolerance", "0.0001"])
        config = trainer.call_args.args[0]
        self.assertEqual(config.actor_learning_rate, 1e-5)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertTrue(config.actor_continuation_guard)
        self.assertEqual(config.actor_guard_states, 6)
        self.assertEqual(config.actor_guard_tolerance, 1e-4)

    def test_invalid_actor_controls_fail_before_an_experiment(self):
        for options in (dict(actor_learning_rate=0), dict(actor_learning_rate=float("nan")),
                        dict(actor_guard_states=0), dict(actor_guard_tolerance=-1),
                        dict(actor_guard_tolerance=float("inf"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainConfig(**options)

    def test_legacy_resume_and_matching_controls_work_but_silent_switch_fails(self):
        original = TrainConfig()
        validate_actor_resume(original, {})
        guarded = replace(original, actor_learning_rate=1e-5, actor_continuation_guard=True)
        validate_actor_resume(guarded, config_dict(guarded))
        for current, saved in ((guarded, {}), (original, config_dict(guarded)),
                               (replace(guarded, actor_guard_states=2), config_dict(guarded))):
            with self.assertRaisesRegex(ValueError, "Resume requires unchanged actor_"):
                validate_actor_resume(current, saved)

    def test_ppo_applies_guard_after_attempts_and_marks_actor_rollback(self):
        torch.set_num_threads(2)
        state = toy_state().state.index(torch.tensor([0]))
        model = MKPModel(ModelConfig(embed_dim=16, num_heads=4, num_layers=1))
        with torch.no_grad():
            policy, value, _, _ = model(state)
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool), remaining=state.remaining[None],
                       selected=state.selected[None], actions=torch.tensor([[0]]),
                       log_probs=policy.log_prob(torch.tensor([0]))[None], values=value[None],
                       rewards=torch.tensor([[.4]]), dones=torch.ones(1, 1, dtype=torch.bool))
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config = TrainConfig(ppo_epochs=1, ranking_coefficient=0, actor_continuation_guard=True)
        trainer.device = torch.device("cpu")
        optimizer = Mock()
        optimizer.zero_grad.side_effect = lambda **kwargs: model.zero_grad(set_to_none=True)
        events = []
        optimizer.step.side_effect = lambda: events.append("attempted_step")
        guard = Mock()
        guard.finish.side_effect = lambda: (events.append("guard_finished") or
            dict(dict.fromkeys(ACTOR_GUARD_FIELDS, 0), actor_guard_enabled=1, actor_guard_rejected=1))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with patch("mkp_Trainer.ActorUpdateGuard", return_value=guard) as constructor:
            stats = trainer.ppo_update(model, optimizer, (state.profits, state.weights, state.capacities), rollout)
        constructor.assert_called_once()
        self.assertEqual(events, ["attempted_step", "guard_finished"])
        self.assertEqual(stats["minibatches"], 1)
        self.assertEqual(stats["actor_minibatches_applied"], 0)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
