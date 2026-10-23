"""Tiny Actor/Adam rollback fixtures; no production training or benchmark."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from mkp_actor_guard import ACTOR_GUARD_FIELDS, ActorUpdateGuard, build_optimizer
from mkp_env import MKPEnv


class ToyActor(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float64))
        self.register_buffer("marker", torch.tensor(0.))


class ToyPolicy:
    def __init__(self, state, logits):
        feasible = state.feasible
        self.owners, self.items = feasible.nonzero(as_tuple=True)
        self.log_probs = torch.empty_like(self.items, dtype=torch.float64)
        self.actions = torch.full((len(state.profits),), -1, device=logits.device, dtype=torch.long)
        for row in range(len(state.profits)):
            mask = self.owners == row
            if mask.any():
                scores = logits[self.items[mask]]
                self.log_probs[mask] = scores.log_softmax(0)
                self.actions[row] = self.items[mask][scores.argmax()]

    def choose(self, stochastic=False):
        if stochastic:
            raise AssertionError("Guard must use deterministic greedy continuation")
        return self.actions


class ToyModel(nn.Module):
    def __init__(self, logits=(2., 1., 0.)):
        super().__init__()
        self.actor = ToyActor(logits)
        self.critic = nn.Parameter(torch.tensor(2., dtype=torch.float64))
        self.probe_modes = []
        self.incidental_random = False

    def policy(self, state):
        self.probe_modes.append((self.training, self.actor.training))
        if self.incidental_random:
            torch.rand(())
        return ToyPolicy(state, self.actor.logits), None, None


def config(**changes):
    fields = dict(learning_rate=.1, actor_learning_rate=None,
                  actor_continuation_guard=True, actor_guard_states=8,
                  actor_guard_tolerance=0., target_kl=100.)
    fields.update(changes)
    return SimpleNamespace(**fields)


def fixture(profits=None):
    profits = torch.tensor(profits if profits is not None else [[6., 5., 5.]], dtype=torch.float64)
    batch = len(profits)
    weights = torch.tensor([[[3.], [2.], [2.]]], dtype=torch.float64).expand(batch, -1, -1).clone()
    capacities = torch.full((batch, 1), 4., dtype=torch.float64)
    data = profits, weights, capacities
    state = MKPEnv(*data).state
    return data, dict(active=state.active[None], remaining=state.remaining[None], selected=state.selected[None])


def toy_adam_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    (model.actor.logits.square().sum() + model.critic.square()).backward()
    optimizer.step()


class ActorGuardTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            else:
                self.assertEqual(actual[key], value)

    def test_optimizer_preserves_legacy_or_splits_only_actor_rate(self):
        model = ToyModel()
        legacy = build_optimizer(model, config())
        self.assertEqual(len(legacy.param_groups), 1)
        self.assertNotIn("name", legacy.param_groups[0])
        self.assertEqual(legacy.param_groups[0]["lr"], .1)
        self.assertEqual(legacy.param_groups[0]["weight_decay"], 0.)
        self.assertEqual([id(p) for p in legacy.param_groups[0]["params"]], [id(p) for p in model.parameters()])
        split = build_optimizer(model, config(actor_learning_rate=1e-5))
        groups = {group["name"]: group for group in split.param_groups}
        self.assertEqual(groups["actor"]["lr"], 1e-5)
        self.assertEqual(groups["critic"]["lr"], .1)
        self.assertEqual([id(p) for p in groups["actor"]["params"]], [id(p) for p in model.actor.parameters()])
        self.assertEqual([id(p) for p in groups["critic"]["params"]], [id(model.critic)])
        self.assertTrue(all(group["weight_decay"] == 0 for group in split.param_groups))
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_optimizer(model, config(actor_learning_rate=invalid))

    def test_disabled_guard_does_not_inspect_any_inputs(self):
        guard = ActorUpdateGuard(None, None, None, None, config(actor_continuation_guard=False))
        self.assertEqual(guard.finish(), dict.fromkeys(ACTOR_GUARD_FIELDS, 0))

    def test_acceptance_uses_complete_suffix_not_first_item_profit(self):
        model = ToyModel((2., 1., 0.))
        data, rollout = fixture()
        optimizer = build_optimizer(model, config())
        guard = ActorUpdateGuard(model, optimizer, data, rollout, config())
        with torch.no_grad():
            model.actor.logits.copy_(torch.tensor([1., 2., 0.]))
        metrics = guard.finish()
        # First-item profit falls 6 -> 5, but the full suffix rises 6 -> 10.
        self.assertAlmostEqual(metrics["actor_guard_before_return"], 6 / 16)
        self.assertAlmostEqual(metrics["actor_guard_after_return"], 10 / 16)
        self.assertEqual(metrics["actor_guard_rejected"], 0)
        torch.testing.assert_close(model.actor.logits, torch.tensor([1., 2., 0.], dtype=torch.float64))
        self.assertEqual(metrics, guard.finish())  # finishing twice is harmless

    def test_rejection_restores_actor_buffers_and_adam_but_keeps_critic(self):
        model = ToyModel((1., 2., 0.))
        optimizer = build_optimizer(model, config())
        toy_adam_step(model, optimizer)  # populate both Adam states
        data, rollout = fixture()
        old_actor = deepcopy(model.actor.state_dict())
        old_adam = deepcopy(optimizer.state[model.actor.logits])
        old_critic = model.critic.detach().clone()
        guard = ActorUpdateGuard(model, optimizer, data, rollout, config())
        toy_adam_step(model, optimizer)
        with torch.no_grad():
            model.actor.logits.copy_(torch.tensor([2., 1., 0.]))
            model.actor.marker.fill_(99.)
        updated_critic = model.critic.detach().clone()
        updated_critic_adam = deepcopy(optimizer.state[model.critic])
        metrics = guard.finish()
        # First-item profit rises 5 -> 6, while the true suffix drops 10 -> 6.
        self.assertEqual(metrics["actor_guard_return_rejected"], 1)
        self.assertEqual(metrics["actor_guard_kl_rejected"], 0)
        self.assert_state_equal(model.actor.state_dict(), old_actor)
        self.assert_state_equal(optimizer.state[model.actor.logits], old_adam)
        torch.testing.assert_close(model.critic, updated_critic, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.critic, old_critic))
        self.assert_state_equal(optimizer.state[model.critic], updated_critic_adam)

    def test_rejection_removes_only_new_actor_optimizer_state(self):
        model = ToyModel((1., 2., 0.))
        optimizer = build_optimizer(model, config())
        data, rollout = fixture()
        guard = ActorUpdateGuard(model, optimizer, data, rollout, config())
        self.assertNotIn(model.actor.logits, optimizer.state)
        toy_adam_step(model, optimizer)
        with torch.no_grad():
            model.actor.logits.copy_(torch.tensor([2., 1., 0.]))
        self.assertEqual(guard.finish()["actor_guard_rejected"], 1)
        self.assertNotIn(model.actor.logits, optimizer.state)
        self.assertIn(model.critic, optimizer.state)

    def test_accepted_update_keeps_actor_adam_progress(self):
        model = ToyModel()
        optimizer = build_optimizer(model, config(actor_learning_rate=.01))
        toy_adam_step(model, optimizer)
        data, rollout = fixture()
        guard = ActorUpdateGuard(model, optimizer, data, rollout, config())
        before_step = optimizer.state[model.actor.logits]["step"].clone()
        toy_adam_step(model, optimizer)  # small change preserves the greedy suffix
        accepted_actor = deepcopy(model.actor.state_dict())
        accepted_adam = deepcopy(optimizer.state[model.actor.logits])
        self.assertEqual(guard.finish()["actor_guard_rejected"], 0)
        self.assert_state_equal(model.actor.state_dict(), accepted_actor)
        self.assert_state_equal(optimizer.state[model.actor.logits], accepted_adam)
        self.assertEqual(float(optimizer.state[model.actor.logits]["step"]), float(before_step) + 1)

    def test_kl_guard_rejects_probability_drift_even_with_equal_returns(self):
        model = ToyModel()
        data, rollout = fixture()
        settings = config(target_kl=1e-5)
        optimizer = build_optimizer(model, settings)
        old_actor = deepcopy(model.actor.state_dict())
        guard = ActorUpdateGuard(model, optimizer, data, rollout, settings)
        with torch.no_grad():
            model.actor.logits.mul_(4.)
        metrics = guard.finish()
        self.assertEqual(metrics["actor_guard_delta"], 0.)
        self.assertEqual(metrics["actor_guard_return_rejected"], 0)
        self.assertEqual(metrics["actor_guard_kl_rejected"], 1)
        self.assertGreater(metrics["actor_guard_max_kl"], settings.target_kl)
        self.assert_state_equal(model.actor.state_dict(), old_actor)

    def test_nonfinite_actor_is_rejected_and_restored(self):
        model = ToyModel()
        data, rollout = fixture()
        optimizer = build_optimizer(model, config())
        old_actor = deepcopy(model.actor.state_dict())
        guard = ActorUpdateGuard(model, optimizer, data, rollout, config())
        with torch.no_grad():
            model.actor.logits[0] = float("nan")
        metrics = guard.finish()
        self.assertEqual(metrics["actor_guard_rejected"], 1)
        self.assertEqual(metrics["actor_guard_return_rejected"], 1)
        self.assert_state_equal(model.actor.state_dict(), old_actor)

    def test_suffix_precision_excludes_large_selected_prefix(self):
        model = ToyModel((0., 2., 1.))
        data = (torch.tensor([[1e12, 1., 1.0001]], dtype=torch.float64),
                torch.tensor([[[0.], [1.], [1.]]], dtype=torch.float64),
                torch.tensor([[1.]], dtype=torch.float64))
        state = MKPEnv(*data).state
        state.selected[0, 0] = True
        rollout = dict(active=state.active[None], remaining=state.remaining[None], selected=state.selected[None])
        guard = ActorUpdateGuard(model, build_optimizer(model, config()), data, rollout, config())
        with torch.no_grad():
            model.actor.logits.copy_(torch.tensor([0., 1., 2.]))
        metrics = guard.finish()
        total = float(data[0].sum())
        self.assertAlmostEqual(metrics["actor_guard_before_return"] * total, 1., places=12)
        self.assertAlmostEqual(metrics["actor_guard_after_return"] * total, 1.0001, places=12)
        self.assertGreater(metrics["actor_guard_delta"], 0.)

    def test_mean_acceptance_reports_individual_worsening(self):
        model = ToyModel()
        data, rollout = fixture([[6., 5., 5.], [6., 5., 0.]])
        guard = ActorUpdateGuard(model, build_optimizer(model, config()), data, rollout, config())
        with torch.no_grad():
            model.actor.logits.copy_(torch.tensor([1., 2., 0.]))
        metrics = guard.finish()
        self.assertEqual(metrics["actor_guard_rejected"], 0)
        self.assertGreater(metrics["actor_guard_delta"], 0.)
        self.assertEqual(metrics["actor_guard_worsened_states"], 1)
        self.assertAlmostEqual(metrics["actor_guard_min_delta"], -1 / 11)

    def test_prefixes_span_instances_and_stages_without_rng_or_heldout(self):
        model = ToyModel()
        model.train()
        model.actor.eval()  # preserve independently configured child modes
        model.incidental_random = True
        data, rollout = fixture([[6., 5., 5.]] * 10)
        rollout = {key: value.expand(9, *value.shape[1:]).clone() for key, value in rollout.items()}
        # A guard has no validation input. Reject reads beyond physical rollout fields.
        class TrainingRollout(dict):
            def __getitem__(self, key):
                if key not in ("active", "remaining", "selected"):
                    raise AssertionError("Guard must not read heldout data or rollout labels")
                return super().__getitem__(key)
        rollout = TrainingRollout(rollout)
        original = {key: value.clone() for key, value in rollout.items()}
        rng = torch.get_rng_state().clone()
        guard = ActorUpdateGuard(model, build_optimizer(model, config()), data, rollout, config())
        metrics = guard.finish()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(metrics["actor_guard_states"], 8)
        self.assertEqual(len({batch for _, batch in guard.prefix_indices}), 8)
        self.assertEqual({time for time, _ in guard.prefix_indices}, {1, 4, 7})
        self.assertTrue(all(mode == (False, False) for mode in model.probe_modes))
        self.assertTrue(model.training)
        self.assertFalse(model.actor.training)
        for key, value in original.items():
            torch.testing.assert_close(rollout[key], value, rtol=0, atol=0)

    def test_small_rollout_does_not_duplicate_prefixes_or_select_inactive(self):
        model = ToyModel()
        data, rollout = fixture([[6., 5., 5.]] * 2)
        rollout = {key: value.expand(5, *value.shape[1:]).clone() for key, value in rollout.items()}
        rollout["active"][2:, 0] = False
        settings = config(actor_guard_states=20)
        guard = ActorUpdateGuard(model, build_optimizer(model, settings), data, rollout, settings)
        self.assertEqual(len(guard.prefix_indices), 7)
        self.assertEqual(len(set(guard.prefix_indices)), 7)
        self.assertTrue(all(rollout["active"][time, batch] for time, batch in guard.prefix_indices))


if __name__ == "__main__":
    unittest.main()
