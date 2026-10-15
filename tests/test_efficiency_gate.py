"""Profit/efficiency contracts on tiny synthetic states; no training or benchmarks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mkp_basis import companion_basis, has_feasible_pair, gate_statistics, GATE_METRICS
from mkp_config import (ModelConfig, TrainConfig, BenchmarkConfig, CHECKPOINT_VERSION,
                        CONTROLLER_VERSION, DECODER_VERSION, config_dict)
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, WeakDecoder
from mkp_performance import prepare_state, cached_features
from mkp_ranking import ranking_objective
from mkp_Trainer import collect_rollout, main
from mkp_utils import atomic_checkpoint, construct_solution, load_checkpoint
from test_opportunity_ranking import critical_state, labeled_batch


def final_choice_state():
    # Either item ends the episode. Higher density A gives 10; B gives 11.
    return MKPEnv(torch.tensor([[10., 11.]]), torch.tensor([[[1.] * 5, [10.] * 5]]),
                  torch.full((1, 5), 10.)).state


class EfficiencyGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1, combination_chunk_size=1)

    def test_previously_unreachable_higher_profit_becomes_greedy_choice(self):
        state = final_choice_state()
        old = WeakDecoder(replace(self.config, efficiency_gate=False))
        theta = old.fixed_theta(state)
        theta.profit_exponent.fill_(2.)
        theta.bottleneck_weight.zero_()
        theta.opportunity_weight.fill_(4.)
        legacy = old(state, theta)
        self.assertEqual(legacy.choose().item(), 0)
        self.assertLess((legacy.score_terms["score"][1] - legacy.score_terms["score"][0]).item(), -1.9)
        # The learned gate alone exposes B, independent of the hard terminal rule.
        theta.efficiency_gate.zero_()
        learned = WeakDecoder(replace(self.config, terminal_profit_override=False))(state, theta)
        self.assertEqual(learned.choose().item(), 1)
        # The terminal rule also works with g=1 and all other corrections enabled.
        theta.efficiency_gate.fill_(1.)
        theta.bottleneck_weight.fill_(2.)
        theta.reserve_weight.fill_(4.)
        theta.resource_interactions.fill_(.2)
        terminal = WeakDecoder(self.config)(state, theta)
        self.assertEqual(terminal.choose().item(), 1)
        self.assertEqual(gate_statistics(terminal).tolist(), [[1., 0., 1.]])
        env = MKPEnv.from_state(state)
        env.step(terminal.choose())
        self.assertEqual(env.objective.item(), 11.)
        self.assertFalse(env.state.active.any())

    def test_gate_interpolates_all_score_terms_and_one_recovers_exact_old_score(self):
        state = critical_state()
        decoder = WeakDecoder(replace(self.config, terminal_profit_override=False))
        theta = decoder.fixed_theta(state)
        theta.opportunity_weight.fill_(1.2)
        theta.reserve_weight.fill_(.7)
        theta.resource_interactions.fill_(.2)
        theta.companion_weights.fill_(.3)
        old = WeakDecoder(replace(self.config, efficiency_gate=False))(state, theta)
        profit = theta.profit_exponent[0] * (state.profits.float()[0] / state.profits.float().mean()).log()
        for g in (0., .35, 1.):
            theta.efficiency_gate.fill_(g)
            actual = decoder(state, theta)
            expected = profit + g * (old.score_terms["score"] - profit)
            torch.testing.assert_close(actual.score_terms["score"], expected)
            torch.testing.assert_close(actual.score_terms["interaction_penalty"],
                                       g * old.score_terms["interaction_penalty"])
            torch.testing.assert_close(actual.score_terms["companion_bonus"], g * old.score_terms["companion_bonus"])
            if g == 1.:
                self.assertTrue(torch.equal(actual.logits, old.logits))
        theta.efficiency_gate.zero_()
        disabled = WeakDecoder(replace(self.config, efficiency_gate=False))(state, theta)
        self.assertTrue(torch.equal(disabled.logits, old.logits))

    def test_pair_geometry_uses_float64_excludes_self_selected_and_infeasible_items(self):
        weights = torch.tensor([[[5.], [5. + 1e-10], [11.]], [[5.], [5.], [11.]]], dtype=torch.float64)
        state = MKPEnv(torch.ones(2, 3), weights, torch.full((2, 1), 10.)).state
        owners, items = state.feasible.nonzero(as_tuple=True)
        self.assertEqual(has_feasible_pair(state, owners, items, 1).tolist(), [False, True])
        state.selected[1, 1] = True
        self.assertEqual(has_feasible_pair(state, *state.feasible.nonzero(as_tuple=True), 7).tolist(), [False, False])
        policy = WeakDecoder(self.config)(state, WeakDecoder(self.config).fixed_theta(state))
        self.assertEqual(policy.score_terms["terminal_profit_override"].tolist(), [True, True])

    def test_pair_detection_does_not_use_underflowed_or_zero_profit_companion_scores(self):
        state = MKPEnv(torch.tensor([[1., 2.]]), torch.ones(1, 2, 2), torch.full((1, 2), 1e8)).state
        owners, items = state.feasible.nonzero(as_tuple=True)
        basis, exists = companion_basis(state, owners, items, self.config, return_has_pair=True)
        self.assertEqual(basis.abs().sum().item(), 0.)
        self.assertTrue(exists.item())
        decoder = WeakDecoder(self.config)
        policy = decoder(state, decoder.fixed_theta(state))
        self.assertFalse(policy.score_terms["terminal_profit_override"].item())
        # A selected positive item can leave a feasible pair with zero pair profit.
        zero = MKPEnv(torch.tensor([[1., 0., 0.]]), torch.ones(1, 3, 1), torch.full((1, 1), 4.)).state
        zero.selected[0, 0] = True
        basis, exists = companion_basis(zero, *zero.feasible.nonzero(as_tuple=True), self.config, return_has_pair=True)
        self.assertEqual(basis.abs().sum().item(), 0.)
        self.assertTrue(exists.item())

    def test_cache_reuses_pair_geometry_and_invalidates_after_step_or_switch_change(self):
        for maximum in (0., self.config.companion_weight_max):
            config = replace(self.config, companion_weight_max=maximum)
            model = MKPModel(config)
            state = critical_state()
            expected = model.policy(state)[0]
            prepare_state(state, config)
            self.assertTrue(state._prepared.tensors["has_feasible_pair"].item())
            with patch("mkp_Model.has_feasible_pair", side_effect=AssertionError("pair recomputed")), \
                 patch("mkp_Model.companion_basis", side_effect=AssertionError("basis recomputed")):
                actual = model.policy(state)[0]
            torch.testing.assert_close(actual.logits, expected.logits)
            torch.testing.assert_close(gate_statistics(actual), gate_statistics(expected))
            self.assertIsNone(cached_features(state, replace(config, terminal_profit_override=False)))
            self.assertIsNone(cached_features(state, replace(config, efficiency_gate=False)))
            env = MKPEnv.from_state(state)
            prepare_state(env.state, config)
            env.step(torch.tensor([1]))
            self.assertIsNone(cached_features(env.state, config))
            prepare_state(env.state, config)
            self.assertFalse(env.state._prepared.tensors["has_feasible_pair"].item())
            self.assertEqual(model.policy(env.state)[0].score_terms["effective_efficiency_gate"].item(), 0.)

    def test_disabling_gate_skips_terminal_pair_detection(self):
        config = replace(self.config, efficiency_gate=False, companion_weight_max=0.)
        decoder = WeakDecoder(config)
        state = final_choice_state()
        with patch("mkp_Model.has_feasible_pair", side_effect=AssertionError("disabled")):
            policy = decoder(state, decoder.fixed_theta(state))
        self.assertEqual(policy.choose().item(), 0)
        self.assertEqual(gate_statistics(policy).tolist(), [[1., 1., 0.]])
        prepare_state(state, config)
        self.assertNotIn("has_feasible_pair", state._prepared.tensors)

    def test_ranking_loss_trains_gate_and_actor_but_not_temperature_or_critic(self):
        model = MKPModel(self.config)
        # The default zero head initializes g=.95; exercise the learned state-dependent path.
        with torch.no_grad():
            model.actor.efficiency_gate_head.weight.normal_(0, .1)
        state = critical_state()
        loss, _, _ = ranking_objective(model, labeled_batch(model, state), TrainConfig())
        loss.backward()
        self.assertGreater(model.actor.efficiency_gate_head.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.actor.efficiency_gate_head.bias.grad.abs().sum().item(), 0.)
        self.assertGreater(model.actor.encoder.item_input.weight.grad.abs().sum().item(), 0.)
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value_head.parameters()))

    def test_terminal_override_blocks_efficiency_gradients_but_keeps_profit_gradient(self):
        model = MKPModel(self.config)
        policy, _, _ = model.policy(final_choice_state())
        (policy.score_terms["score"][1] - policy.score_terms["score"][0]).backward()
        self.assertEqual(model.actor.efficiency_gate_head.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(model.actor.price_head.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(model.actor.strategy_head.bias.grad[1:].abs().sum().item(), 0.)
        self.assertGreater(model.actor.strategy_head.bias.grad[0].abs().item(), 0.)

    def test_gate_is_state_level_bounded_and_permutation_invariant(self):
        model = MKPModel(self.config).eval()
        state = MKPEnv(torch.rand(2, 5) + .1, torch.rand(2, 5, 3), torch.ones(2, 3) * 2).state
        theta = model.actor(state)[0]
        torch.testing.assert_close(theta.efficiency_gate, torch.full((2,), .95))
        with torch.no_grad():
            model.actor.efficiency_gate_head.weight.normal_(0, .2)
        io, ro = torch.tensor([4, 2, 0, 1, 3]), torch.tensor([2, 0, 1])
        perm = MKPState(state.profits[:, io], state.weights[:, io][:, :, ro], state.capacities[:, ro],
                        state.remaining[:, ro], state.selected[:, io])
        original, theta, _ = model.policy(state)
        reordered, next_theta, _ = model.policy(perm)
        self.assertEqual(theta.efficiency_gate.shape, (2,))
        self.assertTrue(((theta.efficiency_gate > 0) & (theta.efficiency_gate < 1)).all())
        torch.testing.assert_close(theta.efficiency_gate, next_theta.efficiency_gate)
        torch.testing.assert_close(original.logits.view(2, 5)[:, io], reordered.logits.view(2, 5))
        self.assertFalse(theta.detach().efficiency_gate.requires_grad)
        self.assertFalse(isinstance(model.decoder, torch.nn.Module))

    def test_gate_diagnostics_average_active_steps_and_preserve_fixed_baseline(self):
        data = (torch.tensor([[10., 11.], [10., 11.]]),
                torch.tensor([[[1., 1.], [10., 10.]], [[2., 2.], [3., 3.]]]), torch.full((2, 2), 10.))
        model = MKPModel(self.config)
        result = construct_solution(data, model)
        torch.testing.assert_close(result["objective"], torch.tensor([11., 21.], dtype=torch.float64))
        self.assertEqual(result["theta_mean"].shape, (2, 10))
        torch.testing.assert_close(result["gate_mean"], torch.tensor([[.95, 0., 1.], [.95, .475, .5]]))
        rollout = collect_rollout(model, data)
        self.assertEqual(set(rollout["gate_metrics"]), set(GATE_METRICS))
        for key, expected in zip(GATE_METRICS, (.95, .95/3, 2/3)):
            self.assertAlmostEqual(rollout["gate_metrics"][key], expected, places=6)
        fixed = construct_solution(data, mode="fixed", model_config=self.config)
        self.assertEqual(fixed["objective"][0].item(), 10.)
        torch.testing.assert_close(fixed["gate_mean"], torch.tensor([[1., 1., 0.], [1., 1., 0.]]))

    def test_zero_weight_exhausted_resources_singleton_and_empty_rows(self):
        state = MKPEnv(torch.tensor([[1., 2.], [1., 2.]]),
                       torch.tensor([[[0., 1.], [0., 0.]], [[0., 1.], [0., 0.]]]),
                       torch.tensor([[1., 2.], [1., 2.]])).state
        state.remaining[:, 0] = 0
        state.selected[1] = True
        model = MKPModel(self.config)
        policy, theta, _ = model.policy(state)
        self.assertEqual(policy.score_terms["terminal_profit_override"].tolist(), [False, False])
        self.assertTrue(torch.isfinite(policy.logits).all())
        self.assertEqual(policy.choose()[1].item(), -1)
        state.selected[0, 1] = True
        policy = model.decoder(state, theta)
        self.assertEqual(policy.score_terms["terminal_profit_override"].tolist(), [True, False])
        self.assertEqual(policy.choose().tolist(), [0, -1])
        state.selected.fill_(True)
        empty = model.decoder(state, theta)
        self.assertEqual(empty.choose().tolist(), [-1, -1])
        self.assertTrue(torch.isfinite(gate_statistics(empty)).all())

    def test_config_cli_checkpoint_round_trip_and_old_schema_rejection(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--no-efficiency-gate", "--no-terminal-profit-override", "--efficiency-gate-initial", ".8"])
        train, config = trainer.call_args.args
        self.assertEqual(train.checkpoint_dir, BenchmarkConfig().checkpoint_dir)
        self.assertEqual(train.checkpoint_dir.name, "profit_efficiency_v6")
        self.assertFalse(config.efficiency_gate or config.terminal_profit_override)
        self.assertEqual(config.efficiency_gate_initial, .8)
        for value in (0., 1., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(config, efficiency_gate_initial=value)
        for kwargs in (dict(efficiency_gate=1), dict(terminal_profit_override="yes")):
            with self.assertRaises(ValueError):
                replace(config, **kwargs)
        model = MKPModel(self.config)
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, controller_version=CONTROLLER_VERSION,
                       decoder_version=DECODER_VERSION, training_data_source="synthetic_only",
                       num_resources=5, model_config=config_dict(self.config), model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.pt"
            atomic_checkpoint(path, payload)
            restored, _ = load_checkpoint(path, "cpu", 5)
            torch.testing.assert_close(model.policy(final_choice_state())[0].logits,
                                       restored.policy(final_choice_state())[0].logits)
            payload["checkpoint_version"] = 5
            atomic_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "v6 requires a fresh training run"):
                load_checkpoint(path, "cpu", 5)


if __name__ == "__main__":
    unittest.main()
