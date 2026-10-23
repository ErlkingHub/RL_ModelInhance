"""Small analytic fixtures only; no optimizer steps or dataset benchmarks."""
from dataclasses import fields, replace
import math
import unittest
from unittest.mock import patch

import torch

from mkp_basis import companion_basis, interaction_loads, decoder_statistics, COMBINATION_METRICS
from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, WeakDecoder, CandidatePolicy
from mkp_ranking import ranking_objective
from mkp_Trainer import collect_rollout, main
from test_opportunity_ranking import critical_state, labeled_batch
from mkp_test_helpers import synthetic_theta


def complementary_state(device="cpu"):
    # A has better immediate density (12/6 vs 8/5), but blocks B and C.
    # B+C fits exactly and yields 16 rather than A's 12.
    return MKPEnv(torch.tensor([[12., 8., 8.]], device=device),
                  torch.tensor([[[6., 6.], [10., 0.], [0., 10.]]], device=device),
                  torch.tensor([[10., 10.]], device=device)).state


class CombinationThetaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1, combination_chunk_size=2)

    def test_pair_bonus_exposes_better_combination_than_high_density_single_item(self):
        state = complementary_state()
        decoder = WeakDecoder(self.config)
        theta = synthetic_theta(state, self.config)
        self.assertEqual(decoder(state, theta).choose().item(), 0)
        theta.companion_weights[0, 0] = 2
        policy = decoder(state, theta)
        self.assertEqual(policy.choose().item(), 1)
        # Analytic evidence of the fixture's consequence, not a benchmark claim.
        env = MKPEnv.from_state(state)
        env.step(policy.choose())
        env.step(decoder(env.state, theta).choose())
        self.assertEqual(env.objective.item(), 16)
        self.assertFalse(env.state.active.any())
        basis = companion_basis(state, *state.feasible.nonzero(as_tuple=True), self.config)
        self.assertTrue(torch.equal(basis[0], torch.zeros(3)))
        torch.testing.assert_close(basis[1:], torch.full((2, 3), 16 / (16 + 2 * (28 / 3))))

    def test_interaction_has_nonzero_mixed_difference_and_changes_order(self):
        state = MKPEnv(torch.ones(1, 4) * 10,
                       torch.tensor([[[1., 1.], [1., 3.], [3., 1.], [3., 3.]]]),
                       torch.ones(1, 2) * 8).state
        decoder = WeakDecoder(self.config)
        theta = synthetic_theta(state, self.config)
        theta.resource_interactions[0, 0, 1] = theta.resource_interactions[0, 1, 0] = 1
        penalty = decoder(state, theta).score_terms["interaction_penalty"]
        mixed = penalty[3] - penalty[2] - penalty[1] + penalty[0]
        self.assertGreater(mixed.item(), .05)
        # Equal profit and summed resource cost; old formula favors A on bottleneck.
        state = MKPEnv(torch.tensor([[10., 10.]]), torch.tensor([[[6., 6.], [10., 2.]]]),
                       torch.tensor([[10., 10.]])).state
        theta = synthetic_theta(state, self.config)
        self.assertEqual(decoder(state, theta).choose().item(), 0)
        theta.resource_interactions[0, 0, 1] = theta.resource_interactions[0, 1, 0] = 2
        self.assertEqual(decoder(state, theta).choose().item(), 1)

    def test_formula_matches_independent_scalar_reference(self):
        state = critical_state()
        decoder = WeakDecoder(self.config)
        theta = synthetic_theta(state, self.config)
        theta.resource_interactions[0] = torch.tensor([[.3, .8], [.8, .2]])
        theta.companion_weights[0] = torch.tensor([.2, .6, .7])
        policy = decoder(state, theta)
        weights, profits, rem = state.weights[0].tolist(), state.profits[0].tolist(), state.remaining[0].tolist()
        mean_w = [sum(w[j] for w in weights) / 3 for j in range(2)]
        mean_p = sum(profits) / 3
        for i in range(3):
            z = [weights[i][j] / (weights[i][j] + mean_w[j]) for j in range(2)]
            expected_penalty = .3*z[0]**2 + .8*z[0]*z[1] + .2*z[1]**2
            expected_bonus = 0
            for scale, coefficient in zip(self.config.companion_scales, (.2, .6, .7)):
                best = 0.
                for j in range(3):
                    after = [(rem[r] - weights[i][r]) - weights[j][r] for r in range(2)]
                    if i == j or min(after) < 0:
                        continue
                    gap = sum(after[r] / mean_w[r] for r in range(2)) / 2
                    quality = (profits[i] + profits[j]) / (profits[i] + profits[j] + 2*mean_p)
                    best = max(best, quality * math.exp(-gap/scale))
                expected_bonus += coefficient * best
            self.assertAlmostEqual(policy.score_terms["interaction_penalty"][i].item(), expected_penalty, places=6)
            self.assertAlmostEqual(policy.score_terms["companion_bonus"][i].item(), expected_bonus, places=6)
        expected = policy.score_terms["base"] - policy.score_terms["interaction_penalty"] + policy.score_terms["companion_bonus"]
        torch.testing.assert_close(policy.logits, expected / theta.temperature[policy.owners])

    def test_zero_controls_recover_original_score_and_flags_skip_new_basis(self):
        state = critical_state()
        model = MKPModel(self.config)
        theta = model.actor(state)[0]
        theta.resource_interactions = torch.zeros_like(theta.resource_interactions)
        theta.companion_weights = torch.zeros_like(theta.companion_weights)
        first = model.decoder(state, theta)
        disabled = WeakDecoder(replace(self.config, interaction_weight_max=0, companion_weight_max=0))
        with patch("mkp_Model.interaction_loads", side_effect=AssertionError("disabled")), \
             patch("mkp_Model.companion_basis", side_effect=AssertionError("disabled")):
            second = disabled(state, theta)
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        torch.testing.assert_close(first.logits, first.score_terms["base"] / theta.temperature[first.owners], rtol=0, atol=0)

    def test_coefficients_are_bounded_and_equivariant_with_nonuniform_heads(self):
        model = MKPModel(self.config).eval()
        with torch.no_grad():
            model.actor.combination_head.resource_pair[-1].weight.normal_(0, .2)
            model.actor.combination_head.companion_mix.weight.normal_(0, .2)
        state = MKPEnv(torch.rand(2, 6) + .1, torch.rand(2, 6, 5), torch.ones(2, 5)*2).state
        order_i, order_r = torch.tensor([4, 2, 0, 5, 1, 3]), torch.tensor([4, 2, 0, 3, 1])
        reordered = MKPState(state.profits[:, order_i], state.weights[:, order_i][:, :, order_r],
                             state.capacities[:, order_r], state.remaining[:, order_r], state.selected[:, order_i])
        with torch.no_grad():
            policy, theta, _ = model.policy(state)
            next_policy, next_theta, _ = model.policy(reordered)
        torch.testing.assert_close(theta.resource_interactions[:, order_r][:, :, order_r], next_theta.resource_interactions)
        torch.testing.assert_close(theta.companion_weights, next_theta.companion_weights)
        torch.testing.assert_close(theta.resource_interactions, theta.resource_interactions.transpose(1, 2))
        self.assertTrue((theta.resource_interactions >= 0).all())
        self.assertTrue((theta.resource_interactions.triu().sum((-1, -2)) <= self.config.interaction_weight_max).all())
        self.assertTrue((theta.companion_weights.sum(-1) <= self.config.companion_weight_max).all())
        torch.testing.assert_close(policy.logits.reshape(2, 6)[:, order_i], next_policy.logits.reshape(2, 6))
        self.assertEqual(theta.resource_interactions.triu().nonzero().shape[0], 2 * 5 * 6 // 2)

    def test_scale_chunk_and_ineligible_item_invariance(self):
        state = critical_state()
        rows, items = state.feasible.nonzero(as_tuple=True)
        original = companion_basis(state, rows, items, self.config)
        scale = torch.tensor([100., .01], dtype=torch.float64)
        rescaled = MKPState(state.profits*13, state.weights*scale, state.capacities*scale,
                            state.remaining*scale, state.selected)
        torch.testing.assert_close(original, companion_basis(rescaled, rows, items, replace(self.config, combination_chunk_size=1)))
        torch.testing.assert_close(interaction_loads(state, rows, items, self.config.eps),
                                   interaction_loads(rescaled, rows, items, self.config.eps))
        extra = MKPEnv(torch.cat((state.profits, torch.tensor([[9999., 9999.]])), 1),
                       torch.cat((state.weights, torch.tensor([[[1., 1.], [100., 100.]]])), 1), state.capacities).state
        extra.selected[0, 3] = True
        torch.testing.assert_close(original, companion_basis(extra, rows, items, self.config))
        # Duplicating every candidate at the same state leaves mean-demand loads unchanged.
        duplicate = MKPState(state.profits.repeat(1, 2), state.weights.repeat(1, 2, 1),
                             state.capacities, state.remaining, state.selected.repeat(1, 2))
        torch.testing.assert_close(interaction_loads(state, rows, items, self.config.eps),
                                   interaction_loads(duplicate, rows, items, self.config.eps))

    def test_float64_pair_feasibility_and_zero_demand_resource(self):
        state = MKPEnv(torch.tensor([[10., 10., 5.]], dtype=torch.float64),
                       torch.tensor([[[5.00000001, 0.], [5., 0.], [5., 0.]]], dtype=torch.float64),
                       torch.tensor([[10., 1.]], dtype=torch.float64)).state
        state.remaining[0, 1] = 0
        basis = companion_basis(state, *state.feasible.nonzero(as_tuple=True), self.config)
        self.assertEqual(basis[0].sum().item(), 0)
        self.assertGreater(basis[1].sum().item(), 0)
        self.assertTrue(torch.isfinite(basis).all())

    def test_ranking_gradients_reach_both_controls_and_encoder_but_not_critic(self):
        model = MKPModel(self.config)
        with torch.no_grad():
            model.actor.combination_head.resource_pair[-1].weight.normal_(0, .1)
            model.actor.combination_head.companion_mix.weight.normal_(0, .1)
        loss = ranking_objective(model, labeled_batch(model, critical_state()), TrainConfig())[0]
        loss.backward()
        head = model.actor.combination_head
        for parameter in (head.resource_pair[0].weight, head.resource_pair[-1].weight,
                          head.interaction_gate.weight, head.companion_mix.weight, head.companion_gate.weight,
                          model.actor.encoder.item_input.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertNotIsInstance(model.decoder, torch.nn.Module)
        for field in fields(model.actor(critical_state())[0]):
            self.assertFalse(getattr(model.actor(critical_state())[0].detach(), field.name).requires_grad)

    def test_terminal_single_candidate_and_detached_score_statistics(self):
        state = MKPEnv(torch.ones(2, 1), torch.tensor([[[0.]], [[3.]]]), torch.ones(2, 1)).state
        model = MKPModel(self.config)
        policy, theta, _ = model.policy(state)
        self.assertEqual(policy.choose().tolist(), [0, -1])
        self.assertEqual(theta.resource_interactions.shape, (2, 1, 1))
        stats = decoder_statistics(policy)
        self.assertTrue(torch.isfinite(stats).all())
        self.assertFalse(stats.requires_grad)
        self.assertEqual(stats[:, -1].sum().item(), 0)
        (theta.resource_interactions.sum() + theta.companion_weights.sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        state.selected.fill_(True)
        policy = model.policy(state)[0]
        self.assertEqual(decoder_statistics(policy).abs().sum().item(), 0)

    def test_diagnostics_compare_score_margins_not_common_offsets(self):
        state = critical_state()
        model = MKPModel(self.config)
        policy = model.policy(state)[0]
        first = decoder_statistics(policy)
        shifted = dict(policy.score_terms)
        shifted["base"] = shifted["base"] + 20
        shifted["score"] = shifted["score"] + 20
        other = CandidatePolicy(policy.owners, policy.items, policy.logits, policy.batch_size, shifted)
        torch.testing.assert_close(first, decoder_statistics(other), atol=1e-5, rtol=1e-5)
        rollout = collect_rollout(model, (state.profits, state.weights, state.capacities))
        self.assertEqual(set(rollout["combination_metrics"]), set(COMBINATION_METRICS))
        self.assertTrue(all(math.isfinite(x) for x in rollout["combination_metrics"].values()))

    def test_cli_and_invalid_basis_settings(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--interaction-weight-max", "0", "--companion-weight-max", "1",
                  "--combination-initial-weight", ".01", "--companion-scales", ".5", "4",
                  "--combination-chunk-size", "3"])
        _, config = trainer.call_args.args
        self.assertEqual(config.interaction_weight_max, 0)
        self.assertEqual(config.companion_scales, (.5, 4.))
        self.assertEqual(config.combination_chunk_size, 3)
        for kwargs in (dict(interaction_weight_max=-1), dict(companion_weight_max=float("nan")),
                       dict(companion_scales=()), dict(companion_scales=(1, 1)), dict(companion_scales=(2, 1)),
                       dict(companion_scales=(0,)), dict(combination_initial_weight=0), dict(combination_chunk_size=0)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_deterministic_nonuniform_heads_backward(self):
        from mkp_utils import seed_everything
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(101, deterministic=True, cpu_threads=2)
            model = MKPModel(self.config).cuda()
            with torch.no_grad():
                model.actor.combination_head.resource_pair[-1].weight.normal_(0, .1)
                model.actor.combination_head.companion_mix.weight.normal_(0, .1)
            state = critical_state("cuda")
            loss = ranking_objective(model, labeled_batch(model, state), TrainConfig())[0]
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
