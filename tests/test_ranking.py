"""Tiny exact counterexamples and backward-only integration; no parameter updates."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from mkp_config import TrainConfig, ModelConfig, CHECKPOINT_VERSION, DECODER_VERSION, config_dict
from mkp_env import MKPEnv
from mkp_Model import MKPModel, StrategyTheta, WeakDecoder
from mkp_ranking import (RankingBatch, paired_terminal_returns, collect_ranking_batch,
                         ranking_loss, ranking_metrics)
from mkp_Trainer import MKPTrainer, collect_rollout
from mkp_utils import atomic_checkpoint, load_checkpoint, construct_solution


def trap():
    # Immediate profit and density prefer item 0, but it excludes BOTH other items.
    # Forced 0 -> 8; forced 1 -> 6+6=12. The continuation has no ambiguity.
    return MKPEnv(torch.tensor([[8., 6., 6.]]), torch.tensor([[[6.], [5.], [5.]]]),
                  torch.tensor([[10.]]))


class RankingContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.mc = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)
        self.tc = TrainConfig(counterfactual_states=4, counterfactual_repeats=2,
                              ppo_epochs=1, minibatch_size=2)

    def batch(self, model):
        state = trap().state
        actions = torch.tensor([[0, 1]])
        samples = paired_terminal_returns(model, state, actions, 2)
        with torch.no_grad():
            policy = model.policy(state)[0]
            old = torch.zeros_like(state.profits, dtype=torch.float32)
            old[policy.owners, policy.items] = policy.log_probs
        return RankingBatch(state, actions, samples, old, torch.tensor([True]))

    def test_effective_weights_absorb_price_and_scarcity(self):
        env = MKPEnv(torch.tensor([[5., 6., 7.]]),
                      torch.tensor([[[2., 1.], [1., 2.], [1., 1.]]]), torch.tensor([[10., 12.]]))
        env.state.remaining[:] = torch.tensor([[4., 3.]])
        s = env.state
        price, beta = torch.tensor([[0.4, 1.6]]), 1.7
        log_effective = (price.log() + (beta - 1) * (s.capacities / s.remaining).float().log()).log_softmax(-1)
        theta = StrategyTheta(log_effective, torch.tensor([1.2]), torch.tensor([0.3]))
        new = WeakDecoder(self.mc)(s, theta)
        w, c, rem = s.weights.float(), s.capacities.float(), s.remaining.float()
        old_cost = (price[:, None] * 3 * w / c[:, None] * (c / rem)[:, None].pow(beta)).mean(-1)
        old_scores = 1.2 * (s.profits.float() / s.profits.float().mean(-1, keepdim=True)).log()
        old_scores -= old_cost.log() + 0.3 * (w / rem[:, None]).amax(-1)
        self.assertTrue(torch.allclose(new.log_probs, (old_scores / self.mc.sampling_temperature).log_softmax(-1)[0], atol=2e-6))
        actor = MKPModel(self.mc).actor
        self.assertIsNone(actor.price_head.bias)
        self.assertEqual(actor.price_head.in_features, self.mc.embed_dim)
        self.assertEqual(actor.strategy_head.out_features, 2)

    def test_long_horizon_reverses_immediate_preference_and_preserves_state(self):
        model = MKPModel(self.mc).train()
        state = trap().state
        rem, selected = state.remaining.clone(), state.selected.clone()
        samples = paired_terminal_returns(model, state, torch.tensor([[0, 1]]), 3)
        self.assertTrue(torch.allclose(samples, torch.tensor([[[0.4, 0.6]]], dtype=torch.float64).expand(1, 3, 2)))
        self.assertFalse(samples.requires_grad)
        self.assertTrue(torch.equal(rem, state.remaining))
        self.assertTrue(torch.equal(selected, state.selected))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_shared_noise_identical_branches_and_prefix_exclusion(self):
        model = MKPModel(self.mc)
        env = MKPEnv(torch.tensor([[1., 8., 6., 6.]]),
                      torch.tensor([[[0.], [6.], [5.], [5.]]]), torch.tensor([[10.]]))
        env.step(torch.tensor([0]))
        samples = paired_terminal_returns(model, env.state, torch.tensor([[1, 2]]), 3)
        self.assertTrue(torch.allclose(samples[0, 0], torch.tensor([8 / 21, 12 / 21], dtype=torch.float64)))
        # Nontrivial stochastic tails share item-indexed noise even after branching.
        env = MKPEnv(torch.tensor([[1., 8., 6., 6., 7.]]),
                      torch.tensor([[[0.], [6.], [5.], [5.], [7.]]]), torch.tensor([[12.]]))
        same = paired_terminal_returns(model, env.state, torch.tensor([[0, 0]]), 8)
        self.assertTrue(torch.equal(same[:, :, 0], same[:, :, 1]))

    def test_preference_gradient_reaches_controller_only(self):
        model = MKPModel(self.mc)
        batch = self.batch(model)
        loss, kl = ranking_loss(model, batch, torch.tensor([0]), self.tc)
        self.assertAlmostEqual(kl.item(), 0., places=6)
        loss.backward()
        # Increasing alpha favors the tempting immediate profit: descent must reduce it.
        self.assertGreater(model.actor.strategy_head.bias.grad[0].item(), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertIsNone(model.actor.feasibility_head.weight.grad)
        metrics = ranking_metrics(model, batch)
        self.assertEqual(metrics["ranking_pair_accuracy"], 0)
        self.assertAlmostEqual(metrics["ranking_pair_regret"], 0.2)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_unreliable_and_tied_pairs_have_no_ranking_gradient(self):
        model = MKPModel(self.mc)
        batch = self.batch(model)
        batch.reliable[:] = False
        loss, _ = ranking_loss(model, batch, torch.tensor([0]), self.tc)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()))

    def test_collection_budget_distinct_feasible_and_disabled(self):
        model = MKPModel(self.mc)
        env = trap()
        data = (env.state.profits, env.state.weights, env.state.capacities)
        rollout = collect_rollout(model, data)
        batch = collect_ranking_batch(model, data, rollout, self.tc)
        self.assertLessEqual(batch.actions.shape[0], self.tc.counterfactual_states)
        self.assertTrue((batch.actions[:, 0] != batch.actions[:, 1]).all())
        self.assertTrue(batch.state.feasible.gather(1, batch.actions).all())
        self.assertFalse(batch.old_log_probs.requires_grad)
        self.tc.ranking_coefficient = 0
        self.assertIsNone(collect_ranking_batch(model, data, rollout, self.tc))

    def test_single_candidate_skips_pair_collection(self):
        model = MKPModel(self.mc)
        env = MKPEnv(torch.tensor([[1., 3.]]), torch.tensor([[[1.], [9.]]]), torch.tensor([[2.]]))
        data = (env.state.profits, env.state.weights, env.state.capacities)
        self.assertIsNone(collect_ranking_batch(model, data, collect_rollout(model, data), self.tc))

    def test_equal_terminal_returns_are_filtered(self):
        model = MKPModel(self.mc)
        env = MKPEnv(torch.tensor([[2., 3., 4.]]), torch.ones(1, 3, 1), torch.tensor([[3.]]))
        data = (env.state.profits, env.state.weights, env.state.capacities)
        batch = collect_ranking_batch(model, data, collect_rollout(model, data), self.tc)
        self.assertTrue(torch.equal(batch.delta, torch.zeros_like(batch.delta)))
        self.assertFalse(batch.reliable.any())

    def test_pair_kl_stops_update_even_when_sampled_ppo_kl_is_zero(self):
        model = MKPModel(self.mc)
        batch = self.batch(model)
        # A normalized old distribution deliberately different from the current one.
        batch.old_log_probs[:] = torch.tensor([[-100., 100., -100.]]).log_softmax(-1)
        trainer = object.__new__(MKPTrainer)
        trainer.config, trainer.device = self.tc, torch.device("cpu")
        env = trap()
        data = (env.state.profits, env.state.weights, env.state.capacities)
        rollout = collect_rollout(model, data)
        with patch("mkp_Trainer.collect_ranking_batch", return_value=batch):
            # None deliberately fails if any optimizer operation happens.
            stats = trainer.ppo_update(model, None, data, rollout)
        self.assertEqual(stats["kl_stopped"], 1)
        self.assertEqual(stats["minibatches"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_paired_continuation_and_ranking_backward(self):
        from mkp_utils import seed_everything
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(5, deterministic=True, cpu_threads=2)
            model = MKPModel(self.mc).cuda()
            env = trap()
            data = tuple(x.cuda() for x in (env.state.profits, env.state.weights, env.state.capacities))
            batch = collect_ranking_batch(model, data, collect_rollout(model, data), self.tc)
            loss, kl = ranking_loss(model, batch, torch.arange(batch.actions.shape[0], device="cuda"), self.tc)
            loss.backward()
            self.assertTrue(torch.isfinite(kl))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(was_deterministic)

    def test_combined_ppo_backward_without_parameter_update(self):
        model = MKPModel(self.mc)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        class BackwardOnly:
            calls = 0
            def zero_grad(self, **kwargs):
                model.zero_grad(**kwargs)
            def step(self):
                self.calls += 1  # Deliberately no optimizer and no weight mutation.
        optimizer = BackwardOnly()
        trainer = object.__new__(MKPTrainer)
        trainer.config, trainer.device = self.tc, torch.device("cpu")
        env = trap()
        data = (env.state.profits, env.state.weights, env.state.capacities)
        stats = trainer.ppo_update(model, optimizer, data, collect_rollout(model, data))
        self.assertGreater(optimizer.calls, 0)
        self.assertGreater(stats["ranking_reliable_pairs"], 0)
        self.assertGreater(stats["ranking_loss"], 0)
        self.assertTrue(all(torch.equal(v, initial[k]) for k, v in model.state_dict().items()))

    def test_new_checkpoint_and_inference_statistics(self):
        model = MKPModel(self.mc)
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, decoder_version=DECODER_VERSION,
                       training_data_source="synthetic_only", num_resources=1,
                       model_config=config_dict(self.mc), model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_checkpoint(path, payload)
            restored, _ = load_checkpoint(path, "cpu", 1)
            env = trap()
            data = (env.state.profits, env.state.weights, env.state.capacities)
            for mode in ("fixed", "learned_static", "learned_dynamic"):
                result = construct_solution(data, restored, mode=mode)
                self.assertEqual(result["theta_mean"].shape, (1, 3))
                self.assertTrue(torch.isfinite(result["theta_mean"]).all())
            payload["checkpoint_version"] = 1
            atomic_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "checkpoint schema"):
                load_checkpoint(path, "cpu", 1)

    def test_ranking_configuration_validation(self):
        for kwargs in (dict(counterfactual_repeats=1), dict(ranking_return_scale=0),
                       dict(ranking_coefficient=-1), dict(gamma=0.9)):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)
        TrainConfig(ranking_coefficient=0, gamma=0.9)


if __name__ == "__main__":
    unittest.main()
