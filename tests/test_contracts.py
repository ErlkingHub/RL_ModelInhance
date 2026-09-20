"""Small interface/gradient checks only: no optimizer steps, training or benchmark runs."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from Benchmark_zeroShot import verify_solution
from mkp_config import CHECKPOINT_VERSION, CONTROLLER_VERSION, DECODER_VERSION, GeneratorConfig, ModelConfig, config_dict
from mkp_data import MKPInstance, synthetic_batch
from mkp_env import MKPEnv, MKPState
from mkp_Model import CandidatePolicy, MKPModel, StrategyTheta, WeakDecoder
from mkp_Trainer import generalized_advantage
from mkp_utils import atomic_checkpoint, load_checkpoint


def toy_state(device="cpu"):
    p = torch.tensor([[8., 5., 2., 1.], [3., 2., 1., 1.]], device=device)
    w = torch.tensor([[[4., 1.], [1., 3.], [1., 1.], [8., 8.]],
                      [[9., 9.], [8., 8.], [7., 7.], [6., 6.]]], device=device)
    c = torch.tensor([[5., 4.], [1., 1.]], device=device)
    return MKPEnv(p, w, c)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    def test_environment_rejects_infeasible_and_terminal_actions(self):
        env = toy_state()
        with self.assertRaises(ValueError):
            env.step(torch.tensor([3, -1]))
        with self.assertRaises(ValueError):
            env.step(torch.tensor([-1, -1]))
        reward, done = env.step(torch.tensor([0, -1]))
        self.assertAlmostEqual(reward[0].item(), 8 / 16)
        self.assertTrue(done[1])
        self.assertTrue(torch.equal(env.state.remaining[0], torch.tensor([1., 3.], dtype=torch.float64)))
        with self.assertRaises(ValueError):
            env.step(torch.tensor([0, -1]))

    def test_ragged_distribution_matches_categorical(self):
        owners, items = torch.tensor([0, 0, 2]), torch.tensor([1, 4, 2])
        logits = torch.tensor([0.3, 1.7, 8.], requires_grad=True)
        policy = CandidatePolicy(owners, items, logits, 3)
        expected = torch.distributions.Categorical(logits=logits[:2])
        self.assertTrue(torch.allclose(policy.log_probs[:2], expected.logits))
        self.assertTrue(torch.allclose(policy.entropy()[0], expected.entropy()))
        self.assertEqual(policy.choose().tolist(), [4, -1, 2])
        self.assertEqual(policy.choose(stochastic=True)[1].item(), -1)
        loss = -policy.log_prob(torch.tensor([1, -1, 2]))[0]
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_decoder_agrees_with_literal_formula(self):
        state = toy_state().state
        decoder = WeakDecoder(self.config)
        theta = decoder.fixed_theta(state)
        actual = decoder(state, theta)
        rows, cols = state.feasible.nonzero(as_tuple=True)
        w, c, rem = state.weights.float()[rows, cols], state.capacities.float()[rows], state.remaining.float()[rows]
        cost = (theta.resource_prices[rows] * (4 * w / c) * (c / rem).pow(theta.scarcity_exponent[rows, None])).mean(-1)
        expected = (theta.profit_exponent[rows] * (state.profits.float()[rows, cols] / state.profits.float().mean(-1)[rows]).log()
                    - cost.log() - theta.bottleneck_weight[rows] * (w / rem).amax(-1)) / theta.temperature[rows]
        self.assertTrue(torch.allclose(actual.logits, expected, atol=1e-5))

    def test_exhausted_resource_zero_weights_and_no_candidates(self):
        env = MKPEnv(torch.tensor([[1., 2.]]), torch.tensor([[[0., 1.], [0., 0.]]]), torch.tensor([[1., 2.]]))
        env.state.remaining[0, 0] = 0
        model = MKPModel(self.config)
        policy, _, theta, _ = model(env.state)
        self.assertTrue(torch.isfinite(policy.log_probs).all())
        (-policy.log_prob(torch.tensor([0])) + policy.entropy()).sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        # All rows terminal: the empty distribution is a legal no-op.
        env.state.selected.fill_(True)
        empty = model.decoder(env.state, theta)
        self.assertEqual(empty.choose().tolist(), [-1])
        self.assertEqual(empty.entropy().tolist(), [0.0])

    def test_model_invariance_and_independent_critic(self):
        model = MKPModel(self.config).eval()
        # Nonzero heads exercise learned permutation equivariance beyond initial uniform theta.
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, 0.1)
            model.actor.strategy_head.weight.normal_(0, 0.1)
        state = toy_state().state
        items, resources = torch.tensor([2, 0, 3, 1]), torch.tensor([1, 0])
        permuted = MKPState(state.profits[:, items], state.weights[:, items][:, :, resources],
                            state.capacities[:, resources], state.remaining[:, resources], state.selected[:, items])
        with torch.no_grad():
            original, value, theta, _ = model(state)
            reordered, perm_value, perm_theta, _ = model(permuted)
        self.assertTrue(torch.allclose(theta.resource_prices[:, resources], perm_theta.resource_prices, atol=1e-5))
        self.assertTrue(torch.allclose(theta.profit_exponent, perm_theta.profit_exponent, atol=1e-5))
        self.assertTrue(torch.allclose(value, perm_value, atol=1e-5))
        probs = {(b, i): p for b, i, p in zip(original.owners.tolist(), original.items.tolist(), original.log_probs.exp().tolist())}
        for b, i, probability in zip(reordered.owners.tolist(), reordered.items.tolist(), reordered.log_probs.exp().tolist()):
            self.assertAlmostEqual(probability, probs[b, items[i].item()], places=5)
        actor_ids = {id(p) for p in model.actor.parameters()}
        self.assertFalse(actor_ids.intersection(id(p) for p in model.critic_encoder.parameters()))

    def test_high_level_gradient_and_candidate_validity(self):
        model = MKPModel(self.config)
        state = toy_state().state
        policy, value, theta, auxiliary = model(state)
        actions = policy.choose(stochastic=True)
        self.assertTrue(state.feasible[0, actions[0]])
        self.assertEqual(actions[1].item(), -1)
        self.assertTrue(torch.allclose(theta.resource_prices.mean(-1), torch.ones(2)))
        loss = -policy.log_prob(actions)[0] + value.square().mean() + auxiliary.square().mean()
        loss.backward()
        self.assertGreater(model.actor.strategy_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.actor.price_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_gae_termination_and_padding(self):
        rewards = torch.tensor([[0.2, 0.5], [0.3, 0.0]])
        values = torch.tensor([[0.1, 0.4], [0.2, 123.]])
        dones = torch.tensor([[False, True], [True, True]])
        active = torch.tensor([[True, True], [True, False]])
        advantage, returns = generalized_advantage(rewards, values, dones, active, 1., 1.)
        self.assertTrue(torch.allclose(returns[active], torch.tensor([0.5, 0.5, 0.3])))
        self.assertEqual(advantage[1, 1].item(), 0)

    def test_synthetic_reproducibility_and_shape(self):
        first = synthetic_batch(2, 5, 3, GeneratorConfig(), np.random.default_rng(7), "cpu")
        second = synthetic_batch(2, 5, 3, GeneratorConfig(), np.random.default_rng(7), "cpu")
        self.assertEqual(first[1].shape, (2, 5, 3))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_checkpoint_contract(self):
        model = MKPModel(self.config)
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, decoder_version=DECODER_VERSION,
                       controller_version=CONTROLLER_VERSION,
                       training_data_source="synthetic_only", num_resources=2,
                       model_config=config_dict(self.config), model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_checkpoint(path, payload)
            restored, _ = load_checkpoint(path, "cpu", 2)
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(model.parameters(), restored.parameters())))
            with self.assertRaises(ValueError):
                load_checkpoint(path, "cpu", 5)

    def test_independent_solution_audit(self):
        instance = MKPInstance("toy", [8, 5, 2, 1], [[4, 1], [1, 3], [1, 1], [8, 8]], [5, 4], 13)
        result = dict(selected=torch.tensor([True, True, False, False]),
                      sequence=torch.tensor([0, 1]), objective=torch.tensor(13.))
        objective, utilization, violation, sequence = verify_solution(instance, result, 1e-7)
        self.assertEqual(objective, 13)
        self.assertEqual(violation, 0)
        self.assertEqual(sequence, [0, 1])
        self.assertTrue(np.all(utilization == 1))
        result["selected"] = torch.tensor([True, True, True, False])
        result["sequence"] = torch.tensor([0, 1, 2])
        result["objective"] = torch.tensor(15.)
        with self.assertRaises(ValueError):
            verify_solution(instance, result, 1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_one_state_gradient_with_deterministic_kernels(self):
        # Interface compatibility only: no rollout, optimizer, training or benchmark.
        from mkp_utils import seed_everything
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(13, deterministic=True, cpu_threads=2)
            model = MKPModel(self.config).cuda()
            state = toy_state("cuda").state
            policy, value, _, aux = model(state)
            action = policy.choose(stochastic=True)
            (-policy.log_prob(action).sum() + value.square().mean() + aux.square().mean()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(was_deterministic)


if __name__ == "__main__":
    unittest.main()
