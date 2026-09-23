"""Analytic theta-action/continuation contracts, no optimizer steps or benchmarks."""
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch

from mkp_config import ModelConfig, TrainConfig
from mkp_env import MKPEnv, MKPState
from mkp_Model import MKPModel, StrategyTheta
from mkp_theta_candidates import propose_thetas, decode_proposals, deduplicate_actions
from mkp_ranking import (candidate_terminal_returns, action_comparisons, collect_ranking_batch,
                         ranking_objective, THETA_RANKING_VERSION)
from mkp_Trainer import MKPTrainer, main
from test_opportunity_ranking import critical_state


class ThetaRankingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(59)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)
        self.train = TrainConfig(ranking_states=2, ranking_pool_size=4, ranking_repeats=3,
                                 ppo_epochs=1, device="cpu")

    def fixture(self, device="cpu"):
        state = critical_state(device)
        model = MKPModel(self.config).to(device)
        with torch.no_grad():
            model.actor.strategy_head.bias[4:].fill_(-20)
            model.actor.combination_head.interaction_gate.bias.fill_(-20)
            model.actor.combination_head.companion_gate.bias.fill_(-20)
            theta = model.policy(state)[1].detach()
        self.assertEqual(model.decoder(state, theta).choose().item(), 0)
        better = replace(theta, opportunity_weight=torch.ones_like(theta.opportunity_weight))
        same_action = replace(better, opportunity_weight=better.opportunity_weight * .8)
        self.assertEqual(model.decoder(state, better).choose().item(), 1)
        self.assertEqual(model.decoder(state, same_action).choose().item(), 1)
        proposals = [theta, theta.detach(), better, same_action]
        data = state.profits, state.weights, state.capacities
        rollout = dict(active=torch.ones(1, 1, dtype=torch.bool, device=device),
                       remaining=state.remaining[None], selected=state.selected[None])
        return model, state, proposals, data, rollout

    def test_proposals_are_legal_detached_and_include_unmodified_current_theta(self):
        for m in (1, 5, 10):
            state = MKPEnv(torch.rand(2, 7) + 1, torch.rand(2, 7, m), torch.ones(2, m) * 3).state
            model = MKPModel(self.config)
            theta = model.policy(state)[1]
            rng = torch.get_rng_state().clone()
            proposals = propose_thetas(theta, self.config, 8, 1., torch.Generator().manual_seed(19))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for field in fields(theta):
                self.assertTrue(torch.equal(getattr(proposals[0], field.name), getattr(theta, field.name)))
            for p in proposals:
                for field in fields(p):
                    self.assertFalse(getattr(p, field.name).requires_grad)
                    self.assertTrue(torch.isfinite(getattr(p, field.name)).all())
                self.assertTrue((p.resource_prices >= self.config.price_floor).all())
                torch.testing.assert_close(p.resource_prices.sum(-1), torch.full((2,), float(m)))
                for name, lo, hi in (("profit_exponent", .5, 2.), ("scarcity_exponent", 0., 2.),
                                     ("bottleneck_weight", 0., 2.), ("opportunity_weight", 0., 4.),
                                     ("reserve_weight", 0., 4.)):
                    self.assertTrue(((getattr(p, name) >= lo) & (getattr(p, name) <= hi)).all())
                torch.testing.assert_close(p.temperature, theta.temperature)
                torch.testing.assert_close(p.resource_interactions, p.resource_interactions.transpose(-1, -2))
                self.assertTrue((p.resource_interactions >= 0).all())
                self.assertTrue((p.resource_interactions.triu().sum((-1, -2)) <= 2.000001).all())
                self.assertTrue((p.companion_weights >= 0).all())
                self.assertTrue((p.companion_weights.sum(-1) <= 2.000001).all())

    def test_disabled_controls_and_equal_bounds_stay_disabled(self):
        config = replace(self.config, opportunity_weight_max=0, reserve_weight_max=0,
                         interaction_weight_max=0, companion_weight_max=0,
                         scarcity_exponent_max=0, bottleneck_weight_max=0,
                         profit_exponent_min=1., profit_exponent_max=1.)
        model = MKPModel(config)
        theta = model.policy(critical_state())[1]
        proposals = propose_thetas(theta, config, 8, 100., torch.Generator().manual_seed(5))
        for p in proposals:
            for name in ("opportunity_weight", "reserve_weight", "resource_interactions", "companion_weights",
                         "scarcity_exponent", "bottleneck_weight"):
                self.assertEqual(getattr(p, name).abs().sum().item(), 0.)
            self.assertEqual(p.profit_exponent.item(), 1.)

    def test_dedup_uses_action_not_theta_and_keeps_current_action(self):
        model, state, proposals, _, _ = self.fixture()
        decoded = decode_proposals(model, state, proposals)
        self.assertEqual(decoded.tolist(), [[0, 0, 1, 1]])
        induced = deduplicate_actions(decoded, 4, torch.Generator().manual_seed(2))
        self.assertEqual(induced.items.tolist(), [0, 1])
        self.assertEqual(induced.witnesses.tolist(), [0, 2])
        self.assertEqual(induced.counts.tolist(), [2])
        for _ in range(2):
            capped = deduplicate_actions(torch.tensor([[9, 9, 4, 4, 6, 7, 6], [2, 2, 2, 2, 2, 2, 2]]),
                                         3, torch.Generator().manual_seed(13))
            self.assertEqual(capped.counts.tolist(), [3, 1])
            self.assertEqual(capped.items[0].item(), 9)
            self.assertEqual(capped.items[-1].item(), 2)
            self.assertEqual(capped.items[:3].unique().numel(), 3)

    def test_q_excludes_prefix_and_continuation_reads_true_successors(self):
        model, state, _, _, _ = self.fixture()
        observed = []
        original = model.policy

        def inspect(successor):
            observed.append((successor.remaining.clone(), successor.selected.clone()))
            return original(successor)

        with patch.object(model, "policy", side_effect=inspect):
            q = candidate_terminal_returns(model, state, torch.tensor([0, 0]), torch.tensor([0, 1]), 1, False)
        torch.testing.assert_close(q, torch.tensor([[10.05], [15.]], dtype=torch.float64))
        torch.testing.assert_close(observed[0][0], torch.tensor([[4.8, 5.], [5., 4.8]], dtype=torch.float64))
        self.assertEqual(observed[0][1].tolist(), [[True, False, False], [False, True, False]])
        env = MKPEnv.from_state(state)
        env.step(torch.tensor([2]))
        q = candidate_terminal_returns(model, env.state, torch.tensor([0]), torch.tensor([1]), 1, False)
        self.assertEqual(q.item(), 10.)  # 5 already selected; only the newly earned 10 is Q.
        self.assertFalse(state.selected.any())

    def test_ragged_shared_noise_duplicate_actions_and_action_order(self):
        env = MKPEnv(torch.tensor([[3., 2., 4., 3., 2.], [3., 2., 4., 3., 2.]]),
                     torch.tensor([[[2., 2.], [1., 4.], [3., 1.], [2., 3.], [1., 2.]]] * 2),
                     torch.tensor([[6., 6.], [6., 6.]]))
        model = MKPModel(self.config)
        owners, actions = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 2])
        q = candidate_terminal_returns(model, env.state, owners, actions, 24, True, torch.Generator().manual_seed(21))
        self.assertTrue(torch.equal(q[0], q[1]))
        perm = torch.tensor([3, 2, 0, 1])
        reordered = candidate_terminal_returns(model, env.state, owners[perm], actions[perm], 24, True,
                                               torch.Generator().manual_seed(21))
        torch.testing.assert_close(reordered, q[perm], rtol=0, atol=0)
        self.assertGreater(q[0].unique().numel(), 1)

    def test_collect_evaluates_each_action_once_and_labels_measured_profit_gain(self):
        model, state, proposals, data, rollout = self.fixture()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch("mkp_theta_candidates.propose_thetas", return_value=proposals), \
             patch("mkp_ranking.candidate_terminal_returns", wraps=candidate_terminal_returns) as returns:
            batch, metrics = collect_ranking_batch(model, data, rollout, self.train, seed=11)
        self.assertEqual(returns.call_count, 2)  # One greedy batch, one repeated stochastic batch.
        for call in returns.call_args_list:
            self.assertEqual(call.args[3].tolist(), [0, 1])
        self.assertEqual(metrics["ranking_unique_actions"], 2)
        self.assertEqual(metrics["ranking_improved_pairs"], 1)
        self.assertEqual(metrics["ranking_rollout_branches"], 8)
        self.assertAlmostEqual(metrics["ranking_raw_improvement_mean"], 4.95, places=5)
        self.assertAlmostEqual(batch.delta.item(), -4.95 / 25.05, places=6)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(weights[name], value) for name, value in model.state_dict().items()))
        self.assertFalse(batch.delta.requires_grad)
        self.assertFalse(state.selected.any())

    def test_same_induced_action_and_disabled_ranking_skip_all_continuations(self):
        model, _, proposals, data, rollout = self.fixture()
        with patch("mkp_theta_candidates.propose_thetas", return_value=[proposals[0]] * 8), \
             patch("mkp_ranking.candidate_terminal_returns", side_effect=AssertionError("duplicate action")):
            batch, metrics = collect_ranking_batch(model, data, rollout, self.train, seed=8)
        self.assertIsNone(batch)
        self.assertEqual(metrics["ranking_no_alternative"], 1)
        with patch.object(model, "policy", side_effect=AssertionError("disabled")):
            batch, metrics = collect_ranking_batch(model, data, rollout,
                replace(self.train, ranking_coefficient=0), seed=8)
        self.assertIsNone(batch)
        self.assertEqual(metrics["ranking_theta_evaluations"], 0)

    def test_generated_proposals_collect_and_preserve_rng(self):
        model, _, _, data, rollout = self.fixture()
        rng = torch.get_rng_state().clone()
        batch, metrics = collect_ranking_batch(model, data, rollout, self.train, seed=23)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(metrics["ranking_unique_actions"], 1)
        self.assertIsNotNone(batch)
        self.assertEqual(metrics["ranking_theta_evaluations"], 8)

    def test_audit_reproduces_theta_actions_and_raw_returns(self):
        model, _, proposals, data, rollout = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.pt"
            with patch("mkp_theta_candidates.propose_thetas", return_value=proposals):
                collect_ranking_batch(model, data, rollout, self.train, seed=23, audit_path=path)
            audit = torch.load(path, weights_only=True)
            self.assertEqual(audit["ranking_protocol"], THETA_RANKING_VERSION)
            frozen = MKPModel(ModelConfig(**audit["model_config"]))
            frozen.load_state_dict(audit["model_state"], strict=True)
            state = MKPState(**audit["state"])
            thetas = [StrategyTheta(**p) for p in audit["theta_proposals"]]
            torch.testing.assert_close(decode_proposals(frozen, state, thetas), audit["decoded_actions"])
            rng = torch.Generator().set_state(audit["continuation_rng_state"])
            q = candidate_terminal_returns(frozen, state, audit["owners"], audit["actions"],
                                            self.train.ranking_repeats, True, rng)
            torch.testing.assert_close(q, audit["sampled_q"], rtol=0, atol=0)
            from verify_ranking_audit import verify_audit
            verified = verify_audit(path)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["improved_pairs"], 1)

    def test_multi_action_targets_retain_both_better_and_worse_choices(self):
        state = critical_state()
        induced = deduplicate_actions(torch.tensor([[0, 1, 2, 1]]), 4, torch.Generator())
        greedy = torch.tensor([[10.], [15.], [7.]], dtype=torch.float64)
        sampled = greedy.expand(-1, 3).clone()
        owners, anchors, alternatives, delta, se, keep = action_comparisons(state, induced, greedy, sampled, self.train)
        self.assertEqual(anchors.tolist(), [0, 0])
        self.assertEqual(alternatives.tolist(), [1, 2])
        self.assertEqual(keep.tolist(), [True, True])
        torch.testing.assert_close(delta, torch.tensor([-5 / 25.05, 3 / 25.05]))
        self.assertTrue((se == 0).all())
        # Noisy/conflicting direction is still rejected in the multi-action path.
        sampled[1] = torch.tensor([1., 1., 1.])
        self.assertFalse(action_comparisons(state, induced, greedy, sampled, self.train)[-1][0])

    def test_ranking_gradient_is_on_decoder_action_scores_not_proposal_theta(self):
        model, _, proposals, data, rollout = self.fixture()
        with patch("mkp_theta_candidates.propose_thetas", return_value=proposals):
            batch, _ = collect_ranking_batch(model, data, rollout, self.train, seed=3)
        with torch.no_grad():
            model.actor.price_head.weight.normal_(0, .03)
            model.actor.strategy_head.weight.normal_(0, .03)
        loss, _, _ = ranking_objective(model, batch, self.train)
        loss.backward()
        self.assertGreater(model.actor.price_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.actor.encoder.item_input.weight.grad.abs().sum().item(), 0)
        self.assertLess(model.actor.strategy_head.bias.grad[3].abs().item(), 1e-6)
        self.assertTrue(all(p.grad is None for p in model.critic_encoder.parameters()))
        self.assertIsNone(model.value_head.weight.grad)
        self.assertFalse(hasattr(model.decoder, "parameters"))
        self.assertTrue(all(not getattr(p, field.name).requires_grad for p in proposals for field in fields(p)))

    def test_ppo_uses_actual_theta_labels_without_optimizer_updates(self):
        model, state, proposals, data, rollout = self.fixture()
        with torch.no_grad():
            policy, value, _, _ = model(state)
        rollout.update(actions=torch.tensor([[0]]), log_probs=policy.log_prob(torch.tensor([0]))[None],
                       values=value[None], rewards=torch.tensor([[.4]]), dones=torch.ones(1, 1, dtype=torch.bool))
        trainer = MKPTrainer.__new__(MKPTrainer)
        trainer.config, trainer.device = self.train, torch.device("cpu")
        optimizer = Mock()
        optimizer.zero_grad.side_effect = lambda **kwargs: model.zero_grad(set_to_none=True)
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        with patch("mkp_theta_candidates.propose_thetas", return_value=proposals):
            stats = trainer.ppo_update(model, optimizer, data, rollout)
        self.assertEqual(stats["ranking_improved_pairs"], 1)
        self.assertEqual(stats["ranking_minibatches"], 1)
        optimizer.step.assert_called_once()  # Mock only; no parameter update.
        self.assertTrue(all(torch.equal(weights[name], value) for name, value in model.state_dict().items()))

    def test_cli_and_invalid_candidate_settings(self):
        with patch("mkp_Trainer.MKPTrainer") as trainer:
            main(["--ranking-mode", "theta", "--ranking-theta-candidates", "12",
                  "--ranking-theta-local-scale", ".7", "--ranking-max-actions", "3", "--ranking-audit-every", "0"])
        train = trainer.call_args.args[0]
        self.assertEqual((train.ranking_mode, train.ranking_theta_candidates, train.ranking_max_actions), ("theta", 12, 3))
        self.assertEqual(train.ranking_audit_every, 0)
        self.assertEqual(train.ranking_theta_local_scale, .7)
        for changes in (dict(ranking_mode="oracle"), dict(ranking_theta_candidates=1),
                        dict(ranking_max_actions=1), dict(ranking_theta_local_scale=float("nan")),
                        dict(ranking_theta_local_scale=-1), dict(ranking_audit_every=-1)):
            with self.assertRaises(ValueError):
                TrainConfig(**changes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_theta_collection_and_variable_size_interfaces(self):
        from mkp_utils import seed_everything
        from mkp_performance import configure_training_runtime
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(59, deterministic=True, cpu_threads=2)
            model, _, proposals, data, rollout = self.fixture("cuda")
            configure_training_runtime(model, replace(self.train, performance_profile="memory24"), "cuda")
            rng = torch.cuda.get_rng_state().clone()
            with patch("mkp_theta_candidates.propose_thetas", return_value=proposals):
                batch, metrics = collect_ranking_batch(model, data, rollout, self.train, seed=7)
            self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
            self.assertEqual(metrics["ranking_improved_pairs"], 1)
            ranking_objective(model, batch, self.train)[0].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            # Candidate generation/decoding only, no long rollout at these sizes.
            for n, m in ((100, 5), (250, 10), (500, 5)):
                state = MKPEnv(torch.rand(1, n, device="cuda") + 1,
                               torch.rand(1, n, m, device="cuda"), torch.ones(1, m, device="cuda") * 4).state
                with torch.no_grad():
                    theta = model.policy(state)[1]
                candidates = propose_thetas(theta, self.config, 3, 1., torch.Generator(device="cuda").manual_seed(2))
                actions = decode_proposals(model, state, candidates)
                self.assertEqual(actions.shape, (1, 3))
                self.assertTrue(state.feasible[0, actions[0]].all())
        finally:
            torch.use_deterministic_algorithms(previous)
