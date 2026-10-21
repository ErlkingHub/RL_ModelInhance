"""Physical geometry and actor residual contracts; no optimizer or training."""
import unittest

import torch

from mkp_config import ModelConfig
from mkp_env import MKPEnv, MKPState
from mkp_resource import (RESOURCE_GEOMETRY_DIM, RESOURCE_RELATION_DIM,
                          ResourceRefinement, resource_geometry)


class ResourceGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)
        self.config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)

    def state(self):
        return MKPEnv(torch.tensor([[8., 4., 3., 100.], [5., 2., 1., 3.]]),
            torch.tensor([[[4., 1., 0.], [1., 3., 0.], [2., 2., 0.], [9., 9., 0.]],
                          [[1., 2., 0.], [2., 1., 0.], [1., 1., 0.], [3., 2., 0.]]]),
            torch.tensor([[6., 5., 4.], [6., 5., 4.]])).state

    def embeddings(self, state):
        b, n = state.profits.shape
        m = state.capacities.shape[1]
        d = self.config.embed_dim
        return torch.randn(b, n, d), torch.randn(b, m, d), torch.randn(b, d)

    def test_exact_demands_empty_and_zero_demand(self):
        state = self.state()
        state.selected[1] = True
        geometry = resource_geometry(state, self.config)
        self.assertEqual(geometry.resources.shape, (2, 3, RESOURCE_GEOMETRY_DIM))
        self.assertEqual(geometry.relations.shape, (2, 3, 3, RESOURCE_RELATION_DIM))
        self.assertAlmostEqual(geometry.resources[0, 0, 2].item(), 7 / 18, places=6)
        self.assertAlmostEqual(geometry.resources[0, 0, 5].item(), (8 * 4 + 4 + 3 * 2) / (15 * 6), places=6)
        self.assertAlmostEqual(geometry.pressure[0, 0].item(), torch.log1p(torch.tensor(7 / 6)).item(), places=6)
        for tensor in (geometry.resources, geometry.relations, geometry.pressure):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(tensor[1].abs().sum().item(), 0)
        self.assertEqual(geometry.resources[:, 2].abs().sum().item(), 0)
        self.assertEqual(geometry.pressure[:, 2].abs().sum().item(), 0)

    def test_shared_and_complementary_demands_are_distinguished(self):
        profits = torch.ones(1, 4)
        capacity = torch.tensor([[10., 10.]])
        shared = MKPEnv(profits, torch.tensor([[[5., 5.], [5., 5.], [0., 0.], [0., 0.]]]), capacity).state
        disjoint = MKPEnv(profits, torch.tensor([[[5., 0.], [5., 0.], [0., 5.], [0., 5.]]]), capacity).state
        a, b = (resource_geometry(s, self.config) for s in (shared, disjoint))
        torch.testing.assert_close(a.resources, b.resources)
        self.assertAlmostEqual(a.relations[0, 0, 1, 0].item(), 1.0)
        self.assertAlmostEqual(b.relations[0, 0, 1, 0].item(), 0.0)
        self.assertAlmostEqual(a.relations[0, 0, 1, 2].item(), 1.0)
        self.assertAlmostEqual(b.relations[0, 0, 1, 2].item(), -1.0)
        self.assertEqual(a.relations[0, 0, 1, 5].item(), 0.5)
        self.assertEqual(b.relations[0, 0, 1, 5].item(), 0.0)

    def test_cumulative_bottleneck_has_positive_pressure_without_argmax_mass(self):
        # Resource 0 is cumulatively most demanded, but never a per-item max.
        state = MKPEnv(torch.ones(1, 4),
            torch.tensor([[[4., 5., 0.], [4., 5., 0.], [4., 0., 5.], [4., 0., 5.]]]),
            torch.full((1, 3), 10.)).state
        geometry = resource_geometry(state, self.config)
        self.assertEqual(geometry.pressure.argmax(-1).item(), 0)
        self.assertGreater(geometry.pressure[0, 0].item(), 0)
        self.assertTrue(torch.isfinite(geometry.pressure).all())

    def test_geometry_and_refinement_permutation_equivariance(self):
        state = self.state()
        items, resources, pooled = self.embeddings(state)
        item_order, resource_order = torch.tensor([2, 0, 3, 1]), torch.tensor([1, 2, 0])
        reordered = MKPState(state.profits[:, item_order], state.weights[:, item_order][:, :, resource_order],
            state.capacities[:, resource_order], state.remaining[:, resource_order], state.selected[:, item_order])
        original = resource_geometry(state, self.config)
        permuted = resource_geometry(reordered, self.config)
        torch.testing.assert_close(permuted.resources, original.resources[:, resource_order])
        torch.testing.assert_close(permuted.relations, original.relations[:, resource_order][:, :, resource_order])
        module = ResourceRefinement(self.config).eval()
        a_resources, a_pool = module(items, resources, pooled, original, state.feasible)
        b_resources, b_pool = module(items[:, item_order], resources[:, resource_order], pooled,
                                     permuted, reordered.feasible)
        torch.testing.assert_close(b_resources, a_resources[:, resource_order], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(b_pool, a_pool, atol=2e-6, rtol=2e-6)

    def test_independent_resource_units_and_profit_units(self):
        state = self.state()
        factors = torch.tensor([1e-9, 3e8, 0.17], dtype=torch.float64)
        scaled = MKPState(state.profits * 1e-12, state.weights * factors,
            state.capacities * factors, state.remaining * factors, state.selected.clone())
        original, changed = (resource_geometry(s, self.config) for s in (state, scaled))
        for key in ("resources", "relations", "pressure"):
            torch.testing.assert_close(getattr(original, key), getattr(changed, key), atol=2e-6, rtol=2e-6)

    def test_invalid_item_data_and_embeddings_are_masked(self):
        state = self.state()
        state.selected[1, 3] = True
        changed = MKPState(*(x.clone() for x in
            (state.profits, state.weights, state.capacities, state.remaining, state.selected)))
        invalid = ~state.feasible
        changed.profits[invalid] *= 100000
        changed.weights[invalid] *= 100000
        original, modified = (resource_geometry(s, self.config) for s in (state, changed))
        for key in ("resources", "relations", "pressure"):
            torch.testing.assert_close(getattr(original, key), getattr(modified, key), atol=0, rtol=0)
        items, resources, pooled = self.embeddings(state)
        altered_items = items.clone()
        altered_items[invalid] = torch.randn_like(altered_items[invalid]) * 10000
        module = ResourceRefinement(self.config).eval()
        actual = module(items, resources, pooled, original, state.feasible)
        expected = module(altered_items, resources, pooled, modified, changed.feasible)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_float64_feasibility_and_zero_remaining(self):
        state = MKPEnv(torch.tensor([[2., 1., 4.]]),
            torch.tensor([[[0., 1.], [0., 2.], [0., 3.]]], dtype=torch.float64),
            torch.tensor([[1., 2.]], dtype=torch.float64)).state
        state.weights[0, 2, 1] = 2 + 1e-10
        state.remaining[0, 0] = 0
        self.assertEqual(state.feasible.tolist(), [[True, True, False]])
        geometry = resource_geometry(state, self.config)
        self.assertEqual(geometry.pressure[0, 0].item(), 0)
        self.assertAlmostEqual(geometry.resources[0, 1, 2].item(), 0.75)
        self.assertAlmostEqual(geometry.resources[0, 1, 5].item(), 2 / 3)
        self.assertTrue(torch.isfinite(geometry.relations).all())

    def test_terminal_and_all_zero_weights_have_identity_residuals(self):
        state = self.state()
        state.selected[1] = True
        state.weights[0].zero_()
        state.remaining[0].zero_()
        geometry = resource_geometry(state, self.config)
        items, resources, pooled = self.embeddings(state)
        module = ResourceRefinement(self.config)
        new_resources, new_pool = module(items, resources, pooled, geometry, state.feasible)
        torch.testing.assert_close(new_resources, resources, atol=0, rtol=0)
        torch.testing.assert_close(new_pool, pooled, atol=0, rtol=0)
        (new_resources.sum() + new_pool.sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_gradients_reach_items_geometry_relations_and_bounded_gates(self):
        state = self.state()
        items, resources, pooled = (x.requires_grad_() for x in self.embeddings(state))
        module = ResourceRefinement(self.config)
        geometry = resource_geometry(state, self.config)
        refined, summary = module(items, resources, pooled, geometry, state.feasible)
        (refined.square().mean() + summary.square().mean()).backward()
        for name in ("geometry_input.0.weight", "item_query.weight", "item_key.weight",
                     "item_value.weight", "relation_bias.weight", "relation_value.weight",
                     "resource_gate", "pooled_gate"):
            parameter = dict(module.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(items.grad[state.feasible].abs().sum().item(), 0)
        self.assertEqual(items.grad[~state.feasible].abs().sum().item(), 0)
        self.assertLessEqual((refined - resources).abs().max().item(), self.config.actor_resource_gate_init + 1e-6)
        self.assertLessEqual((summary - pooled).abs().max().item(), self.config.actor_resource_gate_init + 1e-6)
        self.assertFalse(geometry.resources.requires_grad)
        with torch.no_grad():
            module.resource_gate.fill_(100)
            module.pooled_gate.fill_(100)
            refined, summary = module(items, resources, pooled, geometry, state.feasible)
        self.assertLessEqual((refined - resources).abs().max().item(), self.config.actor_resource_gate_max + 1e-6)
        self.assertLessEqual((summary - pooled).abs().max().item(), self.config.actor_resource_gate_max + 1e-6)


if __name__ == "__main__":
    unittest.main()
