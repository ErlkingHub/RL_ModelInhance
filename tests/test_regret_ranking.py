"""Small loss/gradient contracts; no optimizer updates or model training."""
import unittest

import torch

from mkp_ranking import regret_margin_losses


class RegretRankingTests(unittest.TestCase):
    def test_satisfied_positive_comparison_has_zero_loss_and_zero_gradient(self):
        margin = torch.tensor([1., 2.], requires_grad=True)
        loss = regret_margin_losses(margin, torch.tensor([.01, .02]), .002)
        torch.testing.assert_close(loss, torch.zeros(2))
        loss.sum().backward()
        torch.testing.assert_close(margin.grad, torch.zeros(2))

    def test_negative_return_difference_reverses_the_required_order(self):
        margin = torch.tensor([-1., .5], requires_grad=True)
        loss = regret_margin_losses(margin, torch.tensor([-.01, -.01]), .002)
        self.assertEqual(loss[0].item(), 0.)
        self.assertGreater(loss[1].item(), 0.)
        loss.sum().backward()
        self.assertEqual(margin.grad[0].item(), 0.)
        self.assertGreater(margin.grad[1].item(), 0.)  # descent decreases wrong positive margin

    def test_return_importance_and_required_margin_are_bounded(self):
        margin = torch.zeros(3, requires_grad=True)
        losses = regret_margin_losses(margin, torch.tensor([.001, .002, 200.]), .002)
        torch.testing.assert_close(losses, torch.tensor([.0625, .5, .5]))
        capped = regret_margin_losses(torch.tensor([0.]), torch.tensor([200.]), .002, margin_cap=.2)
        torch.testing.assert_close(capped, torch.tensor([.02]))

    def test_labels_are_detached_and_ties_produce_no_gradient(self):
        delta = torch.tensor([.001, 0.], requires_grad=True)
        margin = torch.tensor([0., -100.], requires_grad=True)
        loss = regret_margin_losses(margin, delta, .002).sum()
        loss.backward()
        self.assertIsNone(delta.grad)
        self.assertLess(margin.grad[0].item(), 0.)
        self.assertEqual(margin.grad[1].item(), 0.)

    def test_extreme_wrong_margins_have_bounded_score_derivatives(self):
        margin = torch.tensor([-1000., 1000.], requires_grad=True)
        regret_margin_losses(margin, torch.tensor([1., -1.]), .002).sum().backward()
        torch.testing.assert_close(margin.grad, torch.tensor([-1., 1.]))

    def test_invalid_scale_and_margin_cap_are_rejected(self):
        for scale, cap in ((0., 1.), (float("nan"), 1.), (1., 0.), (1., float("inf"))):
            with self.subTest(scale=scale, cap=cap), self.assertRaises(ValueError):
                regret_margin_losses(torch.zeros(1), torch.ones(1), scale, cap)


if __name__ == "__main__":
    unittest.main()
