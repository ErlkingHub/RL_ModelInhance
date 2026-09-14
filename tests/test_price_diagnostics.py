"""Regression checks without training, optimizer updates, or benchmark inference."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from mkp_config import CHECKPOINT_VERSION, DECODER_VERSION, ModelConfig, config_dict
from mkp_Model import ConditionalPriceHead, MKPModel
from mkp_utils import atomic_checkpoint, load_checkpoint
from mkp_diagnostics import critic_diagnostics, create_training_diagnostics, read_metrics


class PriceAndDiagnostics(unittest.TestCase):
    def test_context_changes_relative_prices_and_receives_gradient(self):
        torch.manual_seed(3)
        config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1)
        head = ConditionalPriceHead(config)
        resources = torch.randn(2, 5, 16)
        context = torch.randn(2, 16, requires_grad=True)
        logits = head(resources, context)
        prices = config.price_floor + (1 - config.price_floor) * 5 * logits.softmax(-1)
        changed = head(resources, context + 2).softmax(-1)
        self.assertFalse(torch.allclose(logits.softmax(-1), changed, atol=1e-6))
        prices[:, 0].sum().backward()
        self.assertGreater(context.grad.abs().sum().item(), 1e-6)
        self.assertGreater(head.context.weight.grad.abs().sum().item(), 1e-6)
        self.assertTrue(torch.all(prices > config.price_floor))
        self.assertTrue(torch.allclose(prices.mean(-1), torch.ones(2)))
        order = torch.tensor([2, 0, 4, 1, 3])
        self.assertTrue(torch.allclose(head(resources[:, order], context), logits[:, order]))

    def test_legacy_checkpoint_retains_linear_architecture(self):
        config = ModelConfig(embed_dim=16, num_heads=4, num_layers=1, price_head_type="linear", encoder_type="transformer")
        model = MKPModel(config)
        settings = config_dict(config)
        del settings["price_head_type"]
        del settings["price_head_init_std"]
        del settings["encoder_type"]
        del settings["policy_head"]
        del settings["direct_logit_clip"]
        payload = dict(checkpoint_version=CHECKPOINT_VERSION, decoder_version=DECODER_VERSION,
                       training_data_source="synthetic_only", num_resources=2,
                       model_config=settings, model_state=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            atomic_checkpoint(path, payload)
            loaded, _ = load_checkpoint(path, "cpu", 2)
        self.assertEqual(loaded.config.price_head_type, "linear")
        self.assertIsInstance(loaded.actor.price_head, torch.nn.Linear)
        self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_mc_targets_ignore_padding_and_bootstrap(self):
        rewards = torch.tensor([[0.2, 0.5], [0.3, 0.0]])
        values = torch.tensor([[0.1, 0.4], [0.2, 999.]])
        dones = torch.tensor([[False, True], [True, True]])
        active = torch.tensor([[True, True], [True, False]])
        result = critic_diagnostics(rewards, values, dones, active, 1.0)
        self.assertAlmostEqual(result["critic_mc_rmse"], math.sqrt((0.4**2 + 0.1**2 + 0.1**2) / 3), places=6)
        self.assertTrue(math.isfinite(result["critic_mc_ev"]))
        result_discounted = critic_diagnostics(rewards, values, dones, active, 0.5)
        self.assertAlmostEqual(result_discounted["critic_mc_rmse"], math.sqrt((0.25**2 + 0.1**2 + 0.1**2) / 3), places=6)

    def test_legacy_and_new_metrics_render_separately_by_resource_count(self):
        fields = ["M", "N", "update", "policy_loss", "value_loss", "entropy", "validation_return",
                  "total_loss", "stop_kl", "critic_mc_rmse", "critic_mc_ev", "theta_alpha"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(M=5, N=100, update=1, policy_loss=-0.01, value_loss=0.1,
                                     entropy=0.8, validation_return=0.5))
                writer.writerow(dict(M=10, N=100, update=1, policy_loss=-0.02, value_loss=0.02,
                                     entropy=0.5, total_loss=-0.01, stop_kl=0.035, critic_mc_rmse=0.01,
                                     critic_mc_ev=0.9, theta_alpha=1.1))
            (path / "config.json").write_text(json.dumps({"train": {"target_kl": 0.03, "max_grad_norm": 1}}))
            parsed = read_metrics(path / "metrics.csv")
            self.assertTrue(math.isnan(parsed[1]["validation_return"]))
            outputs = create_training_diagnostics(path, smoothing_window=2, dpi=40)
            self.assertEqual({p.name for p in outputs}, {"training_curves_M5.png", "training_curves_M10.png"})
            self.assertTrue(all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in outputs))


if __name__ == "__main__":
    unittest.main()
