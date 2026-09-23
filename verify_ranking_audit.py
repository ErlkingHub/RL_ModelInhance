"""Replay a saved training-label snapshot. No optimizer or benchmark data."""
import argparse
import json
import torch
from mkp_config import ModelConfig, TrainConfig, CONTROLLER_VERSION, DECODER_VERSION
from mkp_env import MKPState
from mkp_Model import MKPModel, StrategyTheta
from mkp_theta_candidates import InducedActions, decode_proposals
from mkp_ranking import THETA_RANKING_VERSION, candidate_terminal_returns, action_comparisons
from mkp_utils import resolve_device, seed_everything


@torch.no_grad()
def verify_audit(path, device=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("ranking_protocol") != THETA_RANKING_VERSION
            or saved.get("controller_version") != CONTROLLER_VERSION
            or saved.get("decoder_version") != DECODER_VERSION):
        raise ValueError("Audit protocol/controller/decoder does not match this implementation")
    device = resolve_device(device or saved["device"])
    if device.type != torch.device(saved["device"]).type:
        raise ValueError("Exact stochastic replay requires the original CPU/CUDA RNG backend")
    seed_everything(0, deterministic=saved["deterministic"])
    model = MKPModel(ModelConfig(**saved["feature_config"])).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["model_state"], strict=True)
    model.reuse_state_features = saved["reuse_state_features"]
    state = MKPState(**{key: value.to(device) for key, value in saved["state"].items()})
    proposals = [StrategyTheta(**{key: value.to(device) for key, value in p.items()})
                 for p in saved["theta_proposals"]]
    decoded = decode_proposals(model, state, proposals)
    torch.testing.assert_close(decoded.cpu(), saved["decoded_actions"], rtol=0, atol=0)
    # The first proposal must really be the saved behavior controller's theta.
    from dataclasses import fields
    current = model.policy(state)[1]
    for field in fields(current):
        torch.testing.assert_close(getattr(current, field.name), getattr(proposals[0], field.name),
                                   rtol=2e-5, atol=2e-6)
    induced = InducedActions(*(saved[key].to(device) for key in
                              ("owners", "actions", "theta_witnesses", "action_counts")))
    torch.testing.assert_close(decoded[induced.owners, induced.witnesses], induced.items, rtol=0, atol=0)
    config = TrainConfig(**saved["train_config"])
    rng = torch.Generator(device=device).set_state(saved["continuation_rng_state"])
    greedy = candidate_terminal_returns(model, state, induced.owners, induced.items, 1, False)
    sampled = candidate_terminal_returns(model, state, induced.owners, induced.items,
                                         config.ranking_repeats, True, rng)
    torch.testing.assert_close(greedy.cpu(), saved["greedy_q"], rtol=0, atol=0)
    torch.testing.assert_close(sampled.cpu(), saved["sampled_q"], rtol=0, atol=0)
    _, anchors, alternatives, delta, se, keep = action_comparisons(state, induced, greedy, sampled, config)
    for actual, key in ((anchors, "anchor_indices"), (alternatives, "alternative_indices"),
                        (delta, "delta"), (se, "standard_error"), (keep, "keep")):
        torch.testing.assert_close(actual.cpu(), saved[key], rtol=0, atol=0)
    return dict(verified=True, protocol=THETA_RANKING_VERSION, device=str(device),
                states=state.profits.shape[0], unique_actions=induced.items.numel(),
                accepted_pairs=int(keep.sum()), improved_pairs=int((keep & (delta < 0)).sum()),
                saved_torch=saved["torch_version"], replay_torch=str(torch.__version__),
                note="Reproduces frozen-policy labels, not optimal Q values or held-out improvement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--device", default=None, help="Default: original backend; CPU/CUDA must match the snapshot")
    args = parser.parse_args()
    print(json.dumps(verify_audit(args.snapshot, args.device), indent=2))
