"""Offline fit objectives and fit-only selection, separate from inference."""
import math

import torch

from mkp_ranking import regret_margin_losses


def action_fit_terms(policy, record, *, objective="regret_margin", anchor=None, score_scale=.2,
                     return_scale=.002, margin_cap=1.):
    """Only queried values define ordering; unknown actions get no value label.

    KL uses every feasible action, but treats it as an observed behavior
    probability, never an invented reward. Scores, not temperature, set order.
    """
    scores = policy.score_terms["score"] / score_scale
    targets = torch.isin(policy.items, record["target_actions"].to(policy.items.device))
    if not targets.any():
        raise ValueError("No feasible diagnostic target")
    if objective == "set_ce":
        loss = torch.logsumexp(scores, 0) - torch.logsumexp(scores[targets], 0)
    elif objective == "regret_margin":
        values = record["future_values"].to(scores.device)[policy.items].detach()
        peers = torch.isfinite(values) & ~targets
        if peers.any():
            delta = (record["reference_future_value"] - values[peers]) / max(record["reference_future_value"], 1e-12)
            margin = scores[targets].amax() - scores[peers]
            loss = regret_margin_losses(margin, delta, return_scale, margin_cap).mean()
        else:
            loss = scores.sum() * 0
    else:
        raise ValueError("Unknown diagnostic objective")
    if anchor is None:
        kl = scores.sum() * 0
    else:
        if not torch.equal(policy.items, anchor["items"].to(policy.items.device)):
            raise ValueError("Anchor action support changed")
        old = anchor["log_probs"].to(policy.log_probs).detach()
        kl = (old.exp() * (old - policy.log_probs)).sum().clamp_min(0)
    return loss, kl


def fit_candidate_key(rows, label_loss, max_kl, kl_budget):
    """Rank candidates on fit only; never select with heldout measurements.

    First maximize actual continuation gain / initial total profit (the PPO
    objective scale), then hit rate, then loss. Reject drift beyond the budget.
    The untouched source actor is always a candidate, so selection cannot lower
    the measured mean fit normalized continuation return. No heldout guarantee.
    """
    if not rows or any(row["split"] != "fit" for row in rows):
        raise ValueError("Actor selection accepts only nonempty fit measurements")
    if not math.isfinite(max_kl) or max_kl > kl_budget or not math.isfinite(label_loss):
        return None
    return (sum(row["greedy_delta_over_initial_profit"] for row in rows) / len(rows),
            sum(row["target_hit"] for row in rows) / len(rows), -label_loss)
