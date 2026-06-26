"""
Counterfactual Evaluation — Inverse Propensity Scoring (IPS)

Standard AUC overestimates model quality because training data is biased
by the previous ranking system (position bias). IPS corrects for this.

IPS Estimator:  R_IPS = (1/N) Σ (y_i * r_i) / p_i
  where p_i = propensity (probability of observation given position)
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def estimate_propensities(positions, alpha=0.5, beta=0.5):
    """
    Estimate propensities using the inverse of position-based exposure.

    p(position) ∝ (1 / position)^alpha  *  exp(beta * position)

    Uses the standard position-based model (Joachims et al. 2017).
    """
    positions = np.maximum(positions, 1)
    props = (1.0 / positions) ** alpha * np.exp(-beta * positions)
    props = props / props.max()
    return np.clip(props, 0.01, 1.0)


def ips_weighted_auc(y_true, y_pred, propensities, session_ids=None):
    """
    Compute IPS-weighted AUC.

    Each observation is weighted by 1/propensity to correct for position bias.
    """
    weights = 1.0 / np.maximum(propensities, 0.01)
    weights = weights / weights.mean()

    if session_ids is not None:
        unique_sessions = np.unique(session_ids)
        aucs = []
        for sid in unique_sessions:
            mask = session_ids == sid
            if mask.sum() < 2:
                continue
            if len(np.unique(y_true[mask])) < 2:
                continue
            try:
                session_weights = weights[mask]
                auc = roc_auc_score(y_true[mask], y_pred[mask], sample_weight=session_weights)
                aucs.append(auc)
            except ValueError:
                continue
        return np.mean(aucs) if aucs else 0.0
    else:
        return roc_auc_score(y_true, y_pred, sample_weight=weights)


def ips_estimator(y_true, y_pred, propensities, reward_func=None):
    """
    Inverse Propensity Scoring estimator.

    Args:
        y_true:  Ground truth relevance (0/1)
        y_pred:  Predicted scores
        propensities: Observation propensities
        reward_func: Function to compute reward from prediction

    Returns:
        IPS estimate of the reward
    """
    if reward_func is None:
        reward_func = lambda y, yhat: (y == 1).astype(float)

    rewards = reward_func(y_true, y_pred)
    ips = np.mean(rewards / np.maximum(propensities, 0.01))
    return ips


def self_normalized_ips(y_true, y_pred, propensities, reward_func=None):
    """Self-normalized IPS estimator for improved variance."""
    if reward_func is None:
        reward_func = lambda y, yhat: (y == 1).astype(float)

    rewards = reward_func(y_true, y_pred)
    w = 1.0 / np.maximum(propensities, 0.01)
    snips = np.sum(w * rewards) / np.sum(w)
    return snips


def counterfactual_evaluation(model, data, position_col="rank_position"):
    """
    Full counterfactual evaluation pipeline.

    1. Estimate propensities from position bias
    2. Compute standard AUC
    3. Compute IPS-weighted AUC
    4. Report the gap (overestimation due to position bias)
    """
    y_true = data["clicked"].values
    y_pred = model.predict(data.drop(columns=["clicked", "ordered"]))
    positions = data[position_col].values
    session_ids = data.get("session_id", np.arange(len(data)))

    props = estimate_propensities(positions)

    std_auc = roc_auc_score(y_true, y_pred)
    ips_auc = ips_weighted_auc(y_true, y_pred, props, session_ids)
    ips_est = ips_estimator(y_true, y_pred, props)
    snips_est = self_normalized_ips(y_true, y_pred, props)

    print("Counterfactual Evaluation Results")
    print(f"  Standard AUC:         {std_auc:.4f}")
    print(f"  IPS-Weighted AUC:     {ips_auc:.4f}")
    print(f"  Overestimation gap:   {std_auc - ips_auc:.4f}")
    print(f"  IPS Estimate:         {ips_est:.4f}")
    print(f"  Self-Normalized IPS:  {snips_est:.4f}")

    return {
        "standard_auc": std_auc,
        "ips_auc": ips_auc,
        "bias_gap": std_auc - ips_auc,
        "ips_estimate": ips_est,
        "snips_estimate": snips_est,
    }


if __name__ == "__main__":
    print("Counterfactual Evaluation — Inverse Propensity Scoring")
    print()
    print("Propensity model:  p(pos) ∝ (1/pos)^α * exp(-β * pos)")
    print(f"  Default: α=0.5, β=0.5")
    print()
    print("Usage:")
    print("  props = estimate_propensities(positions)")
    print("  results = counterfactual_evaluation(model, data)")
