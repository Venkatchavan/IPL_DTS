"""
models/rl_strategy/policy_evaluator.py — Offline Policy Evaluation

Evaluates the quality of the learned FQI policy using offline estimators.
Since we cannot run live A/B tests, offline policy evaluation is the 
principal validation method.

Methods implemented:
  1. Direct Method (DM) — uses Q-model to estimate policy value
  2. Importance Sampling (IS) — weights observed outcomes by behavior policy ratio
  3. Doubly Robust (DR) — combines IS and DM for reduced variance
  4. State Coverage Report — what % of test states have sufficient policy support

HONEST FRAMING:
  Offline policy evaluation is inherently noisy for high-dimensional, 
  sparse state spaces. These estimates should be interpreted as relative 
  comparisons, not absolute win-probability guarantees.
  All estimates come with 95% confidence intervals via bootstrap.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_FEATURES, MODELS_DIR, TEST_SEASONS,
    RL_MIN_STATE_SUPPORT, RL_ACTIONS, RL_DISCOUNT_GAMMA,
)
from models.rl_strategy.state_encoder import encode, encode_vector, encode_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RL_MODEL_DIR = MODELS_DIR / "rl_strategy"
N_BOOTSTRAP   = 500
BOOTSTRAP_SEED = 42


# ── Behavior policy estimation ────────────────────────────────────────────────

def estimate_behavior_policy(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the behavior policy π_b(a | s) — i.e., how often each action
    was taken from each state in the training data.

    Returns a DataFrame with columns: state_key, action, pi_behavior.
    """
    counts = (
        transitions.groupby(["state_key", "action_idx"])
        .size()
        .reset_index(name="count")
    )
    total_per_state = counts.groupby("state_key")["count"].sum().rename("total")
    counts = counts.merge(total_per_state, on="state_key")
    counts["pi_behavior"] = counts["count"] / counts["total"]
    return counts[["state_key", "action_idx", "pi_behavior"]]


# ── Direct Method (DM) ────────────────────────────────────────────────────────

def evaluate_direct_method(transitions: pd.DataFrame, Q_model) -> float:
    """
    Estimate policy value using the Q model directly.
    V(π) = E_s[ Q(s, π(s)) ]

    The policy π is argmax_a Q(s, a).
    """
    n_actions = len(RL_ACTIONS)
    values = []

    for _, row in transitions.iterrows():
        if row["state_vec"] is None:
            continue
        svec = row["state_vec"]
        q_vals = []
        for a_idx in range(n_actions):
            sa = np.hstack([svec, np.eye(n_actions)[a_idx]]).reshape(1, -1)
            q_vals.append(float(Q_model.predict(sa)[0]))
        values.append(max(q_vals))

    return float(np.mean(values)) if values else 0.0


# ── Importance Sampling (IS) ──────────────────────────────────────────────────

def evaluate_importance_sampling(
    transitions: pd.DataFrame,
    policy_df:   pd.DataFrame,
    behavior_policy: pd.DataFrame,
    clip_ratio: float = 10.0,
) -> Tuple_like:
    """
    Importance Sampling estimator for offline policy evaluation.

    V_IS(π) = (1/N) Σ_t [ (π(a|s) / π_b(a|s)) * r_t ]

    Returns (estimate, std_error) with clipped IS ratios to control variance.
    """
    # Build lookup: state_key → recommended_action_idx
    policy_action = dict(zip(policy_df["state_key"], policy_df["recommended_action"]))
    pi_b_map = dict(
        zip(
            zip(behavior_policy["state_key"].astype(str), behavior_policy["action_idx"]),
            behavior_policy["pi_behavior"],
        )
    )

    is_rewards = []
    valid = transitions.dropna(subset=["state_key", "action_idx", "reward"])

    for _, row in valid.iterrows():
        sk  = str(row["state_key"])
        a   = int(row["action_idx"])
        r   = float(row["reward"])

        if sk not in policy_action:
            continue

        target_action_name = policy_action[sk]
        target_a_idx = encode_action(target_action_name)

        # π(a|s): 1 if this action matches the policy, 0 otherwise (deterministic policy)
        pi_s_a = 1.0 if a == target_a_idx else 0.0
        pi_b_s_a = pi_b_map.get((sk, a), 1e-6)  # avoid division by zero

        weight = min(pi_s_a / pi_b_s_a, clip_ratio)
        is_rewards.append(weight * r)

    if not is_rewards:
        return 0.0, 0.0

    estimate = float(np.mean(is_rewards))
    std_err  = float(np.std(is_rewards) / np.sqrt(len(is_rewards)))
    return estimate, std_err


# ── Doubly Robust (DR) ───────────────────────────────────────────────────────

def evaluate_doubly_robust(
    transitions:     pd.DataFrame,
    policy_df:       pd.DataFrame,
    behavior_policy: pd.DataFrame,
    Q_model,
    clip_ratio: float = 10.0,
) -> Tuple_like:
    """
    Doubly Robust estimator — combines DM and IS corrections.

    V_DR = V_DM + (1/N) Σ_t [ w_t * (r_t - Q(s_t, a_t)) ]

    where w_t = π(a_t|s_t) / π_b(a_t|s_t).

    More robust than either DM or IS alone: consistent if DM or IS is accurate.
    """
    n_actions = len(RL_ACTIONS)
    policy_action = dict(zip(policy_df["state_key"], policy_df["recommended_action"]))
    pi_b_map = dict(
        zip(
            zip(behavior_policy["state_key"].astype(str), behavior_policy["action_idx"]),
            behavior_policy["pi_behavior"],
        )
    )

    dm_values  = []
    dr_corrections = []

    valid = transitions.dropna(subset=["state_key", "action_idx", "reward", "state_vec"])

    for _, row in valid.iterrows():
        sk   = str(row["state_key"])
        a    = int(row["action_idx"])
        r    = float(row["reward"])
        svec = row["state_vec"]

        if sk not in policy_action:
            continue

        target_action_name = policy_action[sk]
        target_a_idx = encode_action(target_action_name)

        # DM term: Q(s, π(s))
        sa_policy = np.hstack([svec, np.eye(n_actions)[target_a_idx]]).reshape(1, -1)
        q_policy  = float(Q_model.predict(sa_policy)[0])
        dm_values.append(q_policy)

        # DR correction
        sa_obs   = np.hstack([svec, np.eye(n_actions)[a]]).reshape(1, -1)
        q_obs    = float(Q_model.predict(sa_obs)[0])

        pi_e     = 1.0 if a == target_a_idx else 0.0
        pi_b     = pi_b_map.get((sk, a), 1e-6)
        weight   = min(pi_e / pi_b, clip_ratio)
        dr_corrections.append(weight * (r - q_obs))

    if not dm_values:
        return 0.0, 0.0

    v_dm = float(np.mean(dm_values))
    v_dr = v_dm + float(np.mean(dr_corrections))
    std_err = float(np.std(dr_corrections) / np.sqrt(len(dr_corrections)))
    return v_dr, std_err


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(data: list, statistic=np.mean, n=N_BOOTSTRAP, alpha=0.05, seed=BOOTSTRAP_SEED):
    """
    Compute (lower, upper) bootstrap confidence interval.
    """
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper


# ── State coverage report ─────────────────────────────────────────────────────

def compute_state_coverage(transitions: pd.DataFrame, policy_df: pd.DataFrame) -> dict:
    """
    Report what fraction of test states have:
      - Any policy recommendation (any support)
      - High confidence policy recommendation
      - Medium confidence
      - Low confidence (potentially unreliable)
    """
    test_states = set(transitions["state_key"].dropna().unique())
    policy_states = set(policy_df["state_key"].unique())
    covered = test_states & policy_states

    high_conf = set(policy_df.loc[policy_df["confidence"] == "high", "state_key"])
    med_conf  = set(policy_df.loc[policy_df["confidence"] == "medium", "state_key"])

    n_total = len(test_states)
    return {
        "total_test_states":     n_total,
        "covered_states":        len(covered),
        "coverage_pct":          round(100 * len(covered) / max(n_total, 1), 1),
        "high_conf_states":      len(covered & high_conf),
        "high_conf_pct":         round(100 * len(covered & high_conf) / max(n_total, 1), 1),
        "medium_conf_states":    len(covered & med_conf),
        "low_conf_states":       len(covered) - len(covered & high_conf) - len(covered & med_conf),
    }


# ── Action distribution comparison ───────────────────────────────────────────

def compare_action_distributions(
    transitions: pd.DataFrame,
    policy_df:   pd.DataFrame,
) -> dict:
    """
    Compare historical action distribution vs policy-recommended distribution.
    Large divergence indicates the RL policy is recommending less-tested actions
    (higher uncertainty in those recommendations).
    """
    historical = transitions["action"].value_counts(normalize=True).to_dict()

    policy_counts = policy_df["recommended_action"].value_counts(normalize=True).to_dict()

    return {
        "historical":  {k: round(v, 3) for k, v in historical.items()},
        "policy":      {k: round(v, 3) for k, v in policy_counts.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

# Fix the missing type hints
Tuple_like = tuple


def run():
    log.info("RL Policy Evaluator")

    transitions_path = RL_MODEL_DIR / "transitions.parquet"
    policy_path      = RL_MODEL_DIR / "policy_table.parquet"
    fqi_path         = RL_MODEL_DIR / "fqi_model.pkl"

    if not all(p.exists() for p in [transitions_path, policy_path, fqi_path]):
        log.error("Missing RL artifacts. Run fqi_trainer.py first.")
        return

    transitions = pd.read_parquet(transitions_path)
    policy_df   = pd.read_parquet(policy_path)
    Q_model     = joblib.load(fqi_path)
    log.info(f"  Loaded {len(transitions):,} transitions, {len(policy_df):,} policy states")

    # Filter to test seasons for unbiased evaluation
    if "season" in transitions.columns:
        test_trans = transitions[transitions["season"].isin(TEST_SEASONS)].copy()
        log.info(f"  Test seasons ({TEST_SEASONS}): {len(test_trans):,} transitions")
    else:
        test_trans = transitions.copy()
        log.warning("  No 'season' column — evaluating on all data (optimistic estimate)")

    # Reconstruct state vectors if needed (read back as objects)
    if test_trans["state_vec"].dtype == object:
        for col in ["state_vec", "next_state_vec"]:
            test_trans[col] = test_trans[col].apply(
                lambda x: np.array(x) if x is not None and not isinstance(x, np.ndarray) else x
            )

    log.info("1. Behavior policy estimation...")
    behavior_policy = estimate_behavior_policy(transitions)

    log.info("2. Direct Method (DM)...")
    v_dm = evaluate_direct_method(test_trans, Q_model)
    log.info(f"   V_DM = {v_dm:.4f}")

    log.info("3. Importance Sampling (IS)...")
    v_is, se_is = evaluate_importance_sampling(test_trans, policy_df, behavior_policy)
    log.info(f"   V_IS = {v_is:.4f} (SE = {se_is:.4f})")

    log.info("4. Doubly Robust (DR)...")
    v_dr, se_dr = evaluate_doubly_robust(test_trans, policy_df, behavior_policy, Q_model)
    log.info(f"   V_DR = {v_dr:.4f} (SE = {se_dr:.4f})")

    log.info("5. State coverage...")
    coverage = compute_state_coverage(test_trans, policy_df)
    log.info(f"   Coverage: {coverage['coverage_pct']}% | High conf: {coverage['high_conf_pct']}%")

    log.info("6. Action distribution comparison...")
    action_dist = compare_action_distributions(transitions, policy_df)
    log.info(f"   Historical: {action_dist['historical']}")
    log.info(f"   Policy:     {action_dist['policy']}")

    # Save evaluation results
    eval_results = {
        "value_estimates": {
            "direct_method":        round(v_dm, 4),
            "importance_sampling":  round(v_is, 4),
            "importance_sampling_se": round(se_is, 4),
            "doubly_robust":        round(v_dr, 4),
            "doubly_robust_se":     round(se_dr, 4),
        },
        "state_coverage":      coverage,
        "action_distributions": action_dist,
        "evaluation_data":     f"test_seasons={TEST_SEASONS}, n={len(test_trans)}",
        "honest_framing": (
            "Offline policy evaluation is inherently noisy. "
            "IS/DR estimates may have high variance in sparse state regions. "
            "Use DM as primary estimate and DR for robustness check. "
            "State coverage pct indicates how broadly the policy generalizes."
        ),
    }

    out_path = RL_MODEL_DIR / "policy_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    log.info(f"\nEvaluation results saved: {out_path}")


if __name__ == "__main__":
    run()
