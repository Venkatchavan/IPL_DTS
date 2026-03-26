"""
models/rl_strategy/fqi_trainer.py — Fitted Q-Iteration (FQI) Trainer

Implements offline RL via Fitted Q-Iteration on historical IPL ball-by-ball data.

HONEST FRAMING (must be understood before using this module):
  - This is NOT a real-time RL agent or simulation-based optimizer.
  - This is a historical policy estimator — it asks:
    "From states like this one, which action (conservative / balanced / aggressive)
     was historically associated with better outcomes?"
  - The learned Q-function approximates expected cumulative reward under 
    historical team behavior distributions.
  - Distributional shift warning: if a match state is rarely observed in
    training data, Q-values for that state will be unreliable.
  - All recommendations from this module should be displayed with confidence 
    levels based on state support counts.

Algorithm: Fitted Q-Iteration (Ernst, Geurts, Wehenkel 2005)
  - Build dataset of (state, action, reward, next_state) tuples from historical data
  - Iteratively fit a regression function Q(s, a) → expected return
  - After K iterations, the policy = argmax_a Q(s, a)

Action inference (heuristic — must be disclosed in dashboard):
  Actions are inferred from observed batting behavior per over:
    - aggressive: run_rate > RRR + 2.0 OR boundary_pct > 35%
    - conservative: run_rate < RRR - 1.5 AND dot_pct > 40%
    - balanced: everything else
  This labeling is a proxy — actual intent is not directly observable.
"""

import sys
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    BALL_STATES_FILE, DATA_FEATURES, MODELS_DIR,
    RL_DISCOUNT_GAMMA, RL_MAX_ITERATIONS, RL_MIN_STATE_SUPPORT,
    RL_ACTIONS, RL_ACTION_CONSERVATIVE, RL_ACTION_BALANCED, RL_ACTION_AGGRESSIVE,
)
from models.rl_strategy.state_encoder import encode, encode_vector, encode_action, decode_action
from models.rl_strategy.reward_function import build_reward_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RL_MODEL_DIR = MODELS_DIR / "rl_strategy"
RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Action inference from observed behavior ───────────────────────────────────

def infer_action(row: pd.Series) -> str:
    """
    Infer the batting action intent from observed delivery behavior.
    This is a heuristic approximation of strategic intent.

    Over-level context is preferred; ball-level fallback used here.
    """
    batsman_runs = float(row.get("batsman_runs", 0))
    is_boundary  = (batsman_runs == 4 or batsman_runs == 6)
    is_dot       = (batsman_runs == 0 and float(row.get("extra_runs", 0)) == 0)
    rrr          = float(row.get("pre_rrr", 6.0) or 6.0)
    crr          = float(row.get("pre_crr", 6.0) or 6.0)

    # Aggressive: hit a boundary OR run rate well above RRR
    if is_boundary or (batsman_runs >= 2 and crr > rrr + 1.5):
        return RL_ACTION_AGGRESSIVE

    # Conservative: dot ball when under pressure (should have scored)
    if is_dot and rrr > crr + 1.5:
        return RL_ACTION_CONSERVATIVE

    return RL_ACTION_BALANCED


# ── Build transition dataset ──────────────────────────────────────────────────

def build_transitions(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Build (state, action, reward, next_state) transition table for inning 2.

    Each row represents one legal delivery with:
      - state_key: encoded state tuple before delivery
      - state_vec: encoded state vector before delivery
      - action_idx: inferred action (0=conservative, 1=balanced, 2=aggressive)
      - reward: computed reward for this delivery
      - next_state_key: encoded state after delivery
      - next_state_vec: encoded state vector after delivery
      - is_terminal: True if last legal ball of innings
    """
    log.info("Building RL transition dataset...")

    inning2 = ball_states[
        (ball_states["inning"] == 2) &
        (ball_states["is_legal_ball"] == 1)
    ].copy().reset_index(drop=True)

    # Compute rewards
    log.info("  Computing rewards...")
    rewards = build_reward_series(ball_states)
    inning2["reward"] = rewards.reindex(inning2.index).values

    # Infer venue tier (requires venue_features or fallback)
    inning2["venue_scoring_tier"] = "medium"  # fallback; enhanced in full pipeline

    # Encode states (vectorized via apply)
    log.info("  Encoding states...")

    def row_to_state_dict(row):
        return {
            "runs_needed":        row.get("pre_runs_needed", 0),
            "balls_remaining":    row.get("pre_balls_remaining", 120),
            "wickets_in_hand":    row.get("pre_wickets_in_hand", 10),
            "phase":              row.get("phase", "middle"),
            "rrr_crr_delta":      row.get("pre_rrr_crr_delta", 0),
            "venue_scoring_tier": row.get("venue_scoring_tier", "medium"),
            "dot_ball_streak":    row.get("pre_dot_ball_streak", 0),
        }

    state_dicts = inning2.apply(row_to_state_dict, axis=1)
    state_keys  = [encode(s) for s in state_dicts]
    state_vecs  = [encode_vector(s) for s in state_dicts]

    inning2["state_key"] = state_keys
    inning2["state_vec"] = state_vecs

    # Infer actions
    inning2["action"]     = inning2.apply(infer_action, axis=1)
    inning2["action_idx"] = inning2["action"].map(encode_action)

    # Next-state: the state of the following delivery in the same innings
    inning2 = inning2.sort_values(["match_id", "global_ball_number"]).reset_index(drop=True)

    next_keys = state_keys[1:] + [None]   # shift by 1
    next_vecs = state_vecs[1:] + [None]

    # Reset next_state at end of each match-inning
    for i in range(len(inning2) - 1):
        if (inning2.loc[i, "match_id"]    != inning2.loc[i + 1, "match_id"] or
            inning2.loc[i, "inning"] != inning2.loc[i + 1, "inning"]):
            next_keys[i] = None
            next_vecs[i] = None

    inning2["next_state_key"] = next_keys
    inning2["next_state_vec"] = next_vecs

    # Terminal flag: no valid next state
    inning2["is_terminal"] = inning2["next_state_key"].isna()

    # Fill null rewards with 0
    inning2["reward"] = inning2["reward"].fillna(0.0)

    log.info(f"  Transitions built: {len(inning2):,} records")
    return inning2


# ── Fitted Q-Iteration ────────────────────────────────────────────────────────

def run_fqi(transitions: pd.DataFrame, n_actions: int = 3) -> object:
    """
    Run Fitted Q-Iteration to learn Q(s, a) → expected return.

    Algorithm:
      1. Initialize Q targets as immediate rewards
      2. For K iterations:
         a. Compute Bellman targets: y = r + gamma * max_a Q(s', a)
         b. Refit Q regressor on (s, a) → y
      3. Return final Q model

    Uses XGBoost as the Q function approximator.
    State-action concatenation: [state_vec, onehot(action)]
    """
    try:
        import xgboost as xgb
        Q_model_class = xgb.XGBRegressor
        model_kwargs = dict(n_estimators=100, learning_rate=0.1, max_depth=5,
                            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        Q_model_class = GradientBoostingRegressor
        model_kwargs = dict(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    # Safe null checks — state_vec column holds numpy arrays, so .isna()/.dropna()
    # would misfire; use explicit Python-level None checks instead.
    sv_valid_mask = np.array([x is not None for x in transitions["state_vec"].values])
    rw_valid_mask = transitions["reward"].notna().values
    valid = transitions[sv_valid_mask & rw_valid_mask].copy().reset_index(drop=True)

    # Build state-action feature matrix
    def sa_features(state_vecs, action_idxs):
        X_s = np.vstack(state_vecs)
        X_a = np.eye(n_actions)[action_idxs.astype(int)]  # one-hot
        return np.hstack([X_s, X_a])

    X = sa_features(valid["state_vec"].values, valid["action_idx"].values)
    y = valid["reward"].values.astype(float)

    log.info(f"  FQI: {len(valid):,} transitions, {X.shape[1]} features")
    log.info(f"  Starting {RL_MAX_ITERATIONS} iterations...")

    Q = Q_model_class(**model_kwargs)
    Q.fit(X, y)

    # Precompute next-state feature matrices for all 3 actions (vectorized Bellman)
    next_vecs = valid["next_state_vec"].values
    # Safe None-check on object column containing numpy arrays
    next_is_none = np.array([x is None for x in next_vecs])
    terminal  = valid["is_terminal"].values | next_is_none
    rewards_v  = valid["reward"].values.astype(float)
    eye3 = np.eye(n_actions)

    for iteration in range(RL_MAX_ITERATIONS):
        # Build next-state matrices for each action at once
        non_term_mask = ~terminal
        if non_term_mask.any():
            next_stacked = np.vstack([v for v, t in zip(next_vecs, terminal) if not t])
            q_next_all = np.column_stack([
                Q.predict(np.hstack([next_stacked, np.tile(eye3[a], (len(next_stacked), 1))]))
                for a in range(n_actions)
            ])  # shape: (n_non_terminal, n_actions)
            max_q_next = q_next_all.max(axis=1)

        y_new = rewards_v.copy()
        nt_idx = 0
        for i, is_term in enumerate(terminal):
            if not is_term:
                y_new[i] = rewards_v[i] + RL_DISCOUNT_GAMMA * max_q_next[nt_idx]
                nt_idx += 1

        Q.fit(X, y_new)

        if (iteration + 1) % 10 == 0:
            log.info(f"  Iteration {iteration + 1}/{RL_MAX_ITERATIONS} complete")

    log.info("  FQI training complete.")
    return Q


# ── Policy extraction ─────────────────────────────────────────────────────────

def build_policy_table(transitions: pd.DataFrame, Q_model) -> pd.DataFrame:
    """
    Build a state → recommended_action lookup table.
    Also computes state support counts and confidence flags.

    For each unique state_key, evaluates Q(s, a) for all actions and
    returns the argmax action as the policy recommendation.
    """
    n_actions = len(RL_ACTIONS)

    # Support counts per state
    state_support = (
        transitions.groupby("state_key").size()
        .reset_index(name="support_count")
    )

    # Build state→vec map (safe None check — state_vec is object column of arrays)
    sv_mask = np.array([x is not None for x in transitions["state_vec"].values])
    t_valid = transitions[sv_mask & transitions["state_key"].notna()].copy()
    state_vec_map = (
        t_valid.groupby("state_key")["state_vec"]
        .first()
        .to_dict()
    )

    # Batch predict Q-values for all unique states at once
    eye3 = np.eye(n_actions)
    unique_keys  = list(state_vec_map.keys())
    state_matrix = np.vstack([state_vec_map[k] for k in unique_keys])  # (N, state_dim)

    q_cols = {}
    for a_idx in range(n_actions):
        a_tile = np.tile(eye3[a_idx], (len(state_matrix), 1))
        sa_mat  = np.hstack([state_matrix, a_tile])
        q_cols[RL_ACTIONS[a_idx]] = Q_model.predict(sa_mat)

    q_matrix     = np.column_stack([q_cols[a] for a in RL_ACTIONS])  # (N, 3)
    best_idxs    = q_matrix.argmax(axis=1)
    best_actions = [RL_ACTIONS[i] for i in best_idxs]

    policy_rows = [
        {
            "state_key":          str(k),
            "recommended_action": best_actions[i],
            "q_conservative":     round(float(q_cols["conservative"][i]), 4),
            "q_balanced":         round(float(q_cols["balanced"][i]), 4),
            "q_aggressive":       round(float(q_cols["aggressive"][i]), 4),
        }
        for i, k in enumerate(unique_keys)
    ]

    policy_df = pd.DataFrame(policy_rows)
    policy_df = policy_df.merge(
        state_support.assign(state_key=state_support["state_key"].astype(str)),
        on="state_key", how="left"
    )
    policy_df["confidence"] = policy_df["support_count"].apply(
        lambda n: "high" if n >= RL_MIN_STATE_SUPPORT * 2
        else "medium" if n >= RL_MIN_STATE_SUPPORT
        else "low"
    )

    log.info(f"  Policy table: {len(policy_df):,} distinct states")
    log.info(f"  High confidence states: {(policy_df['confidence']=='high').sum()}")
    log.info(f"  Low confidence states:  {(policy_df['confidence']=='low').sum()}")

    return policy_df


# ── Historical win rates per action ───────────────────────────────────────────

def compute_historical_win_rates(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute empirical win rates under each action per state.
    These are used in the Strategy Lab dashboard as empirical validation
    alongside Q-values.

    Important: This is observational win rate, not causal.
    """
    df = transitions.dropna(subset=["action", "state_key"]).copy()
    df["match_won"] = df["match_won"].fillna(0).astype(int)

    win_rates = (
        df.groupby(["state_key", "action"])
        .agg(
            n_instances =("match_id", "count"),
            win_rate    =("match_won", "mean"),
        )
        .reset_index()
    )
    win_rates["state_key"] = win_rates["state_key"].astype(str)
    return win_rates


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("RL Strategy Module: FQI Training")

    scored_path = DATA_FEATURES / "ball_states_scored.parquet"
    if scored_path.exists():
        ball_states = pd.read_parquet(scored_path)
        log.info(f"  Loaded scored ball states: {len(ball_states):,} rows")
    else:
        log.warning("ball_states_scored.parquet not found — using unscored ball_states (WP-dependent reward will be empty)")
        ball_states = pd.read_parquet(BALL_STATES_FILE)
        # Add placeholder win prob columns so reward function doesn't crash
        ball_states["pre_win_prob"]  = np.nan
        ball_states["post_win_prob"] = np.nan

    transitions = build_transitions(ball_states)
    transitions.to_parquet(RL_MODEL_DIR / "transitions.parquet", index=False)
    log.info(f"  Saved transitions: {RL_MODEL_DIR / 'transitions.parquet'}")

    log.info("Training FQI model...")
    Q_model = run_fqi(transitions)
    joblib.dump(Q_model, RL_MODEL_DIR / "fqi_model.pkl")
    log.info(f"  Saved FQI model: {RL_MODEL_DIR / 'fqi_model.pkl'}")

    log.info("Building policy table...")
    policy_df = build_policy_table(transitions, Q_model)
    policy_df.to_parquet(RL_MODEL_DIR / "policy_table.parquet", index=False)
    log.info(f"  Saved policy table: {RL_MODEL_DIR / 'policy_table.parquet'}")

    log.info("Computing historical win rates per action...")
    win_rates = compute_historical_win_rates(transitions)
    win_rates.to_parquet(RL_MODEL_DIR / "historical_win_rates.parquet", index=False)

    # Save metadata
    action_dist = transitions["action"].value_counts().to_dict()
    meta = {
        "algorithm":           "Fitted Q-Iteration (FQI)",
        "discount_gamma":      RL_DISCOUNT_GAMMA,
        "iterations":          RL_MAX_ITERATIONS,
        "n_transitions":       len(transitions),
        "n_unique_states":     len(transitions["state_key"].unique()),
        "action_distribution": action_dist,
        "honest_framing": (
            "This module estimates historically associated action quality, "
            "not causally optimal strategy. "
            "Recommendations should be treated as pattern-based priors, "
            "not guaranteed optimal plays."
        ),
    }
    with open(RL_MODEL_DIR / "rl_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("\nRL training complete.")
    log.info(f"  Action distribution: {action_dist}")


if __name__ == "__main__":
    run()
