"""
models/rl_strategy/reward_function.py

Reward function for the offline RL chase strategy module.

Design principles:
  - Reward is computed per delivery (immediate) plus terminal bonus.
  - Balances run scoring against wicket preservation.
  - Chase completion is rewarded with a terminal bonus.
  - Reward is risk-adjusted: bonus diminishes if achieved with heavy wicket cost.

Reward components:
  1. runs_above_expectation: batsman_runs vs baseline expected runs from state
  2. wicket_penalty: fixed penalty per wicket lost
  3. win_prob_shift: delta in win probability (from model) — forward-looking signal
  4. terminal_bonus: large positive reward if chase completed (added to final ball)

Units: dimensionless float. Interpreted only relative to other rewards in the dataset.
Not interpretable as "runs" or probability directly.

This reward design is a heuristic and must be disclosed as such.
RL agents trained on it are policy-estimators from historical behavior, not
agents that have discovered globally optimal cricket strategy.
"""

import numpy as np
import pandas as pd

from config import WICKET_PENALTY, RL_CHASE_WIN_BONUS


def compute_baseline_expected_runs(state: dict) -> float:
    """
    Heuristic: expected runs per ball from the current state.
    Approximated as RRR / 6 (runs needed per ball to stay on track).

    A more sophisticated version uses the EFS model prediction,
    but for reward computation we use this simple baseline to avoid
    coupling the reward function to a model.
    """
    rrr = state.get("pre_rrr", None)
    if rrr is None or np.isnan(float(rrr)):
        # Fallback: use CRR if no RRR (inning 1)
        rrr = state.get("pre_crr", 6.0)
    return float(rrr) / 6.0  # runs per ball to maintain rate


def reward(
    batsman_runs: int,
    extra_runs: int,
    is_wicket: bool,
    pre_win_prob: float,
    post_win_prob: float,
    is_terminal: bool,
    chase_success: bool,
    state: dict,
    weights: dict = None,
) -> float:
    """
    Compute reward for a single delivery in a chase context.

    Args:
        batsman_runs: runs scored by batter on this ball
        extra_runs: extra runs (not attributed to batter aggression directly)
        is_wicket: whether a wicket fell
        pre_win_prob: win probability before ball
        post_win_prob: win probability after ball
        is_terminal: True if this is the last ball of the innings
        chase_success: True if batting team won the match
        state: pre-delivery state dictionary (for expected baseline)
        weights: optional dict to override component weights

    Returns:
        float reward value
    """
    w = weights or {
        "runs_above": 0.40,
        "wp_shift":   0.30,
        "wicket_pen": 1.0,    # multiplied by WICKET_PENALTY
        "terminal":   1.0,    # multiplied by RL_CHASE_WIN_BONUS
    }

    # Component 1: Runs above expected baseline
    baseline  = compute_baseline_expected_runs(state)
    runs_delta = (batsman_runs - baseline) * w["runs_above"]

    # Component 2: Win probability shift (forward-looking signal)
    wp_shift = 0.0
    if pre_win_prob is not None and post_win_prob is not None:
        if not (np.isnan(pre_win_prob) or np.isnan(post_win_prob)):
            wp_shift = (float(post_win_prob) - float(pre_win_prob)) * w["wp_shift"]

    # Component 3: Wicket penalty
    wicket_cost = float(is_wicket) * WICKET_PENALTY * w["wicket_pen"]

    # Component 4: Terminal bonus
    terminal_bonus = 0.0
    if is_terminal and chase_success:
        terminal_bonus = RL_CHASE_WIN_BONUS * w["terminal"]

    total_reward = runs_delta + wp_shift - wicket_cost + terminal_bonus
    return round(float(total_reward), 4)


def build_reward_series(ball_states: pd.DataFrame) -> pd.Series:
    """
    Apply reward function to all inning-2 deliveries in the ball_states table.
    Returns a Series aligned to ball_states index.

    Assumptions:
      - pre_win_prob and post_win_prob columns must exist (from pipeline 05)
      - match_won column must exist (1/0)
      - is_terminal is proxied as pre_balls_remaining == 1 (last legal ball)
    """
    df = ball_states.copy()
    mask = df["inning"] == 2

    rewards = pd.Series(np.nan, index=df.index)

    for idx, row in df[mask].iterrows():
        state = {
            "pre_rrr": row.get("pre_rrr"),
            "pre_crr": row.get("pre_crr"),
        }
        r = reward(
            batsman_runs=int(row.get("batsman_runs", 0)),
            extra_runs=int(row.get("extra_runs", 0)),
            is_wicket=bool(row.get("is_wicket", 0)),
            pre_win_prob=row.get("pre_win_prob"),
            post_win_prob=row.get("post_win_prob"),
            is_terminal=bool(row.get("pre_balls_remaining", 99) <= 1),
            chase_success=bool(row.get("match_won", 0) == 1),
            state=state,
        )
        rewards.loc[idx] = r

    return rewards
