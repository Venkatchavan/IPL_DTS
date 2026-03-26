"""
state_difficulty.py — State Difficulty Score (SDS)

SDS quantifies the objective difficulty of the batting situation on arrival
at a given ball. Uses only pre-ball features (no leakage).

Unlike Pressure Index (which focuses on momentum pressure), SDS also accounts
for phase difficulty, venue context, and bowling strength.

Formula:
  SDS = w1 * normalized(rrr_crr_delta)        [chase pressure]
      + w2 * normalized(wickets_lost)           [resource depletion]
      + w3 * phase_difficulty                   [phase baseline difficulty]
      + w4 * normalized(dot_ball_streak)        [incoming pressure]
      + w5 * normalized(venue_difficulty)       [venue scoring difficulty]

Output: SDS in range 0–100. High score = harder state for batting team.

Application: Context-adjust all player contribution metrics by dividing
by SDS (or using as weight) to identify players who perform in hard states.

Limitations:
  - Bowling strength proxy is a season-average; not delivery-specific
  - Venue difficulty uses historical averages; conditions vary per match
"""

import numpy as np
import pandas as pd

# Normalization constants
RRR_CRR_DELTA_MAX  = 8.0
RRR_CRR_DELTA_MIN  = -4.0
DOT_STREAK_MAX     = 12.0

SDS_WEIGHTS = {
    "rrr_crr":    0.30,
    "wickets":    0.25,
    "phase":      0.20,
    "dot_streak": 0.15,
    "venue":      0.10,
}

PHASE_DIFFICULTY = {
    "powerplay": 0.40,  # Fielding restrictions — batting is less difficult
    "middle":    0.65,
    "death":     0.90,
}


def compute_state_difficulty(df: pd.DataFrame, venue_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute State Difficulty Score for every ball state row.
    Appends 'state_difficulty' column (0–100 float).

    Args:
        df: ball_states DataFrame
        venue_features: optional venue baseline DataFrame for venue difficulty scoring
    """
    df = df.copy()

    # ── Component 1: RRR-CRR delta ────────────────────────────────────────────
    rrr_crr = df["pre_rrr_crr_delta"].fillna(0.0).clip(RRR_CRR_DELTA_MIN, RRR_CRR_DELTA_MAX)
    norm_rrr = (rrr_crr - RRR_CRR_DELTA_MIN) / (RRR_CRR_DELTA_MAX - RRR_CRR_DELTA_MIN)
    norm_rrr = norm_rrr.where(df["inning"] == 2, 0.0)

    # ── Component 2: Wickets lost ─────────────────────────────────────────────
    norm_wickets = (df["pre_wickets"] / 10.0).clip(0, 1)

    # ── Component 3: Phase difficulty ─────────────────────────────────────────
    phase_diff = df["phase"].map(PHASE_DIFFICULTY).fillna(0.65)

    # ── Component 4: Dot ball streak ──────────────────────────────────────────
    norm_dot = (df["pre_dot_ball_streak"].clip(0, DOT_STREAK_MAX) / DOT_STREAK_MAX)

    # ── Component 5: Venue difficulty (inverse of scoring ease) ───────────────
    # High-scoring venues are EASIER to bat at → lower difficulty
    # Low-scoring venues are HARDER → higher difficulty
    if venue_features is not None and "venue_scoring_tier" in venue_features.columns:
        tier_map = (
            venue_features[["venue", "venue_scoring_tier"]]
            .drop_duplicates("venue")
            .set_index("venue")["venue_scoring_tier"]
            .map({"low": 0.80, "medium": 0.50, "high": 0.20})
            .to_dict()
        )
        venue_diff = df["venue"].map(tier_map).fillna(0.50)
    else:
        venue_diff = pd.Series(0.50, index=df.index)  # neutral if no venue data

    # ── Combine ───────────────────────────────────────────────────────────────
    sds = (
        SDS_WEIGHTS["rrr_crr"]    * norm_rrr +
        SDS_WEIGHTS["wickets"]    * norm_wickets +
        SDS_WEIGHTS["phase"]      * phase_diff +
        SDS_WEIGHTS["dot_streak"] * norm_dot +
        SDS_WEIGHTS["venue"]      * venue_diff
    )

    df["state_difficulty"] = (sds * 100).clip(0, 100).round(2)
    return df
