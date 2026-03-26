"""
pressure_index.py — Pressure Index computation.

Pressure Index quantifies how difficult the current match state is
for the batting team on a 0–100 scale. Used to contextualize player
contributions and identify high-leverage moments.

Formula:
  PI = alpha * normalized(rrr_crr_delta)
     + beta  * normalized(wickets_lost / 10)
     + gamma * phase_weight
     + delta * normalized(dot_ball_streak)

Weights (from config.py):
  alpha=0.35 (chase pressure gap)
  beta=0.30  (resource depletion)
  gamma=0.20 (phase context — death overs inherently harder)
  delta=0.15 (incoming momentum compression)

Scope:
  - Inning 2: full formula using RRR-CRR delta
  - Inning 1: partial formula (no RRR term); alpha weight redistributed to beta

Limitations:
  - Weights are heuristically chosen, not empirically optimized
  - Does not account for batting lineup quality (treated separately)
  - Normalization bounds are clipped at observed dataset extremes
"""

import numpy as np
import pandas as pd

from config import (
    PI_WEIGHT_RRR_CRR, PI_WEIGHT_WICKETS,
    PI_WEIGHT_PHASE, PI_WEIGHT_DOT_STREAK,
    PHASE_WEIGHT_MAP,
)

# Normalization bounds (clipped to these ranges before scaling to 0–1)
RRR_CRR_DELTA_MAX = 8.0    # delta > 8 runs/over = maximum pressure state
RRR_CRR_DELTA_MIN = -4.0   # delta < -4 = batting extremely comfortably
DOT_STREAK_MAX    = 12.0   # 12+ dot balls in a row = maximum streak pressure


def compute_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pressure Index for every ball state row.
    Appends 'pressure_index' column (0–100 float).

    Input: ball_states DataFrame (from pipeline 02)
    Output: same DataFrame with pressure_index added
    """
    df = df.copy()

    # ── Component 1: RRR-CRR delta (inning 2 only) ────────────────────────────
    rrr_crr = df["pre_rrr_crr_delta"].fillna(0.0).clip(RRR_CRR_DELTA_MIN, RRR_CRR_DELTA_MAX)
    norm_rrr_crr = (rrr_crr - RRR_CRR_DELTA_MIN) / (RRR_CRR_DELTA_MAX - RRR_CRR_DELTA_MIN)
    # For inning 1, this term contributes 0 (no chase pressure)
    norm_rrr_crr = norm_rrr_crr.where(df["inning"] == 2, 0.0)

    # ── Component 2: Wickets lost (resource depletion) ────────────────────────
    norm_wickets = df["pre_wickets"] / 10.0  # 0=none lost, 1=all gone

    # ── Component 3: Phase weight ──────────────────────────────────────────────
    phase_weight = df["phase"].map(PHASE_WEIGHT_MAP).fillna(0.75)

    # ── Component 4: Dot ball streak ──────────────────────────────────────────
    dot_streak = df["pre_dot_ball_streak"].clip(0, DOT_STREAK_MAX) / DOT_STREAK_MAX

    # ── Combine ───────────────────────────────────────────────────────────────
    # For inning 1: redistribute alpha (rrr_crr) to beta (wickets)
    inning_1_mask = df["inning"] == 1

    pressure = (
        PI_WEIGHT_RRR_CRR * norm_rrr_crr +
        PI_WEIGHT_WICKETS  * norm_wickets +
        PI_WEIGHT_PHASE    * phase_weight +
        PI_WEIGHT_DOT_STREAK * dot_streak
    )

    # For inning 1, replace rrr_crr weight with extra wicket weight
    pressure_inning1 = (
        (PI_WEIGHT_RRR_CRR + PI_WEIGHT_WICKETS) * norm_wickets +
        PI_WEIGHT_PHASE * phase_weight +
        PI_WEIGHT_DOT_STREAK * dot_streak
    )

    df["pressure_index"] = np.where(
        inning_1_mask,
        pressure_inning1,
        pressure
    )

    # Scale to 0–100
    df["pressure_index"] = (df["pressure_index"] * 100).clip(0, 100).round(2)

    return df


def pressure_index_summary(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    """
    Summarize average pressure context across a groupby (e.g., per player, venue, phase).
    Returns mean, std, and count of pressure_index.
    """
    if "pressure_index" not in df.columns:
        df = compute_pressure_index(df)

    summary = (
        df.groupby(groupby_cols)["pressure_index"]
        .agg(
            avg_pressure_faced=("mean"),
            std_pressure_faced=("std"),
            n_balls=("count"),
        )
        .reset_index()
    )
    return summary
