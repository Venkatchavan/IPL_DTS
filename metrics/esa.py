"""
esa.py — Expected Score Added (ESA)

ESA measures the shift in expected final innings score caused by a single delivery.
Primarily used for inning 1 (where Win Probability is not available until
the innings total is set).

Formula (per delivery):
  ESA = EFS(state_after) - EFS(state_before)

Where EFS = expected final score from the expected score model (Model 1).

Positive ESA:
  - Batter: scored above what the state model projected
  - Inning 1 context: built the score faster than expected

Negative ESA:
  - Dot ball in an accelerating state
  - Dismissal that collapses projected final score

ESA for bowlers in inning 1:
  Bowler ESA = -(batter ESA on that delivery)
  A good bowler ball suppresses expected final score → negative ESA from batting view
  → positive bowler ESA

Requirements:
  - ball_states must have pre_expected_score and post_expected_score columns
    (populated by Model 1 scoring in pipeline 05)

Limitations:
  - EFS model has uncertainty; ESA inherits that uncertainty
  - Not useful in inning 2 (use WPA instead)
  - Treats all wickets identically in terms of score delta — this understates
    the value of losing a top-order wicket vs a tail wicket
"""

import pandas as pd
import numpy as np


def compute_esa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ball ESA, requiring pre_expected_score and post_expected_score.

    Adds columns:
      - esa: ESA for batting team
      - esa_bowler: ESA credit for bowler (inverted)
    """
    if "pre_expected_score" not in df.columns or "post_expected_score" not in df.columns:
        raise ValueError(
            "DataFrame must have 'pre_expected_score' and 'post_expected_score' columns. "
            "Run model scoring (pipeline 05) before computing ESA."
        )

    df = df.copy()
    # ESA is principally inning 1; valid for inning 2 as secondary metric
    df["esa"]        = df["post_expected_score"] - df["pre_expected_score"]
    df["esa_bowler"] = -df["esa"]

    return df


def batter_esa_summary(df: pd.DataFrame, inning: int = 1, groupby_extra: list = None) -> pd.DataFrame:
    """
    Aggregate ESA per batter for a given inning.

    Args:
        df: ball_states with esa column
        inning: 1 (default, inning 1 context for first-innings builders)
        groupby_extra: additional groupby dimensions

    Returns:
        DataFrame with batter ESA leaderboard
    """
    inning_df = df[(df["inning"] == inning) & df["esa"].notna()].copy()

    groupby = ["striker"] + (groupby_extra or [])

    summary = (
        inning_df.groupby(groupby)
        .agg(
            total_esa     =("esa", "sum"),
            mean_esa      =("esa", "mean"),
            balls_faced   =("esa", "count"),
        )
        .reset_index()
        .rename(columns={"striker": "player"})
    )
    summary["total_esa"] = summary["total_esa"].round(2)
    summary["mean_esa"]  = summary["mean_esa"].round(4)

    return summary.sort_values("total_esa", ascending=False).reset_index(drop=True)


def bowler_esa_summary(df: pd.DataFrame, inning: int = 1, groupby_extra: list = None) -> pd.DataFrame:
    """
    Aggregate ESA per bowler. Higher value = bowler suppressed scoring effectively.
    """
    inning_df = df[(df["inning"] == inning) & df["esa_bowler"].notna()].copy()

    groupby = ["bowler"] + (groupby_extra or [])

    summary = (
        inning_df.groupby(groupby)
        .agg(
            total_esa_bowler=("esa_bowler", "sum"),
            mean_esa_bowler =("esa_bowler", "mean"),
            balls_bowled    =("esa_bowler", "count"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )
    summary["total_esa_bowler"] = summary["total_esa_bowler"].round(2)
    summary["mean_esa_bowler"]  = summary["mean_esa_bowler"].round(4)

    return summary.sort_values("total_esa_bowler", ascending=False).reset_index(drop=True)
