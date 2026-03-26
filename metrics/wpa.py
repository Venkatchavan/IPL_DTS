"""
wpa.py — Win Probability Added (WPA)

WPA measures the shift in chase win probability caused by a single delivery.
For batters: positive WPA = delivery moved batting team closer to winning.
For bowlers: the sign is inverted — a good bowling ball has negative WPA from
the batting team's view, so bowler WPA = -batter_WPA_on_that_ball.

Formula (per delivery):
  WPA = WP(state_after) - WP(state_before)

Where WP = predicted win probability from the chase win probability model (Model 2).

WPA summaries (per player, phase, season):
  - total_wpa: sum of WPA across all qualifying balls
  - mean_wpa: average WPA per ball (rate metric, comparable across sample sizes)
  - wpa_positive: sum of +WPA contributions only
  - wpa_negative: sum of negative WPA contributions only
  - high_leverage_wpa: WPA sum in balls where |WPA delta| > threshold (key moment impact)

Requirements:
  - ball_states must have pre_win_prob and post_win_prob columns
    (populated by the win probability model in pipeline 05)
  - This module is called AFTER model scoring, not before.

Limitations:
  - WPA is only meaningful in inning 2 (chase scenarios).
    Inning 1 contributions are captured by ESA (expected_score_added.py).
  - WPA inherits all uncertainty from the win probability model.
  - Sample size matters: minimum 100 balls for stable WPA estimate (see config.py).
"""

import pandas as pd
import numpy as np


HIGH_LEVERAGE_THRESHOLD = 0.05  # WPA delta > 5% WP shift = high-leverage ball


def compute_wpa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ball WPA, assuming the DataFrame already has:
      - pre_win_prob: win probability before this delivery
      - post_win_prob: win probability after this delivery

    Adds columns:
      - wpa: WPA for batting team on this ball
      - wpa_bowler: WPA credit for bowler (inverted)
      - is_high_leverage: flag for |wpa| > HIGH_LEVERAGE_THRESHOLD
    """
    if "pre_win_prob" not in df.columns or "post_win_prob" not in df.columns:
        raise ValueError(
            "DataFrame must have 'pre_win_prob' and 'post_win_prob' columns. "
            "Run model scoring (pipeline 05) before computing WPA."
        )

    df = df.copy()
    inning2_mask = df["inning"] == 2

    df["wpa"]         = np.where(inning2_mask, df["post_win_prob"] - df["pre_win_prob"], np.nan)
    df["wpa_bowler"]  = np.where(inning2_mask, -(df["post_win_prob"] - df["pre_win_prob"]), np.nan)
    df["is_high_leverage"] = (df["wpa"].abs() > HIGH_LEVERAGE_THRESHOLD).astype("Int8")

    return df


def batter_wpa_summary(df: pd.DataFrame, groupby_extra: list = None) -> pd.DataFrame:
    """
    Aggregate WPA per batter (inning 2 only).

    Args:
        df: ball_states with wpa column
        groupby_extra: additional groupby columns (e.g., ["phase"], ["season"])

    Returns:
        DataFrame with batter WPA leaderboard
    """
    inning2 = df[(df["inning"] == 2) & df["wpa"].notna()].copy()

    groupby = ["striker"] + (groupby_extra or [])

    summary = (
        inning2.groupby(groupby)
        .agg(
            total_wpa       =("wpa", "sum"),
            mean_wpa        =("wpa", "mean"),
            wpa_positive    =("wpa", lambda x: x[x > 0].sum()),
            wpa_negative    =("wpa", lambda x: x[x < 0].sum()),
            balls_in_chase  =("wpa", "count"),
            high_lev_balls  =("is_high_leverage", "sum"),
        )
        .reset_index()
        .rename(columns={"striker": "player"})
    )
    summary["total_wpa"]    = summary["total_wpa"].round(4)
    summary["mean_wpa"]     = summary["mean_wpa"].round(6)
    summary["wpa_positive"] = summary["wpa_positive"].round(4)
    summary["wpa_negative"] = summary["wpa_negative"].round(4)

    return summary.sort_values("total_wpa", ascending=False).reset_index(drop=True)


def bowler_wpa_summary(df: pd.DataFrame, groupby_extra: list = None) -> pd.DataFrame:
    """
    Aggregate WPA per bowler (inning 2 only).
    Higher bowler_wpa = better performance (bowler suppressed win probability growth).
    """
    inning2 = df[(df["inning"] == 2) & df["wpa_bowler"].notna()].copy()

    groupby = ["bowler"] + (groupby_extra or [])

    summary = (
        inning2.groupby(groupby)
        .agg(
            total_wpa_bowler  =("wpa_bowler", "sum"),
            mean_wpa_bowler   =("wpa_bowler", "mean"),
            balls_bowled_chase=("wpa_bowler", "count"),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )
    summary["total_wpa_bowler"] = summary["total_wpa_bowler"].round(4)
    summary["mean_wpa_bowler"]  = summary["mean_wpa_bowler"].round(6)

    return summary.sort_values("total_wpa_bowler", ascending=False).reset_index(drop=True)
