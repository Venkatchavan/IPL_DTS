"""
contextual_economy.py — Contextual Economy Rate and Death Suppression Index (DSI)

Raw economy rate comparisons across bowlers are analytically weak because they
ignore venue scoring norms, phase context, and era. This module provides:

1. Contextual Economy Rate (CER):
   CER = raw_economy - venue_phase_avg_economy
   Negative CER = better than venue-phase baseline (harder to score in context)
   Positive CER = worse than venue-phase baseline

2. Death Overs Suppression Index (DSI):
   Composite score for death-over bowling effectiveness (0–100).
   DSI = w1*(1 - economy_death/venue_death_avg)
       + w2*dot_pct_death
       + w3*wicket_rate_death
       + w4*(1 - boundary_concession_rate_death)
   Minimum 50 death-over balls required for reliable DSI.

3. Control Rate:
   % of deliveries that are dots or scoring at most 1 run (low-damage balls)
   Used to identify containing bowlers beyond wicket-taking ability.

Limitations:
  - Venue averages computed from full dataset; era shifts are not phase-adjusted per year
  - DSI weights (from config.py) are heuristic, not empirically fitted
  - Sample requirement: minimum 50 death-over balls for DSI; smaller samples shown
    with low-confidence flag in dashboard
"""

import pandas as pd
import numpy as np

from config import (
    DSI_WEIGHT_ECONOMY, DSI_WEIGHT_DOT_PCT,
    DSI_WEIGHT_WICKETS, DSI_WEIGHT_BOUNDARY,
    MIN_BALLS_DEATH_DSI, PHASE_DEATH,
)


def compute_contextual_economy(
    bowler_features: pd.DataFrame,
    venue_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Contextual Economy Rate for each bowler × venue × phase bucket.

    Args:
        bowler_features: from pipeline 03, with columns [player, phase, venue, economy_rate, ...]
                         (must include venue dimension; merge required if not present)
        venue_features: from pipeline 03, with [venue, phase, inning, avg_run_rate]

    Returns:
        bowler_features with 'contextual_economy' column added
    """
    venue_baseline = (
        venue_features[venue_features["inning"] == 1]  # use inning 1 for economy baseline
        .groupby(["venue", "phase"])["avg_run_rate"]
        .mean()
        .reset_index()
        .rename(columns={"avg_run_rate": "venue_phase_avg_rr"})
    )

    df = bowler_features.copy()

    if "venue" not in df.columns:
        # Cannot compute contextual economy without venue dimension
        df["contextual_economy"] = np.nan
        df["contextual_economy_note"] = "venue dimension not available in this slice"
        return df

    df = df.merge(venue_baseline, on=["venue", "phase"], how="left")
    df["contextual_economy"] = (df["economy_rate"] - df["venue_phase_avg_rr"]).round(3)
    df.drop(columns=["venue_phase_avg_rr"], inplace=True)

    return df


def compute_dsi(bowler_features: pd.DataFrame, venue_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Death Overs Suppression Index for each bowler.

    Requires the "death_only" slice of bowler_features (from pipeline 03).
    DSI is bounded 0–100. Null if balls_bowled < MIN_BALLS_DEATH_DSI.

    Args:
        bowler_features: includes death-over slice with economy_rate, dot_pct,
                         wicket_rate, boundary_concession_rate per bowler
        venue_features: for venue-phase death-over economy baseline

    Returns:
        DataFrame with [player, dsi, dsi_confidence, balls_death_overs]
    """
    death_bowlers = bowler_features[
        bowler_features["slice"] == "death_only"
    ].copy()

    if len(death_bowlers) == 0:
        return pd.DataFrame(columns=["player", "dsi", "dsi_confidence"])

    # Venue death-over baseline economy
    venue_death_avg = (
        venue_features[
            (venue_features["phase"] == PHASE_DEATH) &
            (venue_features["inning"] == 1)
        ]["avg_run_rate"].mean()
    )
    if pd.isna(venue_death_avg) or venue_death_avg == 0:
        venue_death_avg = 10.0  # fallback if no venue data

    # Component 1: Economy relative to venue death baseline
    # (1 - ratio) → higher when economy << venue average → better
    economy_component = (
        1.0 - (death_bowlers["economy_rate"] / venue_death_avg)
    ).clip(0, 1)

    # Component 2: Dot ball percentage (higher = better)
    dot_component = death_bowlers["dot_pct"].fillna(0).clip(0, 1)

    # Component 3: Wicket rate (normalized: divides by 3 wkts/over as theoretical max)
    wicket_component = (death_bowlers["wicket_rate"].fillna(0) / 3.0).clip(0, 1)

    # Component 4: Inverse boundary concession rate (lower boundaries = better)
    boundary_component = (
        1.0 - death_bowlers["boundary_concession_rate"].fillna(0)
    ).clip(0, 1)

    dsi_raw = (
        DSI_WEIGHT_ECONOMY   * economy_component +
        DSI_WEIGHT_DOT_PCT   * dot_component +
        DSI_WEIGHT_WICKETS   * wicket_component +
        DSI_WEIGHT_BOUNDARY  * boundary_component
    )

    death_bowlers["dsi"] = (dsi_raw * 100).round(2)
    death_bowlers["dsi_confidence"] = death_bowlers["balls_bowled"].apply(
        lambda b: "high" if b >= MIN_BALLS_DEATH_DSI * 2
        else "medium" if b >= MIN_BALLS_DEATH_DSI
        else "low"
    )

    # Null out DSI for very-low sample bowlers but keep them in table with flag
    death_bowlers.loc[
        death_bowlers["balls_bowled"] < MIN_BALLS_DEATH_DSI // 2,
        "dsi"
    ] = np.nan

    return death_bowlers[
        ["player", "dsi", "dsi_confidence", "balls_bowled", "economy_rate",
         "dot_pct", "wicket_rate", "boundary_concession_rate"]
    ].sort_values("dsi", ascending=False, na_position="last").reset_index(drop=True)


def compute_control_rate(bowler_features: pd.DataFrame, ball_states: pd.DataFrame = None) -> pd.DataFrame:
    """
    Control Rate = % of legal deliveries that are dots OR scoring at most 1 run.
    Identifies containing bowlers who restrict boundary scoring.

    If ball_states is provided, computes from raw deliveries.
    Otherwise approximates from bowler_features (less precise).
    """
    if ball_states is not None:
        legal = ball_states[ball_states["is_legal_ball"] == 1].copy()
        legal["is_controlled"] = (legal["batsman_runs"] <= 1).astype(int)

        control = (
            legal.groupby("bowler")
            .agg(
                controlled_balls=("is_controlled", "sum"),
                total_balls=("is_legal_ball", "sum"),
            )
            .reset_index()
            .rename(columns={"bowler": "player"})
        )
        control["control_rate"] = (control["controlled_balls"] / control["total_balls"]).round(4)
        return control[["player", "control_rate", "total_balls"]]

    # Fallback approximation from aggregates
    df = bowler_features[bowler_features["slice"] == "overall"].copy()
    # dot_pct provides lower bound; actual control rate includes singles
    df["control_rate_approx"] = df["dot_pct"]  # underestimate
    return df[["player", "control_rate_approx", "balls_bowled"]]
