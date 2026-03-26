"""
04_metrics_compute.py — Metrics Computation Pipeline

Reads ball_states.parquet and model-scored state files to compute
all analytical metrics. Produces two output files:

  1. metrics_ball_level.parquet — per-ball Pressure Index, State Difficulty,
     WPA, ESA (after model scoring in pipeline 05 adds pre/post win prob)

  2. metrics_player_level.parquet — per-player aggregated metrics:
     WPA, ESA, DSI, Contextual Economy, Control Rate,
     Phase Dependability, Venue Sensitivity Index

NOTE: WPA and ESA require model-scored win probabilities and expected scores.
      This pipeline handles the metrics that DO NOT require model output
      (Pressure Index, State Difficulty) in Phase A, and model-dependent
      metrics (WPA, ESA) in Phase B which runs after pipeline 05.

Run full pipeline (after model scoring):
  python pipelines/04_metrics_compute.py

Run Phase A only (pre-model metrics):
  python pipelines/04_metrics_compute.py --phase A

Input:
  data/features/ball_states.parquet  (always)
  data/features/player_features.parquet
  data/features/venue_features.parquet
  data/features/matchup_features.parquet
  [for Phase B]: model-scored ball_states with pre/post win_prob, expected_score

Output:
  data/features/metrics_ball_level.parquet
  data/features/metrics_player_level.parquet
  data/features/metrics_matchup.parquet
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BALL_STATES_FILE, PLAYER_FEATURES, TEAM_FEATURES,
    DATA_FEATURES, METRICS_BALL, METRICS_PLAYER,
    MIN_BALLS_FACED_BATTER, MIN_BALLS_BOWLED_BOWLER,
)
from metrics.pressure_index   import compute_pressure_index
from metrics.state_difficulty import compute_state_difficulty
from metrics.contextual_economy import compute_dsi, compute_control_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Phase A: Pre-model metrics ─────────────────────────────────────────────────

def run_phase_a(ball_states: pd.DataFrame, venue_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics that require only ball state data (no model predictions).
    Returns enriched ball-level DataFrame.
    """
    log.info("Phase A: Computing Pressure Index...")
    ball_states = compute_pressure_index(ball_states)

    log.info("Phase A: Computing State Difficulty Score...")
    ball_states = compute_state_difficulty(ball_states, venue_features)

    return ball_states


# ── Phase B: Model-dependent metrics ──────────────────────────────────────────

def run_phase_b(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WPA and ESA — requires pre/post win probability and expected score
    columns added by pipeline 05 model scoring.
    """
    has_wp  = "pre_win_prob" in ball_states.columns and "post_win_prob" in ball_states.columns
    has_efs = "pre_expected_score" in ball_states.columns and "post_expected_score" in ball_states.columns

    if has_wp:
        log.info("Phase B: Computing WPA...")
        from metrics.wpa import compute_wpa
        ball_states = compute_wpa(ball_states)
    else:
        log.warning("Phase B: Skipping WPA — win probability columns not found. Run pipeline 05 first.")
        ball_states["wpa"]        = np.nan
        ball_states["wpa_bowler"] = np.nan

    if has_efs:
        log.info("Phase B: Computing ESA...")
        from metrics.esa import compute_esa
        ball_states = compute_esa(ball_states)
    else:
        log.warning("Phase B: Skipping ESA — expected score columns not found. Run pipeline 05 first.")
        ball_states["esa"]        = np.nan
        ball_states["esa_bowler"] = np.nan

    return ball_states


# ── Player-level aggregations ─────────────────────────────────────────────────

def build_player_level_metrics(
    ball_states: pd.DataFrame,
    player_features: pd.DataFrame,
    venue_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all per-ball metrics into player-level summaries.
    Returns a wide DataFrame with one row per player (multi-slice).
    """
    results = []

    # ── WPA summaries ─────────────────────────────────────────────────────────
    if "wpa" in ball_states.columns and ball_states["wpa"].notna().sum() > 0:
        from metrics.wpa import batter_wpa_summary, bowler_wpa_summary
        batter_wpa = batter_wpa_summary(ball_states)
        batter_wpa["player_type"] = "batter"
        batter_wpa["metric_slice"] = "wpa_overall"

        bowler_wpa = bowler_wpa_summary(ball_states)
        bowler_wpa.rename(columns={"total_wpa_bowler": "total_wpa", "mean_wpa_bowler": "mean_wpa",
                                    "balls_bowled_chase": "balls_in_chase"}, inplace=True)
        bowler_wpa["player_type"] = "bowler"
        bowler_wpa["metric_slice"] = "wpa_overall"

        results.extend([batter_wpa, bowler_wpa])

    # ── ESA summaries ─────────────────────────────────────────────────────────
    if "esa" in ball_states.columns and ball_states["esa"].notna().sum() > 0:
        from metrics.esa import batter_esa_summary, bowler_esa_summary
        batter_esa = batter_esa_summary(ball_states, inning=1)
        batter_esa["player_type"] = "batter"
        batter_esa["metric_slice"] = "esa_inning1"

        bowler_esa = bowler_esa_summary(ball_states, inning=1)
        bowler_esa.rename(columns={"total_esa_bowler": "total_esa", "mean_esa_bowler": "mean_esa",
                                    "balls_bowled": "balls_faced"}, inplace=True)
        bowler_esa["player_type"] = "bowler"
        bowler_esa["metric_slice"] = "esa_inning1"

        results.extend([batter_esa, bowler_esa])

    # ── Pressure context per batter ───────────────────────────────────────────
    # Average Pressure Index and State Difficulty faced per batter
    if "pressure_index" in ball_states.columns:
        pi_faced = (
            ball_states[ball_states["inning"] == 2]
            .groupby("striker")
            .agg(
                avg_pi_faced=("pressure_index", "mean"),
                avg_sds_faced=("state_difficulty", "mean"),
                balls_in_pressure=("pressure_index", "count"),
            )
            .reset_index()
            .rename(columns={"striker": "player"})
        )
        pi_faced["player_type"]  = "batter"
        pi_faced["metric_slice"] = "pressure_context"
        results.append(pi_faced)

    # ── DSI for bowlers ───────────────────────────────────────────────────────
    bowler_features = player_features[
        (player_features["player_type"] == "bowler")
    ].copy()

    if len(bowler_features) > 0:
        dsi_df = compute_dsi(bowler_features, venue_features)
        dsi_df["player_type"]  = "bowler"
        dsi_df["metric_slice"] = "dsi"
        results.append(dsi_df)

    # ── Control Rate for bowlers ──────────────────────────────────────────────
    control_df = compute_control_rate(bowler_features, ball_states)
    if len(control_df) > 0:
        control_df["player_type"]  = "bowler"
        control_df["metric_slice"] = "control_rate"
        results.append(control_df)

    # ── Venue Sensitivity Index ───────────────────────────────────────────────
    venue_vsi = _compute_vsi(ball_states)
    if venue_vsi is not None:
        venue_vsi["metric_slice"] = "vsi"
        results.append(venue_vsi)

    if not results:
        log.warning("No player-level metric results computed.")
        return pd.DataFrame()

    metrics_df = pd.concat(results, ignore_index=True)
    log.info(f"  Player-level metrics: {len(metrics_df):,} rows across {metrics_df['metric_slice'].nunique()} slices")
    return metrics_df


def _compute_vsi(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Venue Sensitivity Index (VSI) for batters.
    VSI = std(strike_rate_per_venue) / mean(strike_rate_per_venue)

    High VSI = player performs very differently across venues.
    Low VSI = venue-consistent performer.
    Minimum: player must have batted at >= 3 distinct venues.
    """
    legal = ball_states[ball_states["is_legal_ball"] == 1].copy()
    is_four = (legal["batsman_runs"] == 4).astype(int)
    is_six  = (legal["batsman_runs"] == 6).astype(int)
    legal["is_boundary"] = (is_four | is_six).astype(int)

    per_venue = (
        legal.groupby(["striker", "venue"])
        .agg(balls=("is_legal_ball", "sum"), runs=("batsman_runs", "sum"))
        .reset_index()
    )
    per_venue["sr"] = (per_venue["runs"] / per_venue["balls"] * 100).round(2)

    # Only include venue buckets with sufficient balls
    per_venue = per_venue[per_venue["balls"] >= 30]

    vsi = (
        per_venue.groupby("striker")["sr"]
        .agg(
            vsi_std=("std"),
            vsi_mean=("mean"),
            venue_count=("count"),
        )
        .reset_index()
        .rename(columns={"striker": "player"})
    )
    vsi = vsi[vsi["venue_count"] >= 3].copy()
    vsi["vsi"] = (vsi["vsi_std"] / vsi["vsi_mean"].replace(0, np.nan)).round(3)
    vsi["player_type"] = "batter"

    return vsi[["player", "vsi", "venue_count", "player_type"]]


# ── Matchup metrics ───────────────────────────────────────────────────────────

def build_matchup_metrics(player_features: pd.DataFrame) -> pd.DataFrame:
    """Compute Matchup Leverage Scores."""
    from metrics.matchup_leverage import assign_bowler_style_bucket, compute_mls

    matchup_path = DATA_FEATURES / "matchup_features.parquet"
    if not matchup_path.exists():
        log.warning("matchup_features.parquet not found — skipping MLS computation.")
        return pd.DataFrame()

    matchup_features = pd.read_parquet(matchup_path)
    bowler_overall   = player_features[
        (player_features["player_type"] == "bowler") &
        (player_features["slice"] == "overall")
    ].copy()

    log.info("Computing bowler style buckets for MLS...")
    try:
        style_buckets = assign_bowler_style_bucket(bowler_overall)
        mls_df        = compute_mls(matchup_features, style_buckets)
        log.info(f"  MLS computed for {len(mls_df):,} batter-bowler pairs")
        return mls_df
    except ImportError:
        log.warning("scikit-learn not available — skipping MLS computation.")
        return pd.DataFrame()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(phase: str = "all"):
    log.info(f"Pipeline 04: Metrics Computation (phase={phase})")

    ball_states = pd.read_parquet(BALL_STATES_FILE)
    log.info(f"  Loaded {len(ball_states):,} ball states")

    venue_features_path = DATA_FEATURES / "venue_features.parquet"
    venue_features = pd.read_parquet(venue_features_path) if venue_features_path.exists() else pd.DataFrame()

    player_features = pd.read_parquet(PLAYER_FEATURES) if PLAYER_FEATURES.exists() else pd.DataFrame()

    # Phase A: state-based metrics (no model required)
    ball_states = run_phase_a(ball_states, venue_features)

    # Phase B: model-dependent metrics
    if phase in ("all", "B"):
        ball_states = run_phase_b(ball_states)

    # Save ball-level metrics
    ball_states.to_parquet(METRICS_BALL, index=False)
    log.info(f"Saved metrics_ball_level: {METRICS_BALL}  ({len(ball_states):,} rows)")

    # Player-level aggregations
    player_metrics = build_player_level_metrics(ball_states, player_features, venue_features)
    if len(player_metrics) > 0:
        player_metrics.to_parquet(METRICS_PLAYER, index=False)
        log.info(f"Saved metrics_player_level: {METRICS_PLAYER}  ({len(player_metrics):,} rows)")

    # Matchup metrics
    if len(player_features) > 0:
        mls_df = build_matchup_metrics(player_features)
        if len(mls_df) > 0:
            mls_path = DATA_FEATURES / "metrics_matchup.parquet"
            mls_df.to_parquet(mls_path, index=False)
            log.info(f"Saved metrics_matchup: {mls_path}  ({len(mls_df):,} rows)")

    log.info("Pipeline 04 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL INTEL Metrics Computation Pipeline")
    parser.add_argument("--phase", choices=["A", "B", "all"], default="all",
                        help="A=pre-model metrics only; B=model-dependent only; all=both")
    args = parser.parse_args()
    run(phase=args.phase)
