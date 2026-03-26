"""
03_feature_engineering.py — Player and Team Feature Aggregation

Reads the ball_states.parquet produced by pipeline 02.
Builds three output feature stores:

  1. player_features.parquet — per-player × phase × pressure_band aggregations
     for batters and bowlers, including contextual strike rates, economy rates,
     boundary rates, dot rates, and matchup features.

  2. team_features.parquet — per-team × phase performance profiles, chase metrics,
     venue profiles, and bowling attack summaries.

  3. venue_features.parquet — per-venue scoring baselines used for contextual
     normalization across all player and team metrics.

All aggregations are computed both:
  - overall (all seasons) for stable estimates
  - per-season (for trend analysis and recency filtering in dashboard)

Minimum sample thresholds are NOT enforced here — they are enforced at display time.
All records are preserved; sample counts are included as a column in every output.

Run:
  python pipelines/03_feature_engineering.py

Input:
  data/features/ball_states.parquet

Output:
  data/features/player_features.parquet
  data/features/team_features.parquet
  data/features/venue_features.parquet
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BALL_STATES_FILE, PLAYER_FEATURES, TEAM_FEATURES,
    DATA_FEATURES,
    PHASE_POWERPLAY, PHASE_MIDDLE, PHASE_DEATH,
    MIN_BALLS_FACED_BATTER,
    BOWLER_CREDITED_DISMISSALS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VENUE_FEATURES = DATA_FEATURES / "venue_features.parquet"


# ── Helper: boundary flag ─────────────────────────────────────────────────────

def add_boundary_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["is_four"] = (df["batsman_runs"] == 4).astype(int)
    df["is_six"]  = (df["batsman_runs"] == 6).astype(int)
    df["is_boundary"] = (df["is_four"] | df["is_six"]).astype(int)
    df["is_dot_ball"]  = (
        (df["batsman_runs"] == 0) & (df["extra_runs"] == 0) & (df["is_legal_ball"] == 1)
    ).astype(int)
    return df


def add_bowler_wicket_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    A wicket is credited to the bowler only for specific dismissal kinds.
    Run-outs count as a wicket for batting team but not for bowler competence metrics.
    """
    df["is_bowler_wicket"] = (
        df["is_wicket"] &
        df["dismissal_kind"].isin(BOWLER_CREDITED_DISMISSALS)
    ).astype(int)
    return df


# ── 1. Venue baseline features ─────────────────────────────────────────────────

def build_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-venue, per-phase scoring baselines.
    Used to derive contextual economy and contextual strike impact.
    """
    log.info("Building venue features...")

    legal = df[df["is_legal_ball"] == 1].copy()

    agg = (
        legal.groupby(["venue", "phase", "inning"])
        .agg(
            balls_total =("is_legal_ball", "sum"),
            runs_total  =("total_runs", "sum"),
            wickets_total=("is_wicket", "sum"),
            boundary_total=("is_boundary", "sum"),
            dot_total   =("is_dot_ball", "sum"),
            matches_count=("match_id", "nunique"),
        )
        .reset_index()
    )
    agg["avg_run_rate"]    = (agg["runs_total"] / (agg["balls_total"] / 6)).round(4)
    agg["avg_dot_pct"]     = (agg["dot_total"] / agg["balls_total"]).round(4)
    agg["avg_boundary_pct"]= (agg["boundary_total"] / agg["balls_total"]).round(4)
    agg["avg_wicket_rate"] = (agg["wickets_total"] / (agg["balls_total"] / 6)).round(4)

    # Overall first-innings average per venue (for expected score baseline)
    first_innings = legal[legal["inning"] == 1].copy()
    fi_totals = (
        first_innings.groupby(["match_id", "venue"])["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "inning1_score"})
    )
    venue_fi_avg = (
        fi_totals.groupby("venue")["inning1_score"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "venue_avg_fi_score", "std": "venue_std_fi_score", "count": "venue_fi_matches"})
        .reset_index()
    )

    # Assign venue scoring tier (terciles)
    venue_fi_avg["venue_scoring_tier"] = pd.qcut(
        venue_fi_avg["venue_avg_fi_score"],
        q=3, labels=["low", "medium", "high"]
    )

    agg = agg.merge(venue_fi_avg, on="venue", how="left")
    log.info(f"  Venue features: {len(agg):,} rows ({agg['venue'].nunique()} venues)")
    return agg


# ── 2. Batter features ────────────────────────────────────────────────────────

def build_batter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate batter performance across multiple context slices:
      - Overall
      - By phase
      - By pressure_band (for pressure profiling)
      - By phase × pressure_band (for full context-adjusted value)
    """
    log.info("Building batter features...")

    legal = df[df["is_legal_ball"] == 1].copy()

    def batter_agg(group_df, groupby_cols):
        """Generic aggregation function for any batter grouping."""
        agg = (
            group_df.groupby(groupby_cols)
            .agg(
                balls_faced   =("is_legal_ball", "sum"),
                runs_scored   =("batsman_runs", "sum"),
                fours         =("is_four", "sum"),
                sixes         =("is_six", "sum"),
                boundaries    =("is_boundary", "sum"),
                dot_balls     =("is_dot_ball", "sum"),
                dismissals    =("is_wicket", "sum"),
                innings_count =("match_id", "nunique"),
            )
            .reset_index()
        )
        agg["strike_rate"]   = (agg["runs_scored"] / agg["balls_faced"] * 100).round(2)
        agg["dot_pct"]       = (agg["dot_balls"] / agg["balls_faced"]).round(4)
        agg["boundary_pct"]  = (agg["boundaries"] / agg["balls_faced"]).round(4)
        agg["avg_per_inning"]= (agg["runs_scored"] / agg["innings_count"].replace(0, np.nan)).round(2)
        return agg

    # Slice 1: Overall (player level)
    overall = batter_agg(legal, ["striker"])
    overall["slice"] = "overall"
    overall.rename(columns={"striker": "player"}, inplace=True)

    # Slice 2: By phase
    by_phase = batter_agg(legal, ["striker", "phase"])
    by_phase["slice"] = "phase"
    by_phase.rename(columns={"striker": "player"}, inplace=True)

    # Slice 3: By pressure band (inning 2 only — where RRR is meaningful)
    inning2 = legal[legal["inning"] == 2].copy()
    by_pressure = batter_agg(inning2, ["striker", "pressure_band"])
    by_pressure["slice"] = "pressure_band"
    by_pressure.rename(columns={"striker": "player"}, inplace=True)

    # Slice 4: Phase × Pressure (inning 2 only)
    by_phase_pressure = batter_agg(inning2, ["striker", "phase", "pressure_band"])
    by_phase_pressure["slice"] = "phase_pressure"
    by_phase_pressure.rename(columns={"striker": "player"}, inplace=True)

    # Slice 5: By season (for trend analysis)
    by_season = batter_agg(legal, ["striker", "season"])
    by_season["slice"] = "season"
    by_season.rename(columns={"striker": "player"}, inplace=True)

    batter_df = pd.concat(
        [overall, by_phase, by_pressure, by_phase_pressure, by_season],
        ignore_index=True
    )
    batter_df["player_type"] = "batter"

    log.info(f"  Batter features: {len(batter_df):,} aggregate rows")
    return batter_df


# ── 3. Bowler features ────────────────────────────────────────────────────────

def build_bowler_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bowler performance by phase, pressure context (inning 2),
    and season. Includes bowler-credited wickets only for competence metrics.
    """
    log.info("Building bowler features...")

    legal = df[df["is_legal_ball"] == 1].copy()

    def bowler_agg(group_df, groupby_cols):
        agg = (
            group_df.groupby(groupby_cols)
            .agg(
                balls_bowled       =("is_legal_ball", "sum"),
                runs_conceded      =("total_runs", "sum"),
                wickets_taken      =("is_bowler_wicket", "sum"),
                dot_balls          =("is_dot_ball", "sum"),
                fours_conceded     =("is_four", "sum"),
                sixes_conceded     =("is_six", "sum"),
                matches_bowled     =("match_id", "nunique"),
            )
            .reset_index()
        )
        overs = agg["balls_bowled"] / 6
        agg["economy_rate"]          = (agg["runs_conceded"] / overs.replace(0, np.nan)).round(2)
        agg["dot_pct"]               = (agg["dot_balls"] / agg["balls_bowled"]).round(4)
        agg["wicket_rate"]           = (agg["wickets_taken"] / overs.replace(0, np.nan)).round(4)
        agg["boundary_concession_rate"] = (
            (agg["fours_conceded"] + agg["sixes_conceded"]) / agg["balls_bowled"]
        ).round(4)
        # Control rate: % of deliveries that are dots or singles
        # We approximate singles as (batsman_runs==1); add separately
        return agg

    # Slice 1: Overall
    overall = bowler_agg(legal, ["bowler"])
    overall["slice"] = "overall"
    overall.rename(columns={"bowler": "player"}, inplace=True)

    # Slice 2: By phase
    by_phase = bowler_agg(legal, ["bowler", "phase"])
    by_phase["slice"] = "phase"
    by_phase.rename(columns={"bowler": "player"}, inplace=True)

    # Slice 3: By season
    by_season = bowler_agg(legal, ["bowler", "season"])
    by_season["slice"] = "season"
    by_season.rename(columns={"bowler": "player"}, inplace=True)

    # Slice 4: Death overs specifically (for DSI computation in metrics layer)
    death_only = legal[legal["phase"] == PHASE_DEATH].copy()
    by_death = bowler_agg(death_only, ["bowler"])
    by_death["slice"] = "death_only"
    by_death.rename(columns={"bowler": "player"}, inplace=True)

    bowler_df = pd.concat(
        [overall, by_phase, by_season, by_death],
        ignore_index=True
    )
    bowler_df["player_type"] = "bowler"

    log.info(f"  Bowler features: {len(bowler_df):,} aggregate rows")
    return bowler_df


# ── 4. Matchup features ────────────────────────────────────────────────────────

def build_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per batter–bowler pair statistics.
    Only pairs with >= 30 legal balls are meaningful; sample count included.
    Matchup Leverage Score (MLS) is computed in the metrics layer,
    not here — this provides the raw aggregation.
    """
    log.info("Building matchup features...")

    legal = df[df["is_legal_ball"] == 1].copy()

    matchup = (
        legal.groupby(["striker", "bowler"])
        .agg(
            balls_faced   =("is_legal_ball", "sum"),
            runs_scored   =("batsman_runs", "sum"),
            boundaries    =("is_boundary", "sum"),
            dot_balls     =("is_dot_ball", "sum"),
            dismissals    =("is_wicket", "sum"),
        )
        .reset_index()
    )
    matchup["strike_rate"]  = (matchup["runs_scored"] / matchup["balls_faced"] * 100).round(2)
    matchup["dot_pct"]      = (matchup["dot_balls"] / matchup["balls_faced"]).round(4)
    matchup["boundary_pct"] = (matchup["boundaries"] / matchup["balls_faced"]).round(4)
    matchup["dismissal_rate"]= (matchup["dismissals"] / matchup["balls_faced"]).round(4)

    log.info(f"  Matchup features: {len(matchup):,} pairs ({(matchup['balls_faced']>=30).sum()} with >= 30 balls)")
    return matchup


# ── 5. Team features ──────────────────────────────────────────────────────────

def build_team_features(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team aggregations:
      - Phase-by-phase run rate and wicket profiles
      - Chase win rates by entry pressure band
      - Powerplay aggression index
      - Death-over batting and bowling profiles
      - Home/away venue splits
    """
    log.info("Building team features...")

    legal = df[df["is_legal_ball"] == 1].copy()

    # Batting: team × phase
    batting_phase = (
        legal.groupby(["batting_team", "phase"])
        .agg(
            balls=("is_legal_ball", "sum"),
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            boundaries=("is_boundary", "sum"),
            dots=("is_dot_ball", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    overs_ = batting_phase["balls"] / 6
    batting_phase["run_rate"]      = (batting_phase["runs"] / overs_.replace(0, np.nan)).round(3)
    batting_phase["dot_pct"]       = (batting_phase["dots"] / batting_phase["balls"]).round(4)
    batting_phase["boundary_pct"]  = (batting_phase["boundaries"] / batting_phase["balls"]).round(4)
    batting_phase["wicket_rate"]   = (batting_phase["wickets"] / batting_phase["matches"]).round(4)
    batting_phase["team_role"]     = "batting"

    # Bowling: team × phase
    bowling_phase = (
        legal.groupby(["bowling_team", "phase"])
        .agg(
            balls=("is_legal_ball", "sum"),
            runs=("total_runs", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            dots=("is_dot_ball", "sum"),
            boundaries=("is_boundary", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"bowling_team": "batting_team"})
    )
    overs_b = bowling_phase["balls"] / 6
    bowling_phase["economy_rate"]  = (bowling_phase["runs"] / overs_b.replace(0, np.nan)).round(3)
    bowling_phase["dot_pct"]       = (bowling_phase["dots"] / bowling_phase["balls"]).round(4)
    bowling_phase["wicket_rate"]   = (bowling_phase["wickets"] / overs_b.replace(0, np.nan)).round(4)
    bowling_phase["team_role"]     = "bowling"

    team_df = pd.concat([batting_phase, bowling_phase], ignore_index=True)
    team_df.rename(columns={"batting_team": "team"}, inplace=True)

    # Chase win rate by pressure band at phase entry
    chase_df = df[(df["inning"] == 2) & (df["is_legal_ball"] == 1)].copy()
    # Use first ball of each phase per match to get entry pressure band
    phase_entry = (
        chase_df.sort_values(["match_id", "inning", "global_ball_number"])
        .groupby(["match_id", "batting_team", "phase"])
        .first()
        .reset_index()
    )[["match_id", "batting_team", "phase", "pressure_band", "match_won"]]

    chase_win_rate = (
        phase_entry.groupby(["batting_team", "phase", "pressure_band"])
        .agg(
            n_instances=("match_id", "count"),
            win_rate=("match_won", "mean"),
        )
        .reset_index()
        .rename(columns={"batting_team": "team"})
    )

    log.info(f"  Team features: {len(team_df):,} team-phase rows")
    return team_df, chase_win_rate


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    log.info("Pipeline 03: Feature Engineering")

    log.info("Loading ball states...")
    df       = pd.read_parquet(BALL_STATES_FILE)
    log.info(f"  {len(df):,} ball states loaded")

    # Load matches for team features
    from config import MATCHES_CLEAN
    matches = pd.read_parquet(MATCHES_CLEAN)

    # Enrich with helper flags
    df = add_boundary_flag(df)
    df = add_bowler_wicket_flag(df)

    # Build feature stores
    venue_features   = build_venue_features(df)
    batter_features  = build_batter_features(df)
    bowler_features  = build_bowler_features(df)
    matchup_features = build_matchup_features(df)
    team_features, chase_win_features = build_team_features(df, matches)

    # Combine batter + bowler into unified player features
    player_features  = pd.concat([batter_features, bowler_features], ignore_index=True)

    # Save
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)

    venue_features.to_parquet(VENUE_FEATURES, index=False)
    player_features.to_parquet(PLAYER_FEATURES, index=False)
    team_features.to_parquet(TEAM_FEATURES, index=False)
    matchup_features.to_parquet(DATA_FEATURES / "matchup_features.parquet", index=False)
    chase_win_features.to_parquet(DATA_FEATURES / "chase_win_rate_by_pressure.parquet", index=False)

    log.info(f"Saved venue_features:   {len(venue_features):,} rows")
    log.info(f"Saved player_features:  {len(player_features):,} rows")
    log.info(f"Saved team_features:    {len(team_features):,} rows")
    log.info(f"Saved matchup_features: {len(matchup_features):,} rows")
    log.info("Pipeline 03 complete.")


if __name__ == "__main__":
    run()
