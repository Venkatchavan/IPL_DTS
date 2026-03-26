"""
02_state_reconstruction.py — Ball-by-Ball Match State Engine

For every legal delivery in every innings, compute the full match state vector.
This is the primary analytical artifact. All downstream metrics, models, and
dashboard components derive from this table.

State fields computed at the START of each delivery (pre-outcome, to prevent leakage):
  - Score context: runs so far, wickets lost, balls bowled/remaining
  - Phase: powerplay / middle / death
  - Rate context: CRR, RRR, rrr_crr_delta (inning 2 only)
  - Pressure context: dot_ball_streak, last-N-ball run rate, partnership stats
  - Players: striker, non_striker, bowler
  - Match context: venue, season, batting/bowling team, toss info

Then the outcome of the delivery is appended (batsman_runs, is_wicket, etc.)
so the table supports both state → outcome training and state transition analysis.

Run:
  python pipelines/02_state_reconstruction.py

Input:
  data/processed/deliveries_clean.parquet
  data/processed/matches_clean.parquet

Output:
  data/features/ball_states.parquet
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DELIVERIES_CLEAN, MATCHES_CLEAN, BALL_STATES_FILE,
    get_phase, get_pressure_band,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LEGAL_BALLS_PER_INNINGS = 120
MAX_WICKETS = 10
RECENT_BALLS_WINDOW = 12   # ~2 overs for rolling run-rate momentum


# ── Legal ball flag ───────────────────────────────────────────────────────────

def add_is_legal_ball(df: pd.DataFrame) -> pd.DataFrame:
    """
    A delivery is legal if it is NOT a wide or no-ball.
    Legal balls consume one of the 120 over-balls in an innings.
    Extras (byes, leg-byes) that are NOT wides/no-balls ARE legal deliveries.
    """
    if "wide_runs" in df.columns and "noball_runs" in df.columns:
        df["is_legal_ball"] = (
            (df["wide_runs"].fillna(0) == 0) &
            (df["noball_runs"].fillna(0) == 0)
        ).astype(int)
    else:
        # Fallback: if wide/noball columns not present, mark all balls legal
        # This will be overestimated but is safe for fallback
        df["is_legal_ball"] = 1
        log.warning("wide_runs/noball_runs columns not found — all balls marked legal (may overcount)")
    return df


# ── Core state reconstruction ─────────────────────────────────────────────────

def reconstruct_match_states(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Main reconstruction function.
    Iterates over matches × innings sorted by delivery order and
    computes cumulative/rolling state features at each delivery.
    Returns a row per delivery with the full state vector.
    """
    # Merge match metadata
    match_meta = matches[[
        "match_id", "season", "venue", "toss_winner", "toss_decision",
        "winner", "team1", "team2", "dl_applied"
    ]].copy()

    df = deliveries.merge(match_meta, on="match_id", how="left")
    df = add_is_legal_ball(df)

    # Sort into strict delivery order
    df = df.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)

    log.info(f"Reconstructing state for {df['match_id'].nunique():,} matches, {len(df):,} deliveries...")

    # Pre-sort groups for iteration
    results = []

    for (match_id, inning), grp in df.groupby(["match_id", "inning"], sort=True):
        inning_states = _reconstruct_innings(grp, match_id, inning)
        results.append(inning_states)

    log.info("Concatenating state rows...")
    state_df = pd.concat(results, ignore_index=True)
    log.info(f"  Final state table: {len(state_df):,} rows, {len(state_df.columns)} columns")
    return state_df


def _reconstruct_innings(grp: pd.DataFrame, match_id: int, inning: int) -> pd.DataFrame:
    """
    Reconstruct all delivery states within one innings.
    Returns a DataFrame where each row is one delivery with full state context.

    IMPORTANT: State fields marked PRE_ are computed BEFORE this delivery's outcome,
    making them safe features for model training (no leakage).
    """
    rows = []

    # Innings-level constants from first row
    first = grp.iloc[0]
    batting_team  = first.get("batting_team", None)
    bowling_team  = (
        first["team2"] if batting_team == first.get("team1") else first.get("team1")
        if batting_team is not None else None
    )
    venue         = first.get("venue", None)
    season        = first.get("season", None)
    toss_winner   = first.get("toss_winner", None)
    toss_decision = first.get("toss_decision", None)
    dl_applied    = int(first.get("dl_applied", 0) or 0)
    winner        = first.get("winner", None)
    match_won = 1 if winner == batting_team else (0 if winner is not None else None)

    # Determine target for inning 2 (set externally — must come from innings 1 total)
    # We'll backfill this after processing both innings; for now, set None
    target = None  # placeholder; filled by join after both innings processed

    # Cumulative state accumulators
    cum_runs      = 0
    cum_wickets   = 0
    cum_legal     = 0   # legal balls consumed

    # Rolling window for momentum
    recent_runs_window    = []   # list of (runs, is_legal) for last N balls
    recent_wickets_window = []   # parallel list of is_wicket

    # Dot streak tracker
    dot_streak = 0

    # Partnership tracker
    partnership_runs  = 0
    partnership_balls = 0

    for idx, row in grp.iterrows():
        over         = int(row["over"])
        ball_in_over = int(row["ball"])
        is_legal     = int(row.get("is_legal_ball", 1))
        batsman_runs = int(row["batsman_runs"])
        extra_runs   = int(row["extra_runs"])
        total_runs   = int(row["total_runs"])
        is_wicket    = int(row["is_wicket"])
        dismissal    = row.get("dismissal_kind", None)
        dismissed    = row.get("player_dismissed", None)

        # ── State BEFORE this delivery (pre-outcome) ──────────────────────────

        balls_remaining_legal = MAX_LEGAL_BALLS_PER_INNINGS - cum_legal
        overs_bowled = cum_legal / 6.0
        crr = (cum_runs / overs_bowled) if overs_bowled > 0 else 0.0

        phase = get_phase(over)

        # Recent momentum (last RECENT_BALLS_WINDOW legal balls)
        recent_legal = [(r, w) for (r, legal, w) in
                        zip(recent_runs_window, [1]*len(recent_runs_window), recent_wickets_window)
                        ][-RECENT_BALLS_WINDOW:]
        last_n_runs     = sum(r for (r, w) in recent_legal)
        last_n_wickets  = sum(w for (r, w) in recent_legal)
        last_n_balls_rr = (last_n_runs / (len(recent_legal) / 6)) if recent_legal else 0.0

        state = {
            # Identifiers
            "match_id":              match_id,
            "inning":                inning,
            "over":                  over,
            "ball_in_over":          ball_in_over,
            "global_ball_number":    cum_legal + 1,  # 1-indexed legal ball in innings

            # Pre-delivery score state
            "pre_runs":              cum_runs,
            "pre_wickets":           cum_wickets,
            "pre_balls_bowled":      cum_legal,
            "pre_balls_remaining":   balls_remaining_legal,
            "pre_wickets_in_hand":   MAX_WICKETS - cum_wickets,

            # Phase
            "phase":                 phase,

            # Rate context
            "pre_crr":               round(crr, 4),

            # Momentum
            "pre_dot_ball_streak":   dot_streak,
            "pre_last_n_runs":       last_n_runs,
            "pre_last_n_wickets":    last_n_wickets,
            "pre_last_n_balls_rr":   round(last_n_balls_rr, 4),
            "pre_partnership_runs":  partnership_runs,
            "pre_partnership_balls": partnership_balls,

            # Players
            "striker":               row.get("batsman", None),
            "non_striker":           row.get("non_striker", None),
            "bowler":                row.get("bowler", None),

            # Match context
            "batting_team":          batting_team,
            "bowling_team":          bowling_team,
            "venue":                 venue,
            "season":                season,
            "toss_winner":           toss_winner,
            "toss_decision":         toss_decision,
            "dl_applied":            dl_applied,
            "match_won":             match_won,

            # Outcome of this delivery (post-state)
            "batsman_runs":          batsman_runs,
            "extra_runs":            extra_runs,
            "total_runs":            total_runs,
            "is_legal_ball":         is_legal,
            "is_wicket":             is_wicket,
            "dismissal_kind":        dismissal,
            "player_dismissed":      dismissed,
        }
        rows.append(state)

        # ── Update accumulators (post-delivery) ───────────────────────────────

        if is_legal:
            cum_legal        += 1
            recent_runs_window.append(batsman_runs)   # batsman runs only for momentum
            recent_wickets_window.append(is_wicket)
            # Trim to window
            if len(recent_runs_window) > RECENT_BALLS_WINDOW:
                recent_runs_window.pop(0)
                recent_wickets_window.pop(0)

            # Dot ball streak
            if batsman_runs == 0 and extra_runs == 0:
                dot_streak += 1
            else:
                dot_streak = 0

            # Partnership
            partnership_runs  += batsman_runs + extra_runs
            partnership_balls += 1

        cum_runs    += total_runs

        if is_wicket:
            cum_wickets      += 1
            partnership_runs  = 0
            partnership_balls = 0

    innings_df = pd.DataFrame(rows)

    # Compute RRR and related chase fields — placeholder, filled in post-join
    innings_df["target"]        = None
    innings_df["pre_runs_needed"] = None
    innings_df["pre_rrr"]         = None
    innings_df["pre_rrr_crr_delta"] = None

    return innings_df


def add_chase_fields(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    For inning 2, compute:
      - target (inning 1 total + 1)
      - pre_runs_needed
      - pre_rrr
      - pre_rrr_crr_delta
      - pressure_band

    Target = max cumulative runs in inning 1 + 1, per match.
    """
    log.info("Computing inning-1 totals for chase target derivation...")

    inning1 = state_df[state_df["inning"] == 1].copy()

    # Final score per match = max of pre_runs + total_runs on last ball
    inning1["ball_score"] = inning1["pre_runs"] + inning1["total_runs"]
    inning1_totals = (
        inning1.groupby("match_id")["ball_score"]
        .max()
        .rename("inning1_total")
        .reset_index()
    )
    inning1_totals["target"] = inning1_totals["inning1_total"] + 1

    state_df = state_df.merge(
        inning1_totals[["match_id", "target"]], on="match_id", how="left", suffixes=("_old", "")
    )
    # Drop the old placeholder column if it exists
    if "target_old" in state_df.columns:
        state_df.drop(columns=["target_old"], inplace=True)

    # Compute chase fields for inning 2 only
    mask2 = state_df["inning"] == 2
    # Initialise as float so partial-mask assignment keeps numeric dtype (pandas 2.x)
    state_df["pre_runs_needed"] = np.nan
    state_df["pre_rrr"] = np.nan
    state_df["pre_rrr_crr_delta"] = np.nan
    state_df.loc[mask2, "pre_runs_needed"] = (
        state_df.loc[mask2, "target"] - state_df.loc[mask2, "pre_runs"]
    ).clip(lower=0)

    state_df.loc[mask2, "pre_rrr"] = (
        state_df.loc[mask2, "pre_runs_needed"] /
        (state_df.loc[mask2, "pre_balls_remaining"] / 6).replace(0, np.nan)
    ).round(4)

    state_df.loc[mask2, "pre_rrr_crr_delta"] = (
        state_df.loc[mask2, "pre_rrr"] - state_df.loc[mask2, "pre_crr"]
    ).round(4)

    # Pressure band for all rows (inning 1 = neutral by default)
    state_df["pressure_band"] = state_df["pre_rrr_crr_delta"].apply(
        lambda x: get_pressure_band(x) if pd.notna(x) else "neutral"
    )

    return state_df


def add_win_flag(state_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure match_won is correctly populated for all rows.
    match_won = 1 if batting_team won the match, else 0, else None (no result).
    """
    winner_map = matches.set_index("match_id")["winner"].to_dict()
    state_df["match_winner"] = state_df["match_id"].map(winner_map)
    state_df["match_won"] = (
        state_df["match_winner"] == state_df["batting_team"]
    ).astype("Int8")
    state_df.loc[state_df["match_winner"].isna(), "match_won"] = pd.NA
    state_df.drop(columns=["match_winner"], inplace=True)
    return state_df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    log.info("Pipeline 02: Match State Reconstruction")

    log.info("Loading processed data...")
    deliveries = pd.read_parquet(DELIVERIES_CLEAN)
    matches    = pd.read_parquet(MATCHES_CLEAN)

    log.info(f"  {len(deliveries):,} deliveries, {len(matches):,} matches loaded")

    state_df = reconstruct_match_states(deliveries, matches)
    state_df = add_chase_fields(state_df)
    state_df = add_win_flag(state_df, matches)

    # Reorder columns for readability
    id_cols  = ["match_id", "inning", "over", "ball_in_over", "global_ball_number"]
    ctx_cols = ["season", "venue", "batting_team", "bowling_team",
                "toss_winner", "toss_decision", "dl_applied"]
    state_cols = [
        "phase", "pre_runs", "pre_wickets", "pre_balls_bowled",
        "pre_balls_remaining", "pre_wickets_in_hand",
        "pre_crr", "pre_rrr", "pre_rrr_crr_delta", "pressure_band",
        "target", "pre_runs_needed",
    ]
    momentum_cols = [
        "pre_dot_ball_streak", "pre_last_n_runs", "pre_last_n_wickets",
        "pre_last_n_balls_rr", "pre_partnership_runs", "pre_partnership_balls",
    ]
    player_cols  = ["striker", "non_striker", "bowler"]
    outcome_cols = [
        "batsman_runs", "extra_runs", "total_runs", "is_legal_ball",
        "is_wicket", "dismissal_kind", "player_dismissed",
    ]
    result_cols  = ["match_won"]

    ordered_cols = (
        id_cols + ctx_cols + state_cols + momentum_cols +
        player_cols + outcome_cols + result_cols
    )
    # Add any remaining columns not explicitly ordered
    remaining = [c for c in state_df.columns if c not in ordered_cols]
    state_df = state_df[ordered_cols + remaining]

    BALL_STATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    state_df.to_parquet(BALL_STATES_FILE, index=False)

    log.info(f"Saved: {BALL_STATES_FILE}  ({len(state_df):,} rows, {len(state_df.columns)} columns)")
    log.info("Pipeline 02 complete.")

    # Quick sanity print
    log.info("\nSample state row (inning 2, random):")
    sample = state_df[state_df["inning"] == 2].sample(1, random_state=42)
    for col in state_cols + momentum_cols:
        if col in sample.columns:
            log.info(f"  {col}: {sample[col].values[0]}")


if __name__ == "__main__":
    run()
