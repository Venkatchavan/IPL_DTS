"""
data/adapt_ipl_csv.py — Convert the combined IPL.csv into the expected
deliveries.csv + matches.csv format in data/raw/.

The combined IPL.csv has a different schema from the Cricsheet-style
deliveries.csv our pipeline expects. This script normalises it so the
existing pipelines (01-05) run without modification.

Run once before running the pipelines:
    python data/adapt_ipl_csv.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SRC     = ROOT / "data" / "IPL.csv"


def adapt():
    if not SRC.exists():
        log.error(f"Source file not found: {SRC}")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_deliveries = RAW_DIR / "deliveries.csv"
    out_matches    = RAW_DIR / "matches.csv"

    if out_deliveries.exists() and out_matches.exists():
        log.info("deliveries.csv and matches.csv already exist — skipping adaptation.")
        return

    log.info(f"Reading {SRC.name}  ({SRC.stat().st_size / 1e6:.1f} MB)...")
    df = pd.read_csv(SRC, low_memory=False)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns loaded.")

    # ── Normalise over (0-indexed → 1-indexed) ────────────────────────────────
    df["over_1indexed"] = df["over"] + 1

    # ── Derive wide_runs / noball_runs ─────────────────────────────────────────
    df["wide_runs"]   = np.where(df["extra_type"].str.lower().isin(["wides", "wide"]),
                                 df["runs_extras"].fillna(0), 0).astype(int)
    df["noball_runs"] = np.where(df["extra_type"].str.lower().isin(["noballs", "noball", "nb"]),
                                 1, 0).astype(int)

    # ── Derive is_wicket from wicket_kind ─────────────────────────────────────
    df["is_wicket"] = (~df["wicket_kind"].isna() &
                       (df["wicket_kind"].str.strip() != "") &
                       (df["wicket_kind"].str.lower() != "nan")).astype(int)

    # ── Normalise season to integer year ─────────────────────────────────────
    # Source format: "2007/08", "2008", "2022/23", etc.
    # Pipelines expect: 2008, 2009, ... 2025 (year when IPL season ends)
    def _season_to_year(s):
        s = str(s).strip()
        if "/" in s:
            # "2007/08" → take the second part as 2-digit year, attach century
            parts = s.split("/")
            try:
                base_year = int(parts[0])
                suffix    = int(parts[1])
                # Reconstruct: 2007/08 → 2008; 2022/23 → 2023
                return base_year + 1
            except ValueError:
                return None
        else:
            try:
                return int(s)
            except ValueError:
                return None

    df["season_year"] = df["season"].apply(_season_to_year)

    # ── Build deliveries DataFrame ────────────────────────────────────────────
    deliveries = pd.DataFrame({
        "match_id":         df["match_id"],
        "inning":           df["innings"],
        "over":             df["over_1indexed"],
        "ball":             df["ball"],
        "batting_team":     df["batting_team"],
        "bowling_team":     df["bowling_team"],
        "batsman":          df["batter"],           # pipeline expects "batsman"
        "non_striker":      df["non_striker"],
        "bowler":           df["bowler"],
        "batsman_runs":     df["runs_batter"].fillna(0).astype(int),
        "extra_runs":       df["runs_extras"].fillna(0).astype(int),
        "total_runs":       df["runs_total"].fillna(0).astype(int),
        "wide_runs":        df["wide_runs"],
        "noball_runs":      df["noball_runs"],
        "extras_type":      df["extra_type"].fillna(""),
        "is_wicket":        df["is_wicket"],
        "player_dismissed": df["player_out"].fillna(""),
        "dismissal_kind":   df["wicket_kind"].fillna(""),
        # NOTE: season is intentionally omitted here — pipeline 02 gets it
        # from matches_clean via merge on match_id to avoid column conflict.
    })

    log.info(f"Writing deliveries.csv  ({len(deliveries):,} rows)...")
    deliveries.to_csv(out_deliveries, index=False)

    # ── Build matches DataFrame ───────────────────────────────────────────────
    # One row per unique match — take first row per match for match-level fields
    match_cols = [
        "match_id", "date", "season_year", "venue", "city",
        "toss_winner", "toss_decision", "match_won_by", "win_outcome", "method",
    ]
    available_cols = [c for c in match_cols if c in df.columns]
    match_meta = df.groupby("match_id")[available_cols].first().reset_index(drop=True)
    # Rename season_year → season
    match_meta.rename(columns={"season_year": "season"}, inplace=True)

    # Derive team1 / team2: inning 1 batting team = team1, inning 2 = team2
    teams = (
        df[df["innings"].isin([1, 2])]
        .groupby(["match_id", "innings"])["batting_team"]
        .first()
        .unstack(fill_value="")
        .rename(columns={1: "team1", 2: "team2"})
        .reset_index()
    )
    match_meta = match_meta.merge(teams, on="match_id", how="left")

    # Rename winner column
    if "match_won_by" in match_meta.columns:
        match_meta.rename(columns={"match_won_by": "winner"}, inplace=True)

    # DL applied flag
    if "method" in match_meta.columns:
        match_meta["dl_applied"] = match_meta["method"].str.upper().isin(["D/L", "DLS"]).astype(int)
    else:
        match_meta["dl_applied"] = 0

    # Ensure required columns exist, fill missing
    for col in ["team1", "team2", "venue", "toss_winner", "toss_decision", "winner", "dl_applied"]:
        if col not in match_meta.columns:
            match_meta[col] = ""

    log.info(f"Writing matches.csv  ({len(match_meta):,} rows)...")
    match_meta.to_csv(out_matches, index=False)

    log.info("Adaptation complete.")
    log.info(f"  deliveries.csv → {out_deliveries}")
    log.info(f"  matches.csv    → {out_matches}")


if __name__ == "__main__":
    adapt()
