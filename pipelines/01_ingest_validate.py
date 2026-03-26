"""
01_ingest_validate.py — IPL Data Ingestion and Validation Pipeline

Responsibilities:
  1. Load raw deliveries.csv and matches.csv from data/raw/
  2. Enforce schema types and column presence
  3. Apply venue and franchise name normalization
  4. Run 10 explicit validation checks (see validate_deliveries / validate_matches)
  5. Write clean, schema-validated Parquet files to data/processed/
  6. Print a validation report — fail loudly on critical errors, warn on soft issues

Run:
  python pipelines/01_ingest_validate.py

Output:
  data/processed/deliveries_clean.parquet
  data/processed/matches_clean.parquet
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DELIVERIES_FILE, MATCHES_FILE,
    DELIVERIES_CLEAN, MATCHES_CLEAN,
    TEAM_NAME_MAP, VENUE_NAME_MAP,
    SEASON_MIN, SEASON_MAX,
    BOWLER_CREDITED_DISMISSALS, NON_BOWLER_DISMISSALS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Column presence requirements ──────────────────────────────────────────────

REQUIRED_DELIVERY_COLS = {
    "match_id", "inning", "over", "ball", "batsman", "bowler",
    "batsman_runs", "extra_runs", "total_runs", "is_wicket",
}

REQUIRED_MATCH_COLS = {
    "match_id", "season", "date", "team1", "team2",
    "toss_winner", "toss_decision", "winner", "venue",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_deliveries(path: Path) -> pd.DataFrame:
    """Load raw deliveries CSV with minimal type casting."""
    log.info(f"Loading deliveries from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_matches(path: Path) -> pd.DataFrame:
    """Load raw matches CSV with minimal type casting."""
    log.info(f"Loading matches from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Normalizations ────────────────────────────────────────────────────────────

def normalize_team_names(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Apply canonical franchise name mapping to specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: TEAM_NAME_MAP.get(x, x) if pd.notna(x) else x)
    return df


def normalize_venue_names(df: pd.DataFrame, col: str = "venue") -> pd.DataFrame:
    """Apply canonical venue name mapping."""
    if col in df.columns:
        df[col] = df[col].map(lambda x: VENUE_NAME_MAP.get(x, x) if pd.notna(x) else x)
    return df


def standardize_over_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'over' column is 1-indexed (1–20).
    Some dataset versions ship 0-indexed (0–19). Detect and fix.
    """
    if df["over"].min() == 0:
        log.warning("  'over' column detected as 0-indexed — converting to 1-indexed.")
        df["over"] = df["over"] + 1
    return df


def add_batting_team(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Derive batting_team per delivery by joining match toss/result info.
    Inning 1 batting team = team that batted first; inning 2 = team that chased.
    Since raw deliveries may not have batting_team, derive from match metadata.
    """
    # Build a lookup: match_id → {1: batting_team_1, 2: batting_team_2}
    # Assumption: team1 always bats first unless toss_winner elected to field
    # This assumption must be validated against actual results

    # Determine which team batted first per match
    def batting_first(row):
        if row["toss_decision"] == "bat":
            return row["toss_winner"]
        else:
            # toss winner chose to field, so other team batted first
            other = row["team2"] if row["toss_winner"] == row["team1"] else row["team1"]
            return other

    match_batting = matches[["match_id", "team1", "team2", "toss_winner", "toss_decision"]].copy()
    match_batting["inning1_batting_team"] = match_batting.apply(batting_first, axis=1)
    match_batting["inning2_batting_team"] = match_batting.apply(
        lambda r: r["team2"] if r["inning1_batting_team"] == r["team1"] else r["team1"],
        axis=1
    )

    inning_map = match_batting.set_index("match_id")[
        ["inning1_batting_team", "inning2_batting_team"]
    ]

    def get_batting_team(row):
        mid = row["match_id"]
        if mid not in inning_map.index:
            return None
        if row["inning"] == 1:
            return inning_map.loc[mid, "inning1_batting_team"]
        elif row["inning"] == 2:
            return inning_map.loc[mid, "inning2_batting_team"]
        return None  # super over or invalid inning

    deliveries["batting_team"] = deliveries.apply(get_batting_team, axis=1)
    log.info("  Derived batting_team for all deliveries.")
    return deliveries


# ── Validation checks ─────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.errors   = []
        self.warnings = []

    def error(self, msg: str):
        self.errors.append(msg)
        log.error(f"  [ERROR] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        log.warning(f"  [WARN]  {msg}")

    def info(self, msg: str):
        log.info(f"  [OK]    {msg}")

    def summary(self):
        log.info(f"\n{'='*60}")
        log.info(f"VALIDATION SUMMARY: {len(self.errors)} errors, {len(self.warnings)} warnings")
        if self.errors:
            for e in self.errors:
                log.error(f"  ERROR: {e}")
        if self.warnings:
            for w in self.warnings:
                log.warning(f"  WARN:  {w}")
        log.info(f"{'='*60}\n")
        return len(self.errors) == 0


def validate_column_presence(df: pd.DataFrame, required: set, name: str, report: ValidationReport):
    missing = required - set(df.columns)
    if missing:
        report.error(f"{name}: missing required columns: {missing}")
    else:
        report.info(f"{name}: all required columns present")


def validate_deliveries(df: pd.DataFrame, report: ValidationReport):
    """Run all 10 delivery-level validation checks."""

    # CHECK 1: Column presence
    validate_column_presence(df, REQUIRED_DELIVERY_COLS, "deliveries", report)

    # CHECK 2: Over count integrity — each innings should have <= 20 overs
    over_counts = (
        df[df["inning"].isin([1, 2])]
        .groupby(["match_id", "inning"])["over"]
        .nunique()
    )
    bad_overs = over_counts[over_counts > 20]
    if len(bad_overs) > 0:
        report.warn(f"Check 2 — {len(bad_overs)} innings with > 20 distinct overs (may include super overs)")
    else:
        report.info("Check 2 — All innings have <= 20 overs")

    # CHECK 3: Ball values — flag unusually high ball numbers per over
    legal_balls = df[df.get("wide_runs", pd.Series(0, index=df.index)).fillna(0) == 0]
    # Simple proxy: ball > 8 per over is suspicious
    suspicious_balls = df[df["ball"] > 8]
    if len(suspicious_balls) > 0:
        report.warn(f"Check 3 — {len(suspicious_balls)} deliveries with ball > 8 within over")
    else:
        report.info("Check 3 — Ball numbers within expected range")

    # CHECK 4: Orphaned match_ids (deliveries with no match record)
    # This check is run in validate_join() below after matches are loaded

    # CHECK 5: Score continuity — total_runs must be non-negative
    neg_runs = df[df["total_runs"] < 0]
    if len(neg_runs) > 0:
        report.error(f"Check 5 — {len(neg_runs)} deliveries with negative total_runs")
    else:
        report.info("Check 5 — No negative total_runs")

    # CHECK 6: Dismissal consistency
    wickets = df[df["is_wicket"] == 1]
    missing_dismissal = wickets[wickets["dismissal_kind"].isna() | (wickets["dismissal_kind"] == "")]
    if len(missing_dismissal) > 0:
        report.warn(f"Check 6 — {len(missing_dismissal)} wicket rows missing dismissal_kind")
    else:
        report.info("Check 6 — All is_wicket=1 rows have dismissal_kind")

    missing_dismissed = wickets[wickets["player_dismissed"].isna() | (wickets["player_dismissed"] == "")]
    if len(missing_dismissed) > 0:
        report.warn(f"Check 6b — {len(missing_dismissed)} wicket rows missing player_dismissed")
    else:
        report.info("Check 6b — All is_wicket=1 rows have player_dismissed")

    # CHECK 7: Duplicate ball detection
    dup_cols = ["match_id", "inning", "over", "ball", "batsman", "bowler"]
    existing_dup_cols = [c for c in dup_cols if c in df.columns]
    dupes = df[df.duplicated(subset=existing_dup_cols, keep=False)]
    if len(dupes) > 0:
        report.warn(f"Check 7 — {len(dupes)} apparently duplicate delivery rows detected")
    else:
        report.info("Check 7 — No duplicate (match, inning, over, ball, batsman, bowler) combinations")

    # CHECK 8: Inning range — only 1 and 2 for standard matches
    valid_innings = df["inning"].isin([1, 2, 3, 4])  # 3/4 = super over
    invalid_innings = df[~valid_innings]
    if len(invalid_innings) > 0:
        report.warn(f"Check 8 — {len(invalid_innings)} rows with inning not in {{1,2,3,4}}")
    else:
        report.info("Check 8 — All inning values valid")

    # CHECK 9: Runs consistency — batsman_runs + extra_runs should equal total_runs
    run_mismatch = df[df["batsman_runs"] + df["extra_runs"] != df["total_runs"]]
    if len(run_mismatch) > 0:
        report.warn(f"Check 9 — {len(run_mismatch)} rows where batsman_runs + extra_runs != total_runs")
    else:
        report.info("Check 9 — Run components sum correctly")

    # CHECK 10: Null batsman / bowler
    null_batsman = df["batsman"].isna().sum()
    null_bowler  = df["bowler"].isna().sum()
    if null_batsman > 0:
        report.warn(f"Check 10 — {null_batsman} rows with null batsman")
    if null_bowler > 0:
        report.warn(f"Check 10b — {null_bowler} rows with null bowler")
    if null_batsman == 0 and null_bowler == 0:
        report.info("Check 10 — No null batsman or bowler values")


def validate_matches(df: pd.DataFrame, report: ValidationReport):
    """Validate match-level data."""

    validate_column_presence(df, REQUIRED_MATCH_COLS, "matches", report)

    # Toss decision values
    if "toss_decision" in df.columns:
        invalid_toss = df[~df["toss_decision"].isin(["bat", "field"])]
        if len(invalid_toss) > 0:
            report.warn(f"Matches — {len(invalid_toss)} rows with unexpected toss_decision value: {invalid_toss['toss_decision'].unique()}")
        else:
            report.info("Matches — toss_decision values valid")

    # Season range
    if "season" in df.columns:
        out_of_range = df[(df["season"] < SEASON_MIN) | (df["season"] > SEASON_MAX)]
        if len(out_of_range) > 0:
            report.warn(f"Matches — {len(out_of_range)} rows outside season range {SEASON_MIN}–{SEASON_MAX}")

    # DL applied flag
    if "dl_applied" in df.columns:
        dl_count = df["dl_applied"].sum() if df["dl_applied"].dtype != object else (df["dl_applied"] == 1).sum()
        report.info(f"Matches — {dl_count} DL-affected matches flagged (will be excluded from chase modeling)")


def validate_join(deliveries: pd.DataFrame, matches: pd.DataFrame, report: ValidationReport):
    """Check referential integrity between deliveries and matches."""
    delivery_ids = set(deliveries["match_id"].unique())
    match_ids    = set(matches["match_id"].unique())

    orphaned = delivery_ids - match_ids
    if orphaned:
        report.error(f"Join — {len(orphaned)} match_ids in deliveries with no match record: {list(orphaned)[:10]}")
    else:
        report.info(f"Join — All delivery match_ids exist in matches ({len(match_ids)} matches)")

    empty_matches = match_ids - delivery_ids
    if empty_matches:
        report.warn(f"Join — {len(empty_matches)} matches have no delivery records")


# ── Type casting ──────────────────────────────────────────────────────────────

def cast_delivery_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce delivery columns to correct dtypes, filling nulls on numeric cols."""
    int_cols = ["match_id", "inning", "over", "ball", "batsman_runs",
                "extra_runs", "total_runs", "is_wicket"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Nullable int-like extras columns
    extra_cols = ["wide_runs", "bye_runs", "legbye_runs", "noball_runs", "penalty_runs"]
    for col in extra_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # String columns
    str_cols = ["batsman", "non_striker", "bowler", "player_dismissed",
                "dismissal_kind", "fielder"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)

    return df


def cast_match_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce match columns to correct dtypes."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "season" in df.columns:
        # season may be a year like 2008 or a string like "2007/08"
        df["season"] = df["season"].astype(str).str.extract(r"(\d{4})")[0]
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    for col in ["win_by_runs", "win_by_wickets", "dl_applied"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Super over filtering ───────────────────────────────────────────────────────

def remove_super_overs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove super over deliveries from analysis dataset.
    Super overs are innings 3/4 or flagged by is_super_over column.
    This is standard in cricket analytics — super overs have different strategic context.
    """
    original_len = len(df)
    if "is_super_over" in df.columns:
        df = df[df["is_super_over"].fillna(0) == 0]
    df = df[df["inning"].isin([1, 2])]
    removed = original_len - len(df)
    if removed > 0:
        log.info(f"  Removed {removed} super over deliveries")
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    report = ValidationReport()

    # 1. Load
    if not DELIVERIES_FILE.exists():
        log.error(
            f"Deliveries file not found at {DELIVERIES_FILE}.\n"
            "Download the dataset from Kaggle: https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025\n"
            "Place deliveries.csv and matches.csv in data/raw/"
        )
        sys.exit(1)
    if not MATCHES_FILE.exists():
        log.error(f"Matches file not found at {MATCHES_FILE}.")
        sys.exit(1)

    deliveries = load_deliveries(DELIVERIES_FILE)
    matches    = load_matches(MATCHES_FILE)

    # 2. Type casting
    deliveries = cast_delivery_types(deliveries)
    matches    = cast_match_types(matches)

    # 3. Over index normalization
    deliveries = standardize_over_index(deliveries)

    # 4. Normalization
    log.info("Normalizing team and venue names...")
    matches    = normalize_team_names(matches, ["team1", "team2", "toss_winner", "winner"])
    matches    = normalize_venue_names(matches)
    deliveries = normalize_team_names(deliveries, ["batting_team"] if "batting_team" in deliveries.columns else [])

    # 5. Super over removal
    log.info("Removing super over deliveries...")
    deliveries = remove_super_overs(deliveries)

    # 6. Derive batting_team per delivery
    log.info("Deriving batting_team per delivery...")
    deliveries = add_batting_team(deliveries, matches)

    # 7. Validation
    log.info("\nRunning validation checks on deliveries...")
    validate_deliveries(deliveries, report)

    log.info("\nRunning validation checks on matches...")
    validate_matches(matches, report)

    log.info("\nValidating join integrity...")
    validate_join(deliveries, matches, report)

    # 8. Report
    report.summary()
    if not report.summary():
        log.error("Critical validation errors found. Fix before proceeding.")
        sys.exit(1)

    # 9. Save
    log.info("Writing validated data to Parquet...")
    DATA_PROCESSED = DELIVERIES_CLEAN.parent
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    deliveries.to_parquet(DELIVERIES_CLEAN, index=False)
    matches.to_parquet(MATCHES_CLEAN, index=False)

    log.info(f"  Saved: {DELIVERIES_CLEAN}  ({len(deliveries):,} rows)")
    log.info(f"  Saved: {MATCHES_CLEAN}  ({len(matches):,} rows)")
    log.info("Pipeline 01 complete.")


if __name__ == "__main__":
    run()
