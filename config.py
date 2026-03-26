"""
config.py — Project-wide constants, normalization mappings, and configuration values.

All magic numbers, threshold values, and category definitions live here.
Import this module wherever constants are needed — never hardcode values inline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
DATA_FEATURES   = ROOT_DIR / "data" / "features"
MODELS_DIR      = ROOT_DIR / "models"

# Raw data file names (as downloaded from Kaggle)
DELIVERIES_FILE = DATA_RAW / "deliveries.csv"
MATCHES_FILE    = DATA_RAW / "matches.csv"

# Processed parquet outputs
DELIVERIES_CLEAN    = DATA_PROCESSED / "deliveries_clean.parquet"
MATCHES_CLEAN       = DATA_PROCESSED / "matches_clean.parquet"
BALL_STATES_FILE    = DATA_FEATURES  / "ball_states.parquet"
PLAYER_FEATURES     = DATA_FEATURES  / "player_features.parquet"
TEAM_FEATURES       = DATA_FEATURES  / "team_features.parquet"
METRICS_BALL        = DATA_FEATURES  / "metrics_ball_level.parquet"
METRICS_PLAYER      = DATA_FEATURES  / "metrics_player_level.parquet"

# ── Season range ───────────────────────────────────────────────────────────────
SEASON_MIN = 2008
SEASON_MAX = 2025

# Train/validate/test splits (time-aware — no future leakage)
TRAIN_SEASONS    = list(range(2008, 2023))   # 2008–2022
VALIDATE_SEASONS = [2023]
TEST_SEASONS     = [2024, 2025]

# ── Match state phase definitions ──────────────────────────────────────────────
PHASE_POWERPLAY = "powerplay"   # overs 1–6
PHASE_MIDDLE    = "middle"      # overs 7–15
PHASE_DEATH     = "death"       # overs 16–20

PHASE_BOUNDARIES = {
    PHASE_POWERPLAY: (1, 6),
    PHASE_MIDDLE:    (7, 15),
    PHASE_DEATH:     (16, 20),
}

def get_phase(over: int) -> str:
    """Return the phase label for a given over number (1-indexed)."""
    if 1 <= over <= 6:
        return PHASE_POWERPLAY
    elif 7 <= over <= 15:
        return PHASE_MIDDLE
    elif 16 <= over <= 20:
        return PHASE_DEATH
    return "unknown"

# ── Pressure band thresholds (rrr_crr_delta = RRR - CRR) ──────────────────────
# Used to segment batting/bowling context for context-adjusted metrics.
PRESSURE_BANDS = {
    "low":      (-999, -1.5),   # batting well ahead of required rate
    "neutral":  (-1.5,  1.5),   # on track
    "high":     ( 1.5,  3.0),   # under pressure
    "critical": ( 3.0,  999),   # severe pressure
}

def get_pressure_band(rrr_crr_delta: float) -> str:
    for band, (lo, hi) in PRESSURE_BANDS.items():
        if lo <= rrr_crr_delta < hi:
            return band
    return "neutral"

# ── Wicket penalty for reward function ────────────────────────────────────────
WICKET_PENALTY = 5.0

# ── RL constants ──────────────────────────────────────────────────────────────
RL_DISCOUNT_GAMMA    = 0.95
RL_MAX_ITERATIONS    = 50
RL_CHASE_WIN_BONUS   = 10.0
RL_MIN_STATE_SUPPORT = 20   # minimum observations per state for confident Q-value

# ── Minimum sample thresholds for public metric display ───────────────────────
MIN_BALLS_FACED_BATTER   = 100   # batter must have faced >= 100 balls in context window
MIN_BALLS_BOWLED_BOWLER  = 60    # bowler minimum balls (per phase bucket)
MIN_BALLS_MATCHUP        = 30    # batter-bowler pair minimum balls for MLS
MIN_BALLS_DEATH_DSI      = 50    # bowler minimum death-over balls for DSI
MIN_MATCHES_TEAM_METRIC  = 10    # team minimum matches for any team metric

# ── Venue scoring tiers (assigned after computing historical averages) ─────────
VENUE_TIER_LOW    = "low"
VENUE_TIER_MEDIUM = "medium"
VENUE_TIER_HIGH   = "high"

# ── Franchise name normalization ───────────────────────────────────────────────
# Maps historical/old names to current canonical names. Extend as needed.
TEAM_NAME_MAP = {
    "Delhi Daredevils":              "Delhi Capitals",
    "Deccan Chargers":               "Sunrisers Hyderabad",
    "Pune Warriors":                 "Pune Warriors",        # defunct
    "Rising Pune Supergiant":        "Rising Pune Supergiant",  # defunct
    "Rising Pune Supergiants":       "Rising Pune Supergiant",
    "Kochi Tuskers Kerala":          "Kochi Tuskers Kerala",  # defunct
    "Gujarat Lions":                 "Gujarat Lions",         # defunct
    "Kings XI Punjab":               "Punjab Kings",
    "Royal Challengers Bangalore":   "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":   "Royal Challengers Bengaluru",
}

# ── Venue name normalization ───────────────────────────────────────────────────
VENUE_NAME_MAP = {
    "Eden Garden":                           "Eden Gardens",
    "M Chinnaswamy Stadium":                 "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium":                 "M. Chinnaswamy Stadium",
    "Feroz Shah Kotla":                      "Arun Jaitley Stadium",
    "Feroz Shah Kotla Ground":               "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi":           "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "DY Patil Stadium":                      "Dr DY Patil Sports Academy",
    "Brabourne Stadium, Mumbai":             "Brabourne Stadium",
    "Wankhede Stadium, Mumbai":              "Wankhede Stadium",
}

# ── Dismissal kinds that count as wickets for bowling credit ──────────────────
# (run out is ambiguous — it is a wicket but rarely bowler's credit)
BOWLER_CREDITED_DISMISSALS = {
    "bowled", "caught", "lbw", "stumped",
    "caught and bowled", "hit wicket"
}
NON_BOWLER_DISMISSALS = {
    "run out", "obstructing the field", "handled the ball",
    "retired hurt", "timed out"
}

# ── Legal delivery definition ──────────────────────────────────────────────────
# Wides and no-balls do not count as legal deliveries (do not consume an over ball)
ILLEGAL_EXTRA_TYPES = {"wide", "noball"}   # used to filter is_legal_ball

# ── RL action labels ───────────────────────────────────────────────────────────
RL_ACTION_CONSERVATIVE = "conservative"
RL_ACTION_BALANCED     = "balanced"
RL_ACTION_AGGRESSIVE   = "aggressive"
RL_ACTIONS             = [RL_ACTION_CONSERVATIVE, RL_ACTION_BALANCED, RL_ACTION_AGGRESSIVE]

# ── Pressure Index weights (see metrics/pressure_index.py) ────────────────────
PI_WEIGHT_RRR_CRR   = 0.35
PI_WEIGHT_WICKETS   = 0.30
PI_WEIGHT_PHASE     = 0.20
PI_WEIGHT_DOT_STREAK = 0.15

PHASE_WEIGHT_MAP = {
    PHASE_POWERPLAY: 0.60,
    PHASE_MIDDLE:    0.75,
    PHASE_DEATH:     1.00,
}

# ── Death Overs Suppression Index weights ─────────────────────────────────────
DSI_WEIGHT_ECONOMY   = 0.30
DSI_WEIGHT_DOT_PCT   = 0.25
DSI_WEIGHT_WICKETS   = 0.25
DSI_WEIGHT_BOUNDARY  = 0.20
