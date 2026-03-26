"""
app/config.py — Dashboard application configuration

Color palette, chart defaults, phase color maps, and UI constants.
"""

from pathlib import Path
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
FEATURES    = DATA_DIR / "features"
METRICS     = DATA_DIR / "metrics"
MODELS_DIR  = ROOT / "models"
RL_DIR      = MODELS_DIR / "rl_strategy"

# Data files loaded at startup
BALL_STATES_FILE       = FEATURES / "ball_states_scored.parquet"
BALL_STATES_FALLBACK   = FEATURES / "ball_states.parquet"
PLAYER_FEATURES_FILE   = FEATURES / "player_features.parquet"
TEAM_FEATURES_FILE     = FEATURES / "team_features.parquet"
VENUE_FEATURES_FILE    = FEATURES / "venue_features.parquet"
MATCHUP_FEATURES_FILE  = FEATURES / "matchup_features.parquet"
METRICS_BALL_FILE      = METRICS  / "metrics_ball_level.parquet"
METRICS_PLAYER_FILE    = METRICS  / "metrics_player_level.parquet"
METRICS_MATCHUP_FILE   = METRICS  / "metrics_matchup.parquet"
POLICY_TABLE_FILE      = RL_DIR   / "policy_table.parquet"
HIST_WIN_RATES_FILE    = RL_DIR   / "historical_win_rates.parquet"

# ── Color palette (professional, not garish) ──────────────────────────────────

# Phase colors
PHASE_COLORS = {
    "powerplay": "#3B82F6",   # blue
    "middle":    "#F59E0B",   # amber
    "death":     "#EF4444",   # red
}

# Pressure colors
PRESSURE_COLORS = {
    "low":      "#22C55E",   # green
    "neutral":  "#94A3B8",   # slate
    "high":     "#F97316",   # orange
    "critical": "#EF4444",   # red
}

# Action colors (RL strategy)
ACTION_COLORS = {
    "conservative": "#3B82F6",   # blue
    "balanced":     "#22C55E",   # green
    "aggressive":   "#EF4444",   # red
}

# Confidence colors
CONFIDENCE_COLORS = {
    "high":   "#22C55E",
    "medium": "#F59E0B",
    "low":    "#EF4444",
}

# Default chart theme
CHART_BG = "#0F172A"        # very dark navy
CHART_GRID = "#1E293B"      # dark slate
CHART_TEXT = "#CBD5E1"       # light slate
WP_LINE_COLOR = "#38BDF8"    # sky blue
EFS_LINE_COLOR = "#A78BFA"   # violet

# ── UI constants ──────────────────────────────────────────────────────────────

APP_TITLE   = "T20 Decision Intelligence — IPL 2008–2025"
APP_ICON    = "🏏"
LAYOUT      = "wide"
SIDEBAR_WIDTH = 320  # px (informational only; Streamlit controls this)

# Minimum sample thresholds for display (suppress noisy cells)
MIN_DISPLAY_BALLS_BATTER  = 100
MIN_DISPLAY_BALLS_BOWLER  = 60
MIN_DISPLAY_BALLS_MATCHUP = 30

# WPA high-leverage threshold
HIGH_LEVERAGE_WPA = 0.05

# Default metric sort (for leaderboards)
DEFAULT_BATTER_SORT  = "wpa_total"
DEFAULT_BOWLER_SORT  = "wpa_bowler_total"

SEASONS_ALL  = list(range(2008, 2026))
PHASES_ALL   = ["powerplay", "middle", "death"]
INNINGS_ALL  = [1, 2]

# Phase labels for display
PHASE_LABELS = {
    "powerplay": "Powerplay (Ov 1–6)",
    "middle":    "Middle (Ov 7–15)",
    "death":     "Death (Ov 16–20)",
}

# ── Plotly layout defaults ────────────────────────────────────────────────────

PLOTLY_DEFAULTS = dict(
    paper_bgcolor = CHART_BG,
    plot_bgcolor  = CHART_BG,
    font          = dict(color=CHART_TEXT, size=12),
    margin        = dict(l=40, r=20, t=50, b=40),
    xaxis         = dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False),
)

def apply_chart_theme(fig):
    """Apply standard dark chart theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_DEFAULTS)
    return fig
