"""
app/main.py — Streamlit entry point

Loads all parquet datasets once at startup with @st.cache_data,
then routes to the appropriate tab.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (
    APP_TITLE, APP_ICON, LAYOUT,
    BALL_STATES_FILE, BALL_STATES_FALLBACK,
    PLAYER_FEATURES_FILE, TEAM_FEATURES_FILE, VENUE_FEATURES_FILE,
    MATCHUP_FEATURES_FILE, METRICS_BALL_FILE, METRICS_PLAYER_FILE,
    METRICS_MATCHUP_FILE, POLICY_TABLE_FILE, HIST_WIN_RATES_FILE,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = APP_TITLE,
    page_icon   = APP_ICON,
    layout      = LAYOUT,
    initial_sidebar_state = "expanded",
)

# ── Data bootstrap (auto-downloads + runs pipelines if data is missing) ───────
from setup_data import ensure_data_ready
ensure_data_ready()


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading ball states...")
def load_ball_states():
    path = BALL_STATES_FILE if BALL_STATES_FILE.exists() else BALL_STATES_FALLBACK
    if not path.exists():
        st.error("Ball states data not found. Run pipelines 01–05 first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading player features...")
def load_player_features():
    if PLAYER_FEATURES_FILE.exists():
        return pd.read_parquet(PLAYER_FEATURES_FILE)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading team features...")
def load_team_features():
    if TEAM_FEATURES_FILE.exists():
        return pd.read_parquet(TEAM_FEATURES_FILE)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading venue features...")
def load_venue_features():
    if VENUE_FEATURES_FILE.exists():
        return pd.read_parquet(VENUE_FEATURES_FILE)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading matchup features...")
def load_matchup_features():
    if MATCHUP_FEATURES_FILE.exists():
        return pd.read_parquet(MATCHUP_FEATURES_FILE)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading metrics...")
def load_metrics():
    ball_level   = pd.read_parquet(METRICS_BALL_FILE)   if METRICS_BALL_FILE.exists()   else pd.DataFrame()
    player_level = pd.read_parquet(METRICS_PLAYER_FILE) if METRICS_PLAYER_FILE.exists() else pd.DataFrame()
    matchup      = pd.read_parquet(METRICS_MATCHUP_FILE) if METRICS_MATCHUP_FILE.exists() else pd.DataFrame()
    return ball_level, player_level, matchup


@st.cache_data(show_spinner="Loading RL policy...")
def load_rl_data():
    policy = pd.read_parquet(POLICY_TABLE_FILE)          if POLICY_TABLE_FILE.exists()    else pd.DataFrame()
    hist   = pd.read_parquet(HIST_WIN_RATES_FILE)        if HIST_WIN_RATES_FILE.exists()  else pd.DataFrame()
    return policy, hist


# ── Sidebar and navigation ────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {APP_ICON} IPL Intelligence")
        st.caption("T20 Decision Intelligence Platform")
        st.markdown("---")
        st.markdown(
            "**Data**: IPL 2008–2025  \n"
            "**Metrics**: WPA · ESA · PI · SDS · MLS · DSI  \n"
            "**Models**: EFS · CWP · BWP · PCR  \n"
            "**RL**: Fitted Q-Iteration"
        )
        st.markdown("---")
        st.caption("Use the tabs above to explore.")

def render_home():
    """Landing page with brief platform overview."""
    st.title(f"{APP_ICON} T20 Decision Intelligence Platform")
    st.markdown(
        """
        This is an **analyst-grade** exploration platform built on IPL 2008–2025 
        ball-by-ball data. Navigate via the tabs below.

        | # | Tab | What it answers |
        |---|-----|-----------------|
        | 1 | Executive Home | Key decision moments in any match |
        | 2 | Match State Engine | Ball-by-ball state viewer + EFS vs actual |
        | 3 | Team DNA | Phase profiles, chase patterns, collapse rates |
        | 4 | Player Value | WPA/ESA leaderboards, overrated/underrated quadrant |
        | 5 | Pressure Profiles | How players perform under pressure |
        | 6 | Matchup Intelligence | Batter vs bowler leverage table |
        | 7 | Decision Audit | Toss analysis, bowling changes, death allocation |
        | 8 | Scouting & Role Fit | Role clusters, replacement finder |
        | 9 | Strategy Lab | RL-based batting strategy recommendations |
        | 10 | Methodology | All metric and model definitions |
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Seasons covered", "2008–2025")
    with col2:
        st.metric("Deliveries analysed", "~650 K+")
    with col3:
        st.metric("Metrics computed", "12")


# ── Tab routing ───────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    tabs = st.tabs([
        "🏠 Home",
        "1 · Executive",
        "2 · Match State",
        "3 · Team DNA",
        "4 · Player Value",
        "5 · Pressure Profiles",
        "6 · Matchup Intel",
        "7 · Decision Audit",
        "8 · Scouting",
        "9 · Strategy Lab",
        "10 · Methodology",
    ])

    # Load data (cached)
    ball_states    = load_ball_states()
    player_feats   = load_player_features()
    team_feats     = load_team_features()
    venue_feats    = load_venue_features()
    matchup_feats  = load_matchup_features()
    metrics_ball, metrics_player, metrics_matchup = load_metrics()
    policy_df, hist_win_rates = load_rl_data()

    with tabs[0]:
        render_home()

    with tabs[1]:
        from app._pages.p01_executive_home import render
        render(ball_states, metrics_ball)

    with tabs[2]:
        from app._pages.p02_match_state_engine import render
        render(ball_states)

    with tabs[3]:
        from app._pages.p03_team_dna import render
        render(ball_states, team_feats)

    with tabs[4]:
        from app._pages.p04_player_value import render
        render(metrics_player)

    with tabs[5]:
        from app._pages.p05_pressure_profiles import render
        render(ball_states, player_feats)

    with tabs[6]:
        from app._pages.p06_matchup_intelligence import render
        render(metrics_matchup, matchup_feats)

    with tabs[7]:
        from app._pages.p07_decision_audit import render
        render(ball_states)

    with tabs[8]:
        from app._pages.p08_scouting_role_fit import render
        render(player_feats, metrics_player)

    with tabs[9]:
        from app._pages.p09_strategy_lab import render
        render(ball_states, policy_df, hist_win_rates)

    with tabs[10]:
        from app._pages.p10_methodology import render
        render()


if __name__ == "__main__":
    main()
