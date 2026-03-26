"""
app/pages/p04_player_value.py — Player Value Tab

Shows:
  - WPA leaderboard for batters and bowlers
  - ESA leaderboard (innings 1 primary)
  - Overrated/Underrated quadrant (WPA vs Strike Rate for batters)
  - Season filtering, phase filtering
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, MIN_DISPLAY_BALLS_BATTER, MIN_DISPLAY_BALLS_BOWLER


def render(metrics_player: pd.DataFrame):
    st.header("Player Value — WPA & ESA Leaderboards")

    if metrics_player.empty:
        st.warning("Player metrics not available. Run pipeline 04 (metrics compute) first.")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        if "season" in metrics_player.columns:
            seasons = sorted(metrics_player["season"].dropna().unique(), reverse=True)
            sel_seasons = st.multiselect("Seasons", seasons, default=seasons[:5],
                                         key="pv_seasons")
        else:
            sel_seasons = []
    with col_f2:
        if "phase" in metrics_player.columns:
            phases = ["All"] + sorted(metrics_player["phase"].dropna().unique().tolist())
            sel_phase = st.selectbox("Phase", phases, key="pv_phase")
        else:
            sel_phase = "All"
    with col_f3:
        topN = st.slider("Top N players", 10, 50, 20, key="pv_topn")

    filt = metrics_player.copy()
    if sel_seasons:
        filt = filt[filt["season"].isin(sel_seasons)] if "season" in filt.columns else filt
    if sel_phase != "All" and "phase" in filt.columns:
        filt = filt[filt["phase"] == sel_phase]

    # ── Batter WPA Leaderboard ────────────────────────────────────────────────
    st.subheader("Batter WPA Leaderboard")

    wpa_batter_col = next((c for c in ["wpa_total", "wpa"] if c in filt.columns), None)
    batter_col     = next((c for c in ["striker", "batter", "player"] if c in filt.columns), None)
    balls_col      = next((c for c in ["balls_faced", "balls"] if c in filt.columns), None)

    if wpa_batter_col and batter_col:
        batter_df = filt.copy()
        if balls_col:
            batter_df = batter_df[batter_df[balls_col] >= MIN_DISPLAY_BALLS_BATTER]
        top_batters = (
            batter_df.groupby(batter_col)[wpa_batter_col].sum()
            .reset_index()
            .nlargest(topN, wpa_batter_col)
        )
        top_batters[wpa_batter_col] = top_batters[wpa_batter_col].round(3)

        fig1 = px.bar(
            top_batters, x=wpa_batter_col, y=batter_col, orientation="h",
            color=wpa_batter_col, color_continuous_scale="RdYlGn",
            title=f"Top {topN} Batters by WPA",
            labels={batter_col: "Batter", wpa_batter_col: "Win Probability Added"},
        )
        fig1.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        apply_chart_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("WPA batter data not found in metrics. Check pipeline 04 output.")

    # ── Bowler WPA Leaderboard ────────────────────────────────────────────────
    st.subheader("Bowler WPA Leaderboard")

    wpa_bowler_col = next((c for c in ["wpa_bowler_total", "wpa_bowler"] if c in filt.columns), None)
    bowler_col     = next((c for c in ["bowler"] if c in filt.columns), None)
    balls_bowled   = next((c for c in ["balls_bowled", "balls"] if c in filt.columns), None)

    if wpa_bowler_col and bowler_col:
        bowler_df = filt.copy()
        if balls_bowled:
            bowler_df = bowler_df[bowler_df[balls_bowled] >= MIN_DISPLAY_BALLS_BOWLER]
        top_bowlers = (
            bowler_df.groupby(bowler_col)[wpa_bowler_col].sum()
            .reset_index()
            .nlargest(topN, wpa_bowler_col)
        )
        top_bowlers[wpa_bowler_col] = top_bowlers[wpa_bowler_col].round(3)

        fig2 = px.bar(
            top_bowlers, x=wpa_bowler_col, y=bowler_col, orientation="h",
            color=wpa_bowler_col, color_continuous_scale="RdYlGn",
            title=f"Top {topN} Bowlers by WPA Created",
            labels={bowler_col: "Bowler", wpa_bowler_col: "WPA (bowler)"},
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("WPA bowler data not found. Check pipeline 04 output.")

    # ── Overrated / Underrated Quadrant ───────────────────────────────────────
    st.subheader("Overrated / Underrated Quadrant (Batters)")
    st.caption(
        "X-axis: Traditional Strike Rate | Y-axis: WPA per 100 balls  \n"
        "**Top-right**: Genuinely impactful hitters  \n"
        "**Top-left**: High WPA despite moderate SR — underrated  \n"
        "**Bottom-right**: High SR, low WPA — overrated (inflate vs weak states)"
    )

    sr_col  = next((c for c in ["strike_rate", "sr"] if c in filt.columns), None)
    wpa_100 = next((c for c in ["wpa_per_100", "wpa_total"] if c in filt.columns), None)

    if batter_col and sr_col and wpa_100 and balls_col:
        quad_df = filt.groupby(batter_col).agg(
            **{sr_col: (sr_col, "mean"),
               wpa_100: (wpa_100, "sum"),
               balls_col: (balls_col, "sum")}
        ).reset_index()
        quad_df = quad_df[quad_df[balls_col] >= MIN_DISPLAY_BALLS_BATTER]

        fig3 = px.scatter(
            quad_df, x=sr_col, y=wpa_100, text=batter_col,
            size=balls_col, color=wpa_100,
            color_continuous_scale="RdYlGn",
            title="Batter Value Quadrant",
            labels={sr_col: "Strike Rate", wpa_100: "WPA"},
        )
        fig3.add_vline(x=quad_df[sr_col].median(), line_dash="dash", line_color="#94A3B8")
        fig3.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig3.update_traces(textposition="top center", textfont_size=9)
        fig3.update_layout(coloraxis_showscale=False)
        apply_chart_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Strike rate and WPA columns needed for quadrant. Run pipeline 03 + 04.")
