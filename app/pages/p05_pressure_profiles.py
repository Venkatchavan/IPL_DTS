"""
app/pages/p05_pressure_profiles.py — Pressure Profiles Tab

Shows:
  - How individual batters perform across different pressure bands
  - How bowlers perform in high-pressure situations
  - Pressure Index distribution histograms
  - Phase x pressure band heatmap
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, PRESSURE_COLORS, MIN_DISPLAY_BALLS_BATTER, MIN_DISPLAY_BALLS_BOWLER


def render(ball_states: pd.DataFrame, player_feats: pd.DataFrame):
    st.header("Pressure Profiles — Performance Under Stress")

    if ball_states.empty:
        st.warning("No ball-state data available.")
        return

    has_pressure = "pressure_band" in ball_states.columns
    has_pi       = "pressure_index" in ball_states.columns

    if not has_pressure:
        st.info("Pressure band not available — run pipeline 02 (state reconstruction).")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        seasons = sorted(ball_states["season"].dropna().unique(), reverse=True)
        sel_seasons = st.multiselect("Seasons", seasons, default=seasons[:5], key="pp_seasons")
    with col_b:
        sel_inning = st.selectbox("Innings", ["Both", 1, 2], key="pp_inning")
    with col_c:
        top_n = st.slider("Top N players", 10, 30, 15, key="pp_topn")

    df = ball_states.copy()
    if sel_seasons:
        df = df[df["season"].isin(sel_seasons)]
    if sel_inning != "Both":
        df = df[df["inning"] == sel_inning]

    legal = df[df.get("is_legal_ball", pd.Series(1, index=df.index)) == 1].copy() \
        if "is_legal_ball" in df.columns else df.copy()

    # ── Batter performance by pressure band ──────────────────────────────────
    st.subheader("Batter Strike Rate by Pressure Band")

    batter_col = next((c for c in ["striker", "batter"] if c in legal.columns), None)
    if batter_col:
        bat_pb = (
            legal.groupby([batter_col, "pressure_band"])
            .agg(
                balls   = ("batsman_runs", "count"),
                runs    = ("batsman_runs", "sum"),
                dot_pct = ("batsman_runs", lambda x: (x == 0).mean() * 100),
            )
            .reset_index()
        )
        bat_pb["sr"] = (bat_pb["runs"] / bat_pb["balls"] * 100).round(1)
        bat_pb = bat_pb[bat_pb["balls"] >= MIN_DISPLAY_BALLS_BATTER // 4]  # relaxed for pressure band

        # Filter to top N batters by total runs
        top_batters = (
            legal.groupby(batter_col)["batsman_runs"].sum()
            .nlargest(top_n).index.tolist()
        )
        bat_pb_top = bat_pb[bat_pb[batter_col].isin(top_batters)]

        metric_choice = st.selectbox(
            "Metric", ["sr", "dot_pct"],
            format_func={"sr": "Strike Rate", "dot_pct": "Dot Ball %"}.get,
            key="pp_bat_metric"
        )

        if not bat_pb_top.empty:
            fig1 = px.bar(
                bat_pb_top, x=batter_col, y=metric_choice, color="pressure_band",
                barmode="group",
                color_discrete_map=PRESSURE_COLORS,
                category_orders={"pressure_band": ["low", "neutral", "high", "critical"]},
                title=f"Batter {metric_choice} by Pressure Band (top {top_n})",
            )
            apply_chart_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    # ── Bowler economy by pressure band ──────────────────────────────────────
    st.subheader("Bowler Economy by Pressure Band")

    bowler_col = "bowler" if "bowler" in legal.columns else None
    if bowler_col:
        bowl_pb = (
            legal.groupby([bowler_col, "pressure_band"])
            .agg(
                balls   = ("batsman_runs", "count"),
                runs    = ("batsman_runs", lambda x: x.sum() + legal.loc[x.index, "extra_runs"].sum()
                           if "extra_runs" in legal.columns else x.sum()),
            )
            .reset_index()
        )
        bowl_pb["economy"] = (bowl_pb["runs"] / bowl_pb["balls"] * 6).round(2)
        bowl_pb = bowl_pb[bowl_pb["balls"] >= MIN_DISPLAY_BALLS_BOWLER // 4]

        top_bowlers = (
            legal.groupby(bowler_col)["batsman_runs"].count()
            .nlargest(top_n).index.tolist()
        )
        bowl_pb_top = bowl_pb[bowl_pb[bowler_col].isin(top_bowlers)]

        if not bowl_pb_top.empty:
            fig2 = px.bar(
                bowl_pb_top, x=bowler_col, y="economy", color="pressure_band",
                barmode="group",
                color_discrete_map=PRESSURE_COLORS,
                category_orders={"pressure_band": ["low", "neutral", "high", "critical"]},
                title=f"Bowler Economy by Pressure Band (top {top_n})",
            )
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Pressure Index distribution ───────────────────────────────────────────
    if has_pi:
        st.subheader("Pressure Index Distribution (by Phase)")
        fig3 = px.histogram(
            legal, x="pressure_index", color="phase",
            color_discrete_map = {"powerplay": "#3B82F6", "middle": "#F59E0B", "death": "#EF4444"},
            nbins=50, barmode="overlay",
            opacity=0.7,
            title="Distribution of Pressure Index Values",
            labels={"pressure_index": "Pressure Index (0–100)"},
        )
        apply_chart_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Phase × Pressure heatmap ──────────────────────────────────────────────
    if "phase" in legal.columns:
        st.subheader("Phase × Pressure Band — Run Rate Heatmap")

        heat = (
            legal.groupby(["phase", "pressure_band"])["batsman_runs"]
            .mean()
            .mul(6).reset_index()
            .rename(columns={"batsman_runs": "run_rate_per_over"})
        )
        pivot = heat.pivot_table(
            index="phase", columns="pressure_band",
            values="run_rate_per_over"
        ).reindex(
            index=["powerplay", "middle", "death"],
            columns=["low", "neutral", "high", "critical"],
        )
        fig4 = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            text_auto=".2f",
            title="Average Run Rate per Over by Phase × Pressure Band",
            labels=dict(color="RPO"),
        )
        apply_chart_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)
