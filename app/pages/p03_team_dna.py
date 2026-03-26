"""
app/pages/p03_team_dna.py — Team DNA Tab

Shows:
  - Phase-level batting and bowling profile radar chart
  - Chase win rate by pressure band
  - Collapse frequency by phase and season
  - Historical season performance table
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, PHASE_COLORS, PRESSURE_COLORS


def render(ball_states: pd.DataFrame, team_feats: pd.DataFrame):
    st.header("Team DNA — Batting & Bowling Phase Profiles")

    if ball_states.empty:
        st.warning("No ball-state data available.")
        return

    # ── Team selector ─────────────────────────────────────────────────────────
    all_teams = sorted(ball_states["batting_team"].dropna().unique())
    col_t, col_s = st.columns([2, 1])
    with col_t:
        sel_teams = st.multiselect("Select teams (up to 4)", all_teams,
                                   default=all_teams[:2] if len(all_teams) >= 2 else all_teams,
                                   max_selections=4)
    with col_s:
        seasons = sorted(ball_states["season"].dropna().unique())
        sel_seasons = st.multiselect("Seasons", seasons, default=seasons[-5:])

    if not sel_teams:
        st.info("Select at least one team.")
        return

    df = ball_states[
        (ball_states["batting_team"].isin(sel_teams)) &
        (ball_states["season"].isin(sel_seasons if sel_seasons else seasons))
    ].copy()

    # ── Phase batting profiles ────────────────────────────────────────────────
    st.subheader("Batting Phase Profile")

    if "phase" in df.columns and "is_legal_ball" in df.columns:
        bat_phase = (
            df[df["is_legal_ball"] == 1]
            .groupby(["batting_team", "phase"])
            .agg(
                rpo     = ("batsman_runs", lambda x: x.sum() / len(x) * 6),
                dot_pct = ("batsman_runs", lambda x: (x == 0).mean() * 100),
                bdry_pct= ("batsman_runs", lambda x: ((x == 4) | (x == 6)).mean() * 100),
            )
            .reset_index()
        )

        metric = st.selectbox("Metric", ["rpo", "dot_pct", "bdry_pct"],
                              format_func={"rpo": "Run Rate per Over",
                                           "dot_pct": "Dot Ball %",
                                           "bdry_pct": "Boundary %"}.get,
                              key="dna_bat_metric")

        fig = px.bar(
            bat_phase, x="phase", y=metric, color="batting_team",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Batting {metric} by Phase",
            category_orders={"phase": ["powerplay", "middle", "death"]},
        )
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chase win rates ───────────────────────────────────────────────────────
    st.subheader("Chase Win Rate by Pressure Band")

    chase_df = df[(df["inning"] == 2) & ("pressure_band" in df.columns and "match_won" in df.columns)].copy() \
        if "pressure_band" in df.columns and "match_won" in df.columns else pd.DataFrame()

    if not chase_df.empty:
        chase_wr = (
            chase_df.groupby(["batting_team", "pressure_band"])
            .agg(
                matches=("match_id", "nunique"),
                win_rate=("match_won", "mean"),
            )
            .reset_index()
        )
        chase_wr["win_pct"] = (chase_wr["win_rate"] * 100).round(1)

        fig2 = px.bar(
            chase_wr, x="pressure_band", y="win_pct", color="batting_team",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Chase Win % by Pressure Band",
            category_orders={"pressure_band": ["low", "neutral", "high", "critical"]},
            text="win_pct",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Chase win rate data requires pipeline 02 (state reconstruction).")

    # ── Collapse frequency ────────────────────────────────────────────────────
    st.subheader("Collapse Frequency (3+ wickets in a phase)")

    if "phase" in df.columns and "is_wicket" in df.columns:
        wkt_phase = (
            df[df["is_wicket"] == 1]
            .groupby(["batting_team", "match_id", "phase"])
            .size()
            .reset_index(name="wickets_in_phase")
        )
        collapse = wkt_phase[wkt_phase["wickets_in_phase"] >= 3].copy()
        if not collapse.empty:
            coll_summary = (
                collapse.groupby(["batting_team", "phase"])
                .agg(collapse_count=("match_id", "count"))
                .reset_index()
            )
            fig3 = px.bar(
                coll_summary, x="batting_team", y="collapse_count", color="phase",
                color_discrete_map=PHASE_COLORS,
                title="3+ Wicket Collapses by Phase (absolute count)",
                barmode="stack",
            )
            apply_chart_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No collapses (3+ wickets in a phase) found for selected filters.")

    # ── Season performance table ──────────────────────────────────────────────
    st.subheader("Season-Level Batting Summary")

    if "season" in df.columns and "is_legal_ball" in df.columns:
        ssn = (
            df[df["is_legal_ball"] == 1]
            .groupby(["batting_team", "season"])
            .agg(
                runs   = ("batsman_runs", "sum"),
                balls  = ("is_legal_ball", "sum"),
                wickets= ("is_wicket", "sum") if "is_wicket" in df.columns
                         else ("batsman_runs", lambda _: "?"),
            )
            .reset_index()
        )
        ssn["RR"] = (ssn["runs"] / ssn["balls"] * 6).round(2)
        st.dataframe(ssn.rename(columns={
            "batting_team": "Team", "season": "Season",
            "runs": "Runs", "balls": "Balls", "wickets": "Wickets", "RR": "Run Rate"
        }), use_container_width=True, height=350)
