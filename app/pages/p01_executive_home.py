"""
app/pages/p01_executive_home.py — Executive Home Tab

Shows:
  - Match selector (season + match dropdown)
  - Win Probability trajectory with key events annotated
  - Pressure heatmap over the innings
  - Top 5 high-leverage delivery moments
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import (
    apply_chart_theme, PHASE_COLORS, PRESSURE_COLORS,
    WP_LINE_COLOR, HIGH_LEVERAGE_WPA,
)


def render(ball_states: pd.DataFrame, metrics_ball: pd.DataFrame):
    st.header("Executive Home — Match Decision Analysis")

    if ball_states.empty:
        st.warning("No ball-state data available. Run pipelines first.")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    col_s, col_m = st.columns([1, 3])
    with col_s:
        seasons = sorted(ball_states["season"].dropna().unique(), reverse=True)
        sel_season = st.selectbox("Season", seasons, index=0)

    season_df = ball_states[ball_states["season"] == sel_season]
    matches = (
        season_df.groupby("match_id")
        .agg(home=("batting_team", "first"), away=("bowling_team", "first"))
        .reset_index()
    )
    match_labels = {
        row["match_id"]: f"Match {row['match_id']} | {row['home']} vs {row['away']}"
        for _, row in matches.iterrows()
    }

    with col_m:
        sel_match = st.selectbox(
            "Match",
            options=list(match_labels.keys()),
            format_func=lambda x: match_labels.get(x, str(x)),
        )

    match_df = season_df[season_df["match_id"] == sel_match].sort_values("global_ball_number")

    if match_df.empty:
        st.info("No data for this match.")
        return

    # ── Summary row ──────────────────────────────────────────────────────────
    teams = match_df["batting_team"].unique()
    inn1  = match_df[match_df["inning"] == 1]
    inn2  = match_df[match_df["inning"] == 2]
    score1 = inn1["batsman_runs"].sum() + inn1["extra_runs"].sum() if not inn1.empty else 0
    score2 = inn2["batsman_runs"].sum() + inn2["extra_runs"].sum() if not inn2.empty else 0
    wkts1  = inn1["is_wicket"].sum() if "is_wicket" in inn1.columns else "?"
    wkts2  = inn2["is_wicket"].sum() if "is_wicket" in inn2.columns else "?"

    c1, c2, c3 = st.columns(3)
    with c1:
        t = inn1["batting_team"].iloc[0] if not inn1.empty else "Team 1"
        st.metric(f"{t} (Innings 1)", f"{score1}/{wkts1}")
    with c2:
        t = inn2["batting_team"].iloc[0] if not inn2.empty else "Team 2"
        st.metric(f"{t} (Innings 2)", f"{score2}/{wkts2}")
    with c3:
        if "match_won" in match_df.columns:
            winners = match_df.groupby("batting_team")["match_won"].max()
            winner  = winners[winners == 1].index.tolist()
            st.metric("Result", winner[0] if winner else "TBD")

    # ── Win Probability Trajectory ────────────────────────────────────────────
    st.subheader("Win Probability Trajectory")

    wp_col = "pre_win_prob" if "pre_win_prob" in match_df.columns else None

    if wp_col and match_df[wp_col].notna().any():
        fig = go.Figure()
        for inning_n, color in [(1, WP_LINE_COLOR), (2, "#F97316")]:
            idf = match_df[match_df["inning"] == inning_n]
            if idf.empty or idf[wp_col].isna().all():
                continue
            fig.add_trace(go.Scatter(
                x=idf["global_ball_number"],
                y=idf[wp_col] * 100,
                mode="lines",
                name=f"Inning {inning_n} WP",
                line=dict(color=color, width=2),
            ))
            # Annotate wickets
            wkt_rows = idf[idf.get("is_wicket", pd.Series(0, index=idf.index)) == 1] \
                if "is_wicket" in idf.columns else pd.DataFrame()
            if not wkt_rows.empty:
                fig.add_trace(go.Scatter(
                    x=wkt_rows["global_ball_number"],
                    y=wkt_rows[wp_col] * 100,
                    mode="markers",
                    marker=dict(color="#EF4444", size=8, symbol="x"),
                    name=f"Wicket (Inn{inning_n})",
                    showlegend=True,
                ))
        fig.add_hline(y=50, line_dash="dash", line_color="#94A3B8", annotation_text="50%")
        fig.update_layout(
            title="Win Probability (Inning 2 = batting team perspective)",
            xaxis_title="Global Ball #",
            yaxis_title="Win Probability (%)",
            yaxis=dict(range=[0, 100]),
            **{k: v for k, v in vars(apply_chart_theme(go.Figure())).items() if k == "layout"},
        )
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Win probability not available (run pipeline 05 to generate model scores).")
        # Fallback: run rate chart
        if "pre_crr" in match_df.columns:
            fig = px.line(match_df, x="global_ball_number", y="pre_crr",
                          color="inning", title="Current Run Rate by Ball")
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Pressure Heatmap ──────────────────────────────────────────────────────
    st.subheader("Pressure Index Heatmap")

    pi_src = metrics_ball if not metrics_ball.empty and "pressure_index" in metrics_ball.columns else match_df
    pi_col = "pressure_index" if "pressure_index" in pi_src.columns else None

    if pi_col:
        hmap_df = pi_src[pi_src["match_id"] == sel_match].copy() if "match_id" in pi_src.columns else match_df.copy()
        hmap_df = hmap_df.sort_values("global_ball_number")
        hmap_df["over_n"] = (hmap_df["over"].astype(float)).astype(int)
        pivot = hmap_df.pivot_table(index="inning", columns="over_n", values=pi_col, aggfunc="mean")
        phase_lines = [6, 15]

        fig2 = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="Mean Pressure Index per Over (darker = higher pressure)",
            labels=dict(x="Over", y="Inning", color="Pressure Index"),
            zmin=0, zmax=100,
        )
        apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Pressure index not available (run pipeline 04).")

    # ── High-leverage moments ─────────────────────────────────────────────────
    st.subheader("Top High-Leverage Deliveries")

    wpa_col = "wpa" if "wpa" in pi_src.columns else None
    if wpa_col:
        lev_df = pi_src[pi_src["match_id"] == sel_match].copy() if "match_id" in pi_src.columns else match_df.copy()
        lev_df["abs_wpa"] = lev_df[wpa_col].abs()
        top5 = lev_df.nlargest(5, "abs_wpa")[
            ["over", "ball_in_over", "striker", "bowler", "batsman_runs",
             "is_wicket", wpa_col, "pressure_band"]
        ].rename(columns={"over": "Over", "ball_in_over": "Ball", "striker": "Batter",
                           "bowler": "Bowler", "batsman_runs": "Runs",
                           "is_wicket": "Wicket", wpa_col: "WPA",
                           "pressure_band": "Pressure"})
        st.dataframe(top5.style.background_gradient(subset=["WPA"], cmap="RdYlGn"), use_container_width=True)
    else:
        st.info("WPA not available (run pipeline 04 phase B).")
