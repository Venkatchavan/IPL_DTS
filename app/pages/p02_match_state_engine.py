"""
app/pages/p02_match_state_engine.py — Match State Engine Tab

Ball-by-ball state viewer:
  - Over-by-over scorecard
  - Expected Final Score (EFS) vs Actual trajectory
  - Click-through on specific balls to see full state vector
  - Phase segmentation colors
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, PHASE_COLORS, EFS_LINE_COLOR


def render(ball_states: pd.DataFrame):
    st.header("Match State Engine — Ball-by-Ball Viewer")

    if ball_states.empty:
        st.warning("No ball-state data available.")
        return

    # ── Selectors ─────────────────────────────────────────────────────────────
    col_s, col_m, col_i = st.columns([1, 3, 1])
    with col_s:
        seasons = sorted(ball_states["season"].dropna().unique(), reverse=True)
        sel_season = st.selectbox("Season", seasons, key="mse_season")
    with col_i:
        sel_inning = st.selectbox("Inning", [1, 2], key="mse_inning")

    season_df = ball_states[ball_states["season"] == sel_season]
    matches   = (
        season_df.groupby("match_id")
        .agg(teams=("batting_team", lambda x: " vs ".join(x.unique()[:2])), date=("date", "first"))
        .reset_index()
    )
    match_labels = {r["match_id"]: f"{r['date']} | {r['teams']}" for _, r in matches.iterrows()}

    with col_m:
        sel_match = st.selectbox("Match", list(match_labels.keys()),
                                 format_func=lambda x: match_labels.get(x, str(x)),
                                 key="mse_match")

    idf = ball_states[
        (ball_states["match_id"] == sel_match) &
        (ball_states["inning"] == sel_inning)
    ].sort_values("global_ball_number").reset_index(drop=True)

    if idf.empty:
        st.info("No data for this match / innings combination.")
        return

    # ── Over scorecard ────────────────────────────────────────────────────────
    st.subheader("Over-by-Over Scorecard")

    over_sc = idf.groupby("over").agg(
        runs   = ("batsman_runs", lambda x: x.sum() + idf.loc[x.index, "extra_runs"].sum()),
        wickets= ("is_wicket", "sum") if "is_wicket" in idf.columns else ("batsman_runs", lambda _: 0),
        balls  = ("is_legal_ball", "sum") if "is_legal_ball" in idf.columns else ("batsman_runs", "count"),
        phase  = ("phase", "first"),
    ).reset_index()
    over_sc.columns = ["Over", "Runs", "Wickets", "Legal Balls", "Phase"]
    over_sc["Color"] = over_sc["Phase"].map(PHASE_COLORS).fillna("#94A3B8")

    fig_bar = go.Figure(go.Bar(
        x=over_sc["Over"], y=over_sc["Runs"],
        marker_color=over_sc["Color"],
        customdata=over_sc[["Wickets", "Phase"]],
        hovertemplate="Over %{x}<br>Runs: %{y}<br>Wickets: %{customdata[0]}<br>Phase: %{customdata[1]}<extra></extra>",
    ))
    fig_bar.update_layout(
        title=f"Innings {sel_inning} — Runs per Over",
        xaxis_title="Over",
        yaxis_title="Runs",
        showlegend=False,
    )
    apply_chart_theme(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Wicket markers
    if "is_wicket" in idf.columns:
        wkt_overs = idf[idf["is_wicket"] == 1]["over"].value_counts().reset_index()
        wkt_overs.columns = ["Over", "Wickets"]
        if not wkt_overs.empty:
            st.caption(f"Wickets fell in overs: {sorted(wkt_overs['Over'].tolist())}")

    # ── EFS vs Actual trajectory ──────────────────────────────────────────────
    st.subheader("Expected Final Score vs Actual Cumulative Score")

    efs_col    = "pre_expected_score" if "pre_expected_score" in idf.columns else None
    legal      = idf[idf.get("is_legal_ball", pd.Series(1, index=idf.index)) == 1].copy()
    legal["cum_runs"] = (legal["batsman_runs"] + legal["extra_runs"]).cumsum()
    legal["legal_ball_n"] = range(1, len(legal) + 1)

    fig_efs = go.Figure()
    fig_efs.add_trace(go.Scatter(
        x=legal["legal_ball_n"],
        y=legal["cum_runs"],
        mode="lines",
        name="Actual Score",
        line=dict(color="#22C55E", width=2),
    ))
    if efs_col and legal[efs_col].notna().any():
        fig_efs.add_trace(go.Scatter(
            x=legal["legal_ball_n"],
            y=legal[efs_col],
            mode="lines",
            name="EFS (Pre-delivery)",
            line=dict(color=EFS_LINE_COLOR, width=2, dash="dot"),
        ))
    # Final actual score line
    final_score = int(legal["cum_runs"].max()) if not legal.empty else 0
    fig_efs.add_hline(y=final_score, line_dash="dash", line_color="#94A3B8",
                      annotation_text=f"Final: {final_score}")
    fig_efs.update_layout(
        title="EFS vs Actual (legal deliveries)",
        xaxis_title="Legal Ball #",
        yaxis_title="Runs",
    )
    apply_chart_theme(fig_efs)
    st.plotly_chart(fig_efs, use_container_width=True)

    # ── State vector inspector ────────────────────────────────────────────────
    st.subheader("Ball-Level State Inspector")

    cols_show = [
        c for c in [
            "over", "ball", "striker", "non_striker", "bowler",
            "batsman_runs", "extra_runs", "is_wicket",
            "phase", "pressure_band",
            "pre_runs", "pre_wickets", "pre_balls_remaining",
            "pre_crr", "pre_rrr", "pre_rrr_crr_delta",
            "pre_dot_ball_streak", "pre_last_n_balls_rr",
            "pre_win_prob", "pre_wicket_prob",
        ] if c in idf.columns
    ]
    st.dataframe(
        idf[cols_show].rename(columns={
            "pre_runs": "Runs(pre)", "pre_wickets": "Wkts(pre)",
            "pre_balls_remaining": "Balls Left",
            "pre_crr": "CRR", "pre_rrr": "RRR", "pre_rrr_crr_delta": "RRR-CRR",
            "pre_dot_ball_streak": "Dot Streak",
            "pre_last_n_balls_rr": "Last12 RR",
            "pre_win_prob": "WP", "pre_wicket_prob": "WktP",
        }),
        use_container_width=True,
        height=400,
    )
