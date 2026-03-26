"""
app/pages/p09_strategy_lab.py — Strategy Lab Tab

RL-based batting strategy recommendations.
Uses FQI-trained policy to recommend: conservative / balanced / aggressive

CRITICAL HONEST FRAMING (displayed prominently in dashboard):
  This tool does NOT guarantee optimal strategy.
  It reflects historically associated action quality based on IPL 2008-2025 data.
  Treat recommendations as pattern-based priors, not prescriptive decisions.
  Always filter by confidence level — low-confidence states have unreliable estimates.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, ACTION_COLORS, CONFIDENCE_COLORS, PHASE_COLORS


def _build_state_dict(
    runs_needed: int,
    balls_remaining: int,
    wickets_in_hand: int,
    phase: str,
    rrr_crr_delta: float,
    venue_tier: str,
    dot_streak: int,
) -> dict:
    return {
        "runs_needed":        runs_needed,
        "balls_remaining":    balls_remaining,
        "wickets_in_hand":    wickets_in_hand,
        "phase":              phase,
        "rrr_crr_delta":      rrr_crr_delta,
        "venue_scoring_tier": venue_tier,
        "dot_ball_streak":    dot_streak,
    }


def render(ball_states: pd.DataFrame, policy_df: pd.DataFrame, hist_win_rates: pd.DataFrame):
    st.header("Strategy Lab — RL-Based Batting Strategy Recommender")

    # ── HONEST FRAMING BANNER ─────────────────────────────────────────────────
    st.warning(
        "**Analytical Disclaimer:** This strategy lab uses Fitted Q-Iteration (FQI) on historical IPL data. "
        "Recommendations reflect *historically associated action quality*, not causally proven optimal strategy. "
        "Low-confidence states (fewer than 20 historical observations) should be ignored. "
        "This is a portfolio demonstration of offline RL methodology — not a prescriptive tool."
    )

    if policy_df.empty:
        st.info("RL policy not trained. Run `models/rl_strategy/fqi_trainer.py` first.")

        # Still show the state sliders and historical lookup
        st.subheader("Manual State Explorer (Historical Win Rates)")
        if not ball_states.empty and "match_won" in ball_states.columns and "pressure_band" in ball_states.columns:
            ph = st.selectbox("Phase", ["powerplay", "middle", "death"], key="sl_phase_hist")
            pb = st.selectbox("Pressure Band", ["low", "neutral", "high", "critical"], key="sl_pb_hist")
            inn2 = ball_states[
                (ball_states["inning"] == 2) &
                (ball_states.get("phase", pd.Series("")) == ph) &
                (ball_states.get("pressure_band", pd.Series("")) == pb)
            ]
            if not inn2.empty:
                wr = inn2.groupby("match_id")["match_won"].first().mean()
                st.metric("Historical Win Rate in this context", f"{wr*100:.1f}%")
        return

    # ── State input sliders ───────────────────────────────────────────────────
    st.subheader("Define Match State")
    col1, col2, col3 = st.columns(3)
    with col1:
        runs_needed     = st.slider("Runs Needed", 0, 250, 60, key="sl_runs")
        balls_remaining = st.slider("Balls Remaining", 1, 120, 36, key="sl_balls")
    with col2:
        wickets_in_hand = st.slider("Wickets in Hand", 1, 10, 6, key="sl_wkts")
        dot_streak      = st.slider("Current Dot Ball Streak", 0, 12, 2, key="sl_dots")
    with col3:
        phase      = st.selectbox("Phase", ["powerplay", "middle", "death"], index=1, key="sl_phase")
        venue_tier = st.selectbox("Venue Scoring Tier", ["low", "medium", "high"], index=1, key="sl_venue")

    # Compute RRR-CRR automatically
    rrr = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
    balls_played = 120 - balls_remaining
    crr = (((120 - balls_remaining - runs_needed + runs_needed) / max(balls_played, 1)) * 6
           if balls_played > 0 else 0)
    # simpler: use slider
    rrr_crr_delta = st.slider("RRR − CRR (positive = batting team behind)", -5.0, 10.0, 0.0, 0.5, key="sl_rrr_delta")

    st.caption(f"Required Run Rate: **{rrr:.2f}** per over")

    # ── Policy lookup ─────────────────────────────────────────────────────────
    st.subheader("Recommended Action")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from models.rl_strategy.state_encoder import encode

        state = _build_state_dict(runs_needed, balls_remaining, wickets_in_hand,
                                   phase, rrr_crr_delta, venue_tier, dot_streak)
        state_key = str(encode(state))

        match = policy_df[policy_df["state_key"] == state_key]

        if not match.empty:
            rec_action  = match.iloc[0]["recommended_action"]
            confidence  = match.iloc[0]["confidence"]
            q_cons      = match.iloc[0].get("q_conservative", None)
            q_bal       = match.iloc[0].get("q_balanced", None)
            q_agg       = match.iloc[0].get("q_aggressive", None)
            support     = match.iloc[0].get("support_count", 0)

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Recommended Action", rec_action.upper())
            with col_r2:
                conf_color = CONFIDENCE_COLORS.get(confidence, "#94A3B8")
                st.metric("Confidence", confidence.upper())
                if confidence == "low":
                    st.error("Low confidence — insufficient historical data for this state. Ignore recommendation.")
            with col_r3:
                st.metric("Historical Observations", int(support))

            # Q-value comparison
            if all(v is not None for v in [q_cons, q_bal, q_agg]):
                q_df = pd.DataFrame({
                    "Action": ["Conservative", "Balanced", "Aggressive"],
                    "Q-Value (Expected Return)": [round(q_cons, 3), round(q_bal, 3), round(q_agg, 3)],
                })
                fig_q = px.bar(
                    q_df, x="Action", y="Q-Value (Expected Return)",
                    color="Action", color_discrete_map={
                        "Conservative": ACTION_COLORS["conservative"],
                        "Balanced":     ACTION_COLORS["balanced"],
                        "Aggressive":   ACTION_COLORS["aggressive"],
                    },
                    title="Q-Values by Action for This State",
                )
                apply_chart_theme(fig_q)
                st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info(
                "This exact state was not observed in training data. "
                "No recommendation available. Try adjusting the sliders to a nearby state."
            )

    except Exception as e:
        st.error(f"Policy lookup error: {e}")

    # ── Historical win rates for this context ─────────────────────────────────
    if not hist_win_rates.empty:
        st.subheader("Historical Win Rates by Action (Empirical)")
        st.caption(
            "This shows observed win rates in similar historical states, **BEFORE** RL policy correction. "
            "These are correlational, not causal."
        )

        try:
            match_hist = hist_win_rates[hist_win_rates["state_key"] == state_key] \
                if "state_key" in hist_win_rates.columns else pd.DataFrame()

            if not match_hist.empty:
                fig_hist = px.bar(
                    match_hist, x="action", y="win_rate",
                    color="action",
                    color_discrete_map=ACTION_COLORS,
                    text="n_instances",
                    title="Observed Win Rate by Action (this state)",
                    labels={"win_rate": "Win Rate", "action": "Action"},
                )
                fig_hist.update_traces(texttemplate="n=%{text}", textposition="outside")
                fig_hist.update_yaxes(range=[0, 1])
                apply_chart_theme(fig_hist)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No historical win rate data for this exact state.")
        except Exception:
            pass

    # ── Policy action distribution ────────────────────────────────────────────
    st.subheader("Policy Action Distribution Across All States")

    if "recommended_action" in policy_df.columns:
        act_dist = policy_df.groupby(["recommended_action", "confidence"]).size().reset_index(name="count")
        fig_dist = px.bar(
            act_dist, x="recommended_action", y="count", color="confidence",
            color_discrete_map=CONFIDENCE_COLORS,
            title="How often each action is recommended (by confidence level)",
            barmode="stack",
            category_orders={"confidence": ["high", "medium", "low"]},
        )
        apply_chart_theme(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
