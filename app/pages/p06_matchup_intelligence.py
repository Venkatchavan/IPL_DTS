"""
app/pages/p06_matchup_intelligence.py — Matchup Intelligence Tab

Shows:
  - Matchup Leverage Score (MLS) table: batter vs bowler style bucket
  - Matchup heatmap (batter SR vs bowler)
  - Exploitable matchup recommender
  - Head-to-head historical stats
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, MIN_DISPLAY_BALLS_MATCHUP


def render(metrics_matchup: pd.DataFrame, matchup_feats: pd.DataFrame):
    st.header("Matchup Intelligence — Batter vs Bowler")

    src = metrics_matchup if not metrics_matchup.empty else matchup_feats

    if src.empty:
        st.warning("Matchup data not available. Run pipelines 03–04 first.")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    batter_col = next((c for c in ["striker", "batter"] if c in src.columns), None)
    bowler_col = "bowler" if "bowler" in src.columns else None

    col_b, col_w, col_n = st.columns(3)
    with col_b:
        if batter_col:
            batters = sorted(src[batter_col].dropna().unique())
            sel_batter = st.selectbox("Batter filter (optional)", ["All"] + batters, key="mi_batter")
        else:
            sel_batter = "All"
    with col_w:
        if bowler_col:
            bowlers = sorted(src[bowler_col].dropna().unique())
            sel_bowler = st.selectbox("Bowler filter (optional)", ["All"] + bowlers, key="mi_bowler")
        else:
            sel_bowler = "All"
    with col_n:
        min_balls = st.slider("Minimum balls in matchup", 10, 100, MIN_DISPLAY_BALLS_MATCHUP, key="mi_min")

    filt = src.copy()
    balls_col = next((c for c in ["balls_faced", "balls", "n_balls"] if c in filt.columns), None)
    if balls_col:
        filt = filt[filt[balls_col] >= min_balls]
    if sel_batter != "All" and batter_col:
        filt = filt[filt[batter_col] == sel_batter]
    if sel_bowler != "All" and bowler_col:
        filt = filt[filt[bowler_col] == sel_bowler]

    # ── MLS table ─────────────────────────────────────────────────────────────
    st.subheader("Matchup Leverage Score (MLS) Table")

    mls_col = next((c for c in ["mls", "matchup_leverage_score"] if c in filt.columns), None)
    if mls_col and batter_col and bowler_col:
        display_cols = [
            c for c in [batter_col, bowler_col, balls_col, "strike_rate", "sr",
                        "dot_pct", "boundary_pct", "dismissal_rate", mls_col, "confidence"]
            if c in filt.columns
        ]
        mls_table = filt[display_cols].sort_values(mls_col, key=abs, ascending=False).head(50)
        mls_table[mls_col] = mls_table[mls_col].round(3) if mls_col in mls_table else mls_table[mls_col]

        st.dataframe(
            mls_table.style.background_gradient(subset=[mls_col], cmap="RdYlGn"),
            use_container_width=True,
            height=400,
        )

        # Distribution of MLS values
        fig1 = px.histogram(
            filt.dropna(subset=[mls_col]), x=mls_col, nbins=40,
            color_discrete_sequence=["#3B82F6"],
            title="MLS Distribution (positive = batter advantage, negative = bowler advantage)",
            labels={mls_col: "Matchup Leverage Score"},
        )
        fig1.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
        apply_chart_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("MLS column not found. Run pipeline 04 to compute matchup metrics.")

    # ── Batter SR heatmap vs bowlers ──────────────────────────────────────────
    st.subheader("Strike Rate Heatmap (top batters × top bowlers)")

    sr_col = next((c for c in ["strike_rate", "sr"] if c in filt.columns), None)
    if batter_col and bowler_col and sr_col:
        top_batters = filt.groupby(batter_col)[balls_col].sum().nlargest(15).index.tolist() if balls_col else filt[batter_col].unique()[:15]
        top_bowlers = filt.groupby(bowler_col)[balls_col].sum().nlargest(15).index.tolist() if balls_col else filt[bowler_col].unique()[:15]

        heat_df = filt[
            filt[batter_col].isin(top_batters) &
            filt[bowler_col].isin(top_bowlers)
        ]
        pivot = heat_df.pivot_table(index=batter_col, columns=bowler_col, values=sr_col, aggfunc="mean")

        if not pivot.empty:
            fig2 = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                title="Strike Rate Heatmap (batter rows, bowler cols) — greener = batter advantage",
                labels=dict(color="Strike Rate"),
            )
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Exploitable matchups recommender ─────────────────────────────────────
    st.subheader("Exploitable Matchups Recommender")

    if mls_col and batter_col and bowler_col:
        threshold = st.slider("MLS magnitude threshold", 0.5, 3.0, 1.0, 0.1, key="mi_thresh")
        col_a2, col_b2 = st.columns(2)

        with col_a2:
            st.markdown("**Batter-favoured matchups** (positive MLS)")
            batter_fav = (
                filt[filt[mls_col] >= threshold]
                .sort_values(mls_col, ascending=False)
                [([batter_col, bowler_col, mls_col] + ([balls_col] if balls_col else []))]
                .head(15)
            )
            st.dataframe(batter_fav, use_container_width=True)

        with col_b2:
            st.markdown("**Bowler-favoured matchups** (negative MLS)")
            bowler_fav = (
                filt[filt[mls_col] <= -threshold]
                .sort_values(mls_col, ascending=True)
                [([batter_col, bowler_col, mls_col] + ([balls_col] if balls_col else []))]
                .head(15)
            )
            st.dataframe(bowler_fav, use_container_width=True)
