"""
app/pages/p08_scouting_role_fit.py — Scouting & Role Fit Tab

Shows:
  - Player role clustering radar (anchor, aggressor, powerplay specialist, etc.)
  - Replacement finder: find similar players by feature profile
  - Squad gap analysis by role
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, MIN_DISPLAY_BALLS_BATTER


# Role name map for K-Means cluster labels (set after inspection)
ROLE_NAMES = {
    0: "Anchor",
    1: "Powerplay Aggressor",
    2: "Middle-Order Builder",
    3: "Death Finisher",
    4: "Bowling Allrounder",
}


def _profile_radar(player_series: pd.Series, dimensions: list, title: str) -> go.Figure:
    """Create a radar chart for a player profile."""
    vals = [float(player_series.get(d, 0)) for d in dimensions]
    vals += vals[:1]  # close radar
    angles = [i / len(dimensions) * 360 for i in range(len(dimensions))] + [0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=dimensions + [dimensions[0]],
        fill="toself", name=title,
        line=dict(color="#3B82F6"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
    )
    apply_chart_theme(fig)
    return fig


def render(player_feats: pd.DataFrame, metrics_player: pd.DataFrame):
    st.header("Scouting & Role Fit — Clustering & Replacement Finder")

    src = player_feats if not player_feats.empty else metrics_player
    if src.empty:
        st.warning("Player features not available. Run pipeline 03 first.")
        return

    batter_col = next((c for c in ["striker", "batter", "player"] if c in src.columns), None)
    if not batter_col:
        st.warning("Cannot identify player column in features table.")
        return

    # ── Role assignment (K-Means cluster) ────────────────────────────────────
    st.subheader("Player Role Clusters")

    role_cols = [c for c in ["strike_rate", "dot_pct", "boundary_pct", "avg_phase_powerplay",
                              "avg_phase_death", "balls_faced"] if c in src.columns]

    if len(role_cols) >= 3 and "role_cluster" not in src.columns:
        # Compute on-the-fly if not pre-computed in pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans

        ball_thresh = MIN_DISPLAY_BALLS_BATTER
        cluster_src = src.dropna(subset=role_cols).copy()
        balls_col = next((c for c in ["balls_faced", "balls"] if c in cluster_src.columns), None)
        if balls_col:
            cluster_src = cluster_src[cluster_src[balls_col] >= ball_thresh]

        if len(cluster_src) >= 10:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(cluster_src[role_cols])
            n_clusters = min(5, len(cluster_src))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_src["role_cluster"] = km.fit_predict(X)
            cluster_src["role_name"] = cluster_src["role_cluster"].map(ROLE_NAMES).fillna("Other")
            src = src.merge(
                cluster_src[[batter_col, "role_cluster", "role_name"]],
                on=batter_col, how="left"
            )
        else:
            st.info("Not enough player data for clustering.")
    elif "role_cluster" in src.columns:
        src["role_name"] = src["role_cluster"].map(ROLE_NAMES).fillna("Other")

    if "role_name" in src.columns:
        role_counts = src.groupby("role_name")[batter_col].count().reset_index()
        role_counts.columns = ["Role", "Players"]
        fig_roles = px.bar(
            role_counts, x="Role", y="Players",
            title="Player Count by Role",
            color="Role", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        apply_chart_theme(fig_roles)
        st.plotly_chart(fig_roles, use_container_width=True)

        sel_role = st.selectbox("Filter by role", ["All"] + sorted(src["role_name"].dropna().unique().tolist()))
        view_df = src if sel_role == "All" else src[src["role_name"] == sel_role]
        display_cols = [batter_col, "role_name"] + role_cols
        display_cols = [c for c in display_cols if c in view_df.columns]
        st.dataframe(view_df[display_cols].sort_values(role_cols[0], ascending=False).head(50),
                     use_container_width=True)

    # ── Player radar profile ──────────────────────────────────────────────────
    st.subheader("Player Profile Radar")

    players = sorted(src[batter_col].dropna().unique())
    sel_player = st.selectbox("Select player", players, key="srf_player")
    player_row = src[src[batter_col] == sel_player]

    radar_dims = [c for c in ["strike_rate", "dot_pct", "boundary_pct",
                               "avg_phase_powerplay", "avg_phase_death",
                               "pressure_sr_high", "pressure_sr_critical"] if c in src.columns]
    if player_row.empty or not radar_dims:
        st.info("Insufficient data for this player or radar dimensions not available.")
    else:
        # Normalize 0-1 across all players
        normalized = src[radar_dims].copy()
        for col in radar_dims:
            col_min, col_max = normalized[col].min(), normalized[col].max()
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min + 1e-9)
        p_norm = normalized.loc[player_row.index[0]]
        fig_rad = _profile_radar(p_norm, radar_dims, f"{sel_player} — Normalized Profile")
        st.plotly_chart(fig_rad, use_container_width=True)

    # ── Replacement finder ────────────────────────────────────────────────────
    st.subheader("Replacement Finder — Most Similar Players")
    st.caption("Find players with the closest feature profile (cosine similarity).")

    sim_dims = [c for c in ["strike_rate", "dot_pct", "boundary_pct"] if c in src.columns]
    if len(sim_dims) >= 2 and not src.empty:
        balls_col = next((c for c in ["balls_faced", "balls"] if c in src.columns), None)
        sim_src = src.dropna(subset=sim_dims).copy()
        if balls_col:
            sim_src = sim_src[sim_src[balls_col] >= MIN_DISPLAY_BALLS_BATTER // 2]

        if len(sim_src) >= 5:
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics.pairwise import cosine_similarity

            scaler = MinMaxScaler()
            X_sim = scaler.fit_transform(sim_src[sim_dims])
            p_idx = sim_src[batter_col].tolist().index(sel_player) if sel_player in sim_src[batter_col].values else None

            if p_idx is not None:
                sims = cosine_similarity(X_sim[p_idx:p_idx+1], X_sim)[0]
                sim_src = sim_src.copy()
                sim_src["similarity"] = sims
                top_similar = (
                    sim_src[sim_src[batter_col] != sel_player]
                    .nlargest(10, "similarity")
                    [[batter_col, "similarity"] + sim_dims]
                )
                top_similar["similarity"] = top_similar["similarity"].round(3)
                st.dataframe(top_similar, use_container_width=True)
            else:
                st.info(f"{sel_player} not found in filtered feature set (may lack minimum balls).")
