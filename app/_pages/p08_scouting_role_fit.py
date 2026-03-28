"""
app/pages/p08_scouting_role_fit.py — Scouting & Role Fit Tab

Shows:
  - Player role clustering radar (anchor, aggressor, powerplay specialist, etc.)
  - Replacement finder: find similar players by feature profile
  - Squad gap analysis by role
  - Active player filter (2023–2025 IPL seasons)
  - Pakistan nationality exclusion (banned from IPL)
  - Domestic league integration framework
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


# ── Pakistan nationality players (banned from IPL) ────────────────────────────
# These players appeared in early IPL seasons before the BCCI ban on Pakistan
# nationals. Excluded from scouting recommendations by default.
PAKISTAN_NATIONAL_PLAYERS = {
    "Shoaib Akhtar", "Shoaib Malik", "Shoaib Ahmed",
    "Sohail Tanvir", "Umar Gul", "Kamran Akmal", "Kamran Khan",
    "Salman Butt", "Shahid Afridi", "Danish Kaneria", "Abdul Razzaq",
    "Iftikhar Ahmed", "Yasir Shah", "Shadab Khan", "Fakhar Zaman",
    "Mohammad Aamer", "Mohammad Asif", "Umar Akmal",
}

# ── Activity threshold default ───────────────────────────────────────────────
# Players whose last IPL appearance was before this year are treated as retired/inactive.
ACTIVE_SINCE_DEFAULT = 2022  # 4 seasons back from 2026

# ── Role name map ─────────────────────────────────────────────────────────────
ROLE_NAMES = {
    0: "Anchor",
    1: "Powerplay Aggressor",
    2: "Middle-Order Builder",
    3: "Death Finisher",
    4: "Bowling Allrounder",
}

# ── Domestic league labels (for future integration) ───────────────────────────
DOMESTIC_LEAGUES = {
    "SMAT": "Syed Mushtaq Ali Trophy (India)",
    "BBL":  "Big Bash League (Australia)",
    "CPL":  "Caribbean Premier League (West Indies)",
    "SA20": "SA20 (South Africa)",
    "ILT20": "ILT20 (UAE)",
    "MLC":  "Major League Cricket (USA)",
}


def _profile_radar(player_series: pd.Series, dimensions: list, title: str) -> go.Figure:
    """Create a radar chart for a player profile."""
    vals = [float(player_series.get(d, 0)) for d in dimensions]
    vals += vals[:1]

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


def _get_player_last_season(player_feats: pd.DataFrame, player_col: str) -> dict:
    """
    Return {player_name: last_season_int} for every player in the data.
    Players with no season data get last_season = 0 (excluded by any threshold).
    """
    if "season" not in player_feats.columns:
        return {}
    valid = player_feats[[player_col, "season"]].dropna()
    last = (
        valid.groupby(player_col)["season"]
        .max()
        .astype(float)
        .apply(lambda x: int(x) if not np.isnan(x) else 0)
        .to_dict()
    )
    return last


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

    # ── Pre-compute last season per player (from season-sliced rows) ──────────
    last_season_map = _get_player_last_season(src, batter_col)  # {player: int year}
    valid_years = sorted({y for y in last_season_map.values() if y >= 2019})
    available_thresholds = valid_years if valid_years else [2019, 2020, 2021, 2022, 2023, 2024]

    # ── Filters sidebar ───────────────────────────────────────────────────────
    st.subheader("Filters")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        active_since = st.select_slider(
            "Last played in IPL (minimum year)",
            options=available_thresholds,
            value=min(ACTIVE_SINCE_DEFAULT, max(available_thresholds)),
            help="Excludes players whose last IPL appearance was before this season. "
                 "Default 2022 = active within the past 4 seasons.",
            key="srf_active_since",
        )

    with col_f2:
        exclude_pak = st.toggle("Exclude Pakistan nationals", value=True,
                                help="Pakistan players are currently ineligible for IPL selection")

    with col_f3:
        player_type_filter = "All"
        if "player_type" in src.columns:
            types = ["All"] + sorted(src["player_type"].dropna().unique().tolist())
            player_type_filter = st.selectbox("Player type", types, key="srf_ptype")

    # Work from 'overall' slice only (career aggregate, one row per player)
    if "slice" in src.columns:
        work_df = src[src["slice"] == "overall"].copy()
    else:
        work_df = src.copy()

    # Attach last_season to every row so it shows in tables and can filter
    work_df["last_season"] = work_df[batter_col].map(last_season_map).fillna(0).astype(int)

    # Apply recency filter — excludes retired / long-inactive players
    work_df = work_df[work_df["last_season"] >= active_since]

    # Apply Pakistan exclusion
    if exclude_pak:
        work_df = work_df[~work_df[batter_col].isin(PAKISTAN_NATIONAL_PLAYERS)]

    # Apply player type filter
    if player_type_filter != "All" and "player_type" in work_df.columns:
        work_df = work_df[work_df["player_type"] == player_type_filter]

    if work_df.empty:
        st.warning("No players match the current filters.")
        return

    n_2025 = int((work_df["last_season"] == 2025).sum())
    n_2024 = int((work_df["last_season"] == 2024).sum())
    st.caption(
        f"**{work_df[batter_col].nunique()} players** — "
        f"{n_2025} played in IPL 2025, {n_2024} last played in 2024"
        f"{' · Pakistan nationals excluded' if exclude_pak else ''}"
    )

    # ── Role clustering ───────────────────────────────────────────────────────
    st.subheader("Player Role Clusters")

    role_cols = [c for c in ["strike_rate", "dot_pct", "boundary_pct",
                              "balls_faced"] if c in work_df.columns]

    if len(role_cols) >= 3 and "role_cluster" not in work_df.columns:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans

        ball_thresh = MIN_DISPLAY_BALLS_BATTER
        cluster_src = work_df.dropna(subset=role_cols).copy()
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
            work_df = work_df.merge(
                cluster_src[[batter_col, "role_cluster", "role_name"]],
                on=batter_col, how="left"
            )
        else:
            st.info("Not enough player data for clustering with current filters.")
    elif "role_cluster" in work_df.columns:
        work_df["role_name"] = work_df["role_cluster"].map(ROLE_NAMES).fillna("Other")

    if "role_name" in work_df.columns:
        role_counts = work_df.groupby("role_name")[batter_col].count().reset_index()
        role_counts.columns = ["Role", "Players"]
        fig_roles = px.bar(
            role_counts, x="Role", y="Players",
            title="Player Count by Role (filtered)",
            color="Role", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        apply_chart_theme(fig_roles)
        st.plotly_chart(fig_roles, use_container_width=True)

        sel_role = st.selectbox("Filter by role", ["All"] + sorted(
            work_df["role_name"].dropna().unique().tolist()), key="srf_role")
        view_df = work_df if sel_role == "All" else work_df[work_df["role_name"] == sel_role]
        display_cols = [batter_col, "last_season", "role_name"] + role_cols
        display_cols = [c for c in display_cols if c in view_df.columns]
        st.dataframe(
            view_df[display_cols].sort_values(role_cols[0], ascending=False).head(50),
            use_container_width=True
        )

    # ── Player radar profile ──────────────────────────────────────────────────
    st.subheader("Player Profile Radar")

    players = sorted(work_df[batter_col].dropna().unique())
    if not players:
        st.info("No players available with current filters.")
        return

    sel_player = st.selectbox("Select player", players, key="srf_player")
    player_row = work_df[work_df[batter_col] == sel_player]

    radar_dims = [c for c in ["strike_rate", "dot_pct", "boundary_pct",
                               "avg_per_inning"] if c in work_df.columns]

    if player_row.empty or not radar_dims:
        st.info("Insufficient data for this player or radar dimensions not available.")
    else:
        normalized = work_df[radar_dims].copy()
        for col in radar_dims:
            col_min, col_max = normalized[col].min(), normalized[col].max()
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min + 1e-9)
        if not player_row.empty and player_row.index[0] in normalized.index:
            p_norm = normalized.loc[player_row.index[0]]
            fig_rad = _profile_radar(p_norm, radar_dims, f"{sel_player} — Normalized Profile")
            st.plotly_chart(fig_rad, use_container_width=True)

    # ── Replacement finder ────────────────────────────────────────────────────
    st.subheader("Replacement Finder — Most Similar Players")
    st.caption(
        "Cosine similarity on batting profile. "
        "Only players matching your active/exclusion filters are considered as replacements."
    )

    sim_dims = [c for c in ["strike_rate", "dot_pct", "boundary_pct"] if c in work_df.columns]
    if len(sim_dims) >= 2:
        balls_col = next((c for c in ["balls_faced", "balls"] if c in work_df.columns), None)
        sim_src = work_df.dropna(subset=sim_dims).copy()
        if balls_col:
            sim_src = sim_src[sim_src[balls_col] >= MIN_DISPLAY_BALLS_BATTER // 2]

        if len(sim_src) >= 5:
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics.pairwise import cosine_similarity

            scaler = MinMaxScaler()
            X_sim = scaler.fit_transform(sim_src[sim_dims])
            player_list = sim_src[batter_col].tolist()

            if sel_player in player_list:
                p_idx = player_list.index(sel_player)
                sims = cosine_similarity(X_sim[p_idx:p_idx+1], X_sim)[0]
                sim_src = sim_src.copy()
                sim_src["similarity"] = sims
                show_cols = [batter_col, "last_season", "similarity"] + sim_dims
                if "player_type" in sim_src.columns:
                    show_cols.insert(2, "player_type")

                top_similar = (
                    sim_src[sim_src[batter_col] != sel_player]
                    .nlargest(10, "similarity")
                    [[c for c in show_cols if c in sim_src.columns]]
                )
                top_similar["similarity"] = top_similar["similarity"].round(3)
                st.dataframe(top_similar, use_container_width=True)
            else:
                st.info(f"{sel_player} not found in filtered feature set (may lack minimum balls).")

    # ── Domestic league integration framework ─────────────────────────────────
    st.markdown("---")
    with st.expander("🌐 Additional Leagues Integration (Coming Soon)", expanded=False):
        st.markdown(
            "**Currently showing**: IPL 2008–2025 data only.\n\n"
            "**Planned domestic league integrations** (eligible players for IPL selection):\n"
        )
        col_l1, col_l2 = st.columns(2)
        eligible_leagues = {k: v for k, v in DOMESTIC_LEAGUES.items()}
        items = list(eligible_leagues.items())
        with col_l1:
            for abbr, name in items[:3]:
                st.markdown(f"- **{abbr}** — {name}")
        with col_l2:
            for abbr, name in items[3:]:
                st.markdown(f"- **{abbr}** — {name}")

        st.info(
            "To add domestic league data: export player stats in the same schema "
            "as `data/raw/deliveries.csv` (player, strike_rate, dot_pct, boundary_pct, "
            "balls_faced, season, competition) and re-run `pipelines/03_feature_engineering.py`. "
            "Players from **PSL** and those holding Pakistan nationality are excluded "
            "as they are currently ineligible for IPL selection."
        )

