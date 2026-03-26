"""
app/pages/p07_decision_audit.py — Decision Audit Tab

Answers:
  - Did toss decisions (bat/bowl first) correlate with win rates?
  - Which bowling change timings were most impactful (WPA shift)?
  - Death-over bowling allocation by team — specialist usage vs results
  - Powerplay bowling aggression vs phase outcome
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.config import apply_chart_theme, PHASE_COLORS


def render(ball_states: pd.DataFrame):
    st.header("Decision Audit — Toss, Bowling Changes & Death Allocation")

    if ball_states.empty:
        st.warning("No ball-state data available.")
        return

    seasons = sorted(ball_states["season"].dropna().unique(), reverse=True)
    sel_seasons = st.multiselect("Seasons", seasons, default=seasons[:5], key="da_seasons")
    df = ball_states[ball_states["season"].isin(sel_seasons)] if sel_seasons else ball_states.copy()

    # ── Toss decision analysis ────────────────────────────────────────────────
    st.subheader("Toss Decision → Win Rate")

    if "toss_decision" in df.columns and "match_won" in df.columns:
        toss_df = (
            df.groupby(["match_id", "toss_decision"])
            .agg(win=("match_won", "first"), season=("season", "first"))
            .reset_index()
        )
        toss_summary = (
            toss_df.groupby(["toss_decision", "season"])
            .agg(matches=("match_id", "count"), wins=("win", "sum"))
            .reset_index()
        )
        toss_summary["win_pct"] = (toss_summary["wins"] / toss_summary["matches"] * 100).round(1)

        fig1 = px.line(
            toss_summary, x="season", y="win_pct", color="toss_decision",
            markers=True,
            title="Toss Decision Win % by Season (bat vs field first)",
            color_discrete_sequence=["#3B82F6", "#EF4444"],
            labels={"win_pct": "Win %", "season": "Season", "toss_decision": "Toss Decision"},
        )
        fig1.add_hline(y=50, line_dash="dash", line_color="#94A3B8", annotation_text="50%")
        apply_chart_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        overall = toss_df.groupby("toss_decision").agg(
            matches=("match_id", "count"), win_pct=("win", lambda x: round(x.mean() * 100, 1))
        ).reset_index()
        st.dataframe(overall.rename(columns={"toss_decision": "Decision", "matches": "Matches", "win_pct": "Win %"}),
                     use_container_width=True)
    else:
        st.info("Toss decision data not in ball_states. Ensure matches.csv join was applied in pipeline 01.")

    # ── Bowling change impact (WPA shift) ─────────────────────────────────────
    st.subheader("Bowling Change Impact (WPA Shift)")
    st.caption(
        "A bowling change is detected when the bowler changes between overs. "
        "We measure WPA in the 2 overs before vs the 2 overs after the change."
    )

    if "bowler" in df.columns and "wpa_bowler" in df.columns:
        # Detect bowling changes
        over_bowler = (
            df.sort_values(["match_id", "inning", "over"])
            .groupby(["match_id", "inning", "over"])["bowler"]
            .first()
            .reset_index()
        )
        over_bowler["prev_bowler"] = over_bowler.groupby(["match_id", "inning"])["bowler"].shift(1)
        changes = over_bowler[
            over_bowler["bowler"] != over_bowler["prev_bowler"]
        ].dropna(subset=["prev_bowler"])

        if not changes.empty:
            # For each change, compute WPA in adjacent overs
            wpa_over = (
                df.groupby(["match_id", "inning", "over"])["wpa_bowler"]
                .sum()
                .reset_index()
                .rename(columns={"wpa_bowler": "wpa_sum"})
            )
            change_eval = []
            for _, row in changes.iterrows():
                mid, inn, over_n = row["match_id"], row["inning"], int(row["over"])
                before = wpa_over[
                    (wpa_over["match_id"] == mid) &
                    (wpa_over["inning"] == inn) &
                    (wpa_over["over"].isin([over_n - 2, over_n - 1]))
                ]["wpa_sum"].sum()
                after = wpa_over[
                    (wpa_over["match_id"] == mid) &
                    (wpa_over["inning"] == inn) &
                    (wpa_over["over"].isin([over_n, over_n + 1]))
                ]["wpa_sum"].sum()
                change_eval.append({"match_id": mid, "over": over_n,
                                    "wpa_before": before, "wpa_after": after,
                                    "wpa_delta": after - before})

            change_df = pd.DataFrame(change_eval)
            fig2 = px.histogram(
                change_df, x="wpa_delta", nbins=40,
                title="WPA Δ After Bowling Change (positive = change helped bowler)",
                color_discrete_sequence=["#22C55E"],
                labels={"wpa_delta": "WPA After − WPA Before"},
            )
            fig2.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
            apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Median WPA Δ: {change_df['wpa_delta'].median():.3f}  |  "
                f"% positive changes: {(change_df['wpa_delta'] > 0).mean() * 100:.1f}%"
            )
    else:
        st.info("WPA bowler data not found. Run pipeline 04 phase B first.")

    # ── Death bowling allocation ───────────────────────────────────────────────
    st.subheader("Death-Over Bowling Allocation by Team")

    death_df = df[(df.get("phase", pd.Series("", index=df.index)) == "death")] \
        if "phase" in df.columns else pd.DataFrame()

    if not death_df.empty and "bowler" in death_df.columns:
        team_col = "bowling_team" if "bowling_team" in death_df.columns else "batting_team"
        alloc = (
            death_df[death_df["is_legal_ball"] == 1] if "is_legal_ball" in death_df.columns
            else death_df
        ).groupby([team_col, "bowler"])["bowler"].count()
        alloc = alloc.reset_index(name="balls_bowled")

        teams_avail = sorted(alloc[team_col].unique())
        sel_team_audit = st.selectbox("Team (death allocation)", teams_avail, key="da_team")

        team_alloc = alloc[alloc[team_col] == sel_team_audit].nlargest(10, "balls_bowled")
        fig3 = px.bar(
            team_alloc, x="bowler", y="balls_bowled",
            title=f"Death-Over Balls Bowled — {sel_team_audit}",
            color="balls_bowled", color_continuous_scale="Blues",
        )
        fig3.update_layout(coloraxis_showscale=False)
        apply_chart_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Death-over data not available or phase column missing.")
