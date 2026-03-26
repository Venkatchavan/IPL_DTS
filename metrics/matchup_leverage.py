"""
matchup_leverage.py — Matchup Leverage Score (MLS)

MLS quantifies whether a batter has a statistical edge (positive MLS) or
disadvantage (negative MLS) against a specific bowler or bowling style.

Formula:
  MLS = (actual_sr_vs_bowler - expected_sr_vs_bowling_style) /
        std_sr_vs_bowling_style

Where:
  - actual_sr_vs_bowler: batter's observed strike rate vs this specific bowler
  - expected_sr_vs_bowling_style: average SR batters of similar profile achieve
    vs bowlers of this style / hand / pace
  - std_sr_vs_bowling_style: standard deviation of SR distribution for that style bucket

MLS > 0: Batter outperforms expectation (has advantage)
MLS < 0: Batter underperforms expectation (bowler has advantage)
MLS is a Z-score, interpretable as standard deviations from baseline.

Limitations:
  - Bowling "style" is not explicitly tagged in IPL dataset; requires proxy:
    * We cluster bowlers by economy, dot_pct, boundary_rate → style buckets
    * This is a heuristic, not a ground-truth style classification
  - Minimum 30 balls required for reliable MLS (shown in UI as sample count)
  - Small sample pairs will have wide confidence intervals

Deployment:
  - Used in Tab 6 (Matchup Intelligence) of dashboard
  - Filterable by season range, phase, venue
"""

import pandas as pd
import numpy as np

from config import MIN_BALLS_MATCHUP


def assign_bowler_style_bucket(bowler_overall: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Assign proxy style buckets to bowlers using unsupervised clustering
    on economy, dot_pct, boundary_concession_rate, wicket_rate.

    Buckets approximate: pace-attacking / pace-containing / spin-attacking /
    spin-containing / death-specialist (labeling is post-hoc interpretation).

    Args:
        bowler_overall: bowler features with slice=="overall"
        n_clusters: number of style clusters

    Returns:
        DataFrame with bowler + style_bucket columns
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ["economy_rate", "dot_pct", "boundary_concession_rate", "wicket_rate"]
    df = bowler_overall[bowler_overall["slice"] == "overall"].copy()
    df = df.dropna(subset=features)

    feature_matrix = df[features].values
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["style_bucket"] = km.fit_predict(X)

    return df[["player", "style_bucket"]]


def compute_mls(
    matchup_features: pd.DataFrame,
    bowler_style_buckets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Matchup Leverage Score for all batter–bowler pairs with >= MIN_BALLS_MATCHUP.

    Args:
        matchup_features: from pipeline 03 (per striker-bowler aggregation)
        bowler_style_buckets: from assign_bowler_style_bucket()

    Returns:
        matchup_features with mls column added
    """
    df = matchup_features.copy()

    # Enforce minimum sample
    df = df[df["balls_faced"] >= MIN_BALLS_MATCHUP].copy()

    if len(df) == 0:
        df["mls"] = np.nan
        return df

    # Join style bucket of each bowler
    df = df.merge(
        bowler_style_buckets.rename(columns={"player": "bowler"}),
        on="bowler", how="left"
    )

    # Compute style-bucket expected SR and std
    style_stats = (
        df.groupby(["striker", "style_bucket"])["strike_rate"]
        .agg(expected_sr="mean", std_sr="std")
        .reset_index()
    )
    # Fallback: overall mean/std when style bucket has only 1 matchup
    overall_sr_stats = (
        df.groupby("striker")["strike_rate"]
        .agg(expected_sr_overall="mean", std_sr_overall="std")
        .reset_index()
    )

    df = df.merge(style_stats, on=["striker", "style_bucket"], how="left")
    df = df.merge(overall_sr_stats, on="striker", how="left")

    # Use overall stats as fallback where style bucket has only 1 observation
    df["expected_sr"] = df["expected_sr"].fillna(df["expected_sr_overall"])
    df["std_sr"]      = df["std_sr"].fillna(df["std_sr_overall"]).replace(0, np.nan)

    # Compute MLS
    df["mls"] = (
        (df["strike_rate"] - df["expected_sr"]) / df["std_sr"]
    ).round(3)

    # Confidence flag
    df["mls_confidence"] = df["balls_faced"].apply(
        lambda b: "high" if b >= 100 else "medium" if b >= 30 else "low"
    )

    return df.sort_values("mls", ascending=False).reset_index(drop=True)


def get_exploitable_matchups(
    mls_df: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Return matchups where MLS > threshold (batter advantage) or < -threshold (bowler advantage).
    Used to flag strategically exploitable pairings for Tab 6.

    Args:
        mls_df: output from compute_mls()
        threshold: Z-score boundary for flagging

    Returns:
        Filtered DataFrame with batter_advantage and bowler_advantage flags
    """
    df = mls_df.copy()
    df["batter_advantage"] = (df["mls"] > threshold).astype(int)
    df["bowler_advantage"] = (df["mls"] < -threshold).astype(int)

    return df[df["batter_advantage"] | df["bowler_advantage"]].reset_index(drop=True)
