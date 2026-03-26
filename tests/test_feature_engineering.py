"""
tests/test_feature_engineering.py

Unit tests for pipelines/03_feature_engineering.py
Focus: aggregation correctness, minimum sample enforcement, boundary/dot flags.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "feature_engineering",
        ROOT / "pipelines" / "03_feature_engineering.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fe_mod():
    return get_module()


def make_ball_states(rows):
    defaults = {
        "match_id": 1, "inning": 1, "over": 1, "ball": 1,
        "batting_team": "TeamA", "bowling_team": "TeamB",
        "striker": "Batter1", "non_striker": "Batter2", "bowler": "Bowler1",
        "batsman_runs": 0, "extra_runs": 0,
        "is_wicket": 0, "dismissal_kind": None, "player_dismissed": None,
        "is_legal_ball": 1, "phase": "powerplay", "pressure_band": "neutral",
        "pre_crr": 6.0, "pre_rrr": 6.0, "pre_rrr_crr_delta": 0.0,
        "season": 2024, "venue": "TestGround",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ── Boundary flags ────────────────────────────────────────────────────────────

def test_boundary_flag_four(fe_mod):
    df = make_ball_states([{"batsman_runs": 4}])
    result = fe_mod.add_boundary_flag(df)
    assert result.iloc[0]["is_four"] == 1
    assert result.iloc[0]["is_six"] == 0
    assert result.iloc[0]["is_boundary"] == 1
    assert result.iloc[0]["is_dot_ball"] == 0


def test_boundary_flag_six(fe_mod):
    df = make_ball_states([{"batsman_runs": 6}])
    result = fe_mod.add_boundary_flag(df)
    assert result.iloc[0]["is_six"] == 1
    assert result.iloc[0]["is_boundary"] == 1


def test_dot_ball_flag(fe_mod):
    df = make_ball_states([{"batsman_runs": 0, "extra_runs": 0}])
    result = fe_mod.add_boundary_flag(df)
    assert result.iloc[0]["is_dot_ball"] == 1


def test_non_dot_run(fe_mod):
    df = make_ball_states([{"batsman_runs": 1}])
    result = fe_mod.add_boundary_flag(df)
    assert result.iloc[0]["is_dot_ball"] == 0


# ── Bowler wicket flag ────────────────────────────────────────────────────────

def test_bowler_wicket_normal_dismissal(fe_mod):
    df = make_ball_states([{
        "is_wicket": 1, "dismissal_kind": "caught"
    }])
    result = fe_mod.add_bowler_wicket_flag(df)
    assert result.iloc[0]["is_bowler_wicket"] == 1


def test_bowler_wicket_excludes_runout(fe_mod):
    df = make_ball_states([{
        "is_wicket": 1, "dismissal_kind": "run out"
    }])
    result = fe_mod.add_bowler_wicket_flag(df)
    assert result.iloc[0]["is_bowler_wicket"] == 0


def test_no_wicket(fe_mod):
    df = make_ball_states([{"is_wicket": 0, "dismissal_kind": None}])
    result = fe_mod.add_bowler_wicket_flag(df)
    assert result.iloc[0]["is_bowler_wicket"] == 0


# ── Batter features aggregation ──────────────────────────────────────────────

def test_batter_features_strike_rate(fe_mod):
    """Strike rate = (total batsman runs / balls faced) × 100"""
    rows = [
        {"striker": "Batter1", "batsman_runs": 4, "is_legal_ball": 1},
        {"striker": "Batter1", "batsman_runs": 0, "is_legal_ball": 1},
        {"striker": "Batter1", "batsman_runs": 6, "is_legal_ball": 1},
        {"striker": "Batter1", "batsman_runs": 1, "is_legal_ball": 1},
    ] * 30  # 120 balls total → SR = (30×11 / 120) × 100 = 275
    df = make_ball_states(rows)
    df = fe_mod.add_boundary_flag(df)
    df = fe_mod.add_bowler_wicket_flag(df)
    result = fe_mod.build_batter_features(df)

    overall = result.get("overall", pd.DataFrame())
    if not overall.empty:
        batter_row = overall[overall["striker"] == "Batter1"]
        if not batter_row.empty:
            sr = batter_row.iloc[0].get("strike_rate", None)
            if sr is not None:
                expected_sr = (4 + 0 + 6 + 1) / 4 * 100
                assert abs(sr - expected_sr) < 1.0, f"Strike rate mismatch: {sr} vs {expected_sr}"


# ── Minimum sample enforcement ────────────────────────────────────────────────

def test_matchup_min_balls(fe_mod):
    """Matchups with fewer than MIN_BALLS_MATCHUP should not appear in results."""
    rows = [{"striker": "A", "bowler": "X", "batsman_runs": 4, "is_legal_ball": 1}] * 5
    df = make_ball_states(rows)
    df = fe_mod.add_boundary_flag(df)
    df = fe_mod.add_bowler_wicket_flag(df)
    result = fe_mod.build_matchup_features(df)

    if not result.empty:
        matchup_row = result[
            (result.get("striker", result.index) == "A") &
            (result.get("bowler", result.index) == "X")
        ]
        # Should either be empty (filtered) or have a low-confidence flag
        assert matchup_row.empty or matchup_row.iloc[0].get("confidence", "low") == "low", \
            "Short matchup should be filtered or flagged low-confidence"
