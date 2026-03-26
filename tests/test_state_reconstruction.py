"""
tests/test_state_reconstruction.py

Unit tests for pipelines/02_state_reconstruction.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_deliveries(rows):
    """Build a minimal deliveries DataFrame for testing."""
    defaults = {
        "match_id": 1,
        "inning": 2,
        "over": 1,
        "ball": 1,
        "batting_team": "TeamA",
        "bowling_team": "TeamB",
        "batter": "Batter1",
        "non_striker": "Batter2",
        "bowler": "Bowler1",
        "batsman_runs": 0,
        "extra_runs": 0,
        "total_runs": 0,
        "wide_runs": 0,
        "noball_runs": 0,
        "is_wicket": 0,
        "player_dismissed": None,
        "dismissal_kind": None,
        "season": 2024,
    }
    records = []
    for r in rows:
        row = {**defaults, **r}
        records.append(row)
    return pd.DataFrame(records)


def get_matches_stub():
    """Minimal matches DataFrame required by reconstruct_match_states."""
    return pd.DataFrame([{
        "match_id": 1,
        "season": 2024,
        "venue": "TestVenue",
        "toss_winner": "TeamA",
        "toss_decision": "bat",
        "winner": "TeamA",
        "team1": "TeamA",
        "team2": "TeamB",
        "dl_applied": 0,
    }])


# ── Import module ─────────────────────────────────────────────────────────────

def get_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "state_reconstruction",
        ROOT / "pipelines" / "02_state_reconstruction.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sr_mod():
    return get_module()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_is_legal_ball_wide(sr_mod):
    df = make_deliveries([
        {"wide_runs": 1, "extra_runs": 1, "batsman_runs": 0},
        {"wide_runs": 0, "noball_runs": 0, "batsman_runs": 4},
    ])
    result = sr_mod.add_is_legal_ball(df)
    assert result.iloc[0]["is_legal_ball"] == 0, "Wide should not be legal"
    assert result.iloc[1]["is_legal_ball"] == 1, "Normal delivery should be legal"


def test_is_legal_ball_noball(sr_mod):
    df = make_deliveries([
        {"noball_runs": 1, "extra_runs": 1, "batsman_runs": 0},
    ])
    result = sr_mod.add_is_legal_ball(df)
    assert result.iloc[0]["is_legal_ball"] == 0, "No-ball should not be legal"


def test_cumulative_runs_increment(sr_mod):
    """Cumulative runs should increase correctly across legal balls."""
    rows = [
        {"over": 1, "ball": 1, "batsman_runs": 4, "extra_runs": 0, "total_runs": 4},
        {"over": 1, "ball": 2, "batsman_runs": 0, "extra_runs": 0, "total_runs": 0},
        {"over": 1, "ball": 3, "batsman_runs": 6, "extra_runs": 0, "total_runs": 6},
    ]
    df = make_deliveries(rows)
    matches = get_matches_stub()
    result = sr_mod.reconstruct_match_states(df, matches)

    pre_runs = result["pre_runs"].tolist()
    assert pre_runs[0] == 0, "First ball: 0 runs before"
    assert pre_runs[1] == 4, "Second ball: 4 runs before"
    assert pre_runs[2] == 4, "Third ball: 4 runs before (dot ball after)"


def test_wicket_counting(sr_mod):
    """Wickets in hand should decrease after dismissal."""
    rows = [
        {"over": 1, "ball": 1, "batsman_runs": 0, "total_runs": 0, "is_wicket": 0},
        {"over": 1, "ball": 2, "batsman_runs": 0, "total_runs": 0, "is_wicket": 1,
         "dismissal_kind": "caught", "player_dismissed": "Batter1"},
        {"over": 1, "ball": 3, "batsman_runs": 0, "total_runs": 0, "is_wicket": 0},
    ]
    df = make_deliveries(rows)
    matches = get_matches_stub()
    result = sr_mod.reconstruct_match_states(df, matches)

    pre_wickets = result["pre_wickets"].tolist()
    assert pre_wickets[0] == 0
    assert pre_wickets[1] == 0, "Before the wicket ball, still 0 wickets"
    assert pre_wickets[2] == 1, "After wicket, 1 wicket recorded"


def test_no_future_leakage(sr_mod):
    """pre_runs must equal the actual runs BEFORE each delivery."""
    rows = [
        {"over": 1, "ball": 1, "batsman_runs": 4, "extra_runs": 0, "total_runs": 4},
        {"over": 1, "ball": 2, "batsman_runs": 2, "extra_runs": 0, "total_runs": 2},
        {"over": 1, "ball": 3, "batsman_runs": 0, "extra_runs": 0, "total_runs": 0},
    ]
    df = make_deliveries(rows)
    matches = get_matches_stub()
    result = sr_mod.reconstruct_match_states(df, matches)

    # pre_runs[i] should match cumulative total before ball i
    expected_pre_runs = [0, 4, 6]
    for i, expected in enumerate(expected_pre_runs):
        actual = int(result.iloc[i]["pre_runs"])
        assert actual == expected, f"Ball {i}: expected pre_runs={expected}, got {actual}"


def test_balls_remaining_decreases(sr_mod):
    """Balls remaining should decrease by 1 for each legal ball."""
    rows = [
        {"over": i + 1, "ball": b + 1, "batsman_runs": 0, "extra_runs": 0, "total_runs": 0}
        for i in range(3) for b in range(6)
    ]
    df = make_deliveries(rows)
    matches = get_matches_stub()
    result = sr_mod.reconstruct_match_states(df, matches)

    pre_balls = result["pre_balls_remaining"].tolist()
    assert pre_balls[0] == 120, "Start: 120 balls remaining"
    assert pre_balls[1] == 119, "After first legal ball: 119 remaining"


def test_chase_fields_rrr(sr_mod):
    """RRR should be runs_needed / balls_remaining * 6."""
    # add_chase_fields derives target from inning 1 data; need both innings
    # Inning 1: one ball, 4 runs → inning1_total=4, target=5
    rows_inn1 = [{"inning": 1, "over": 1, "ball": 1, "batsman_runs": 4,
                  "extra_runs": 0, "total_runs": 4}]
    # Inning 2: one ball, 0 runs → pre_runs_needed=5, pre_balls_remaining=120
    rows_inn2 = [{"inning": 2, "over": 1, "ball": 1, "batsman_runs": 0,
                  "extra_runs": 0, "total_runs": 0}]
    df = make_deliveries(rows_inn1 + rows_inn2)
    matches = get_matches_stub()
    state_df = sr_mod.reconstruct_match_states(df, matches)
    state_df = sr_mod.add_chase_fields(state_df)

    row = state_df[state_df["inning"] == 2].iloc[0]
    expected_rrr = (float(row["pre_runs_needed"]) / float(row["pre_balls_remaining"])) * 6
    assert abs(float(row["pre_rrr"]) - expected_rrr) < 0.01
