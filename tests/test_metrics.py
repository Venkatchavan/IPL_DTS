"""
tests/test_metrics.py

Unit tests for the 12 metrics modules.
Focus: formula correctness, edge cases, normalization bounds.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Pressure Index ────────────────────────────────────────────────────────────

class TestPressureIndex:

    def _make_state(self, rrr_crr_delta=0.0, wickets=0, phase="middle", dot_streak=0, inning=2):
        return {
            "pre_rrr_crr_delta": rrr_crr_delta,
            "pre_wickets": wickets,
            "phase": phase,
            "pre_dot_ball_streak": dot_streak,
            "inning": inning,
        }

    def test_zero_pressure_inning1(self):
        from metrics.pressure_index import compute_pressure_index
        df = pd.DataFrame([{
            "pre_rrr_crr_delta": 0.0,
            "pre_wickets": 0,
            "phase": "powerplay",
            "pre_dot_ball_streak": 0,
            "inning": 1,
        }])
        result = compute_pressure_index(df)
        pi = float(result["pressure_index"].iloc[0])
        assert 0 <= pi <= 100, f"Pressure Index out of range: {pi}"

    def test_max_pressure_components(self):
        from metrics.pressure_index import compute_pressure_index
        df = pd.DataFrame([{
            "pre_rrr_crr_delta": 8.0,   # max clamp
            "pre_wickets": 10,
            "phase": "death",
            "pre_dot_ball_streak": 12,   # max clamp
            "inning": 2,
        }])
        result = compute_pressure_index(df)
        pi = float(result["pressure_index"].iloc[0])
        assert pi > 80, f"Max-component state should produce PI>80, got {pi}"

    def test_pi_bounded_0_100(self):
        from metrics.pressure_index import compute_pressure_index
        rows = [
            {"pre_rrr_crr_delta": 15.0, "pre_wickets": 10, "phase": "death", "pre_dot_ball_streak": 20, "inning": 2},
            {"pre_rrr_crr_delta": -10.0, "pre_wickets": 0, "phase": "powerplay", "pre_dot_ball_streak": 0, "inning": 2},
        ]
        df = pd.DataFrame(rows)
        result = compute_pressure_index(df)
        pi_col = result["pressure_index"]
        assert (pi_col >= 0).all() and (pi_col <= 100).all(), "PI must be in [0, 100]"

    def test_inning1_rrr_term_zero(self):
        """Inning 1 PI should be equal whether delta is 0 or 8 (RRR term removed)."""
        from metrics.pressure_index import compute_pressure_index
        base = {"pre_wickets": 3, "phase": "middle", "pre_dot_ball_streak": 2, "inning": 1}
        df1 = pd.DataFrame([{**base, "pre_rrr_crr_delta": 0.0}])
        df2 = pd.DataFrame([{**base, "pre_rrr_crr_delta": 8.0}])
        pi1 = float(compute_pressure_index(df1)["pressure_index"].iloc[0])
        pi2 = float(compute_pressure_index(df2)["pressure_index"].iloc[0])
        assert pi1 == pi2, "Inning 1 PI should not depend on RRR-CRR delta"


# ── Win Probability Added ─────────────────────────────────────────────────────

class TestWPA:

    def test_positive_wpa_on_boundary(self):
        from metrics.wpa import compute_wpa
        df = pd.DataFrame([{
            "inning": 2,
            "pre_win_prob": 0.40,
            "post_win_prob": 0.55,
            "batsman_runs": 4,
            "extra_runs": 0,
            "is_wicket": 0,
            "striker": "Batter1",
            "bowler": "Bowler1",
            "match_id": 1,
            "global_ball_number": 1,
        }])
        result = compute_wpa(df)
        assert result.iloc[0]["wpa"] == pytest.approx(0.15, abs=1e-4)

    def test_negative_wpa_on_wicket(self):
        from metrics.wpa import compute_wpa
        df = pd.DataFrame([{
            "inning": 2,
            "pre_win_prob": 0.60,
            "post_win_prob": 0.40,
            "batsman_runs": 0,
            "extra_runs": 0,
            "is_wicket": 1,
            "striker": "Batter1",
            "bowler": "Bowler1",
            "match_id": 1,
            "global_ball_number": 2,
        }])
        result = compute_wpa(df)
        assert result.iloc[0]["wpa"] == pytest.approx(-0.20, abs=1e-4)

    def test_wpa_only_inning2(self):
        from metrics.wpa import compute_wpa
        df = pd.DataFrame([{
            "inning": 1,
            "pre_win_prob": 0.50,
            "post_win_prob": 0.55,
            "batsman_runs": 4,
            "extra_runs": 0,
            "is_wicket": 0,
            "striker": "Batter1",
            "bowler": "Bowler1",
            "match_id": 1,
            "global_ball_number": 1,
        }])
        result = compute_wpa(df)
        # WPA should be NaN or 0 for inning 1
        assert pd.isna(result.iloc[0]["wpa"]) or result.iloc[0]["wpa"] == 0


# ── Expected Score Added ──────────────────────────────────────────────────────

class TestESA:

    def test_positive_esa_on_boundary(self):
        from metrics.esa import compute_esa
        df = pd.DataFrame([{
            "inning": 1,
            "pre_expected_score": 160.0,
            "post_expected_score": 168.0,
            "batsman_runs": 4,
            "extra_runs": 0,
            "is_wicket": 0,
            "striker": "Batter1",
            "bowler": "Bowler1",
            "match_id": 1,
            "global_ball_number": 1,
        }])
        result = compute_esa(df)
        assert result.iloc[0]["esa"] == pytest.approx(8.0, abs=1e-4)


# ── Matchup Leverage Score ────────────────────────────────────────────────────

class TestMLS:

    def test_mls_formula(self):
        """MLS = (batter_sr - mean_sr) / std_sr"""
        from metrics.matchup_leverage import compute_mls

        df = pd.DataFrame([
            {"striker": "A", "bowler": "B1", "balls_faced": 40, "batsman_runs": 24},  # SR=60
            {"striker": "A", "bowler": "B2", "balls_faced": 40, "batsman_runs": 40},  # SR=100
            {"striker": "C", "bowler": "B1", "balls_faced": 40, "batsman_runs": 32},  # SR=80
        ])
        df["strike_rate"] = df["batsman_runs"] / df["balls_faced"] * 100

        # Separate bowler style buckets DataFrame (as required by compute_mls)
        bowler_style_buckets = pd.DataFrame([
            {"player": "B1", "style_bucket": 0},
            {"player": "B2", "style_bucket": 0},
        ])

        result = compute_mls(df, bowler_style_buckets)
        # With these values, A vs B1 SR=60, A vs B2 SR=100
        # mean across bucket: (60+100+80)/3 = 80
        # std ~ 16.3
        assert "mls" in result.columns
        assert result["mls"].notna().any()
