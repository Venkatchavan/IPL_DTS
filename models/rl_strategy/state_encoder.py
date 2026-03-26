"""
models/rl_strategy/state_encoder.py

Encodes a raw match state dictionary into a discrete state key
for the offline RL Q-table / FQI model.

The state space is deliberately kept interpretable and tractable.
Continuous features are binned; the state key is a tuple of bin indices.

Design decision: we use a mixed representation —
  - discrete state key (tuple) for tabular Q-table lookup
  - vector encoding (numpy array) for FQI function approximation

Both are produced from the same encode() function.
"""

import numpy as np
from typing import Tuple, Dict


# ── State space bin definitions ───────────────────────────────────────────────
# Each feature is mapped to a small number of discrete bins.
# Bin counts are chosen to balance coverage vs tractability.

BINS = {
    "runs_needed": [0, 12, 24, 36, 54, 72, 96, 120, 9999],
    # [0-11, 12-23, 24-35, 36-53, 54-71, 72-95, 96-119, 120+]

    "balls_remaining": [0, 6, 12, 18, 24, 36, 48, 72, 120],
    # [<6, 6-11, 12-17, 18-23, 24-35, 36-47, 48-71, 72-120]

    "wickets_in_hand": [0, 1, 2, 3, 4, 6, 8, 10, 11],
    # 1, 2, 3, 4, 5-6, 7-8, 9-10

    "rrr_crr_delta": [-99, -3, -1.5, 0, 1.5, 3, 5, 99],
    # very_easy, easy, neutral_easy, neutral_hard, hard, very_hard

    "dot_ball_streak": [0, 1, 3, 5, 8, 99],
    # 0, 1-2, 3-4, 5-7, 8+
}

PHASE_BINS = {"powerplay": 0, "middle": 1, "death": 2}

VENUE_TIER_BINS = {"low": 0, "medium": 1, "high": 2}


def _digitize(value: float, bins: list, clip: bool = True) -> int:
    """Return bin index (0-based from right-edge bins)."""
    val = float(value) if value is not None and not np.isnan(float(value)) else 0.0
    if clip:
        val = np.clip(val, bins[0], bins[-2])
    return int(np.digitize(val, bins[1:]))


def encode(state: Dict) -> Tuple:
    """
    Encode a state dictionary into a discrete tuple key.

    Expected state keys:
      - runs_needed (float/int)
      - balls_remaining (float/int)
      - wickets_in_hand (int, 1-10)
      - phase (str: powerplay/middle/death)
      - rrr_crr_delta (float, chase pressure gap)
      - venue_scoring_tier (str: low/medium/high)
      - dot_ball_streak (int)

    Returns: tuple of int bin indices (hashable, lookuable in Q-table)
    """
    state_key = (
        _digitize(state.get("runs_needed", 0),         BINS["runs_needed"]),
        _digitize(state.get("balls_remaining", 120),   BINS["balls_remaining"]),
        _digitize(state.get("wickets_in_hand", 10),    BINS["wickets_in_hand"]),
        PHASE_BINS.get(state.get("phase", "middle"),   1),
        _digitize(state.get("rrr_crr_delta", 0),       BINS["rrr_crr_delta"]),
        VENUE_TIER_BINS.get(state.get("venue_scoring_tier", "medium"), 1),
        _digitize(state.get("dot_ball_streak", 0),     BINS["dot_ball_streak"]),
    )
    return state_key


def encode_vector(state: Dict) -> np.ndarray:
    """
    Encode state as a flat float32 numpy vector for FQI function approximation.
    Normalizes each component to [0, 1] range.
    """
    runs_needed      = np.clip(float(state.get("runs_needed", 0)),       0, 150) / 150
    balls_remaining  = np.clip(float(state.get("balls_remaining", 120)), 0, 120) / 120
    wickets_in_hand  = np.clip(float(state.get("wickets_in_hand", 10)),  1,  10) / 10
    phase_v          = PHASE_BINS.get(state.get("phase", "middle"), 1) / 2
    rrr_crr_delta    = np.clip(float(state.get("rrr_crr_delta", 0)),   -5,   8)
    rrr_crr_norm     = (rrr_crr_delta + 5) / 13   # map [-5, 8] → [0, 1]
    venue_tier_v     = VENUE_TIER_BINS.get(state.get("venue_scoring_tier", "medium"), 1) / 2
    dot_streak       = np.clip(float(state.get("dot_ball_streak", 0)),   0,  12) / 12

    return np.array([
        runs_needed, balls_remaining, wickets_in_hand,
        phase_v, rrr_crr_norm, venue_tier_v, dot_streak
    ], dtype=np.float32)


def encode_action(action: str) -> int:
    """Map action label to integer index."""
    ACTION_MAP = {"conservative": 0, "balanced": 1, "aggressive": 2}
    return ACTION_MAP.get(action, 1)


def decode_action(action_idx: int) -> str:
    """Map action index back to label."""
    REVERSE_MAP = {0: "conservative", 1: "balanced", 2: "aggressive"}
    return REVERSE_MAP.get(action_idx, "balanced")
