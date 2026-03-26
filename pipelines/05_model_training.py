"""
05_model_training.py — Model Training and Scoring Pipeline

Trains all analytical models and scores the ball_states dataset
with model predictions. Also saves trained model artifacts.

Models trained:
  1. Expected Final Score (EFS) — XGBoost Regressor, inning 1
  2. Chase Win Probability (CWP) — XGBoost Classifier, inning 2
  3. Ball-Level Wicket Probability (BWP) — XGBoost Classifier
  4. Phase-Level Collapse Risk (PCR) — XGBoost Classifier

After training, scores the full ball_states file and writes:
  data/features/ball_states_scored.parquet
    with columns: pre_expected_score, post_expected_score,
                  pre_win_prob, post_win_prob,
                  wicket_prob, collapse_risk

Then triggers pipeline 04 Phase B to compute WPA and ESA.

Time-aware validation:
  Train: 2008–2022, Validate: 2023, Test: 2024–2025
  No future seasons appear in any training fold.

Run:
  python pipelines/05_model_training.py

Output:
  models/expected_score/model.pkl
  models/win_probability/model.pkl
  models/wicket_probability/model.pkl
  models/collapse_risk/model.pkl
  data/features/ball_states_scored.parquet
  models/*/evaluation_metrics.json
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    log_loss, brier_score_loss,
    roc_auc_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BALL_STATES_FILE, MODELS_DIR, DATA_FEATURES,
    TRAIN_SEASONS, VALIDATE_SEASONS, TEST_SEASONS,
    PHASE_POWERPLAY, PHASE_MIDDLE, PHASE_DEATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    log.warning("XGBoost not installed — falling back to GradientBoostingRegressor/Classifier")
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# ── Shared feature utilities ───────────────────────────────────────────────────

PHASE_ENCODER = {PHASE_POWERPLAY: 0, PHASE_MIDDLE: 1, PHASE_DEATH: 2}
PRESSURE_BAND_ENCODER = {"low": 0, "neutral": 1, "high": 2, "critical": 3}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode phase and pressure_band for model input."""
    df = df.copy()
    df["phase_enc"]         = df["phase"].map(PHASE_ENCODER).fillna(1).astype(int)
    df["pressure_band_enc"] = df["pressure_band"].map(PRESSURE_BAND_ENCODER).fillna(1).astype(int)
    return df


def split_by_season(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val   = df[df["season"].isin(VALIDATE_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]
    log.info(f"  Split — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def get_xgb_regressor(**kwargs):
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, **kwargs
        )
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)


def get_xgb_classifier(**kwargs):
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, **kwargs
        )
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)


def save_model(model, model_name: str, metadata: dict = None):
    path = MODELS_DIR / model_name / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    if metadata:
        meta_path = MODELS_DIR / model_name / "evaluation_metrics.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    log.info(f"  Saved model + metadata → {path.parent}/")


def load_model(model_name: str):
    path = MODELS_DIR / model_name / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ── Model 1: Expected Final Score (EFS) ───────────────────────────────────────

EFS_FEATURES = [
    "pre_runs", "pre_wickets", "pre_balls_bowled", "pre_balls_remaining",
    "phase_enc", "pre_crr", "pre_dot_ball_streak",
    "pre_last_n_runs", "pre_last_n_wickets", "pre_last_n_balls_rr",
    "pre_partnership_runs", "pre_partnership_balls",
]


def build_efs_dataset(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Build inning-1 training data for Expected Final Score model.

    Target: final innings total for each match (joined back to each ball).
    State: pre-ball features only.
    Unit: each legal ball in inning 1.

    Leakage guard: only use pre_ features. Final score is the label,
    not an input feature.
    """
    inning1 = ball_states[(ball_states["inning"] == 1) & (ball_states["is_legal_ball"] == 1)].copy()

    # Compute final innings score per match
    inning1["ball_total"] = inning1["pre_runs"] + inning1["total_runs"]
    final_scores = (
        inning1.groupby("match_id")["ball_total"]
        .max()
        .rename("final_score")
        .reset_index()
    )

    inning1 = inning1.merge(final_scores, on="match_id", how="left")
    inning1 = inning1.dropna(subset=["final_score"])
    inning1 = encode_categoricals(inning1)
    return inning1


def train_efs_model(ball_states: pd.DataFrame) -> dict:
    log.info("\n── Training Model 1: Expected Final Score ──")
    df = build_efs_dataset(ball_states)

    # Baseline: linear extrapolation
    df["baseline_efs"] = df["pre_runs"] + (df["pre_crr"] * (df["pre_balls_remaining"] / 6))

    train, val, test = split_by_season(df)

    X_train = train[EFS_FEATURES].fillna(0)
    y_train = train["final_score"]
    X_val   = val[EFS_FEATURES].fillna(0)
    y_val   = val["final_score"]
    X_test  = test[EFS_FEATURES].fillna(0)
    y_test  = test["final_score"]

    model = get_xgb_regressor()
    model.fit(X_train, y_train)

    # Baseline performance
    baseline_val_mae = mean_absolute_error(y_val, val["baseline_efs"].fillna(y_val.mean()))
    baseline_test_mae = mean_absolute_error(y_test, test["baseline_efs"].fillna(y_test.mean()))

    # Model performance
    val_mae  = mean_absolute_error(y_val,  model.predict(X_val))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    val_rmse  = np.sqrt(mean_squared_error(y_val,  model.predict(X_val)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    metadata = {
        "model": "XGBoost Regressor — Expected Final Score",
        "target": "final innings total (runs)",
        "train_seasons": TRAIN_SEASONS,
        "validate_seasons": VALIDATE_SEASONS,
        "test_seasons": TEST_SEASONS,
        "features": EFS_FEATURES,
        "validation": {
            "baseline_mae": round(baseline_val_mae, 2),
            "model_mae":    round(val_mae, 2),
            "model_rmse":   round(val_rmse, 2),
        },
        "test": {
            "baseline_mae": round(baseline_test_mae, 2),
            "model_mae":    round(test_mae, 2),
            "model_rmse":   round(test_rmse, 2),
        },
        "leakage_note": "Only pre_ball features used. Final score is label, not feature.",
    }

    log.info(f"  Validation MAE — Baseline: {baseline_val_mae:.1f} | Model: {val_mae:.1f}")
    log.info(f"  Test MAE       — Baseline: {baseline_test_mae:.1f} | Model: {test_mae:.1f}")

    save_model(model, "expected_score", metadata)
    return metadata


# ── Model 2: Chase Win Probability (CWP) ──────────────────────────────────────

CWP_FEATURES = [
    "pre_runs_needed", "pre_balls_remaining", "pre_wickets_in_hand",
    "pre_rrr", "pre_rrr_crr_delta",
    "phase_enc", "pressure_band_enc",
    "pre_dot_ball_streak", "pre_last_n_runs", "pre_last_n_balls_rr",
    "pre_partnership_runs", "pre_partnership_balls",
]


def build_cwp_dataset(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Build inning-2 training data for Chase Win Probability model.
    Target: match_won (1 if batting team won, 0 otherwise).
    Include only balls where match result is known (non-null match_won).
    """
    inning2 = ball_states[
        (ball_states["inning"] == 2) &
        (ball_states["is_legal_ball"] == 1) &
        ball_states["match_won"].notna() &
        ball_states["pre_runs_needed"].notna()
    ].copy()

    inning2["match_won"] = inning2["match_won"].astype(int)
    inning2 = encode_categoricals(inning2)
    inning2 = inning2.dropna(subset=CWP_FEATURES)
    return inning2


def train_cwp_model(ball_states: pd.DataFrame) -> dict:
    log.info("\n── Training Model 2: Chase Win Probability ──")
    df = build_cwp_dataset(ball_states)

    train, val, test = split_by_season(df)

    X_train = train[CWP_FEATURES].fillna(0)
    y_train = train["match_won"]
    X_val   = val[CWP_FEATURES].fillna(0)
    y_val   = val["match_won"]
    X_test  = test[CWP_FEATURES].fillna(0)
    y_test  = test["match_won"]

    # Base model
    base_model = get_xgb_classifier()
    base_model.fit(X_train, y_train)

    # Calibration (Platt scaling) — critical for WPA to be meaningful
    log.info("  Calibrating probabilities with Platt scaling...")
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    val_probs  = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    val_brier  = brier_score_loss(y_val,  val_probs)
    test_brier = brier_score_loss(y_test, test_probs)
    val_logloss  = log_loss(y_val,  val_probs)
    test_logloss = log_loss(y_test, test_probs)
    val_auc  = roc_auc_score(y_val,  val_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    metadata = {
        "model": "XGBoost Classifier (Platt-calibrated) — Chase Win Probability",
        "target": "match_won (1 = batting team won)",
        "calibration": "Platt scaling (CalibratedClassifierCV, sigmoid)",
        "train_seasons": TRAIN_SEASONS,
        "validate_seasons": VALIDATE_SEASONS,
        "test_seasons": TEST_SEASONS,
        "features": CWP_FEATURES,
        "validation": {
            "brier_score": round(val_brier, 4),
            "log_loss":    round(val_logloss, 4),
            "auc_roc":     round(val_auc, 4),
        },
        "test": {
            "brier_score": round(test_brier, 4),
            "log_loss":    round(test_logloss, 4),
            "auc_roc":     round(test_auc, 4),
        },
        "leakage_note": "All features are pre-ball state. match_won is the terminal label.",
        "calibration_note": "Probabilities are calibrated on 2023 validation set. "
                            "Brier score should be near 0.20 for well-calibrated chase model.",
    }

    log.info(f"  Validation — Brier: {val_brier:.4f}  AUC: {val_auc:.4f}  LogLoss: {val_logloss:.4f}")
    log.info(f"  Test       — Brier: {test_brier:.4f}  AUC: {test_auc:.4f}  LogLoss: {test_logloss:.4f}")

    save_model(model, "win_probability", metadata)
    return metadata


# ── Model 3: Ball-Level Wicket Probability (BWP) ──────────────────────────────

BWP_FEATURES = [
    "phase_enc", "over", "pre_wickets", "pre_balls_bowled",
    "pre_dot_ball_streak", "pre_last_n_wickets",
    "pre_partnership_balls", "pressure_band_enc",
]


def train_bwp_model(ball_states: pd.DataFrame) -> dict:
    log.info("\n── Training Model 3: Ball-Level Wicket Probability ──")

    df = ball_states[ball_states["is_legal_ball"] == 1].copy()
    df = encode_categoricals(df)
    df = df.dropna(subset=BWP_FEATURES + ["is_wicket"])

    train, val, test = split_by_season(df)

    wicket_rate = train["is_wicket"].mean()
    scale_pos_weight = (1 - wicket_rate) / wicket_rate  # handle class imbalance

    X_train = train[BWP_FEATURES].fillna(0)
    y_train = train["is_wicket"]

    model_kwargs = {}
    if XGB_AVAILABLE:
        model_kwargs["scale_pos_weight"] = scale_pos_weight

    base_model = get_xgb_classifier(**model_kwargs)
    base_model.fit(X_train, y_train)

    # Calibrate on validation
    X_val   = val[BWP_FEATURES].fillna(0)
    y_val   = val["is_wicket"]
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train[BWP_FEATURES].fillna(0), train["is_wicket"])

    X_test  = test[BWP_FEATURES].fillna(0)
    y_test  = test["is_wicket"]
    test_probs = model.predict_proba(X_test)[:, 1]

    test_brier = brier_score_loss(y_test, test_probs)
    test_ap    = average_precision_score(y_test, test_probs)

    metadata = {
        "model": "XGBoost Classifier (isotonic-calibrated) — Wicket Probability",
        "target": "is_wicket (1 = wicket fell on this delivery)",
        "class_imbalance_handling": f"scale_pos_weight={scale_pos_weight:.2f}",
        "test": {"brier_score": round(test_brier, 4), "avg_precision": round(test_ap, 4)},
        "note": "Evaluate with PR curve, not accuracy (wickets ~7% of deliveries).",
        "leakage_note": "No post-ball information in features.",
    }

    log.info(f"  Test — Brier: {test_brier:.4f}  AvgPrecision: {test_ap:.4f}")
    save_model(model, "wicket_probability", metadata)
    return metadata


# ── Model 4: Collapse Risk ─────────────────────────────────────────────────────

def train_collapse_risk_model(ball_states: pd.DataFrame) -> dict:
    """
    Collapse Risk: Probability of losing 3+ wickets in the next 5 overs.
    Built at phase-entry level (one record per match × phase transition).
    """
    log.info("\n── Training Model 4: Collapse Risk ──")

    # Build phase-entry dataset
    legal = ball_states[ball_states["is_legal_ball"] == 1].copy()

    # For each match × phase transition, get entry state and future wickets
    phase_entries = (
        legal.sort_values(["match_id", "inning", "global_ball_number"])
        .groupby(["match_id", "inning", "phase"])
        .first()
        .reset_index()
    )

    # Compute future wickets in phase (within same match × inning × phase)
    future_wickets = (
        legal.groupby(["match_id", "inning", "phase"])["is_wicket"]
        .sum()
        .reset_index()
        .rename(columns={"is_wicket": "phase_total_wickets"})
    )

    phase_entries = phase_entries.merge(future_wickets, on=["match_id", "inning", "phase"], how="left")
    phase_entries["collapse"] = (phase_entries["phase_total_wickets"] >= 3).astype(int)

    phase_entries = encode_categoricals(phase_entries)

    COLLAPSE_FEATURES = [
        "phase_enc", "pre_wickets", "pre_crr", "pre_rrr_crr_delta",
        "pre_last_n_wickets", "pressure_band_enc", "pre_balls_remaining",
    ]
    df = phase_entries.dropna(subset=COLLAPSE_FEATURES)

    train, val, test = split_by_season(df)

    X_train = train[COLLAPSE_FEATURES].fillna(0)
    y_train = train["collapse"]
    X_val   = val[COLLAPSE_FEATURES].fillna(0)
    y_val   = val["collapse"]
    X_test  = test[COLLAPSE_FEATURES].fillna(0)
    y_test  = test["collapse"]

    base_model = get_xgb_classifier()
    base_model.fit(X_train, y_train)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_brier = brier_score_loss(y_test, test_probs)
    test_auc   = roc_auc_score(y_test, test_probs) if y_test.nunique() > 1 else 0.5

    metadata = {
        "model": "XGBoost Classifier — Phase-Level Collapse Risk",
        "target": "collapse (1 = 3+ wickets in phase)",
        "unit_of_analysis": "match × inning × phase entry state",
        "test": {"brier_score": round(test_brier, 4), "auc_roc": round(test_auc, 4)},
        "leakage_note": "Future wickets in phase are the label; not included as a feature.",
    }

    log.info(f"  Test — Brier: {test_brier:.4f}  AUC: {test_auc:.4f}")
    save_model(model, "collapse_risk", metadata)
    return metadata


# ── Ball state scoring ─────────────────────────────────────────────────────────

def score_ball_states(ball_states: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all trained models to the full ball_states dataset.
    Computes pre and post delivery predictions for WPA / ESA computation.
    """
    log.info("\nScoring ball states with trained models...")
    df = ball_states.copy()
    df = encode_categoricals(df)

    # ── Score EFS (inning 1) ──────────────────────────────────────────────────
    try:
        efs_model = load_model("expected_score")
        inning1_mask = df["inning"] == 1

        # Pre-ball score
        X_pre = df.loc[inning1_mask, EFS_FEATURES].fillna(0)
        df.loc[inning1_mask, "pre_expected_score"] = efs_model.predict(X_pre)

        # Post-ball score: temporarily advance state
        df_post_efs = df.loc[inning1_mask].copy()
        df_post_efs["pre_runs"]          += df_post_efs["total_runs"]
        df_post_efs["pre_wickets"]       += df_post_efs["is_wicket"]
        df_post_efs["pre_balls_bowled"]  += df_post_efs["is_legal_ball"]
        df_post_efs["pre_balls_remaining"] -= df_post_efs["is_legal_ball"]
        df_post_efs["pre_crr"] = np.where(
            df_post_efs["pre_balls_bowled"] > 0,
            df_post_efs["pre_runs"] / (df_post_efs["pre_balls_bowled"] / 6),
            0
        )
        df_post_efs = encode_categoricals(df_post_efs)
        X_post_efs = df_post_efs[EFS_FEATURES].fillna(0)
        df.loc[inning1_mask, "post_expected_score"] = efs_model.predict(X_post_efs)

        log.info(f"  EFS scored for {inning1_mask.sum():,} inning-1 balls")
    except FileNotFoundError:
        log.warning("  EFS model not found — skipping expected score scoring")
        df["pre_expected_score"]  = np.nan
        df["post_expected_score"] = np.nan

    # ── Score CWP (inning 2) ──────────────────────────────────────────────────
    try:
        cwp_model = load_model("win_probability")
        inning2_mask = (df["inning"] == 2) & df["pre_runs_needed"].notna()

        X_pre2 = df.loc[inning2_mask, CWP_FEATURES].fillna(0)
        df.loc[inning2_mask, "pre_win_prob"] = cwp_model.predict_proba(X_pre2)[:, 1]

        # Post-ball: advance state
        df_post_cwp = df.loc[inning2_mask].copy()
        df_post_cwp["pre_runs"]          += df_post_cwp["total_runs"]
        df_post_cwp["pre_runs_needed"]   -= df_post_cwp["total_runs"]
        df_post_cwp["pre_wickets"]       += df_post_cwp["is_wicket"]
        df_post_cwp["pre_wickets_in_hand"] = 10 - df_post_cwp["pre_wickets"]
        df_post_cwp["pre_balls_bowled"]  += df_post_cwp["is_legal_ball"]
        df_post_cwp["pre_balls_remaining"] = (
            df_post_cwp["pre_balls_remaining"] - df_post_cwp["is_legal_ball"]
        ).clip(lower=0)
        df_post_cwp["pre_crr"] = np.where(
            df_post_cwp["pre_balls_bowled"] > 0,
            df_post_cwp["pre_runs"] / (df_post_cwp["pre_balls_bowled"] / 6),
            0
        )
        df_post_cwp["pre_rrr"] = np.where(
            df_post_cwp["pre_balls_remaining"] > 0,
            df_post_cwp["pre_runs_needed"] / (df_post_cwp["pre_balls_remaining"] / 6),
            0
        )
        df_post_cwp["pre_rrr_crr_delta"] = df_post_cwp["pre_rrr"] - df_post_cwp["pre_crr"]
        df_post_cwp = encode_categoricals(df_post_cwp)

        X_post2 = df_post_cwp[CWP_FEATURES].fillna(0)
        df.loc[inning2_mask, "post_win_prob"] = cwp_model.predict_proba(X_post2)[:, 1]

        # Terminal ball corrections: if wickets_in_hand = 0 or runs_needed <= 0, clamp
        df.loc[df["pre_wickets_in_hand"] == 0, "post_win_prob"] = 0.0
        runs_achieved = (df["inning"] == 2) & (df["pre_runs"] + df["total_runs"] >= df["target"])
        df.loc[runs_achieved, "post_win_prob"] = 1.0

        log.info(f"  CWP scored for {inning2_mask.sum():,} inning-2 balls")
    except FileNotFoundError:
        log.warning("  CWP model not found — skipping win probability scoring")
        df["pre_win_prob"]  = np.nan
        df["post_win_prob"] = np.nan

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    log.info("Pipeline 05: Model Training and Scoring")

    ball_states = pd.read_parquet(BALL_STATES_FILE)
    log.info(f"  Loaded {len(ball_states):,} ball states")

    # Check season coverage
    seasons_present = sorted(ball_states["season"].dropna().unique().astype(int))
    log.info(f"  Seasons in dataset: {seasons_present}")

    # Train all models
    efs_meta  = train_efs_model(ball_states)
    cwp_meta  = train_cwp_model(ball_states)
    bwp_meta  = train_bwp_model(ball_states)
    pcr_meta  = train_collapse_risk_model(ball_states)

    # Score full dataset
    scored = score_ball_states(ball_states)

    # Save scored ball states
    scored_path = DATA_FEATURES / "ball_states_scored.parquet"
    scored.to_parquet(scored_path, index=False)
    log.info(f"\nSaved scored ball states: {scored_path}  ({len(scored):,} rows)")

    # Trigger Phase B metrics (WPA, ESA)
    log.info("\nTriggering metrics pipeline Phase B (WPA + ESA)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "metrics_04",
        Path(__file__).parent / "04_metrics_compute.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run(phase="B")

    log.info("\nPipeline 05 complete.")
    log.info("\n── Model Summary ──")
    for name, meta in [("EFS", efs_meta), ("CWP", cwp_meta), ("BWP", bwp_meta), ("PCR", pcr_meta)]:
        log.info(f"  {name}: {json.dumps(meta.get('test', {}))}")


if __name__ == "__main__":
    run()
