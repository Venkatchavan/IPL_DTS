# Model Cards

Detailed documentation for all 4 machine learning models in the platform.

---

## Model 1: Expected Final Score (EFS)

**Type**: Regression  
**Algorithm**: XGBoost (`XGBRegressor`) with fallback to `GradientBoostingRegressor`  
**File**: `pipelines/05_model_training.py` → `models/efs/`

### Purpose

Predicts the projected final innings score given the current in-game state.  
Used to compute **ESA (Expected Score Added)** per delivery.

### Features (12)

| Feature | Description |
|---------|-------------|
| `pre_runs` | Runs scored before this delivery |
| `pre_wickets` | Wickets fallen before this delivery |
| `pre_balls_remaining` | Balls remaining in innings |
| `phase_powerplay` | Binary: is current phase powerplay? |
| `phase_middle` | Binary: is current phase middle? |
| `phase_death` | Binary: is current phase death? |
| `pre_crr` | Current run rate |
| `pre_last_n_balls_rr` | Run rate over last 12 balls |
| `pre_dot_ball_streak` | Current consecutive dot ball count |
| `pre_partnership_runs` | Current partnership runs |
| `venue_scoring_tier` | Encoded venue run-scoring difficulty (0/1/2) |
| `season_normalized` | Season normalized 0–1 |

### Training Strategy

- **Train**: Seasons 2008–2022
- **Validation**: Season 2023
- **Test**: Seasons 2024–2025
- Target: `innings_final_score` (computed per match-inning)

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error on test set |
| RMSE | Root Mean Squared Error on test set |
| Baseline MAE | Linear extrapolation: `(pre_runs / balls_played) × 120` |

### Known Limitations

- Doesn't account for specific batting lineup (no player identity)
- Less accurate in rain-affected or DL-method situations
- Late innings predictions more reliable than early (more state info)

---

## Model 2: Chase Win Probability (CWP)

**Type**: Binary Classification (calibrated)  
**Algorithm**: XGBoost (`XGBClassifier`) + Platt scaling (`CalibratedClassifierCV(method='sigmoid')`)  
**File**: `pipelines/05_model_training.py` → `models/cwp/`

### Purpose

Real-time estimate of the probability that the batting team (inning 2) wins.  
Used to compute **WPA (Win Probability Added)** per delivery.

### Features (12)

| Feature | Description |
|---------|-------------|
| `pre_runs_needed` | Runs still required to win |
| `pre_balls_remaining` | Balls remaining |
| `pre_wickets_in_hand` | Wickets not yet fallen |
| `pre_rrr` | Required run rate |
| `pre_crr` | Current run rate |
| `pre_rrr_crr_delta` | RRR − CRR (deficit indicator) |
| `phase_powerplay/middle/death` | Binary phase indicators |
| `pre_last_n_balls_rr` | Recent run rate (last 12 balls) |
| `venue_scoring_tier` | Encoded venue run-scoring environment |
| `season_normalized` | Season (trend capture) |

### Training Strategy

- **Context**: Inning 2 only
- **Calibration**: Fit on validation set (Season 2023) using Platt scaling
- **Class balance**: Approximately 50/50 by design (win/loss)

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Brier Score | Mean squared error of probability estimates (lower = better) |
| Log-Loss | Cross-entropy of predicted probabilities |
| AUC-ROC | Discrimination ability |
| Calibration curve | Reliability of probability magnitudes |

### Known Limitations

- Inning 1 model not included — WPA uses ESA for inning 1
- Does not account for player quality in lineup
- Probability estimates less reliable for extremely rare states

---

## Model 3: Ball-Level Wicket Probability (BWP)

**Type**: Binary Classification (calibrated)  
**Algorithm**: XGBoost + isotonic regression calibration  
**File**: `pipelines/05_model_training.py` → `models/bwp/`

### Purpose

Estimates the probability that a given delivery results in a wicket.  
Used upstream in ESA/WPA computation and for matchup decisions.

### Features

Bowler historical stats (economy, dot%, strike rate), batter historical stats, 
matchup strikes (if available), phase, pressure band, inning.

### Class Imbalance Handling

Wickets occur on ~7% of legal deliveries.  
Addressed via `scale_pos_weight = (1 − wicket_rate) / wicket_rate` in XGBoost.

### Calibration

Isotonic regression (more flexible than Platt for asymmetric distributions with extreme class imbalance).

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Brier Score | Probability calibration quality |
| Average Precision | Area under precision-recall curve |
| Calibration at 5%, 10%, 15% | Reliability at action-relevant thresholds |

---

## Model 4: Phase Collapse Risk (PCR)

**Type**: Binary Classification (calibrated)  
**Algorithm**: XGBoost + Platt scaling  
**File**: `pipelines/05_model_training.py` → `models/pcr/`

### Purpose

Predicts the probability that a batting team loses ≥3 wickets within the current phase.  
Used in the Decision Audit tab and as a risk signal in team/state analysis.

### Target Definition

`collapse = 1` if 3 or more wickets fall in the current phase for this match-inning, else `0`.  
Computed at phase entry (first ball of the phase).

### Features

Phase-entry wickets, recent wicket rate (last 3 overs), bowler quality aggregates, 
batting team collapse history, venue.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Brier Score | Probability calibration |
| AUC-ROC | Discrimination |
| Precision@0.3 | Practical relevance at alert threshold |

---

## General Notes

### Saved Artifacts

Each model is saved as:
```
models/{name}/
  model.pkl                 ← trained estimator (joblib)
  evaluation_metrics.json   ← test-set metrics
```

### Loading Models

```python
import joblib
model = joblib.load("models/cwp/model.pkl")
```

### Versioning

Model artifacts are not committed to git (too large).  
Reproduce by running `pipelines/05_model_training.py` from scratch.
