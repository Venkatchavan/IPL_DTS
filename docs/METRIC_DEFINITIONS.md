# Metric Definitions

Authoritative reference for all 12 computed metrics in the T20 Decision Intelligence Platform.

---

## 1. Pressure Index (PI)

**Type**: Per-delivery  
**Range**: 0–100  
**File**: `metrics/pressure_index.py`

### Formula

```
PI = α × norm(RRR − CRR)  +  β × norm(Wickets Lost)  +  γ × Phase_Weight  +  δ × norm(Dot_Streak)
```

Default weights: α=0.35, β=0.30, γ=0.20, δ=0.15

### Component Details

| Component | Normalization | Notes |
|-----------|--------------|-------|
| RRR − CRR | clamp(x, −4, 8) → (x − −4) / 12 | Inning 2 only; inning 1 redistributes this weight to β |
| Wickets Lost | `wkts / 10` | Linear 0–10 |
| Phase Weight | powerplay=0.40, middle=0.65, death=0.90 | Captures phase sensitivity |
| Dot Streak | clamp(x, 0, 12) / 12 | Recent consecutive dot balls |

### Interpretation

- 0–30: Low pressure
- 30–60: Moderate pressure
- 60–80: High pressure
- 80–100: Critical pressure

---

## 2. State Difficulty Score (SDS)

**Type**: Per-delivery  
**Range**: 0–100  
**File**: `metrics/state_difficulty.py`

### Formula

```
SDS = 0.30 × norm(RRR)  +  0.25 × norm(Wickets)  +  0.20 × Phase_Difficulty  +  0.15 × norm(Dot_Streak)  +  0.10 × Venue_Difficulty
```

### Phase Difficulty Values

| Phase | Score |
|-------|-------|
| Powerplay | 0.40 |
| Middle | 0.65 |
| Death | 0.90 |

### Interpretation

Higher SDS = harder match state to bat in.  
Use alongside PI to separate "pressure" (match context) from "difficulty" (inherent state challenge).

---

## 3. Win Probability Added (WPA)

**Type**: Per-delivery  
**Context**: Inning 2 primary; inning 1 uses ESA instead  
**File**: `metrics/wpa.py`

### Formula

```
WPA_batter = post_win_prob − pre_win_prob
WPA_bowler = pre_win_prob − post_win_prob
```

`pre_win_prob` / `post_win_prob` from Model 2 (CWP).

### High-Leverage Threshold

`|WPA| > 0.05` (5% swing) flags a delivery as high-leverage.

### Aggregation

Player-level: `WPA_total = Σ WPA_per_delivery` (across all innings 2 appearances).

---

## 4. Expected Score Added (ESA)

**Type**: Per-delivery  
**Context**: Inning 1 primary  
**File**: `metrics/esa.py`

### Formula

```
ESA_batter = post_expected_score − pre_expected_score
ESA_bowler = pre_expected_score − post_expected_score
```

`pre_expected_score` / `post_expected_score` from Model 1 (EFS).

---

## 5. Contextual Economy Rate (CER)

**Type**: Bowler-level  
**File**: `metrics/contextual_economy.py`

### Formula

```
CER = raw_economy − venue_phase_avg_economy
```

Where `venue_phase_avg_economy` is the historical average economy at the same venue in the same phase.

### Interpretation

- CER < 0: Bowler conceding less than venue average → good
- CER > 0: Bowler conceding more than venue average → poor (in context)

---

## 6. Death Suppression Index (DSI)

**Type**: Bowler-level  
**Range**: 0–100  
**Minimum**: 50 death-over balls bowled  
**File**: `metrics/contextual_economy.py`

### Formula

```
DSI = 0.30 × econ_norm  +  0.25 × dot_norm  +  0.25 × boundary_suppression  +  0.20 × wicket_norm
```

Where all components are normalized 0–1 relative to all death bowlers in the dataset.

### Confidence Levels

| Support (death balls) | Confidence |
|-----------------------|-----------|
| ≥ 100 | high |
| 50–99 | medium |
| < 50 | low (suppressed in display) |

---

## 7. Matchup Leverage Score (MLS)

**Type**: Batter–bowler pair  
**File**: `metrics/matchup_leverage.py`

### Formula

```
MLS = (batter_SR_vs_bowler − mean_SR_vs_bowler_bucket) / std_SR_vs_bowler_bucket
```

Bowler style buckets are assigned via K-Means clustering (k=5) on:
- economy rate
- dot ball %
- boundary rate
- wicket rate

### Interpretation

- MLS > 1.5: Strong batter advantage (exploitable)
- MLS < −1.5: Strong bowler advantage (exploitable)

Minimum 30 balls required; otherwise confidence = low.

---

## 8. Boundary Conversion Rate (BCR)

**Type**: Batter-level  
**File**: Computed in `pipelines/03_feature_engineering.py`

### Formula

```
BCR = (balls where batsman_runs ∈ {4, 6}) / (balls where batsman_runs ≥ 1)
```

Measures "quality of contact" — how often attacking shots become boundaries.

---

## 9. Venue Sensitivity Index (VSI)

**Type**: Bowler-level  
**File**: `pipelines/04_metrics_compute.py`

### Formula

```
VSI = std_dev(economy_rate_per_venue) / mean_economy_rate
```

High VSI = bowler is inconsistent across venues (environment-sensitive).  
Low VSI = bowler performs similarly at all venues.

---

## 10. Phase Collapse Risk (PCR)

**Type**: Per-phase-entry, team-level  
**Output**: Probability 0–1  
**File**: Model 4 in `pipelines/05_model_training.py`

Model probability that ≥3 wickets fall in the current phase (powerplay / middle / death).

Input features: wickets at phase entry, recent wickets rate, bowler quality, batting team history.

---

## 11. Partnership Run Rate (PRR)

**Type**: Per-ball (running)  
**File**: `pipelines/02_state_reconstruction.py`

```
PRR = partnership_runs / partnership_balls × 6
```

Computed during state reconstruction. Resets on every wicket.

---

## 12. Control Rate

**Type**: Bowler-level  
**File**: `metrics/contextual_economy.py`

```
Control_Rate = % of deliveries where batsman_runs ≤ 1
```

Higher control rate = more "dot-or-single" deliveries → bowling dominance.
