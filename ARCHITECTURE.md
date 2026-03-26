# IPL INTEL — T20 Decision Intelligence Platform
## Full Analytical Architecture Blueprint
**Version:** 1.0 | **Dataset:** IPL 2008–2025 Ball-by-Ball + Match-Level | **Status:** Design Phase

---

## 0. Platform Positioning

This is not a cricket dashboard. It is a **T20 Decision Intelligence Platform** — a public-grade
analytical product designed to answer questions that teams, analysts, and performance consultancies
actually care about:

- Which lineup decisions were suboptimal relative to historical state expectations?
- Which players add win probability in high-difficulty match states?
- What does a high-quality death-over bowling resource allocation look like?
- Which roles are replaceable without loss of strategic function?
- What chase strategy does historical data support in specific state conditions?

Every module, metric, model, and chart must be traceable to a decision, a question, or an
analytical purpose. Decorative analytics are explicitly excluded.

---

## 1. Repository Structure

```
ipl-intel/
│
├── data/
│   ├── raw/                        # Original Kaggle CSVs (unmodified)
│   ├── processed/                  # Validated, typed, schema-enforced parquet files
│   ├── features/                   # Engineered feature stores (ball-level, over-level, innings-level)
│   └── schema/
│       ├── ball_by_ball_schema.json
│       └── match_schema.json
│
├── pipelines/
│   ├── 01_ingest_validate.py       # Load raw CSVs, enforce schema, flag anomalies
│   ├── 02_state_reconstruction.py  # Ball-by-ball match state engine
│   ├── 03_feature_engineering.py   # All feature derivation logic
│   ├── 04_metrics_compute.py       # All analytical metric scoring
│   └── 05_model_training.py        # Model training entry point
│
├── models/
│   ├── expected_score/             # Expected final score model (regression)
│   ├── win_probability/            # Chase win probability model (classification)
│   ├── wicket_probability/         # Ball-level wicket risk model
│   ├── collapse_risk/              # Phase-level collapse model
│   ├── role_clustering/            # Unsupervised role assignment
│   ├── replacement_similarity/     # Player role-replacement engine
│   └── rl_strategy/                # Offline RL chase policy module
│       ├── state_encoder.py
│       ├── reward_function.py
│       ├── fqi_trainer.py          # Fitted Q-Iteration
│       └── policy_evaluator.py
│
├── metrics/
│   ├── wpa.py                      # Win Probability Added
│   ├── esa.py                      # Expected Score Added
│   ├── pressure_index.py
│   ├── state_difficulty.py
│   ├── collapse_risk.py
│   ├── contextual_economy.py
│   ├── phase_dependability.py
│   └── matchup_leverage.py
│
├── app/
│   ├── main.py                     # Streamlit entry point
│   ├── pages/
│   │   ├── 01_executive_home.py
│   │   ├── 02_match_state_engine.py
│   │   ├── 03_team_dna.py
│   │   ├── 04_player_value.py
│   │   ├── 05_pressure_profiles.py
│   │   ├── 06_matchup_intelligence.py
│   │   ├── 07_decision_audit.py
│   │   ├── 08_scouting_role_fit.py
│   │   ├── 09_strategy_lab.py
│   │   └── 10_methodology.py
│   ├── components/                 # Reusable UI components (charts, filters, tooltips)
│   └── config.py                   # App-level constants, color palette, defaults
│
├── notebooks/
│   ├── EDA_01_data_validation.ipynb
│   ├── EDA_02_state_reconstruction_audit.ipynb
│   ├── EDA_03_metric_distributions.ipynb
│   ├── MODEL_01_win_probability.ipynb
│   ├── MODEL_02_expected_score.ipynb
│   ├── MODEL_03_collapse_risk.ipynb
│   ├── RL_01_offline_policy_design.ipynb
│   └── PORTFOLIO_summary.ipynb
│
├── tests/
│   ├── test_state_reconstruction.py
│   ├── test_metrics.py
│   └── test_feature_engineering.py
│
├── docs/
│   ├── METRIC_DEFINITIONS.md
│   ├── MODEL_CARDS.md
│   ├── RL_DESIGN_NOTES.md
│   └── ANALYTICAL_DECISIONS.md
│
├── ARCHITECTURE.md                 # This document
├── README.md
├── requirements.txt
├── Dockerfile
└── .streamlit/
    └── config.toml
```

---

## 2. Data Pipeline

### 2.1 Source Data

**Dataset:** IPL 2008–2025 (Kaggle — chaitu20/ipl-dataset2008-2025)

Expected files:
- `matches.csv` — match-level metadata (teams, toss, venue, result, season)
- `deliveries.csv` — ball-by-ball records (inning, over, ball, batsman, bowler, runs, extras, dismissal)

### 2.2 Schema Enforcement

**matches.csv critical columns:**

| Column | Type | Notes |
|---|---|---|
| match_id | int | Primary key |
| season | int | 2008–2025 |
| date | date | Match date |
| venue | string | Normalize venue names (aliases exist) |
| team1 / team2 | string | Normalize franchise names across rebrands |
| toss_winner / toss_decision | string | |
| winner | string | Nullable (ties, no-results) |
| result_margin | float | Runs or wickets |
| target_runs | int | First-innings total + 1 |

**deliveries.csv critical columns:**

| Column | Type | Notes |
|---|---|---|
| match_id | int | FK to matches |
| inning | int | 1 or 2 |
| over | int | 0-indexed or 1-indexed — validate |
| ball | int | Ball within over |
| batsman / non_striker | string | Normalize player names |
| bowler | string | |
| batsman_runs | int | |
| extra_runs | int | |
| total_runs | int | |
| extras_type | string | wide / noball / bye / legbye |
| is_wicket | bool/int | |
| dismissal_kind | string | |
| player_dismissed | string | |

### 2.3 Validation Checks (non-negotiable)

```python
# pipeline/01_ingest_validate.py responsibilities:

# 1. Over count integrity: each innings should have <= 20 overs
# 2. Ball integrity: 1–6 per over (7+ only valid on extras)
# 3. No orphaned match_ids: all delivery match_ids must exist in matches
# 4. Score continuity: cumulative runs must be monotonic per innings
# 5. Dismissal consistency: is_wicket=1 must have dismissal_kind and player_dismissed filled
# 6. Venue normalization: map known aliases (e.g. "Eden Gardens" vs "Eden Garden")
# 7. Team name normalization: map franchise renames (e.g. "Delhi Daredevils" -> "Delhi Capitals")
# 8. Season derivation: extract from date if season column has gaps
# 9. Target validation: second-innings target must equal first-innings total + 1
# 10. Duplicate ball detection: flag repeated (match_id, inning, over, ball) combos
```

### 2.4 State Reconstruction Engine

This is the core data artifact. Every analysis derives from the ball-level match state.

For each ball, compute and store:

```python
ball_state = {
    # Identifiers
    "match_id": int,
    "inning": int,               # 1 or 2
    "over": int,                 # 1–20
    "ball": int,                 # 1–6 (+extras)
    "global_ball_number": int,   # 1–120 for legal deliveries

    # Score state
    "runs_scored_so_far": int,
    "wickets_lost": int,
    "balls_bowled_legal": int,
    "balls_remaining_legal": int, # 120 - balls_bowled_legal (inning 1) or target-aware (inning 2)

    # Phase
    "phase": str,                # "powerplay" (1-6), "middle" (7-15), "death" (16-20)

    # Rate state
    "current_run_rate": float,   # runs / overs_bowled
    "required_run_rate": float,  # inning 2 only: runs_needed / overs_remaining
    "target": int,               # inning 2 only
    "runs_needed": int,          # inning 2 only

    # Players
    "striker": str,
    "non_striker": str,
    "bowler": str,

    # Delivery outcome
    "batsman_runs": int,
    "extra_runs": int,
    "total_runs": int,
    "is_wicket": bool,
    "dismissal_kind": str,

    # Rolling context features
    "last_5_balls_runs": int,       # runs in last 5 legal balls
    "last_5_balls_wickets": int,    # wickets in last 5 legal balls
    "dot_ball_streak": int,         # consecutive dots ending at this ball
    "partnership_runs": int,        # current partnership runs
    "partnership_balls": int,       # current partnership legal balls

    # Venue / match context
    "venue": str,
    "season": int,
    "batting_team": str,
    "bowling_team": str,
    "toss_winner": str,
    "toss_decision": str,

    # Derived pressure indicators
    "rrr_crr_delta": float,         # required_run_rate - current_run_rate (inning 2 only)
    "wickets_in_hand": int,         # 10 - wickets_lost

    # Model outputs (populated later in pipeline)
    "expected_final_score": float,  # from Model 1
    "win_probability": float,       # from Model 2 (inning 2 only)
    "wicket_probability": float,    # from Model 3
    "pressure_index": float,        # from Metrics layer
    "state_difficulty": float,      # from Metrics layer
}
```

---

## 3. Feature Engineering

### 3.1 Ball-Level Features

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `phase` | over 1–6: PW, 7–15: MD, 16–20: DT | Phase segmentation |
| `crr` | cumulative_runs / (balls_bowled / 6) | Scoring rate so far |
| `rrr` | runs_needed / (balls_remaining / 6) | Chase pressure proxy |
| `rrr_crr_delta` | rrr - crr | Pressure gap |
| `dot_ball_streak` | count consecutive dots up to this ball | Momentum compression |
| `last_n_balls_rr` | sum(runs, last n balls) / (n/6) | Recent momentum |
| `wickets_in_hand` | 10 - wickets_lost | Resource depth |
| `balls_per_wicket` | balls_bowled / max(wickets_lost, 1) | Batting stability |
| `score_pressure` | runs_needed / wickets_in_hand | Combined resource pressure |
| `partnership_rr` | partnership_runs / (partnership_balls / 6) | Current pair momentum |
| `venue_avg_score_phase` | historical phase avg at venue | Contextual baseline |

### 3.2 Player Aggregation Features

Built at multiple granularities: per-phase, per-venue, per-season, per-pressure-band.

**Pressure bands (for batter context):**
- Low: rrr_crr_delta < -1.5 (batting comfortably ahead)
- Neutral: -1.5 to +1.5
- High: > 1.5 (batting under pressure)
- Critical: > 3.0 (severe pressure)

**Batter features per pressure band:**

| Feature | Definition |
|---|---|
| `sr_pressure_band` | strike rate within pressure band |
| `dot_rate_pressure_band` | dot ball % within pressure band |
| `boundary_rate_pressure_band` | boundary % within pressure band |
| `dismissal_rate_pressure_band` | P(dismissal) within pressure band |
| `wpa_mean` | mean Win Probability Added per ball faced |
| `esa_mean` | mean Expected Score Added per ball |
| `phase_sr_delta` | strike rate in phase vs venue baseline |

**Bowler features per phase:**

| Feature | Definition |
|---|---|
| `economy_phase` | economy rate restricted to phase |
| `dot_ball_pct_phase` | dot ball % in phase |
| `wicket_rate_phase` | wickets per over in phase |
| `boundary_concession_rate` | (4s + 6s) / balls_bowled |
| `control_rate` | dot % + singles % (low-damage delivery rate) |
| `contextual_economy` | economy adjusted for venue scoring baseline |
| `death_suppression_index` | composite death-over suppression score |
| `pressure_creation_score` | dot streak induction rate |

### 3.3 Match-Level Features

| Feature | Definition |
|---|---|
| `venue_avg_first_innings` | historical average first-innings score at venue |
| `pitch_pace_index` | (actual_score - venue_avg) / venue_std (normalized) |
| `toss_bat_win_pct_venue` | historical win% for batting first at venue |
| `team_form_last_5` | win rate in last 5 matches (rolling, time-aware) |
| `bowling_attack_strength` | mean bowler economy (relative to dataset average) |
| `batting_depth_proxy` | top-6 average position-adjusted batting average |

---

## 4. Analytical Metrics Layer

Each metric is an independently computed, stored, and queryable score.

### 4.1 Win Probability Added (WPA)

**Unit of analysis:** Ball  
**Applicable to:** Batter, Bowler (inning 2 primarily, inning 1 derivable from Expected Score path)

```
WPA_ball = WP(state_after_ball) - WP(state_before_ball)
```

Where WP = chase win probability from Model 2.

- Positive WPA for batters: delivery moved win probability up
- Negative WPA for batters: dismissal, dot in critical state
- For bowlers: invert sign — negative WPA output = good bowler ball

**Aggregation:** sum over all balls faced / bowled in a context window

**Why a coach cares:** Identifies players who create or destroy match-winning probability, not just raw output.

---

### 4.2 Expected Score Added (ESA)

**Unit of analysis:** Ball  
**Applicable to:** Batter, Bowler (inning 1 primary)

```
ESA_ball = EFS(state_after_ball) - EFS(state_before_ball)
```

Where EFS = expected final score from Model 1.

Positive ESA: delivery improved projected innings total  
Negative ESA: wasteful delivery (dot, dismissal from Model 1 perspective)

**Why a coach cares:** Measures contribution to innings building in first innings where WP is undefined until target is set.

---

### 4.3 State Difficulty Score (SDS)

**Unit of analysis:** Match state (over + ball + context)

```
SDS = f(
    rrr_crr_delta,         # chase pressure gap
    wickets_in_hand,       # resource depth (inverted)
    phase,                 # death = hardest
    dot_ball_streak,       # incoming pressure
    venue_baseline_delta,  # vs pitch context
    bowler_strength_proxy  # who is bowling
)
```

Normalized 0–100. High SDS = difficult state for batting team.

**Interpretation:** A batter scoring 30 in a state with SDS > 80 is more valuable than one scoring 40 in SDS < 30. Context-adjusted contribution.

---

### 4.4 Pressure Index

**Unit of analysis:** Ball or player delivery window

```
Pressure_Index = alpha * (rrr_crr_delta / max_delta) +
                 beta  * (1 - wickets_in_hand/10) +
                 gamma * (phase_weight) +
                 delta * (dot_ball_streak / max_streak_observed)
```

Weights (alpha=0.35, beta=0.30, gamma=0.20, delta=0.15) — tunable, must be documented.

---

### 4.5 Collapse Risk Score

**Unit of analysis:** Team × phase × state window  
**Output:** Probability of losing 3+ wickets in next 5 overs

Computed from Model 3 (collapse risk classifier). Input features:
- wickets already lost in phase
- phase
- current_rr vs venue_avg_rr
- recent wicket frequency (last 3 overs)
- bowling attack strength

---

### 4.6 Contextual Economy Rate

```
Contextual_Economy = raw_economy - venue_phase_avg_economy
```

A bowler with economy 9.0 in death overs at a high-scoring venue (avg 11.5) is phenomenal.  
A bowler with economy 7.0 at a low-scoring venue in middle overs may be ordinary.  
Raw economy comparisons without venue-phase context are analytically weak.

---

### 4.7 Death Overs Suppression Index (DSI)

**Component of:** Bowler evaluation

```
DSI = w1 * (1 - economy_death / venue_death_avg) +
      w2 * dot_pct_death +
      w3 * wicket_rate_death +
      w4 * (1 - boundary_concession_rate_death)
```

Normalized 0–100. Requires minimum 50 death-over balls to be reportable.

---

### 4.8 Dot-Ball Pressure Conversion Rate (DBPCR)

**Unit of analysis:** Bowler or phase

Measures how often dot-ball streaks result in wickets shortly after.

```
DBPCR = P(wicket within 3 balls | preceded by dot_streak >= 3)
```

High DBPCR indicates a bowler who creates and converts pressure, not just tight bowling.

---

### 4.9 Chase Control Score (CCS)

**Unit of analysis:** Batting team or individual batter

Measures how effectively a team controlled a chase at each phase transition:

```
CCS_phase = 1 - |RRR_at_phase_start - RRR_at_phase_end| / RRR_at_phase_start
```

High CCS = chase was managed smoothly (RRR stayed controlled).  
Low CCS = chase accelerated dangerously or collapsed.

---

### 4.10 Matchup Leverage Score (MLS)

**Unit of analysis:** Batter–Bowler pair (minimum 30 balls faced)

```
MLS_batter_advantage =
    (actual_sr_vs_bowler - expected_sr_vs_bowler_type) /
    std_sr_vs_bowler_type

Where expected_sr is derived from similar bowling-style group averages.
```

Positive MLS = batter has statistical edge vs this bowler type.  
Negative MLS = bowler has control over this batter type.

---

### 4.11 Venue Sensitivity Index (VSI)

**Unit of analysis:** Player

```
VSI = std(player_metric_across_venues) / mean(player_metric_across_venues)
```

High VSI = player performance varies materially by venue (specialist or venue-dependent).  
Low VSI = consistent performer regardless of venue context.

---

### 4.12 Role Stability Score (RSS)

**Unit of analysis:** Batter across seasons

Measures consistency of batting role (position, phase, pressure band) across seasons.  
Used to identify players whose value is durable vs situationally constructed.

---

## 5. Model Layer

### Model 1 — Expected Final Score (EFS)

**Type:** Regression  
**Unit of analysis:** Ball (inning 1 primary)  
**Target:** final_innings_score  
**Features:**
- runs_scored_so_far
- wickets_lost
- phase
- over_number
- balls_remaining
- current_run_rate
- venue_avg_first_innings
- bowling_attack_strength
- dot_ball_streak
- last_5_balls_rr
- season (or binned era)
- partnership_runs / partnership_balls

**Baseline:** linear extrapolation (`crr * remaining_overs`)  
**Model:** XGBoost Regressor  
**Validation:** time split — train 2008–2022, validate 2023, test 2024–2025  
**Metric:** MAE (runs), RMSE, % within ±10 runs  
**Leakage risk:** Do not use post-ball features. State must be constructed at ball START.

---

### Model 2 — Chase Win Probability (CWP)

**Type:** Binary classification  
**Unit of analysis:** Ball (inning 2 only)  
**Target:** batting_team_won (1/0)  
**Features:**
- runs_needed
- balls_remaining_legal
- wickets_in_hand
- required_run_rate
- current_run_rate
- rrr_crr_delta
- phase
- venue
- batting_team_strength_proxy
- bowling_team_strength_proxy
- dot_ball_streak
- last_5_balls_rr
- score_pressure
- season_era

**Calibration:** Required — use Platt scaling or isotonic regression.  
**Validation:** time split — test strictly on 2024–2025  
**Metric:** Brier Score, Calibration Curve, Log-Loss  
**Note:** This is the probability engine for WPA computation.

---

### Model 3 — Ball-Level Wicket Probability (BWP)

**Type:** Binary classification  
**Unit of analysis:** Ball  
**Target:** is_wicket (1/0)  
**Features:**
- phase
- over_number
- bowler_wicket_rate_phase
- batter_dismissal_rate_pressure_band
- dot_ball_streak (pressure = vulnerability)
- batter_balls_faced_this_innings (new vs settled)
- batting_team_wickets_lost
- partnership_balls

**Note:** Class imbalance handling required (wickets are ~7% of balls).  
Use precision-recall analysis, not accuracy.

---

### Model 4 — Phase-Level Collapse Risk (PCR)

**Type:** Binary classification (3+ wickets in next N overs)  
**Unit of analysis:** Phase entry state  
**Target:** collapse_flag  
**Features:** (see Section 4.5)

Calibrated probability output.  
Used to trigger Collapse Risk alerts in the dashboard.

---

### Model 5 — Player Role Clustering

**Type:** Unsupervised (K-Means or HDBSCAN)  
**Unit of analysis:** Player × season  
**Feature matrix:**
- phase-weighted strike rate
- dot ball rate by phase
- wicket rate by phase
- aggression index (4s + 6s / balls)
- pressure performance ratio
- role position (avg batting position)
- overs bowled per match (for allrounders)

**Output:** Role labels (anchor, stabilizer, accelerator, finisher, enforcer, etc.)  
**Interpretability:** Centroid profiling + radar chart per cluster

---

### Model 6 — Replacement Similarity Engine

**Type:** Distance-based (cosine or euclidean on normalized feature vectors)  
**Purpose:** Given player X, find the top-K most similar players by role profile.  
**Use case:** Scouting, squad depth analysis, transfer/auction targeting.

---

## 6. Offline RL Strategy Module

### 6.1 Why Offline RL, and Honest Framing

We do not have a simulator. We cannot train an online agent.  
What we can do: learn a **policy** from historical state-action-outcome transitions.  
This is **offline (batch) reinforcement learning**, also called historical policy optimization.

**Honest claim:** "This module estimates which strategic intent (conservative / balanced / aggressive)
was historically associated with better outcomes from similar match states."  
This is a policy recommendation, not a proven optimal strategy.

---

### 6.2 State Space

```python
state = {
    "runs_needed": int,             # binned: <20, 20-40, 40-60, 60-80, 80-100, 100-120+
    "balls_remaining": int,         # binned: <12, 12-24, 24-36, 36-48, 48-72, 72-120
    "wickets_in_hand": int,         # 10, 9, 8, 7, 6, 5, <=4
    "phase": str,                   # powerplay / middle / death
    "rrr_crr_delta": float,         # binned: < -2, -2 to 0, 0 to 2, 2 to 4, > 4
    "venue_scoring_tier": str,      # low / medium / high (venue historical tercile)
    "dot_ball_streak": int,         # binned: 0, 1–2, 3–4, 5+
    "batting_strength_remaining":str # strong / moderate / weak (based on remaining lineup)
}
```

**State space size (approximate):** 6 × 6 × 7 × 3 × 5 × 3 × 4 × 3 ≈ ~68,000 theoretical states.  
In practice, far fewer will be observed (data will cover ~5,000–15,000 unique states).

---

### 6.3 Action Space

Three discrete actions, interpretable and auditable:

| Action | Definition |
|---|---|
| `conservative` | Prioritize wicket preservation; accept lower run rate |
| `balanced` | Maintain required rate; take calculated risks |
| `aggressive` | Attempt to exceed required rate; higher boundary intent |

**Action inferral from data:** Infer action from actual batting behavior per state:
- `aggressive`: over's run rate > RRR + 2.0 OR boundary% > 40% of balls
- `conservative`: run rate < RRR - 1.5 AND dot_pct > 40%
- `balanced`: everything else

This is a heuristic label for supervised offline policy estimation.  
**This assumption must be disclosed.**

---

### 6.4 Reward Function

```python
def compute_reward(ball_state, outcome):
    base_reward = ball_state["batsman_runs"] - ball_state["expected_runs_baseline"]

    # Reward structure
    r = 0
    r += base_reward * 0.4              # immediate runs above expectation
    r -= 5.0 * ball_state["is_wicket"]  # wicket penalty
    r += 0.3 * ball_state["wpa_delta"]  # win probability shift
    if match_won and is_last_ball:
        r += 10.0                        # chase completion bonus

    return r
```

**Discount factor gamma:** 0.95 (balance between immediate and terminal reward)

---

### 6.5 Learning Method: Fitted Q-Iteration (FQI)

FQI is a standard, principled offline RL algorithm that is:
- implementable without a simulator
- interpretable through Q-value tables or learned approximators
- honest about distributional shift limitations

```python
# Pseudocode: FQI iteration
Q = initialize_Q_function()   # e.g., XGBoost regressor

for iteration in range(max_iterations):
    # Build regression targets
    y = []
    X = []
    for (s, a, r, s_next) in historical_transitions:
        target = r + gamma * max(Q(s_next, a_prime) for a_prime in actions)
        X.append(encode(s, a))
        y.append(target)

    Q = fit_regressor(X, y)   # refit Q function

policy = lambda s: argmax(Q(s, a) for a in actions)
```

**Limitation disclosure:** FQI trained on historical data inherits historical team behavior
distributions. States with sparse data (~< 20 observations) will have high Q-value uncertainty.
These states should be flagged with "low confidence" in the UI.

---

### 6.6 Policy Evaluation (Offline, No Simulator)

Use **Importance Sampling (IS) or Doubly Robust (DR) estimator** to estimate counterfactual
policy value without running live episodes.

```
IS_estimate = mean(
    (pi_new(a | s) / pi_behavior(a | s)) * reward
    for (s, a, reward) in logged_data
)
```

Report:
- Policy value estimate with confidence intervals
- State coverage (fraction of states with sufficient support)
- Action distribution comparison: learned policy vs historical behavior

---

### 6.7 RL Module Output (Dashboard: Strategy Lab)

For a given input state (user configures via sliders):
- Recommended action: Conservative / Balanced / Aggressive
- Q-value confidence band
- Historical win rate under each action from similar states
- Sample historical matches where team chose each action and outcome
- Caveat text: "This is a historical pattern recommendation, not a live strategy optimizer."

---

## 7. Dashboard Architecture

### Tab 1 — Executive Home (Situation Room)

**Question answered:** Where is the value and pressure story of a given match or season?

**Components:**
- Match selector (season + match)
- Innings win probability trajectory (line chart, ball-by-ball)
- Pressure Index heatmap (overs × phase)
- Key decision points flagged (largest WP swings per match)
- Top WPA contributors (batter + bowler) for selected match
- Collapse events highlighted
- Season-level summary switcher

---

### Tab 2 — Match State Engine

**Question answered:** What did the match state look like at any given moment?

**Components:**
- Ball-by-ball state table (filterable)
- Expected Score trajectory vs Actual (inning 1)
- Win Probability curve vs actual result (inning 2)
- State reconstruction viewer: click any ball to inspect full state vector
- Momentum metrics panel (rolling run rate, dot-streak, partnership)
- Phase transition summary cards
- State Difficulty score plotted over innings

---

### Tab 3 — Team DNA

**Question answered:** What are each team's structural tendencies, strengths, and failure modes?

**Components:**
- Team phase-by-phase run rate profiling (radar chart)
- Powerplay aggression vs middle-over stability positioning
- Death-over batting and bowling performance bands
- Collapse frequency by phase (waffle chart or bar)
- Chase success rate by pressure band entry
- Toss decision quality by venue (historical win% conditional)
- Season-by-season trend line for key metrics

---

### Tab 4 — Player Value (Context-Adjusted)

**Question answered:** Which players create match-winning value beyond what raw stats show?

**Components:**
- WPA leaderboard (filterable by phase, pressure band, season, role)
- ESA leaderboard (inning 1 focus)
- Player value card: raw totals vs context-adjusted rank overlay
- "Overrated / Underrated" quadrant: plot raw rank vs WPA rank
- Batter phase heat map (SR × phase × pressure band as 3D bubble)
- Bowler suppression matrix (phase × DSI × contextual economy)
- Minimum ball thresholds enforced (shown in UI)

---

### Tab 5 — Pressure Profiles

**Question answered:** Who thrives under difficult match states and who breaks?

**Components:**
- Batter performance by pressure band (SR, boundary%, dot% side-by-side)
- Bowler performance by phase pressure (economy + dot% + wicket rate)
- Pressure survivors list (top performers in SDS > 75 states)
- Pressure collapse list (worst WPA in SDS > 75 states)
- Dot-ball pressure conversion rate table (bowlers)
- Partnership stability under pressure (pairs with high CCS in critical states)
- Phase-specific stress point profiles per team

---

### Tab 6 — Matchup Intelligence

**Question answered:** Which batter–bowler matchups are strategically exploitable?

**Components:**
- Matchup Leverage Score table (batter vs bowler type)
- Batter vs bowling style drilldown (pace / offspin / legbreak / left-arm pace / etc.)
- Bowler vs batting hand analysis (if handedness data available)
- Matchup recommender: given batting lineup, flag high-risk bowler pairings
- Heat map: batter × bowler MLS across selected season range
- Min 30 balls filter enforced; sample size shown
- Historical match outcomes where high-MLS matchups were/weren't used

---

### Tab 7 — Decision Audit

**Question answered:** Were the key in-match decisions (bowling changes, promotions, toss) consistent with what historical data would support?

**Components:**
- Match timeline with decision overlay
- Toss decision quality table by venue (bat-first vs field-first historical win%)
- Bowling change audit: did the change happen in a high-collapse-risk state?
- Batter promotion audit: was the promoted batter a pressure-fit for that state?
- Death-over bowling resource allocation: overs used by bowler × death-over DSI ranking
- Counterfactual comparison panel: "If team had batted first here, historical WP at over 10 = X"
- Caveat: all comparisons are against historical distributions, not causal ground truth

---

### Tab 8 — Scouting & Role Fit

**Question answered:** Which players fit which strategic roles, and who can replace whom?

**Components:**
- Role cluster assignment per player × season
- Radar chart per player (multi-metric profile)
- Replacement similarity finder: input player, get top-5 role-matches with difference score
- Role gap analysis: given a squad, identify missing profiles
- Phase coverage audit: which phases are over/under-resourced in a squad?
- Best XI builder: select from available players, see projected phase coverage
- VSI (Venue Sensitivity Index) flag for players with inconsistent venue profiles

---

### Tab 9 — Strategy Lab (Offline RL)

**Question answered:** What strategic intent does historical data support from this match state?

**Components:**
- State input panel (sliders: runs needed, balls left, wickets in hand, phase, venue tier)
- Policy recommendation: Conservative / Balanced / Aggressive with confidence
- Historical win rate per action from similar states (bar chart)
- Sample match outcomes under each action
- Q-value surface visualization (2D projection: rrr_crr_delta × wickets_in_hand)
- Confidence map (state density from training data)
- Prominent caveat section (methodology, distributional shift, honest limitations)

---

### Tab 10 — Methodology & Definitions

**Question answered:** How are all metrics, models, and RL components defined and measured?

**Components:**
- Metric glossary (all metrics defined with formula, interpretation, limitations)
- Model cards (each model: features, target, training method, validation results)
- RL design disclosure (assumptions, state space, action inference, FQI method)
- Data dictionary (column definitions)
- Known limitations section
- Validation result tables (Brier scores, MAEs, etc.)
- Changelog

---

## 8. Deployment Architecture

### 8.1 Stack

| Layer | Technology | Rationale |
|---|---|---|
| Data processing | Python + Pandas/Polars | Sufficient for dataset size |
| Feature store | Parquet files (local / cloud) | Fast columnar reads |
| Modeling | scikit-learn + XGBoost/LightGBM | Interpretable, production-ready |
| RL | Custom FQI in Python | Transparent, auditable |
| Dashboard | Streamlit | Fastest path to public deployment |
| Deployment | Streamlit Community Cloud or HuggingFace Spaces | Free, public, linkable |
| Version control | Git + GitHub | Portfolio visibility |
| Environment | requirements.txt + optional Dockerfile | Reproducibility |

### 8.2 Data Flow for Dashboard

```
Raw CSVs (Kaggle)
     │
     ▼
01_ingest_validate.py  →  data/processed/  (validated parquet)
     │
     ▼
02_state_reconstruction.py  →  data/features/ball_states.parquet
     │
     ▼
03_feature_engineering.py  →  data/features/player_features.parquet
                              data/features/team_features.parquet
     │
     ▼
04_metrics_compute.py  →  data/features/metrics_ball_level.parquet
                          data/features/metrics_player_level.parquet
     │
     ▼
05_model_training.py  →  models/*/  (saved model artifacts)
     │
     ▼
app/main.py  →  loads parquet + model artifacts at startup
             →  renders Streamlit tabs
             →  all filtering is in-memory (pandas/polars queries)
```

### 8.3 Performance Considerations

- Pre-compute all aggregations during pipeline; do not compute inside `app/` at render time
- Ball-level dataset for 2008–2025 is ~1.2M+ rows — manageable in memory
- Model inference at app startup (not per-query): load models once, score full dataset
- Use `@st.cache_data` aggressively for all data loads and aggregations
- RL policy lookup is a dictionary query — effectively instant

### 8.4 GitHub Repository Hygiene (Portfolio Requirements)

- `README.md`: Professional positioning, architecture diagram, screenshot, methodology summary, installation, live link
- `docs/METRIC_DEFINITIONS.md`: Full metric glossary
- `docs/MODEL_CARDS.md`: Model performance tables, assumptions
- `docs/RL_DESIGN_NOTES.md`: Honest RL framing, design decisions
- `docs/ANALYTICAL_DECISIONS.md`: Why each design choice was made (demonstrates analytical thinking)
- No raw CSV files in repo (data too large; document Kaggle source in README)
- GitHub Actions CI: run tests on push

---

## 9. Build Sequence (Prioritized)

Build in this order to ensure each layer supports the next:

| Phase | Module | Output |
|---|---|---|
| Phase 1 | Data ingestion, validation, state reconstruction | `ball_states.parquet` |
| Phase 2 | Feature engineering | `player_features.parquet`, `team_features.parquet` |
| Phase 3 | Win probability + expected score models | Model artifacts |
| Phase 4 | Metrics computation (WPA, ESA, SDS, PI, DSI) | `metrics_*.parquet` |
| Phase 5 | Dashboard: Tabs 1, 2, 4, 5 | First deployable version |
| Phase 6 | Matchup intelligence + Decision audit | Tabs 6, 7 |
| Phase 7 | Role clustering + Replacement engine | Tab 8 |
| Phase 8 | Offline RL module | Tab 9 |
| Phase 9 | Collapse risk model | Tab 3 + alerts across tabs |
| Phase 10 | Methodology tab, README, deployment polish | Public launch version |

---

## 10. Analytical Standards Enforcement

Every metric and model output must satisfy all of the following before it appears in the dashboard:

1. **Minimum sample threshold** — no metric displayed with < N observations (N documented per metric)
2. **Time-aware validation** — no model trained on data future to its validation set
3. **Honest labeling** — all probabilistic outputs labeled as probabilities, not certainties
4. **Causal disclaimer** — observational data; correlations do not imply strategic mandates
5. **Separability** — descriptive, predictive, and prescriptive outputs are visually distinct
6. **RL disclaimer** — Strategy Lab prominently marks all recommendations as historically-derived, not optimized
7. **No black-box overload** — every non-trivial metric has a tooltip with definition and formula
8. **Leakage audit log** — documented for each model (what features were excluded and why)

---

*This document is the authoritative design specification for the IPL INTEL platform.*  
*All module implementations, metric definitions, and model designs must trace to the decisions described here.*  
*Deviations from this architecture require documented justification.*
