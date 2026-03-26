"""
app/pages/p10_methodology.py — Methodology Tab

Full documentation of:
  - All 12 metrics with formulas
  - 4 model definitions with features, training strategy, evaluation
  - RL design: state/action/reward/algorithm/honest framing
  - Data pipeline stages
  - Known limitations
"""

import streamlit as st


def render():
    st.header("Methodology — Metrics, Models & RL Design")

    with st.expander("📋 Data Pipeline Overview", expanded=False):
        st.markdown("""
        **Source**: [Kaggle IPL Dataset 2008–2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)
        — `deliveries.csv` + `matches.csv`

        | Stage | Script | Output |
        |-------|--------|--------|
        | 01 — Ingest & Validate | `pipelines/01_ingest_validate.py` | `deliveries_clean.parquet`, `matches_clean.parquet` |
        | 02 — State Reconstruction | `pipelines/02_state_reconstruction.py` | `ball_states.parquet` (30+ field state vector) |
        | 03 — Feature Engineering | `pipelines/03_feature_engineering.py` | `player_features.parquet`, `team_features.parquet`, etc. |
        | 04A — Metrics (no model) | `pipelines/04_metrics_compute.py --phase A` | `metrics_ball_level.parquet` |
        | 05 — Model Training | `pipelines/05_model_training.py` | `ball_states_scored.parquet`, 4 model .pkl files |
        | 04B — Metrics (model-dependent) | `pipelines/04_metrics_compute.py --phase B` | `metrics_player_level.parquet` |
        | RL — FQI Training | `models/rl_strategy/fqi_trainer.py` | `policy_table.parquet`, `fqi_model.pkl` |

        **Train/Validation/Test Split**: Seasons 2008–2022 (train), 2023 (validation), 2024–2025 (test).
        All model evaluations are on the held-out test set to prevent leakage.

        **Leakage prevention**: All model features use `pre_` prefix — computed *before* delivery outcome.
        """)

    with st.expander("📐 Metric Definitions", expanded=False):
        st.markdown("""
        #### 1. Pressure Index (PI)
        **Range**: 0–100 | **Context**: Per-delivery

        ```
        PI = 0.35 × norm(RRR−CRR) + 0.30 × norm(Wickets) + 0.20 × Phase_Weight + 0.15 × norm(Dot_Streak)
        ```
        - Inning 1: RRR−CRR term = 0 (no target yet), weight redistributed to wickets
        - RRR−CRR clamped to [−4, 8]; Dot streak clamped to [0, 12]
        - Phase weights: Powerplay=0.40, Middle=0.65, Death=0.90

        #### 2. State Difficulty Score (SDS)
        **Range**: 0–100 | **Context**: Per-delivery (batter/bowler context)

        ```
        SDS = 0.30×norm(RRR) + 0.25×norm(Wickets) + 0.20×Phase_Difficulty + 0.15×norm(Dot_Streak) + 0.10×Venue_Difficulty
        ```

        #### 3. Win Probability Added (WPA)
        **Context**: Per-delivery, inning 2 primary

        ```
        WPA = post_win_prob − pre_win_prob   (batter perspective)
        WPA_bowler = pre_win_prob − post_win_prob  (bowler perspective)
        ```
        Requires Model 2 (CWP) scores. High-leverage threshold: |WPA| > 5%.

        #### 4. Expected Score Added (ESA)
        **Context**: Per-delivery, inning 1 primary

        ```
        ESA = post_expected_score − pre_expected_score
        ```
        Requires Model 1 (EFS) scores.

        #### 5. Contextual Economy Rate (CER)
        ```
        CER = raw_economy − venue_phase_avg_economy
        ```
        Negative CER = bowler conceding *less* than venue average (good).

        #### 6. Death Suppression Index (DSI)
        **Range**: 0–100 | **Min**: 50 death balls bowled

        ```
        DSI = 0.30×econ_norm + 0.25×dot_norm + 0.25×bdry_suppr + 0.20×wicket_norm
        ```
        Higher DSI = more effective death bowler.

        #### 7. Matchup Leverage Score (MLS)
        ```
        Z = (batter_SR_vs_bowler − mean_SR_vs_bucket) / std_SR_vs_bucket
        ```
        Bowler style buckets assigned via K-Means (k=5) on bowler metrics.
        Positive MLS = batter has advantage; negative = bowler has advantage.

        #### 8. Control Rate
        ```
        Control Rate = % of balls where batsman_runs ≤ 1
        ```
        High control = bowler forcing dot/singles.

        #### 9. Venue Sensitivity Index (VSI)
        ```
        VSI = std_dev(team_economy_across_venues) / mean_economy
        ```
        High VSI = bowler's effectiveness varies widely by venue.

        #### 10. Phase Entry Collapse Risk (PCR)
        Model output (Model 4). Probability of 3+ wickets falling in a given phase.
        See Model 4 definition below.

        #### 11. Partnership Run Rate (PRR)
        Running partnership run rate: `partnership_runs / partnership_balls * 6`.

        #### 12. Boundary Conversion Rate (BCR)
        ```
        BCR = (fours + sixes) / total_scoring_shots
        ```
        Measures "quality of contact" — how often attacking shots convert to boundaries.
        """)

    with st.expander("🤖 Model Definitions", expanded=False):
        st.markdown("""
        All models use **XGBoost** as the primary estimator. A scikit-learn
        `GradientBoostingRegressor/Classifier` is used as a fallback if XGBoost
        is not installed.

        #### Model 1: Expected Final Score (EFS) — Regressor
        | Parameter | Value |
        |-----------|-------|
        | Target | Final innings score |
        | Features | `pre_runs`, `pre_wickets`, `pre_balls_remaining`, phase flags, venue tier, season |
        | Evaluation | MAE, RMSE on test set |
        | Baseline | Linear extrapolation: `pre_runs / balls_played * 120` |

        #### Model 2: Chase Win Probability (CWP) — Classifier
        | Parameter | Value |
        |-----------|-------|
        | Target | `match_won` (batting team) |
        | Context | Inning 2 only |
        | Features | `pre_runs_needed`, `pre_balls_remaining`, `pre_wickets`, RRR, CRR, delta, phase, venue |
        | Calibration | Platt scaling (sigmoid) on validation set |
        | Evaluation | Brier score, log-loss, AUC-ROC |

        #### Model 3: Ball-level Wicket Probability (BWP) — Classifier
        | Parameter | Value |
        |-----------|-------|
        | Target | `is_wicket` (0/1) |
        | Features | Bowler stats, batter stats, matchup history, pitch state, phase |
        | Class imbalance | `scale_pos_weight` (wickets ~7% of balls) |
        | Calibration | Isotonic regression |
        | Evaluation | Brier score, Average Precision |

        #### Model 4: Phase Collapse Risk (PCR) — Classifier
        | Parameter | Value |
        |-----------|-------|
        | Target | 3+ wickets in current phase (binary) |
        | Features | Phase entry state: wickets, runs, recent_wickets, bowler quality metrics |
        | Calibration | Platt scaling |
        | Evaluation | Brier score, AUC-ROC |
        """)

    with st.expander("🎯 Offline RL Design", expanded=False):
        st.markdown("""
        #### Algorithm: Fitted Q-Iteration (FQI)
        Ernst, Geurts & Wehenkel (2005). Iteratively approximates the Q-function
        using a supervised regression approach on transition tuples.

        #### State Space (7 dimensions)
        | Dimension | Bins |
        |-----------|------|
        | Runs Needed | 0–10, 11–25, 26–50, 51–80, 81–120, >120 |
        | Balls Remaining | 0–6, 7–18, 19–36, 37–60, 61–90, >90 |
        | Wickets in Hand | 1–2, 3–5, 6–7, 8–10 |
        | Phase | powerplay, middle, death |
        | RRR−CRR delta | <−2, −2–0, 0–2, 2–4, >4 |
        | Venue Scoring Tier | low, medium, high |
        | Dot Ball Streak | 0, 1–2, 3–4, 5–12 |

        #### Actions (3)
        - **conservative** — dot balls, blocked singles
        - **balanced** — singles, twos, occasional boundaries
        - **aggressive** — boundary attempt, big hits

        **Action inference**: Actions are *inferred* from observed batting behavior
        using heuristic thresholds. Actual intent is not observable. This is a 
        documented proxy.

        #### Reward Function (4 components)
        ```
        R = 0.40 × runs_above_baseline 
          + 0.30 × wp_shift 
          − 5.0 × wicket_fell
          + 10.0 × terminal_success
        ```
        Baseline: RRR/6 (runs per ball to stay on track).

        #### Honest Limitations
        - Distributional shift: historical teams ≠ future teams
        - Action labels are heuristic proxies
        - Sparse states have high uncertainty — filter to high/medium confidence only
        - This is observational data, not a simulator — cannot explore counterfactuals
        - Recommendations are pattern-based priors, not prescriptive strategy

        #### Offline Policy Evaluation
        Three estimators used to validate policy quality on held-out test seasons:
        1. **Direct Method (DM)**: Uses Q-model to estimate expected return
        2. **Importance Sampling (IS)**: Weights outcomes by behavior policy ratio (clipped at 10×)
        3. **Doubly Robust (DR)**: Combines DM + IS for reduced variance

        Results stored in `models/rl_strategy/policy_evaluation.json`.
        """)

    with st.expander("⚠️ Known Limitations", expanded=False):
        st.markdown("""
        1. **Missing player metadata**: Player attributes (batting hand, age, role) 
           are not in the Kaggle dataset. Role assignment is purely statistical.

        2. **Team composition changes**: A player's performance may change across 
           teams/seasons. Features aggregate across all seasons — context is approximate.

        3. **DL-adjusted matches**: Matches with Duckworth-Lewis adjustments are 
           retained but flagged. EFS/WP models may be less accurate for these.

        4. **Super overs**: Removed from all analysis (separate competition format).

        5. **Venue granularity**: Venue features aggregate across all seasons at a 
           venue. Stadium conditions change year to year.

        6. **RL counterfactuals**: No simulator exists for IPL. FQI cannot explore 
           states not present in historical data.

        7. **Sample size for rare states**: Many state combinations are rarely 
           observed (e.g., 0 wickets down, 5 runs needed, 1 ball remaining). 
           All displays suppress results below minimum sample thresholds.
        """)

    with st.expander("📚 References & Credits", expanded=False):
        st.markdown("""
        - **Dataset**: [IPL Dataset 2008–2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) — Kaggle
        - **FQI**: Ernst, Geurts, Wehenkel (2005). *Tree-Based Batch Mode Reinforcement Learning*
        - **WPA framework**: Inspired by Win Probability methods in baseball analytics (FanGraphs)
        - **Doubly Robust OPE**: Dudík, Langford, Li (2011). *Doubly Robust Policy Evaluation and Learning*
        - **XGBoost**: Chen & Guestrin (2016)
        - **Streamlit**: [streamlit.io](https://streamlit.io)
        """)
