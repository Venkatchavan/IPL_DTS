# T20 Decision Intelligence Platform

> **Analyst-grade exploration of IPL 2008–2025 ball-by-ball data.**  
> Built as a public portfolio artifact demonstrating advanced sports analytics, ML, and offline RL.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ipl-ntelligence-system.streamlit.app/)

🚀 **[https://ipl-ntelligence-system.streamlit.app/](https://ipl-ntelligence-system.streamlit.app/)**

---

## What this is

This is **not** a fan dashboard. It is a decision intelligence platform that answers questions like:

- Which batting actions, in which match states, were *historically* associated with better outcomes?
- How does a player's strike rate change under critical pressure vs neutral pressure?
- Which batter–bowler matchup most strongly favors the bowler — controlling for state difficulty?
- Was Team X's bowling allocation in death overs optimal given their WPA patterns?

---

## Architecture

```
IPL_INTEL/
├── data/
│   ├── raw/              ← deliveries.csv + matches.csv (from Kaggle, not committed)
│   ├── processed/        ← cleaned parquet files
│   ├── features/         ← ball_states, player/team/venue features
│   └── metrics/          ← computed analytics metrics
├── pipelines/            ← numbered ETL + ML scripts (run in order)
├── metrics/              ← 12 metric computation modules
├── models/
│   ├── rl_strategy/      ← FQI trainer, policy evaluator
│   └── ...               ← EFS, CWP, BWP, PCR model folders
├── app/
│   ├── main.py           ← Streamlit entry point
│   ├── config.py         ← UI constants, color palette
│   └── pages/            ← 10 dashboard tab modules
├── docs/                 ← Metric definitions, model cards, RL design notes
├── tests/                ← Pytest unit tests
└── config.py             ← Project-wide constants
```

---

## Data

**Source**: [IPL Dataset 2008–2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) on Kaggle.

Download `deliveries.csv` and `matches.csv`, place in `data/raw/`.  
The data is not committed to this repository due to Kaggle licensing.

---

## Build Sequence

Run pipelines in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest + validate raw data
python pipelines/01_ingest_validate.py

# 3. Reconstruct ball-by-ball state vector (30+ fields per delivery)
python pipelines/02_state_reconstruction.py

# 4. Engineer batter / bowler / venue / matchup features
python pipelines/03_feature_engineering.py

# 5. Compute Phase-A metrics (no models needed)
python pipelines/04_metrics_compute.py --phase A

# 6. Train all 4 ML models + score full dataset
python pipelines/05_model_training.py

# 7. Compute Phase-B metrics (WPA, ESA — requires model scores)
python pipelines/04_metrics_compute.py --phase B

# 8. Train offline RL policy (Fitted Q-Iteration)
python models/rl_strategy/fqi_trainer.py

# 9. Evaluate RL policy offline
python models/rl_strategy/policy_evaluator.py

# 10. Launch dashboard
streamlit run app/main.py
```

---

## Dashboard Tabs

| # | Tab | Key Questions |
|---|-----|---------------|
| 1 | Executive Home | Match WP trajectory, high-leverage moments |
| 2 | Match State Engine | Ball-by-ball state viewer, EFS vs actual |
| 3 | Team DNA | Phase profiles, chase patterns, collapse frequency |
| 4 | Player Value | WPA/ESA leaderboards, overrated/underrated quadrant |
| 5 | Pressure Profiles | How players perform by pressure band |
| 6 | Matchup Intelligence | MLS table, batter vs bowler heatmap |
| 7 | Decision Audit | Toss analysis, bowling changes, death allocation |
| 8 | Scouting & Role Fit | Role clusters, replacement finder |
| 9 | Strategy Lab | RL-based batting strategy recommender |
| 10 | Methodology | All metric/model/RL definitions |

---

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **PI** — Pressure Index | Per-ball | 4-component pressure score (0–100) |
| **SDS** — State Difficulty Score | Per-ball | How hard is this state to bat in |
| **WPA** — Win Probability Added | Per-ball | Change in win probability per delivery |
| **ESA** — Expected Score Added | Per-ball | Contribution to expected innings total |
| **CER** — Contextual Economy Rate | Bowler | Economy adjusted for venue/phase baseline |
| **DSI** — Death Suppression Index | Bowler | Death-over effectiveness composite (0–100) |
| **MLS** — Matchup Leverage Score | Pair | Batter SR vs expected for bowler style bucket |
| **BCR** — Boundary Conversion Rate | Batter | Attacking shots that go for 4/6 |
| **VSI** — Venue Sensitivity Index | Bowler | Consistency across venues |
| **PCR** — Phase Collapse Risk | Team/State | Probability of 3+ wickets in the phase |
| **PRR** — Partnership Run Rate | Pair | Live partnership run rate |
| **Control Rate** | Bowler | % of balls with batsman_runs ≤ 1 |

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| EFS — Expected Final Score | XGBoost Regressor | Predict innings total from mid-innings state |
| CWP — Chase Win Probability | XGBoost + Platt calibration | Real-time win probability (inning 2) |
| BWP — Ball-level Wicket Probability | XGBoost + isotonic calibration | Per-delivery wicket likelihood |
| PCR — Phase Collapse Risk | XGBoost + Platt calibration | 3+ wickets in a phase (binary) |

**Validation strategy**: Time-aware split — Train: 2008–2022, Val: 2023, Test: 2024–2025.

---

## Offline RL

The Strategy Lab uses **Fitted Q-Iteration (FQI)** — an offline batch RL algorithm.

**Honest framing** (displayed in the dashboard):
- This is NOT a real-time RL agent or simulator
- Actions are inferred from historical behavior using heuristic proxies
- Recommendations reflect historically associated outcomes, not causal optima
- Low-confidence states (< 20 observations) should be ignored

See `docs/RL_DESIGN_NOTES.md` for full methodology.

---

## Tests

```bash
pytest tests/
```

Covers: state reconstruction correctness, metric computation, feature engineering edge cases.

---

## Deployment

Designed for [Streamlit Community Cloud](https://streamlit.io/cloud) or [HuggingFace Spaces](https://huggingface.co/spaces).

For Streamlit Community Cloud:
1. Push repo to GitHub
2. Add Kaggle data (or pre-computed parquets) as HuggingFace dataset or Streamlit secrets
3. Deploy via `app/main.py`

---

## License

MIT — see `LICENSE`
