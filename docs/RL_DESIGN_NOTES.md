# Offline RL Design Notes

Complete methodology for the Strategy Lab's Fitted Q-Iteration (FQI) module.

---

## Motivation

Standard RL requires an environment to explore. We have no IPL simulator.  
**Offline RL** (batch RL) learns from logged historical data without interaction.  
FQI is the canonical offline RL algorithm for tabular/function-approximation settings.

---

## State Space

7-dimensional discrete state, binned for tractability.

| Dimension | Bins | Rationale |
|-----------|------|-----------|
| Runs Needed | [0–10], [11–25], [26–50], [51–80], [81–120], [>120] | Captures game situation granularly |
| Balls Remaining | [0–6], [7–18], [19–36], [37–60], [61–90], [>90] | Phase-grouped |
| Wickets in Hand | [1–2], [3–5], [6–7], [8–10] | Coarse (lineup depth proxy) |
| Phase | powerplay, middle, death | Directly observable |
| RRR − CRR delta | [<−2], [−2–0], [0–2], [2–4], [>4] | Batting deficit/surplus |
| Venue Scoring Tier | low, medium, high | Environmental context |
| Dot Ball Streak | [0], [1–2], [3–4], [5–12] | Momentum proxy |

**Observed state space size**: ~5,000–15,000 unique combinations in IPL 2008–2025 data.

---

## Actions

Three discrete actions:

| Action | Historical Proxy | Code |
|--------|-----------------|------|
| **conservative** | dot ball when RRR far below CRR, or guarded defense | 0 |
| **balanced** | singles, twos, occasional boundary | 1 |
| **aggressive** | boundary-attempted delivery (4 or 6 scored), or high crr vs rrr | 2 |

### Action Inference (HEURISTIC — must disclose)

Actions are *inferred* (not directly labeled) from delivery-level outcomes:

```python
if is_boundary or (batsman_runs >= 2 and crr > rrr + 1.5):
    action = "aggressive"
elif is_dot and rrr > crr + 1.5:
    action = "conservative"  
else:
    action = "balanced"
```

**Limitation**: This conflates outcome with intent. A boundary may be unintentional (edged four);  
a dot may be intentional (defensive) or reactive (great delivery). The dashboard discloses this.

---

## Reward Function

4-component reward per legal delivery in inning 2:

```
R = 0.40 × runs_above_baseline
  + 0.30 × wp_shift
  − 5.0  × wicket_fell
  + 10.0 × terminal_success
```

### Component Details

| Component | Computation | Rationale |
|-----------|-------------|-----------|
| `runs_above_baseline` | `batsman_runs − (RRR / 6)` | Runs above required strike rate |
| `wp_shift` | `post_win_prob − pre_win_prob` | Direct WP contribution |
| `wicket_fell` | Binary: 1 if dismissal | Wicket loss is costly in chase |
| `terminal_success` | 10.0 if final delivery and match_won=1 | Big positive for chase success |

**Inning 1**: Only ESA-based reward (no terminal bonus; no RRR). Not used in FQI (inning 2 only).

---

## Algorithm: Fitted Q-Iteration (FQI)

### References

Ernst, D., Geurts, P., & Wehenkel, L. (2005).  
*Tree-Based Batch Mode Reinforcement Learning.* Journal of Machine Learning Research.

### Procedure

1. Build transition dataset `(s, a, r, s', terminal)` from all inning-2 legal deliveries.
2. Initialize Q-target: `y = r` (immediate reward only).
3. For K iterations:
   - Compute Bellman target: `y_i = r + γ × max_a Q(s', a)` (for non-terminal transitions)
   - Refit Q regressor: `Q ← fit(X=[s‖a_onehot], y=y_i)`

**Q function approximator**: XGBoost Regressor  
**State-action feature**: `[encode_vector(s) ‖ one_hot(a)]`  
**Discount factor**: γ = 0.95  
**Iterations**: 20 (configurable via `RL_MAX_ITERATIONS` in `config.py`)

### Convergence

FQI with XGBoost does not have guaranteed convergence (unlike linear function approximation).  
In practice, Q-values stabilize within 15–25 iterations for this state space size.

---

## Offline Policy Evaluation

Since we cannot run the policy live, we evaluate using three offline estimators on test seasons (2024–2025).

### 1. Direct Method (DM)

```
V_DM = E_s[ max_a Q(s, a) ]
```

Averages the Q-value of the greedy action over test states.  
**Bias**: Assumes Q-model is accurately calibrated.

### 2. Importance Sampling (IS)

```
V_IS = (1/N) Σ [ (π(a|s) / π_b(a|s)) × r ]
```

Weights observed outcomes by the ratio of the learned policy to the behavior policy.  
Ratios are clipped at 10× to control variance.  
**Bias**: High variance in low-support states.

### 3. Doubly Robust (DR)

```
V_DR = V_DM + (1/N) Σ [ w_i × (r_i − Q(s_i, a_i)) ]
```

Combines DM and IS into a variance-reduced estimator.  
Consistent if *either* the Q-model or the behavior policy estimate is accurate.

---

## State Coverage

The policy table maps each observed state → recommended action + confidence level.

| Support Count | Confidence |
|--------------|-----------|
| ≥ 40 observations | high |
| 20–39 | medium |
| < 20 | low (unreliable — suppressed in prod display) |

---

## Limitations and Honest Framing

1. **No counterfactuals**: FQI cannot evaluate actions never taken in that state.
2. **Distribution shift**: Policy was trained on historical teams. Future team compositions differ.
3. **Heuristic action labels**: Conservative/balanced/aggressive are proxies, not ground truth intent.
4. **Sparse states**: Many state combinations appear fewer than 5 times — Q-values are noise.
5. **Confounders**: Player quality, pitch condition, and opposition quality are not fully captured in state.
6. **The learned "optimal" strategy is optimal for the observed data distribution**, not globally optimal.

**This module demonstrates offline RL methodology on real sports data. It is a research artifact, not a deployment system.**
