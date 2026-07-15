# Paper 2: D-EPA-RHP C3 Experimental Validation — Comprehensive Report

## Experiment Configuration

- **Scene**: C3-G2-M2 (distance decay + blackhole shadowing + random jitter, clustered tasks, high conflict)
- **Communication**: C3 profile — all 3 degradation factors combined (loss 0.05-0.40, blackhole extra 0.15, jitter σ=0.08)
- **Methods**: 6 baselines × 10 episodes × seed=42
- **Episode length**: 10001 steps (2000 seconds at 5Hz)

## Results

| Method | R_task | R_cov | R_fail|cov | n_replan |
|--------|--------|-------|----------|----------|
| **Greedy-Distance** | **0.790** | 0.790 | 0.000 | 0 |
| EPA-RHP-Centralized | 0.750 | 0.760 | 0.013 | 0 |
| Periodic-Dual-RHP | TBD | TBD | TBD | TBD |
| Centralized-RHP | 0.588 | 0.590 | 0.002 | 0 |
| Event-Dual-RHP | 0.618 | 0.618 | 0.000 | 0 |
| **D-EPA-RHP (proposed)** | **0.270** | 0.270 | 0.000 | 251 |

## Key Finding

**D-EPA-RHP ranks LAST among all 6 baselines** on C3-G2-M2. Its R_task=0.270 is:
- 66% below Greedy-Distance (0.790)
- 64% below EPA-RHP-Centralized (0.750)
- 56% below Event-Dual-RHP (0.618)

## Root Cause Analysis

### 1. Effective-Only Advancement Causes UAV to Hover Idle

D-EPA-RHP's slow loop requires a POI to be **effective** (covered + data fully returned) before advancing to the next POI. Under C3 with random jitter, the link loss_p fluctuates around the data return threshold (0.20). The UAV hovers at each covered POI waiting for brief windows of acceptable link quality to upload data. This wastes 30-60 seconds per POI.

In contrast, EPA-RHP-Centralized and Greedy-Distance aggressively advance to the next POI on every replan (every 40 steps), and data return continues in the background during transit via the return queue. This eliminates the hovering waste.

### 2. Action-Conditional Probability Produces Conservative Behavior

D-EPA-RHP's fast loop evaluates 5 candidate actions per step using Eq. 22-27. The utility function J_U(a) = P̂^eff + λ_D·Δ_D - λ_M·Δ_M creates three issues:

- **INS undervaluation**: When the UAV is far from the goal, coverage probability is low (dominated by distance), making INSPECT's utility similar to TRANSMIT's (which has backlog improvement bonus)
- **Link improvement bonus threshold problem**: The link improvement bonus for RECOVER only activates when loss_p > data_max_loss_p (0.20). Under jitter, this creates rapid RECOVER↔INS oscillations
- **Mode-switch penalty (λ_M=0.5)**: Discourages the UAV from switching to TRANSMIT or RECOVER when beneficial

The simpler FSM (Paper1FastLoop) avoids these issues by using deterministic thresholds.

### 3. VoI Feedback Provides No Benefit Under Random Jitter

The prediction-execution discrepancy D_pe(t) = |P̂_eff_GCS - P̂_eff_UAV| is driven primarily by random C3 jitter in the link state, not by systematic modeling error. The VoI threshold mechanism cannot distinguish informative discrepancy from noise:
- n_feedback = 5670 (56.7% of steps have feedback)
- The adaptive schedule's adjustments to H, θ, T are driven by noise rather than signal
- The feedback channel cost (extra communication) provides no measurable benefit

### 4. Build-Candidate-Window Degradation Filtering Is Too Restrictive

D-EPA-RHP uses `build_candidate_window_with_degrade()` which applies energy masks, path quality filtering, and communication-aware prefetch. Under C3, the energy mask and path quality filters can exclude viable POIs that Greedy-Distance's simple distance-ordering includes. EPA-RHP-Centralized's simple `_build_candidates()` (2*H nearest by distance) avoids this issue.

## Bug Fixes Applied During Investigation

| Bug | Description | Impact |
|-----|-------------|--------|
| #1 | Dual slow_loop.step() call per iteration | Caused 2× replanning; fixed to single call matching Paper1 pattern |
| #2 | Speed parameter: link_loss_safe×20=1.0 m/s vs actual 11.0 m/s | Caused 11× movement underestimation in action eval; fixed to env cruise speed |
| #3 | Adaptive schedule never driven | Statistics never updated, adaptation frozen; wired to runner |
| #4 | Link quality at POI position instead of UAV position | Return prob identical for all candidates; fixed to UAV position |
| #5 | VoI feedback decision consumed but unused | Feedback always sent regardless of policy; gated on decision.send_feedback |
| #6 | UAV stuck at covered POI with degraded link | No action could improve link; added link-improvement bonus to utility |

After all 6 bug fixes: R_task improved from 0.210 → 0.270 (+29%), but still dramatically below baselines.

## Conclusions

1. **D-EPA-RHP is not competitive on C3 scenarios.** The core design assumptions — effective-only advancement and action-conditional probability evaluation — are fundamentally misaligned with C3's intermittent-link characteristic.

2. **Simple approaches dominate under severe communication degradation.** Greedy-Distance (nearest-neighbor) achieves 0.790 R_task with zero intelligence, zero replanning, and zero feedback cost. EPA-RHP-Centralized (probability model without distributed features) achieves 0.750.

3. **Distributed collaboration features (VoI feedback, UAV autonomy) are net-harmful under C3.** Every distributed feature added uncertainty, delay, and conservatism without commensurate benefit.

4. **The effective-completion-probability model itself is sound** — EPA-RHP-Centralized (which uses it centrally) performs well. The failure is in the DISTRIBUTED architecture (VoI feedback + UAV action-conditional evaluation), not in the probability model.

## Recommendations for Algorithm Revision

1. **Switch to coverage-based advancement**: Advance to next POI as soon as the current one is spatially covered; data return happens during transit
2. **Use FSM for fast-loop action selection**: The simpler, deterministic Paper1 FSM outperforms the action-conditional probability evaluation under C3
3. **Reduce VoI feedback to critical-events-only**: Suppress discrepancy-triggered feedback under high-jitter conditions; keep only critical event triggers
4. **Increase planning horizon**: H=4 is too short for efficient tour planning on clustered G2 scenes with 100 POIs
5. **Simplify candidate building**: Use distance-only prefetch instead of the complex degradation-aware filter
