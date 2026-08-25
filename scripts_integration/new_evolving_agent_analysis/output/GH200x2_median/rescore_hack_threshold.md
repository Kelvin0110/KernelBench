# Uniform-threshold re-score (30x)

Generated 2026-08-25T06:49:31.761362+00:00. Baseline: `NVIDIA_GH200x2_median`.

> **STATUS: PARTIAL.** These runs are incomplete. `ANALYSIS_RULES.md:158` forbids
> partial prefixes as a final comparative result -- read the *deltas* as magnitude,
> not the levels as a leaderboard.

## The seam

`src/kernelbench/eval.py` changed `excessive_speedup_threshold` 10 -> 30 at
**2026-08-24T15:11:45** (commit `588a6a5`). eval.py is re-imported by every eval spawn, so it
reached all live arms with no restart and no log line. `is_hack` gates which iterations
may form a best, so identical kernels scored differently either side of that instant.

- evals whose label changes under a uniform 30x: **271** of 14884

- all flips are pre-seam evals in the (10, 30] band; re-scoring can only *raise* them,
  so this is a one-directional correction toward the post-seam arms.


## Aligned within model

Cross-model contrast is invalid (different endpoints, GPUs, latency), so each model
group is aligned on the problems common to all of its arms.


### gpt-oss-120b -- 17 aligned problems, 9 arms

| arm | fast_p@1.0 stored | uniform | delta | geomean stored | uniform | delta |
|---|---|---|---|---|---|---|
| oss_compress | 0.941 | 0.941 | +0.000 | 1.266 | 1.266 | +0.000 |
| oss_deletion | 0.765 | 0.765 | +0.000 | 1.490 | 1.490 | +0.000 |
| oss_folding | 0.824 | 0.824 | +0.000 | 1.581 | 1.581 | +0.000 |
| oss | 0.941 | 0.941 | +0.000 | 1.488 | 1.488 | +0.000 |
| oss_l2 | 0.824 | 0.824 | +0.000 | 1.536 | 1.536 | +0.000 |
| oss_markov | 0.765 | 0.765 | +0.000 | 1.039 | 1.039 | +0.000 |
| **oss_merge_sim08** | 0.824 | 0.824 | +0.000 | 1.702 | 1.734 | +0.032 |
| oss_refinement | 0.824 | 0.824 | +0.000 | 1.567 | 1.567 | +0.000 |
| oss_selective_r5 | 0.765 | 0.765 | +0.000 | 1.405 | 1.405 | +0.000 |

### terra -- 32 aligned problems, 6 arms

| arm | fast_p@1.0 stored | uniform | delta | geomean stored | uniform | delta |
|---|---|---|---|---|---|---|
| **terra_compress** | 0.781 | 0.875 | +0.094 | 2.466 | 3.074 | +0.608 |
| **terra_deletion** | 0.938 | 0.938 | +0.000 | 2.065 | 2.605 | +0.540 |
| **terra** | 0.906 | 0.906 | +0.000 | 2.358 | 2.923 | +0.565 |
| **terra_l2** | 0.844 | 0.844 | +0.000 | 1.931 | 2.515 | +0.583 |
| **terra_markov** | 0.906 | 0.906 | +0.000 | 1.835 | 2.151 | +0.316 |
| **terra_selective_r5** | 0.812 | 0.844 | +0.031 | 1.866 | 2.205 | +0.339 |

## What this does not fix

Only the reported metric. At run time `is_hack=True` also vetoed `is_new_best`, so a
suppressed kernel was never banked and the agent kept debugging from a worse state.
Re-scoring recovers the number, not the search trajectory.

