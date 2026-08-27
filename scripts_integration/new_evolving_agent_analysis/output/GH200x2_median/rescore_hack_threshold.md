# Uniform-threshold re-score (30x)

Generated 2026-08-27T01:48:14.421269+00:00. Baseline: `NVIDIA_GH200x2_median`.

> **STATUS: COMPLETE.** All 15 arms finished 50/50 with
> `run_summary.json`. Levels are quotable. **But n=1 replicate per cell:** per
> `ANALYSIS_RULES.md` and open item 9 (log-SD 0.147 across identical-config
> replicates), a single replicate cannot support an arm-vs-arm winner claim.

## The seam

`src/kernelbench/eval.py` changed `excessive_speedup_threshold` 10 -> 30 at
**2026-08-24T15:11:45** (commit `588a6a5`). eval.py is re-imported by every eval spawn, so it
reached all live arms with no restart and no log line. `is_hack` gates which iterations
may form a best, so identical kernels scored differently either side of that instant.

- evals whose label changes under a uniform 30x: **268** of 22500

- all flips are pre-seam evals in the (10, 30] band; re-scoring can only *raise* them,
  so this is a one-directional correction toward the post-seam arms.


## Aligned within model

Cross-model contrast is invalid (different endpoints, GPUs, latency), so each model
group is aligned on the problems common to all of its arms.


### gpt-oss-120b -- 50 aligned problems, 9 arms

| arm | fast_p@1.0 stored | uniform | delta | geomean stored | uniform | delta |
|---|---|---|---|---|---|---|
| oss_compress | 0.780 | 0.780 | +0.000 | 1.336 | 1.336 | +0.000 |
| oss_deletion | 0.700 | 0.700 | +0.000 | 1.382 | 1.382 | +0.000 |
| oss_folding | 0.660 | 0.660 | +0.000 | 1.443 | 1.443 | +0.000 |
| **oss** | 0.660 | 0.660 | +0.000 | 1.291 | 1.325 | +0.034 |
| oss_l2 | 0.720 | 0.720 | +0.000 | 1.471 | 1.471 | +0.000 |
| **oss_markov** | 0.700 | 0.700 | +0.000 | 1.189 | 1.196 | +0.007 |
| oss_merge_sim08 | 0.640 | 0.640 | +0.000 | 1.301 | 1.301 | +0.000 |
| oss_refinement | 0.680 | 0.680 | +0.000 | 1.525 | 1.525 | +0.000 |
| oss_selective_r5 | 0.660 | 0.660 | +0.000 | 1.402 | 1.402 | +0.000 |

### terra -- 50 aligned problems, 6 arms

| arm | fast_p@1.0 stored | uniform | delta | geomean stored | uniform | delta |
|---|---|---|---|---|---|---|
| **terra_compress** | 0.840 | 0.900 | +0.060 | 2.901 | 3.308 | +0.407 |
| **terra_deletion** | 0.980 | 0.980 | +0.000 | 2.479 | 2.877 | +0.398 |
| **terra** | 0.860 | 0.860 | +0.000 | 2.417 | 2.773 | +0.356 |
| **terra_l2** | 0.800 | 0.800 | +0.000 | 1.966 | 2.328 | +0.362 |
| **terra_markov** | 0.840 | 0.840 | +0.000 | 1.758 | 1.951 | +0.193 |
| **terra_selective_r5** | 0.820 | 0.840 | +0.020 | 2.049 | 2.276 | +0.227 |

## What this does not fix

Only the reported metric. At run time `is_hack=True` also vetoed `is_new_best`, so a
suppressed kernel was never banked and the agent kept debugging from a worse state.
Re-scoring recovers the number, not the search trajectory.

