# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-28T07:59:09.887619+00:00`
- aggregate_generated_at_utc: `2026-08-28T07:59:09.841448+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-5.6-terra`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2_2nd/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 18
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_5_6_terra_r2_compress_itr30_GH200_2026_08_25_10_43` | complete | compress_trigger | gpt-5.6-terra | inference |
| R2 | `base_agent_gpt_5_6_terra_r2_deletion_itr30_GH200_2026_08_25_10_43` | complete | truncation | gpt-5.6-terra | inference |
| R3 | `base_agent_gpt_5_6_terra_r2_folding_itr30_GH200_2026_08_25_10_43` | complete | folding | gpt-5.6-terra | inference |
| R4 | `base_agent_gpt_5_6_terra_r2_itr30_GH200_2026_08_25_10_41` | complete | truncation | gpt-5.6-terra | inference |
| R5 | `base_agent_gpt_5_6_terra_r2_markov_itr30_GH200_2026_08_25_10_42` | complete | markov_report | gpt-5.6-terra | inference |
| R6 | `base_agent_gpt_5_6_terra_r2_merge_sim08_itr30_GH200_2026_08_25_10_44` | complete | truncation | gpt-5.6-terra | inference |
| R7 | `base_agent_gpt_5_6_terra_r2_merge_sim09_itr30_GH200_2026_08_25_10_45` | complete | truncation | gpt-5.6-terra | inference |
| R8 | `base_agent_gpt_5_6_terra_r2_refinement_itr30_GH200_2026_08_25_10_44` | complete | truncation | gpt-5.6-terra | inference |
| R9 | `base_agent_gpt_5_6_terra_r2_selective_r5_itr30_GH200_2026_08_25_10_42` | complete | selective_retention | gpt-5.6-terra | inference |
| R10 | `base_agent_gpt_5_6_terra_r3_compress_itr30_GH200_2026_08_25_10_43` | complete | compress_trigger | gpt-5.6-terra | inference |
| R11 | `base_agent_gpt_5_6_terra_r3_deletion_itr30_GH200_2026_08_25_10_43` | complete | truncation | gpt-5.6-terra | inference |
| R12 | `base_agent_gpt_5_6_terra_r3_folding_itr30_GH200_2026_08_25_10_43` | complete | folding | gpt-5.6-terra | inference |
| R13 | `base_agent_gpt_5_6_terra_r3_itr30_GH200_2026_08_25_10_41` | complete | truncation | gpt-5.6-terra | inference |
| R14 | `base_agent_gpt_5_6_terra_r3_markov_itr30_GH200_2026_08_25_10_42` | complete | markov_report | gpt-5.6-terra | inference |
| R15 | `base_agent_gpt_5_6_terra_r3_merge_sim07_itr30_GH200_2026_08_25_10_45` | complete | truncation | gpt-5.6-terra | inference |
| R16 | `base_agent_gpt_5_6_terra_r3_merge_sim08_itr30_GH200_2026_08_25_10_44` | complete | truncation | gpt-5.6-terra | inference |
| R17 | `base_agent_gpt_5_6_terra_r3_refinement_itr30_GH200_2026_08_25_10_44` | complete | truncation | gpt-5.6-terra | inference |
| R18 | `base_agent_gpt_5_6_terra_r3_selective_r5_itr30_GH200_2026_08_25_10_42` | complete | selective_retention | gpt-5.6-terra | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 60.55 | 72.7 | 8 |
| R2 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 69.23 | 83.1 | 7 |
| R3 | folding | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 65.44 | 78.5 | 8 |
| R4 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 61.08 | 73.3 | 7 |
| R5 | markov_report | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 60.88 | 73.1 | 3 |
| R6 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 64.12 | 76.9 | 5 |
| R7 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 62.63 | 75.2 | 4 |
| R8 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 63.97 | 76.8 | 6 |
| R9 | selective_retention | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 62.31 | 74.8 | 6 |
| R10 | compress_trigger | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 58.61 | 70.3 | 7 |
| R11 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 62.94 | 75.5 | 6 |
| R12 | folding | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 61.83 | 74.2 | 6 |
| R13 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 57.32 | 68.8 | 6 |
| R14 | markov_report | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 59.17 | 71.0 | 6 |
| R15 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 60.25 | 72.3 | 6 |
| R16 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 63.21 | 75.9 | 4 |
| R17 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 62.40 | 74.9 | 7 |
| R18 | selective_retention | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 59.17 | 71.0 | 9 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | complete | 50/50 | 1.000 | 0.800 | 0.320 | 2.1188 | 50 | 1.000 | 0.880 | 0.440 | 2.6319 | 50 |
| R2 | truncation+deletion | complete | 49/50 | 0.980 | 0.800 | 0.320 | 2.0157 | 49 | 0.980 | 0.860 | 0.540 | 2.7419 | 49 |
| R3 | folding | complete | 50/50 | 1.000 | 0.820 | 0.400 | 2.1869 | 50 | 1.000 | 0.820 | 0.460 | 2.6478 | 50 |
| R4 | truncation | complete | 50/50 | 0.980 | 0.820 | 0.440 | 2.3569 | 49 | 1.000 | 0.900 | 0.500 | 2.8414 | 50 |
| R5 | markov_report | complete | 49/50 | 0.980 | 0.680 | 0.220 | 1.5835 | 49 | 0.980 | 0.760 | 0.260 | 1.8215 | 49 |
| R6 | truncation+merge@0.8 | complete | 50/50 | 1.000 | 0.860 | 0.440 | 2.3790 | 50 | 1.000 | 0.900 | 0.560 | 2.9712 | 50 |
| R7 | truncation+merge@0.9 | complete | 50/50 | 0.980 | 0.820 | 0.300 | 1.9468 | 49 | 1.000 | 0.920 | 0.480 | 2.5947 | 50 |
| R8 | truncation+refine | complete | 50/50 | 1.000 | 0.800 | 0.340 | 1.9157 | 50 | 1.000 | 0.900 | 0.480 | 2.7309 | 50 |
| R9 | selective_retention | complete | 50/50 | 1.000 | 0.840 | 0.360 | 1.9012 | 50 | 1.000 | 0.900 | 0.420 | 2.4738 | 50 |
| R10 | compress_trigger | complete | 50/50 | 0.960 | 0.660 | 0.360 | 1.9464 | 48 | 1.000 | 0.840 | 0.460 | 2.5494 | 50 |
| R11 | truncation+deletion | complete | 50/50 | 1.000 | 0.720 | 0.340 | 1.9716 | 50 | 1.000 | 0.840 | 0.400 | 2.4654 | 50 |
| R12 | folding | complete | 50/50 | 1.000 | 0.740 | 0.360 | 1.8859 | 50 | 1.000 | 0.880 | 0.480 | 2.4031 | 50 |
| R13 | truncation | complete | 50/50 | 1.000 | 0.740 | 0.320 | 1.9489 | 50 | 1.000 | 0.860 | 0.500 | 2.5921 | 50 |
| R14 | markov_report | complete | 50/50 | 0.980 | 0.640 | 0.220 | 1.7332 | 49 | 1.000 | 0.720 | 0.360 | 2.1635 | 50 |
| R15 | truncation+merge@0.7 | complete | 50/50 | 0.980 | 0.820 | 0.300 | 1.8628 | 49 | 1.000 | 0.860 | 0.420 | 2.4642 | 50 |
| R16 | truncation+merge@0.8 | complete | 50/50 | 1.000 | 0.800 | 0.340 | 2.0191 | 50 | 1.000 | 0.920 | 0.480 | 2.6451 | 50 |
| R17 | truncation+refine | complete | 50/50 | 1.000 | 0.880 | 0.400 | 2.3251 | 50 | 1.000 | 0.900 | 0.520 | 2.9101 | 50 |
| R18 | selective_retention | complete | 50/50 | 1.000 | 0.760 | 0.340 | 2.0829 | 50 | 1.000 | 0.820 | 0.480 | 2.7940 | 50 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 5.3818 | 1.7878 | 2.6319 | 50 | 2.4637 | 47 | 29.1667 | 30 | 3 | 1.000 | 1.000 | 0.940 | 0.880 | 0.600 | 0.440 |
| R2 | 30 | 50 | 4.9194 | 2.0504 | 2.7419 | 49 | 2.6659 | 45 | 4.0406 | 40 | 3 | 0.980 | 0.980 | 0.940 | 0.860 | 0.680 | 0.540 |
| R3 | 30 | 50 | 5.4724 | 1.8239 | 2.6478 | 50 | 2.4298 | 50 | 23.0539 | 8 | 2 | 1.000 | 1.000 | 0.940 | 0.820 | 0.620 | 0.460 |
| R4 | 30 | 50 | 5.5127 | 1.9746 | 2.8414 | 50 | 2.5239 | 46 | 26.5517 | 26 | 5 | 1.000 | 1.000 | 0.960 | 0.900 | 0.640 | 0.500 |
| R5 | 30 | 50 | 3.3352 | 1.4330 | 1.8215 | 49 | 1.7122 | 46 | 29.6154 | 31 | 2 | 0.980 | 0.980 | 0.880 | 0.760 | 0.480 | 0.260 |
| R6 | 30 | 50 | 4.8381 | 2.1395 | 2.9712 | 50 | 2.7710 | 50 | 9.5337 | 7 | 3 | 1.000 | 1.000 | 0.940 | 0.900 | 0.740 | 0.560 |
| R7 | 30 | 50 | 4.4598 | 1.9466 | 2.5947 | 50 | 2.0892 | 45 | 4.8242 | 36 | 4 | 1.000 | 1.000 | 0.940 | 0.920 | 0.660 | 0.480 |
| R8 | 30 | 50 | 5.1993 | 1.9080 | 2.7309 | 50 | 2.6572 | 47 | 27.1127 | 21 | 4 | 1.000 | 1.000 | 0.960 | 0.900 | 0.700 | 0.480 |
| R9 | 30 | 50 | 4.6186 | 1.7353 | 2.4738 | 50 | 2.3797 | 49 | 22.3134 | 11 | 3 | 1.000 | 1.000 | 0.960 | 0.900 | 0.640 | 0.420 |
| R10 | 30 | 50 | 4.7257 | 1.9385 | 2.5494 | 50 | 2.1549 | 47 | 19.1542 | 30 | 2 | 1.000 | 1.000 | 0.940 | 0.840 | 0.600 | 0.460 |
| R11 | 30 | 50 | 5.0467 | 1.7913 | 2.4654 | 50 | 2.1170 | 45 | 29.5263 | 22 | 5 | 1.000 | 1.000 | 0.880 | 0.840 | 0.640 | 0.400 |
| R12 | 30 | 50 | 4.3495 | 1.8551 | 2.4031 | 50 | 2.4577 | 44 | 22.8244 | 28 | 2 | 1.000 | 1.000 | 0.920 | 0.880 | 0.620 | 0.480 |
| R13 | 30 | 50 | 4.9586 | 1.9619 | 2.5921 | 50 | 2.3447 | 47 | 4.1895 | 28 | 4 | 1.000 | 1.000 | 0.940 | 0.860 | 0.660 | 0.500 |
| R14 | 30 | 50 | 4.4772 | 1.6293 | 2.1635 | 50 | 1.8028 | 43 | 25.8389 | 46 | 3 | 1.000 | 1.000 | 0.900 | 0.720 | 0.580 | 0.360 |
| R15 | 30 | 50 | 4.9322 | 1.6229 | 2.4642 | 50 | 2.0728 | 47 | 29.6154 | 19 | 2 | 1.000 | 1.000 | 0.960 | 0.860 | 0.620 | 0.420 |
| R16 | 30 | 50 | 4.7429 | 1.9560 | 2.6451 | 50 | 2.4671 | 49 | 26.0930 | 8 | 2 | 1.000 | 1.000 | 0.960 | 0.920 | 0.680 | 0.480 |
| R17 | 30 | 50 | 5.1439 | 2.0770 | 2.9101 | 50 | 2.5746 | 47 | 29.8404 | 23 | 2 | 1.000 | 1.000 | 0.980 | 0.900 | 0.720 | 0.520 |
| R18 | 30 | 50 | 5.4477 | 1.9005 | 2.7940 | 50 | 2.4839 | 47 | 25.6164 | 38 | 4 | 1.000 | 1.000 | 0.940 | 0.820 | 0.700 | 0.480 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 240 | 240 | 0 | 0 | 0 | 0 | 1 |
| R2 | yes | no | no | 326 | 12 | 0 | 314 | 0 | 314 | 3 |
| R3 | no | no | no | 392 | 392 | 0 | 0 | 0 | 0 | 1 |
| R4 | no | no | no | 335 | 335 | 0 | 0 | 0 | 0 | 1 |
| R5 | no | no | no | 354 | 354 | 0 | 0 | 0 | 0 | 1 |
| R6 | no | yes | no | 402 | 90 | 57 | 0 | 0 | 0 | 7 |
| R7 | no | yes | no | 352 | 319 | 14 | 0 | 0 | 0 | 7 |
| R8 | no | no | yes | 365 | 344 | 0 | 0 | 21 | 0 | 2 |
| R9 | no | no | no | 249 | 249 | 0 | 0 | 0 | 0 | 1 |
| R10 | no | no | no | 245 | 245 | 0 | 0 | 0 | 0 | 1 |
| R11 | yes | no | no | 318 | 15 | 0 | 303 | 0 | 303 | 3 |
| R12 | no | no | no | 398 | 398 | 0 | 0 | 0 | 0 | 1 |
| R13 | no | no | no | 345 | 345 | 0 | 0 | 0 | 0 | 1 |
| R14 | no | no | no | 339 | 339 | 0 | 0 | 0 | 0 | 1 |
| R15 | no | yes | no | 362 | 26 | 28 | 0 | 0 | 0 | 7 |
| R16 | no | yes | no | 392 | 122 | 55 | 0 | 0 | 0 | 7 |
| R17 | no | no | yes | 369 | 334 | 0 | 0 | 35 | 0 | 2 |
| R18 | no | no | no | 254 | 254 | 0 | 0 | 0 | 0 | 1 |

## Deltas vs baseline run `base_agent_gpt_5_6_terra_r2_itr30_GH200_2026_08_25_10_41`

### `base_agent_gpt_5_6_terra_r2_compress_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 29.1667 | +2.6149 | +9.8% | better |
| speedup_best_mean | 5.5127 | 5.3818 | -0.1309 | -2.4% | worse |
| speedup_best_median | 1.9746 | 1.7878 | -0.1868 | -9.5% | worse |
| speedup_best_geomean | 2.8414 | 2.6319 | -0.2095 | -7.4% | worse |
| speedup_current_geomean | 2.5239 | 2.4637 | -0.0602 | -2.4% | worse |
| hack_iteration_count | 26 | 30 | +4 | +15.4% | worse |
| problems_with_hack | 5 | 3 | -2 | -40.0% | better |
| l1_entry_count | 335 | 240 | -95 | -28.4% | better |
| total_wall_time_hours | 61.082 | 60.549 | -0.533 | -0.9% | better |
| avg_wall_time_min | 73.299 | 72.659 | -0.640 | -0.9% | better |

### `base_agent_gpt_5_6_terra_r2_deletion_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 49 | -1 | -2.0% | worse |
| correct_rate | 1.0000 | 0.9800 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 26.5517 | 4.0406 | -22.5111 | -84.8% | worse |
| speedup_best_mean | 5.5127 | 4.9194 | -0.5933 | -10.8% | worse |
| speedup_best_median | 1.9746 | 2.0504 | +0.0758 | +3.8% | better |
| speedup_best_geomean | 2.8414 | 2.7419 | -0.0995 | -3.5% | worse |
| speedup_current_geomean | 2.5239 | 2.6659 | +0.1420 | +5.6% | better |
| hack_iteration_count | 26 | 40 | +14 | +53.8% | worse |
| problems_with_hack | 5 | 3 | -2 | -40.0% | better |
| l1_entry_count | 335 | 326 | -9 | -2.7% | better |
| total_wall_time_hours | 61.082 | 69.229 | +8.147 | +13.3% | worse |
| avg_wall_time_min | 73.299 | 83.075 | +9.776 | +13.3% | worse |

### `base_agent_gpt_5_6_terra_r2_folding_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 23.0539 | -3.4978 | -13.2% | worse |
| speedup_best_mean | 5.5127 | 5.4724 | -0.0403 | -0.7% | worse |
| speedup_best_median | 1.9746 | 1.8239 | -0.1507 | -7.6% | worse |
| speedup_best_geomean | 2.8414 | 2.6478 | -0.1936 | -6.8% | worse |
| speedup_current_geomean | 2.5239 | 2.4298 | -0.0941 | -3.7% | worse |
| hack_iteration_count | 26 | 8 | -18 | -69.2% | better |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 392 | +57 | +17.0% | worse |
| total_wall_time_hours | 61.082 | 65.444 | +4.361 | +7.1% | worse |
| avg_wall_time_min | 73.299 | 78.532 | +5.234 | +7.1% | worse |

### `base_agent_gpt_5_6_terra_r2_markov_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 49 | -1 | -2.0% | worse |
| correct_rate | 1.0000 | 0.9800 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 26.5517 | 29.6154 | +3.0637 | +11.5% | better |
| speedup_best_mean | 5.5127 | 3.3352 | -2.1775 | -39.5% | worse |
| speedup_best_median | 1.9746 | 1.4330 | -0.5416 | -27.4% | worse |
| speedup_best_geomean | 2.8414 | 1.8215 | -1.0199 | -35.9% | worse |
| speedup_current_geomean | 2.5239 | 1.7122 | -0.8117 | -32.2% | worse |
| hack_iteration_count | 26 | 31 | +5 | +19.2% | worse |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 354 | +19 | +5.7% | worse |
| total_wall_time_hours | 61.082 | 60.879 | -0.203 | -0.3% | better |
| avg_wall_time_min | 73.299 | 73.055 | -0.244 | -0.3% | better |

### `base_agent_gpt_5_6_terra_r2_merge_sim08_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 9.5337 | -17.0180 | -64.1% | worse |
| speedup_best_mean | 5.5127 | 4.8381 | -0.6746 | -12.2% | worse |
| speedup_best_median | 1.9746 | 2.1395 | +0.1649 | +8.3% | better |
| speedup_best_geomean | 2.8414 | 2.9712 | +0.1298 | +4.6% | better |
| speedup_current_geomean | 2.5239 | 2.7710 | +0.2470 | +9.8% | better |
| hack_iteration_count | 26 | 7 | -19 | -73.1% | better |
| problems_with_hack | 5 | 3 | -2 | -40.0% | better |
| l1_entry_count | 335 | 402 | +67 | +20.0% | worse |
| total_wall_time_hours | 61.082 | 64.120 | +3.038 | +5.0% | worse |
| avg_wall_time_min | 73.299 | 76.944 | +3.645 | +5.0% | worse |

### `base_agent_gpt_5_6_terra_r2_merge_sim09_itr30_GH200_2026_08_25_10_45`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 4.8242 | -21.7275 | -81.8% | worse |
| speedup_best_mean | 5.5127 | 4.4598 | -1.0529 | -19.1% | worse |
| speedup_best_median | 1.9746 | 1.9466 | -0.0280 | -1.4% | worse |
| speedup_best_geomean | 2.8414 | 2.5947 | -0.2467 | -8.7% | worse |
| speedup_current_geomean | 2.5239 | 2.0892 | -0.4347 | -17.2% | worse |
| hack_iteration_count | 26 | 36 | +10 | +38.5% | worse |
| problems_with_hack | 5 | 4 | -1 | -20.0% | better |
| l1_entry_count | 335 | 352 | +17 | +5.1% | worse |
| total_wall_time_hours | 61.082 | 62.630 | +1.548 | +2.5% | worse |
| avg_wall_time_min | 73.299 | 75.156 | +1.858 | +2.5% | worse |

### `base_agent_gpt_5_6_terra_r2_refinement_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 27.1127 | +0.5610 | +2.1% | better |
| speedup_best_mean | 5.5127 | 5.1993 | -0.3134 | -5.7% | worse |
| speedup_best_median | 1.9746 | 1.9080 | -0.0666 | -3.4% | worse |
| speedup_best_geomean | 2.8414 | 2.7309 | -0.1105 | -3.9% | worse |
| speedup_current_geomean | 2.5239 | 2.6572 | +0.1332 | +5.3% | better |
| hack_iteration_count | 26 | 21 | -5 | -19.2% | better |
| problems_with_hack | 5 | 4 | -1 | -20.0% | better |
| l1_entry_count | 335 | 365 | +30 | +9.0% | worse |
| total_wall_time_hours | 61.082 | 63.973 | +2.891 | +4.7% | worse |
| avg_wall_time_min | 73.299 | 76.767 | +3.469 | +4.7% | worse |

### `base_agent_gpt_5_6_terra_r2_selective_r5_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 22.3134 | -4.2383 | -16.0% | worse |
| speedup_best_mean | 5.5127 | 4.6186 | -0.8941 | -16.2% | worse |
| speedup_best_median | 1.9746 | 1.7353 | -0.2393 | -12.1% | worse |
| speedup_best_geomean | 2.8414 | 2.4738 | -0.3676 | -12.9% | worse |
| speedup_current_geomean | 2.5239 | 2.3797 | -0.1442 | -5.7% | worse |
| hack_iteration_count | 26 | 11 | -15 | -57.7% | better |
| problems_with_hack | 5 | 3 | -2 | -40.0% | better |
| l1_entry_count | 335 | 249 | -86 | -25.7% | better |
| total_wall_time_hours | 61.082 | 62.310 | +1.228 | +2.0% | worse |
| avg_wall_time_min | 73.299 | 74.772 | +1.473 | +2.0% | worse |

### `base_agent_gpt_5_6_terra_r3_compress_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 19.1542 | -7.3975 | -27.9% | worse |
| speedup_best_mean | 5.5127 | 4.7257 | -0.7870 | -14.3% | worse |
| speedup_best_median | 1.9746 | 1.9385 | -0.0362 | -1.8% | worse |
| speedup_best_geomean | 2.8414 | 2.5494 | -0.2920 | -10.3% | worse |
| speedup_current_geomean | 2.5239 | 2.1549 | -0.3691 | -14.6% | worse |
| hack_iteration_count | 26 | 30 | +4 | +15.4% | worse |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 245 | -90 | -26.9% | better |
| total_wall_time_hours | 61.082 | 58.608 | -2.474 | -4.1% | better |
| avg_wall_time_min | 73.299 | 70.330 | -2.969 | -4.1% | better |

### `base_agent_gpt_5_6_terra_r3_deletion_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 29.5263 | +2.9746 | +11.2% | better |
| speedup_best_mean | 5.5127 | 5.0467 | -0.4660 | -8.5% | worse |
| speedup_best_median | 1.9746 | 1.7913 | -0.1833 | -9.3% | worse |
| speedup_best_geomean | 2.8414 | 2.4654 | -0.3760 | -13.2% | worse |
| speedup_current_geomean | 2.5239 | 2.1170 | -0.4070 | -16.1% | worse |
| hack_iteration_count | 26 | 22 | -4 | -15.4% | better |
| problems_with_hack | 5 | 5 | +0 | +0.0% | same |
| l1_entry_count | 335 | 318 | -17 | -5.1% | better |
| total_wall_time_hours | 61.082 | 62.939 | +1.857 | +3.0% | worse |
| avg_wall_time_min | 73.299 | 75.527 | +2.229 | +3.0% | worse |

### `base_agent_gpt_5_6_terra_r3_folding_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 22.8244 | -3.7273 | -14.0% | worse |
| speedup_best_mean | 5.5127 | 4.3495 | -1.1632 | -21.1% | worse |
| speedup_best_median | 1.9746 | 1.8551 | -0.1195 | -6.1% | worse |
| speedup_best_geomean | 2.8414 | 2.4031 | -0.4383 | -15.4% | worse |
| speedup_current_geomean | 2.5239 | 2.4577 | -0.0663 | -2.6% | worse |
| hack_iteration_count | 26 | 28 | +2 | +7.7% | worse |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 398 | +63 | +18.8% | worse |
| total_wall_time_hours | 61.082 | 61.830 | +0.748 | +1.2% | worse |
| avg_wall_time_min | 73.299 | 74.196 | +0.898 | +1.2% | worse |

### `base_agent_gpt_5_6_terra_r3_itr30_GH200_2026_08_25_10_41`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 4.1895 | -22.3622 | -84.2% | worse |
| speedup_best_mean | 5.5127 | 4.9586 | -0.5541 | -10.1% | worse |
| speedup_best_median | 1.9746 | 1.9619 | -0.0127 | -0.6% | worse |
| speedup_best_geomean | 2.8414 | 2.5921 | -0.2493 | -8.8% | worse |
| speedup_current_geomean | 2.5239 | 2.3447 | -0.1792 | -7.1% | worse |
| hack_iteration_count | 26 | 28 | +2 | +7.7% | worse |
| problems_with_hack | 5 | 4 | -1 | -20.0% | better |
| l1_entry_count | 335 | 345 | +10 | +3.0% | worse |
| total_wall_time_hours | 61.082 | 57.324 | -3.758 | -6.2% | better |
| avg_wall_time_min | 73.299 | 68.789 | -4.510 | -6.2% | better |

### `base_agent_gpt_5_6_terra_r3_markov_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 25.8389 | -0.7128 | -2.7% | worse |
| speedup_best_mean | 5.5127 | 4.4772 | -1.0355 | -18.8% | worse |
| speedup_best_median | 1.9746 | 1.6293 | -0.3453 | -17.5% | worse |
| speedup_best_geomean | 2.8414 | 2.1635 | -0.6779 | -23.9% | worse |
| speedup_current_geomean | 2.5239 | 1.8028 | -0.7212 | -28.6% | worse |
| hack_iteration_count | 26 | 46 | +20 | +76.9% | worse |
| problems_with_hack | 5 | 3 | -2 | -40.0% | better |
| l1_entry_count | 335 | 339 | +4 | +1.2% | worse |
| total_wall_time_hours | 61.082 | 59.172 | -1.910 | -3.1% | better |
| avg_wall_time_min | 73.299 | 71.007 | -2.292 | -3.1% | better |

### `base_agent_gpt_5_6_terra_r3_merge_sim07_itr30_GH200_2026_08_25_10_45`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 29.6154 | +3.0637 | +11.5% | better |
| speedup_best_mean | 5.5127 | 4.9322 | -0.5805 | -10.5% | worse |
| speedup_best_median | 1.9746 | 1.6229 | -0.3517 | -17.8% | worse |
| speedup_best_geomean | 2.8414 | 2.4642 | -0.3772 | -13.3% | worse |
| speedup_current_geomean | 2.5239 | 2.0728 | -0.4511 | -17.9% | worse |
| hack_iteration_count | 26 | 19 | -7 | -26.9% | better |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 362 | +27 | +8.1% | worse |
| total_wall_time_hours | 61.082 | 60.250 | -0.832 | -1.4% | better |
| avg_wall_time_min | 73.299 | 72.300 | -0.998 | -1.4% | better |

### `base_agent_gpt_5_6_terra_r3_merge_sim08_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 26.0930 | -0.4587 | -1.7% | worse |
| speedup_best_mean | 5.5127 | 4.7429 | -0.7697 | -14.0% | worse |
| speedup_best_median | 1.9746 | 1.9560 | -0.0186 | -0.9% | worse |
| speedup_best_geomean | 2.8414 | 2.6451 | -0.1963 | -6.9% | worse |
| speedup_current_geomean | 2.5239 | 2.4671 | -0.0568 | -2.2% | worse |
| hack_iteration_count | 26 | 8 | -18 | -69.2% | better |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 392 | +57 | +17.0% | worse |
| total_wall_time_hours | 61.082 | 63.211 | +2.129 | +3.5% | worse |
| avg_wall_time_min | 73.299 | 75.853 | +2.554 | +3.5% | worse |

### `base_agent_gpt_5_6_terra_r3_refinement_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 29.8404 | +3.2887 | +12.4% | better |
| speedup_best_mean | 5.5127 | 5.1439 | -0.3687 | -6.7% | worse |
| speedup_best_median | 1.9746 | 2.0770 | +0.1024 | +5.2% | better |
| speedup_best_geomean | 2.8414 | 2.9101 | +0.0687 | +2.4% | better |
| speedup_current_geomean | 2.5239 | 2.5746 | +0.0506 | +2.0% | better |
| hack_iteration_count | 26 | 23 | -3 | -11.5% | better |
| problems_with_hack | 5 | 2 | -3 | -60.0% | better |
| l1_entry_count | 335 | 369 | +34 | +10.1% | worse |
| total_wall_time_hours | 61.082 | 62.400 | +1.318 | +2.2% | worse |
| avg_wall_time_min | 73.299 | 74.880 | +1.581 | +2.2% | worse |

### `base_agent_gpt_5_6_terra_r3_selective_r5_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 26.5517 | 25.6164 | -0.9353 | -3.5% | worse |
| speedup_best_mean | 5.5127 | 5.4477 | -0.0649 | -1.2% | worse |
| speedup_best_median | 1.9746 | 1.9005 | -0.0741 | -3.8% | worse |
| speedup_best_geomean | 2.8414 | 2.7940 | -0.0474 | -1.7% | worse |
| speedup_current_geomean | 2.5239 | 2.4839 | -0.0401 | -1.6% | worse |
| hack_iteration_count | 26 | 38 | +12 | +46.2% | worse |
| problems_with_hack | 5 | 4 | -1 | -20.0% | better |
| l1_entry_count | 335 | 254 | -81 | -24.2% | better |
| total_wall_time_hours | 61.082 | 59.168 | -1.914 | -3.1% | better |
| avg_wall_time_min | 73.299 | 71.001 | -2.297 | -3.1% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) | delta(R7-R4) | delta(R8-R4) | delta(R9-R4) | delta(R10-R4) | delta(R11-R4) | delta(R12-R4) | delta(R13-R4) | delta(R14-R4) | delta(R15-R4) | delta(R16-R4) | delta(R17-R4) | delta(R18-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4432 | 1.3928 | 1.3002 | 1.4745 | 1.1788 | 1.4711 | 1.0853 | 1.2443 | 1.5388 | 1.4649 | 1.3996 | 1.1943 | 1.5839 | 1.4489 | 1.3377 | 1.4359 | 1.7039 | 1.2531 | -0.0314 | -0.0817 | -0.1743 | -0.2957 | -0.0034 | -0.3892 | -0.2303 | +0.0643 | -0.0097 | -0.0750 | -0.2802 | +0.1094 | -0.0256 | -0.1368 | -0.0386 | +0.2293 | -0.2214 |
| 5 | 1.7532 | 1.7877 | 1.8249 | 2.1038 | 1.4508 | 1.9910 | 1.6995 | 1.5886 | 1.6787 | 1.6858 | 1.6514 | 1.6386 | 1.7248 | 1.6045 | 1.6504 | 1.7965 | 1.8779 | 1.7774 | -0.3506 | -0.3161 | -0.2789 | -0.6530 | -0.1128 | -0.4043 | -0.5152 | -0.4251 | -0.4180 | -0.4524 | -0.4652 | -0.3790 | -0.4993 | -0.4534 | -0.3073 | -0.2259 | -0.3264 |
| 10 | 2.1188 | 2.0157 | 2.1869 | 2.3569 | 1.5835 | 2.3790 | 1.9468 | 1.9157 | 1.9012 | 1.9464 | 1.9716 | 1.8859 | 1.9489 | 1.7332 | 1.8628 | 2.0191 | 2.3251 | 2.0829 | -0.2381 | -0.3412 | -0.1700 | -0.7734 | +0.0221 | -0.4101 | -0.4412 | -0.4557 | -0.4105 | -0.3853 | -0.4710 | -0.4080 | -0.6237 | -0.4941 | -0.3378 | -0.0318 | -0.2740 |
| 15 | 2.3187 | 2.2850 | 2.3036 | 2.5873 | 1.6428 | 2.7536 | 2.1450 | 2.3113 | 2.1980 | 2.1438 | 2.1145 | 2.1349 | 2.2319 | 1.8493 | 2.0156 | 2.4278 | 2.4884 | 2.2714 | -0.2686 | -0.3023 | -0.2837 | -0.9445 | +0.1664 | -0.4423 | -0.2759 | -0.3892 | -0.4434 | -0.4727 | -0.4524 | -0.3553 | -0.7379 | -0.5716 | -0.1595 | -0.0989 | -0.3158 |
| 20 | 2.4858 | 2.4907 | 2.4405 | 2.6833 | 1.7073 | 2.9057 | 2.3094 | 2.5672 | 2.2629 | 2.2580 | 2.3197 | 2.2563 | 2.3482 | 1.8820 | 2.2333 | 2.5868 | 2.7574 | 2.6031 | -0.1975 | -0.1926 | -0.2428 | -0.9760 | +0.2224 | -0.3739 | -0.1161 | -0.4205 | -0.4253 | -0.3636 | -0.4270 | -0.3352 | -0.8014 | -0.4500 | -0.0966 | +0.0741 | -0.0802 |
| 25 | 2.5447 | 2.6929 | 2.6048 | 2.7458 | 1.7262 | 2.9219 | 2.5511 | 2.6448 | 2.4523 | 2.3058 | 2.4053 | 2.3673 | 2.3930 | 2.1068 | 2.4124 | 2.6262 | 2.8488 | 2.7490 | -0.2012 | -0.0529 | -0.1410 | -1.0197 | +0.1761 | -0.1947 | -0.1010 | -0.2935 | -0.4400 | -0.3405 | -0.3785 | -0.3528 | -0.6391 | -0.3335 | -0.1196 | +0.1029 | +0.0031 |
| 30 | 2.6319 | 2.7419 | 2.6478 | 2.8414 | 1.8215 | 2.9712 | 2.5947 | 2.7309 | 2.4738 | 2.5494 | 2.4654 | 2.4031 | 2.5921 | 2.1635 | 2.4642 | 2.6451 | 2.9101 | 2.7940 | -0.2095 | -0.0995 | -0.1936 | -1.0199 | +0.1298 | -0.2467 | -0.1105 | -0.3676 | -0.2920 | -0.3760 | -0.4383 | -0.2493 | -0.6779 | -0.3772 | -0.1963 | +0.0687 | -0.0474 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) | delta(R7-R4) | delta(R8-R4) | delta(R9-R4) | delta(R10-R4) | delta(R11-R4) | delta(R12-R4) | delta(R13-R4) | delta(R14-R4) | delta(R15-R4) | delta(R16-R4) | delta(R17-R4) | delta(R18-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.400 | 0.380 | 0.380 | 0.380 | 0.340 | 0.360 | 0.320 | 0.300 | 0.340 | 0.320 | 0.380 | 0.260 | 0.460 | 0.360 | 0.300 | 0.360 | 0.340 | 0.300 | +0.020 | +0.000 | +0.000 | -0.040 | -0.020 | -0.060 | -0.080 | -0.040 | -0.060 | +0.000 | -0.120 | +0.080 | -0.020 | -0.080 | -0.020 | -0.040 | -0.080 |
| 5 | 0.640 | 0.720 | 0.760 | 0.760 | 0.580 | 0.760 | 0.680 | 0.660 | 0.780 | 0.620 | 0.660 | 0.640 | 0.680 | 0.600 | 0.720 | 0.720 | 0.800 | 0.600 | -0.120 | -0.040 | +0.000 | -0.180 | +0.000 | -0.080 | -0.100 | +0.020 | -0.140 | -0.100 | -0.120 | -0.080 | -0.160 | -0.040 | -0.040 | +0.040 | -0.160 |
| 10 | 0.800 | 0.800 | 0.820 | 0.820 | 0.680 | 0.860 | 0.820 | 0.800 | 0.840 | 0.660 | 0.720 | 0.740 | 0.740 | 0.640 | 0.820 | 0.800 | 0.880 | 0.760 | -0.020 | -0.020 | +0.000 | -0.140 | +0.040 | +0.000 | -0.020 | +0.020 | -0.160 | -0.100 | -0.080 | -0.080 | -0.180 | +0.000 | -0.020 | +0.060 | -0.060 |
| 15 | 0.840 | 0.840 | 0.820 | 0.860 | 0.700 | 0.880 | 0.900 | 0.820 | 0.860 | 0.800 | 0.780 | 0.820 | 0.820 | 0.660 | 0.840 | 0.860 | 0.880 | 0.780 | -0.020 | -0.020 | -0.040 | -0.160 | +0.020 | +0.040 | -0.040 | +0.000 | -0.060 | -0.080 | -0.040 | -0.040 | -0.200 | -0.020 | +0.000 | +0.020 | -0.080 |
| 20 | 0.860 | 0.860 | 0.820 | 0.880 | 0.700 | 0.880 | 0.900 | 0.880 | 0.860 | 0.800 | 0.840 | 0.840 | 0.840 | 0.660 | 0.860 | 0.920 | 0.900 | 0.800 | -0.020 | -0.020 | -0.060 | -0.180 | +0.000 | +0.020 | +0.000 | -0.020 | -0.080 | -0.040 | -0.040 | -0.040 | -0.220 | -0.020 | +0.040 | +0.020 | -0.080 |
| 25 | 0.880 | 0.860 | 0.820 | 0.880 | 0.700 | 0.900 | 0.900 | 0.900 | 0.900 | 0.800 | 0.840 | 0.880 | 0.840 | 0.720 | 0.860 | 0.920 | 0.900 | 0.820 | +0.000 | -0.020 | -0.060 | -0.180 | +0.020 | +0.020 | +0.020 | +0.020 | -0.080 | -0.040 | +0.000 | -0.040 | -0.160 | -0.020 | +0.040 | +0.020 | -0.060 |
| 30 | 0.880 | 0.860 | 0.820 | 0.900 | 0.760 | 0.900 | 0.920 | 0.900 | 0.900 | 0.840 | 0.840 | 0.880 | 0.860 | 0.720 | 0.860 | 0.920 | 0.900 | 0.820 | -0.020 | -0.040 | -0.080 | -0.140 | +0.000 | +0.020 | +0.000 | +0.000 | -0.060 | -0.060 | -0.020 | -0.040 | -0.180 | -0.040 | +0.020 | +0.000 | -0.080 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_5_6_terra_r2_itr30_GH200_2026_08_25_10_41`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | best_geomean | 30 | 2.8414 | 2.6319 | -0.2095 | -7.4% |
| R1 | fast_p_best@1.0 | 30 | 0.900 | 0.880 | -0.020 | -2.2% |
| R2 | best_geomean | 30 | 2.8414 | 2.7419 | -0.0995 | -3.5% |
| R2 | fast_p_best@1.0 | 30 | 0.900 | 0.860 | -0.040 | -4.4% |
| R3 | best_geomean | 30 | 2.8414 | 2.6478 | -0.1936 | -6.8% |
| R3 | fast_p_best@1.0 | 30 | 0.900 | 0.820 | -0.080 | -8.9% |
| R5 | best_geomean | 30 | 2.8414 | 1.8215 | -1.0199 | -35.9% |
| R5 | fast_p_best@1.0 | 30 | 0.900 | 0.760 | -0.140 | -15.6% |
| R6 | best_geomean | 30 | 2.8414 | 2.9712 | +0.1298 | +4.6% |
| R6 | fast_p_best@1.0 | 30 | 0.900 | 0.900 | +0.000 | +0.0% |
| R7 | best_geomean | 30 | 2.8414 | 2.5947 | -0.2467 | -8.7% |
| R7 | fast_p_best@1.0 | 30 | 0.900 | 0.920 | +0.020 | +2.2% |
| R8 | best_geomean | 30 | 2.8414 | 2.7309 | -0.1105 | -3.9% |
| R8 | fast_p_best@1.0 | 30 | 0.900 | 0.900 | +0.000 | +0.0% |
| R9 | best_geomean | 30 | 2.8414 | 2.4738 | -0.3676 | -12.9% |
| R9 | fast_p_best@1.0 | 30 | 0.900 | 0.900 | +0.000 | +0.0% |
| R10 | best_geomean | 30 | 2.8414 | 2.5494 | -0.2920 | -10.3% |
| R10 | fast_p_best@1.0 | 30 | 0.900 | 0.840 | -0.060 | -6.7% |
| R11 | best_geomean | 30 | 2.8414 | 2.4654 | -0.3760 | -13.2% |
| R11 | fast_p_best@1.0 | 30 | 0.900 | 0.840 | -0.060 | -6.7% |
| R12 | best_geomean | 30 | 2.8414 | 2.4031 | -0.4383 | -15.4% |
| R12 | fast_p_best@1.0 | 30 | 0.900 | 0.880 | -0.020 | -2.2% |
| R13 | best_geomean | 30 | 2.8414 | 2.5921 | -0.2493 | -8.8% |
| R13 | fast_p_best@1.0 | 30 | 0.900 | 0.860 | -0.040 | -4.4% |
| R14 | best_geomean | 30 | 2.8414 | 2.1635 | -0.6779 | -23.9% |
| R14 | fast_p_best@1.0 | 30 | 0.900 | 0.720 | -0.180 | -20.0% |
| R15 | best_geomean | 30 | 2.8414 | 2.4642 | -0.3772 | -13.3% |
| R15 | fast_p_best@1.0 | 30 | 0.900 | 0.860 | -0.040 | -4.4% |
| R16 | best_geomean | 30 | 2.8414 | 2.6451 | -0.1963 | -6.9% |
| R16 | fast_p_best@1.0 | 30 | 0.900 | 0.920 | +0.020 | +2.2% |
| R17 | best_geomean | 30 | 2.8414 | 2.9101 | +0.0687 | +2.4% |
| R17 | fast_p_best@1.0 | 30 | 0.900 | 0.900 | +0.000 | +0.0% |
| R18 | best_geomean | 30 | 2.8414 | 2.7940 | -0.0474 | -1.7% |
| R18 | fast_p_best@1.0 | 30 | 0.900 | 0.820 | -0.080 | -8.9% |
