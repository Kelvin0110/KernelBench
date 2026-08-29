# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-28T07:59:13.781649+00:00`
- aggregate_generated_at_utc: `2026-08-28T07:59:13.737620+00:00`
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

## Deltas vs baseline run `base_agent_gpt_5_6_terra_r3_itr30_GH200_2026_08_25_10_41`

### `base_agent_gpt_5_6_terra_r2_compress_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 29.1667 | +24.9772 | +596.2% | better |
| speedup_best_mean | 4.9586 | 5.3818 | +0.4232 | +8.5% | better |
| speedup_best_median | 1.9619 | 1.7878 | -0.1741 | -8.9% | worse |
| speedup_best_geomean | 2.5921 | 2.6319 | +0.0398 | +1.5% | better |
| speedup_current_geomean | 2.3447 | 2.4637 | +0.1190 | +5.1% | better |
| hack_iteration_count | 28 | 30 | +2 | +7.1% | worse |
| problems_with_hack | 4 | 3 | -1 | -25.0% | better |
| l1_entry_count | 345 | 240 | -105 | -30.4% | better |
| total_wall_time_hours | 57.324 | 60.549 | +3.225 | +5.6% | worse |
| avg_wall_time_min | 68.789 | 72.659 | +3.870 | +5.6% | worse |

### `base_agent_gpt_5_6_terra_r2_deletion_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 49 | -1 | -2.0% | worse |
| correct_rate | 1.0000 | 0.9800 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 4.1895 | 4.0406 | -0.1489 | -3.6% | worse |
| speedup_best_mean | 4.9586 | 4.9194 | -0.0391 | -0.8% | worse |
| speedup_best_median | 1.9619 | 2.0504 | +0.0885 | +4.5% | better |
| speedup_best_geomean | 2.5921 | 2.7419 | +0.1497 | +5.8% | better |
| speedup_current_geomean | 2.3447 | 2.6659 | +0.3212 | +13.7% | better |
| hack_iteration_count | 28 | 40 | +12 | +42.9% | worse |
| problems_with_hack | 4 | 3 | -1 | -25.0% | better |
| l1_entry_count | 345 | 326 | -19 | -5.5% | better |
| total_wall_time_hours | 57.324 | 69.229 | +11.905 | +20.8% | worse |
| avg_wall_time_min | 68.789 | 83.075 | +14.286 | +20.8% | worse |

### `base_agent_gpt_5_6_terra_r2_folding_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 23.0539 | +18.8644 | +450.3% | better |
| speedup_best_mean | 4.9586 | 5.4724 | +0.5139 | +10.4% | better |
| speedup_best_median | 1.9619 | 1.8239 | -0.1380 | -7.0% | worse |
| speedup_best_geomean | 2.5921 | 2.6478 | +0.0557 | +2.1% | better |
| speedup_current_geomean | 2.3447 | 2.4298 | +0.0851 | +3.6% | better |
| hack_iteration_count | 28 | 8 | -20 | -71.4% | better |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 392 | +47 | +13.6% | worse |
| total_wall_time_hours | 57.324 | 65.444 | +8.119 | +14.2% | worse |
| avg_wall_time_min | 68.789 | 78.532 | +9.743 | +14.2% | worse |

### `base_agent_gpt_5_6_terra_r2_itr30_GH200_2026_08_25_10_41`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 26.5517 | +22.3622 | +533.8% | better |
| speedup_best_mean | 4.9586 | 5.5127 | +0.5541 | +11.2% | better |
| speedup_best_median | 1.9619 | 1.9746 | +0.0127 | +0.6% | better |
| speedup_best_geomean | 2.5921 | 2.8414 | +0.2493 | +9.6% | better |
| speedup_current_geomean | 2.3447 | 2.5239 | +0.1792 | +7.6% | better |
| hack_iteration_count | 28 | 26 | -2 | -7.1% | better |
| problems_with_hack | 4 | 5 | +1 | +25.0% | worse |
| l1_entry_count | 345 | 335 | -10 | -2.9% | better |
| total_wall_time_hours | 57.324 | 61.082 | +3.758 | +6.6% | worse |
| avg_wall_time_min | 68.789 | 73.299 | +4.510 | +6.6% | worse |

### `base_agent_gpt_5_6_terra_r2_markov_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 49 | -1 | -2.0% | worse |
| correct_rate | 1.0000 | 0.9800 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 4.1895 | 29.6154 | +25.4259 | +606.9% | better |
| speedup_best_mean | 4.9586 | 3.3352 | -1.6234 | -32.7% | worse |
| speedup_best_median | 1.9619 | 1.4330 | -0.5289 | -27.0% | worse |
| speedup_best_geomean | 2.5921 | 1.8215 | -0.7706 | -29.7% | worse |
| speedup_current_geomean | 2.3447 | 1.7122 | -0.6325 | -27.0% | worse |
| hack_iteration_count | 28 | 31 | +3 | +10.7% | worse |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 354 | +9 | +2.6% | worse |
| total_wall_time_hours | 57.324 | 60.879 | +3.555 | +6.2% | worse |
| avg_wall_time_min | 68.789 | 73.055 | +4.266 | +6.2% | worse |

### `base_agent_gpt_5_6_terra_r2_merge_sim08_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 9.5337 | +5.3442 | +127.6% | better |
| speedup_best_mean | 4.9586 | 4.8381 | -0.1204 | -2.4% | worse |
| speedup_best_median | 1.9619 | 2.1395 | +0.1775 | +9.0% | better |
| speedup_best_geomean | 2.5921 | 2.9712 | +0.3791 | +14.6% | better |
| speedup_current_geomean | 2.3447 | 2.7710 | +0.4263 | +18.2% | better |
| hack_iteration_count | 28 | 7 | -21 | -75.0% | better |
| problems_with_hack | 4 | 3 | -1 | -25.0% | better |
| l1_entry_count | 345 | 402 | +57 | +16.5% | worse |
| total_wall_time_hours | 57.324 | 64.120 | +6.795 | +11.9% | worse |
| avg_wall_time_min | 68.789 | 76.944 | +8.155 | +11.9% | worse |

### `base_agent_gpt_5_6_terra_r2_merge_sim09_itr30_GH200_2026_08_25_10_45`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 4.8242 | +0.6348 | +15.2% | better |
| speedup_best_mean | 4.9586 | 4.4598 | -0.4988 | -10.1% | worse |
| speedup_best_median | 1.9619 | 1.9466 | -0.0153 | -0.8% | worse |
| speedup_best_geomean | 2.5921 | 2.5947 | +0.0025 | +0.1% | better |
| speedup_current_geomean | 2.3447 | 2.0892 | -0.2555 | -10.9% | worse |
| hack_iteration_count | 28 | 36 | +8 | +28.6% | worse |
| problems_with_hack | 4 | 4 | +0 | +0.0% | same |
| l1_entry_count | 345 | 352 | +7 | +2.0% | worse |
| total_wall_time_hours | 57.324 | 62.630 | +5.306 | +9.3% | worse |
| avg_wall_time_min | 68.789 | 75.156 | +6.367 | +9.3% | worse |

### `base_agent_gpt_5_6_terra_r2_refinement_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 27.1127 | +22.9232 | +547.2% | better |
| speedup_best_mean | 4.9586 | 5.1993 | +0.2408 | +4.9% | better |
| speedup_best_median | 1.9619 | 1.9080 | -0.0539 | -2.7% | worse |
| speedup_best_geomean | 2.5921 | 2.7309 | +0.1388 | +5.4% | better |
| speedup_current_geomean | 2.3447 | 2.6572 | +0.3125 | +13.3% | better |
| hack_iteration_count | 28 | 21 | -7 | -25.0% | better |
| problems_with_hack | 4 | 4 | +0 | +0.0% | same |
| l1_entry_count | 345 | 365 | +20 | +5.8% | worse |
| total_wall_time_hours | 57.324 | 63.973 | +6.648 | +11.6% | worse |
| avg_wall_time_min | 68.789 | 76.767 | +7.978 | +11.6% | worse |

### `base_agent_gpt_5_6_terra_r2_selective_r5_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 22.3134 | +18.1240 | +432.6% | better |
| speedup_best_mean | 4.9586 | 4.6186 | -0.3400 | -6.9% | worse |
| speedup_best_median | 1.9619 | 1.7353 | -0.2266 | -11.5% | worse |
| speedup_best_geomean | 2.5921 | 2.4738 | -0.1183 | -4.6% | worse |
| speedup_current_geomean | 2.3447 | 2.3797 | +0.0350 | +1.5% | better |
| hack_iteration_count | 28 | 11 | -17 | -60.7% | better |
| problems_with_hack | 4 | 3 | -1 | -25.0% | better |
| l1_entry_count | 345 | 249 | -96 | -27.8% | better |
| total_wall_time_hours | 57.324 | 62.310 | +4.986 | +8.7% | worse |
| avg_wall_time_min | 68.789 | 74.772 | +5.983 | +8.7% | worse |

### `base_agent_gpt_5_6_terra_r3_compress_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 19.1542 | +14.9648 | +357.2% | better |
| speedup_best_mean | 4.9586 | 4.7257 | -0.2328 | -4.7% | worse |
| speedup_best_median | 1.9619 | 1.9385 | -0.0235 | -1.2% | worse |
| speedup_best_geomean | 2.5921 | 2.5494 | -0.0427 | -1.6% | worse |
| speedup_current_geomean | 2.3447 | 2.1549 | -0.1898 | -8.1% | worse |
| hack_iteration_count | 28 | 30 | +2 | +7.1% | worse |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 245 | -100 | -29.0% | better |
| total_wall_time_hours | 57.324 | 58.608 | +1.284 | +2.2% | worse |
| avg_wall_time_min | 68.789 | 70.330 | +1.541 | +2.2% | worse |

### `base_agent_gpt_5_6_terra_r3_deletion_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 29.5263 | +25.3368 | +604.8% | better |
| speedup_best_mean | 4.9586 | 5.0467 | +0.0881 | +1.8% | better |
| speedup_best_median | 1.9619 | 1.7913 | -0.1706 | -8.7% | worse |
| speedup_best_geomean | 2.5921 | 2.4654 | -0.1267 | -4.9% | worse |
| speedup_current_geomean | 2.3447 | 2.1170 | -0.2277 | -9.7% | worse |
| hack_iteration_count | 28 | 22 | -6 | -21.4% | better |
| problems_with_hack | 4 | 5 | +1 | +25.0% | worse |
| l1_entry_count | 345 | 318 | -27 | -7.8% | better |
| total_wall_time_hours | 57.324 | 62.939 | +5.615 | +9.8% | worse |
| avg_wall_time_min | 68.789 | 75.527 | +6.738 | +9.8% | worse |

### `base_agent_gpt_5_6_terra_r3_folding_itr30_GH200_2026_08_25_10_43`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 22.8244 | +18.6350 | +444.8% | better |
| speedup_best_mean | 4.9586 | 4.3495 | -0.6091 | -12.3% | worse |
| speedup_best_median | 1.9619 | 1.8551 | -0.1068 | -5.4% | worse |
| speedup_best_geomean | 2.5921 | 2.4031 | -0.1890 | -7.3% | worse |
| speedup_current_geomean | 2.3447 | 2.4577 | +0.1130 | +4.8% | better |
| hack_iteration_count | 28 | 28 | +0 | +0.0% | same |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 398 | +53 | +15.4% | worse |
| total_wall_time_hours | 57.324 | 61.830 | +4.506 | +7.9% | worse |
| avg_wall_time_min | 68.789 | 74.196 | +5.407 | +7.9% | worse |

### `base_agent_gpt_5_6_terra_r3_markov_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 25.8389 | +21.6495 | +516.8% | better |
| speedup_best_mean | 4.9586 | 4.4772 | -0.4814 | -9.7% | worse |
| speedup_best_median | 1.9619 | 1.6293 | -0.3326 | -17.0% | worse |
| speedup_best_geomean | 2.5921 | 2.1635 | -0.4287 | -16.5% | worse |
| speedup_current_geomean | 2.3447 | 1.8028 | -0.5419 | -23.1% | worse |
| hack_iteration_count | 28 | 46 | +18 | +64.3% | worse |
| problems_with_hack | 4 | 3 | -1 | -25.0% | better |
| l1_entry_count | 345 | 339 | -6 | -1.7% | better |
| total_wall_time_hours | 57.324 | 59.172 | +1.848 | +3.2% | worse |
| avg_wall_time_min | 68.789 | 71.007 | +2.218 | +3.2% | worse |

### `base_agent_gpt_5_6_terra_r3_merge_sim07_itr30_GH200_2026_08_25_10_45`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 29.6154 | +25.4259 | +606.9% | better |
| speedup_best_mean | 4.9586 | 4.9322 | -0.0264 | -0.5% | worse |
| speedup_best_median | 1.9619 | 1.6229 | -0.3390 | -17.3% | worse |
| speedup_best_geomean | 2.5921 | 2.4642 | -0.1279 | -4.9% | worse |
| speedup_current_geomean | 2.3447 | 2.0728 | -0.2719 | -11.6% | worse |
| hack_iteration_count | 28 | 19 | -9 | -32.1% | better |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 362 | +17 | +4.9% | worse |
| total_wall_time_hours | 57.324 | 60.250 | +2.926 | +5.1% | worse |
| avg_wall_time_min | 68.789 | 72.300 | +3.511 | +5.1% | worse |

### `base_agent_gpt_5_6_terra_r3_merge_sim08_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 26.0930 | +21.9035 | +522.8% | better |
| speedup_best_mean | 4.9586 | 4.7429 | -0.2156 | -4.3% | worse |
| speedup_best_median | 1.9619 | 1.9560 | -0.0059 | -0.3% | worse |
| speedup_best_geomean | 2.5921 | 2.6451 | +0.0530 | +2.0% | better |
| speedup_current_geomean | 2.3447 | 2.4671 | +0.1225 | +5.2% | better |
| hack_iteration_count | 28 | 8 | -20 | -71.4% | better |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 392 | +47 | +13.6% | worse |
| total_wall_time_hours | 57.324 | 63.211 | +5.886 | +10.3% | worse |
| avg_wall_time_min | 68.789 | 75.853 | +7.064 | +10.3% | worse |

### `base_agent_gpt_5_6_terra_r3_refinement_itr30_GH200_2026_08_25_10_44`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 29.8404 | +25.6510 | +612.3% | better |
| speedup_best_mean | 4.9586 | 5.1439 | +0.1854 | +3.7% | better |
| speedup_best_median | 1.9619 | 2.0770 | +0.1151 | +5.9% | better |
| speedup_best_geomean | 2.5921 | 2.9101 | +0.3180 | +12.3% | better |
| speedup_current_geomean | 2.3447 | 2.5746 | +0.2299 | +9.8% | better |
| hack_iteration_count | 28 | 23 | -5 | -17.9% | better |
| problems_with_hack | 4 | 2 | -2 | -50.0% | better |
| l1_entry_count | 345 | 369 | +24 | +7.0% | worse |
| total_wall_time_hours | 57.324 | 62.400 | +5.076 | +8.9% | worse |
| avg_wall_time_min | 68.789 | 74.880 | +6.091 | +8.9% | worse |

### `base_agent_gpt_5_6_terra_r3_selective_r5_itr30_GH200_2026_08_25_10_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 50 | 50 | +0 | +0.0% | same |
| correct_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% | same |
| best_speedup_overall | 4.1895 | 25.6164 | +21.4270 | +511.4% | better |
| speedup_best_mean | 4.9586 | 5.4477 | +0.4892 | +9.9% | better |
| speedup_best_median | 1.9619 | 1.9005 | -0.0614 | -3.1% | worse |
| speedup_best_geomean | 2.5921 | 2.7940 | +0.2019 | +7.8% | better |
| speedup_current_geomean | 2.3447 | 2.4839 | +0.1392 | +5.9% | better |
| hack_iteration_count | 28 | 38 | +10 | +35.7% | worse |
| problems_with_hack | 4 | 4 | +0 | +0.0% | same |
| l1_entry_count | 345 | 254 | -91 | -26.4% | better |
| total_wall_time_hours | 57.324 | 59.168 | +1.843 | +3.2% | worse |
| avg_wall_time_min | 68.789 | 71.001 | +2.212 | +3.2% | worse |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | delta(R1-R13) | delta(R2-R13) | delta(R3-R13) | delta(R4-R13) | delta(R5-R13) | delta(R6-R13) | delta(R7-R13) | delta(R8-R13) | delta(R9-R13) | delta(R10-R13) | delta(R11-R13) | delta(R12-R13) | delta(R14-R13) | delta(R15-R13) | delta(R16-R13) | delta(R17-R13) | delta(R18-R13) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4432 | 1.3928 | 1.3002 | 1.4745 | 1.1788 | 1.4711 | 1.0853 | 1.2443 | 1.5388 | 1.4649 | 1.3996 | 1.1943 | 1.5839 | 1.4489 | 1.3377 | 1.4359 | 1.7039 | 1.2531 | -0.1407 | -0.1911 | -0.2837 | -0.1094 | -0.4051 | -0.1128 | -0.4986 | -0.3396 | -0.0451 | -0.1190 | -0.1844 | -0.3896 | -0.1350 | -0.2462 | -0.1480 | +0.1199 | -0.3308 |
| 5 | 1.7532 | 1.7877 | 1.8249 | 2.1038 | 1.4508 | 1.9910 | 1.6995 | 1.5886 | 1.6787 | 1.6858 | 1.6514 | 1.6386 | 1.7248 | 1.6045 | 1.6504 | 1.7965 | 1.8779 | 1.7774 | +0.0284 | +0.0629 | +0.1001 | +0.3790 | -0.2740 | +0.2662 | -0.0253 | -0.1362 | -0.0461 | -0.0390 | -0.0734 | -0.0862 | -0.1203 | -0.0744 | +0.0718 | +0.1531 | +0.0526 |
| 10 | 2.1188 | 2.0157 | 2.1869 | 2.3569 | 1.5835 | 2.3790 | 1.9468 | 1.9157 | 1.9012 | 1.9464 | 1.9716 | 1.8859 | 1.9489 | 1.7332 | 1.8628 | 2.0191 | 2.3251 | 2.0829 | +0.1699 | +0.0668 | +0.2380 | +0.4080 | -0.3654 | +0.4301 | -0.0021 | -0.0331 | -0.0476 | -0.0024 | +0.0228 | -0.0630 | -0.2157 | -0.0861 | +0.0702 | +0.3762 | +0.1340 |
| 15 | 2.3187 | 2.2850 | 2.3036 | 2.5873 | 1.6428 | 2.7536 | 2.1450 | 2.3113 | 2.1980 | 2.1438 | 2.1145 | 2.1349 | 2.2319 | 1.8493 | 2.0156 | 2.4278 | 2.4884 | 2.2714 | +0.0868 | +0.0531 | +0.0717 | +0.3553 | -0.5891 | +0.5217 | -0.0869 | +0.0794 | -0.0339 | -0.0881 | -0.1174 | -0.0970 | -0.3826 | -0.2163 | +0.1959 | +0.2565 | +0.0395 |
| 20 | 2.4858 | 2.4907 | 2.4405 | 2.6833 | 1.7073 | 2.9057 | 2.3094 | 2.5672 | 2.2629 | 2.2580 | 2.3197 | 2.2563 | 2.3482 | 1.8820 | 2.2333 | 2.5868 | 2.7574 | 2.6031 | +0.1377 | +0.1426 | +0.0924 | +0.3352 | -0.6408 | +0.5576 | -0.0388 | +0.2190 | -0.0853 | -0.0902 | -0.0285 | -0.0919 | -0.4662 | -0.1148 | +0.2386 | +0.4093 | +0.2549 |
| 25 | 2.5447 | 2.6929 | 2.6048 | 2.7458 | 1.7262 | 2.9219 | 2.5511 | 2.6448 | 2.4523 | 2.3058 | 2.4053 | 2.3673 | 2.3930 | 2.1068 | 2.4124 | 2.6262 | 2.8488 | 2.7490 | +0.1516 | +0.2999 | +0.2118 | +0.3528 | -0.6669 | +0.5289 | +0.1581 | +0.2518 | +0.0593 | -0.0872 | +0.0123 | -0.0257 | -0.2863 | +0.0193 | +0.2332 | +0.4557 | +0.3559 |
| 30 | 2.6319 | 2.7419 | 2.6478 | 2.8414 | 1.8215 | 2.9712 | 2.5947 | 2.7309 | 2.4738 | 2.5494 | 2.4654 | 2.4031 | 2.5921 | 2.1635 | 2.4642 | 2.6451 | 2.9101 | 2.7940 | +0.0398 | +0.1497 | +0.0557 | +0.2493 | -0.7706 | +0.3791 | +0.0025 | +0.1388 | -0.1183 | -0.0427 | -0.1267 | -0.1890 | -0.4287 | -0.1279 | +0.0530 | +0.3180 | +0.2019 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | delta(R1-R13) | delta(R2-R13) | delta(R3-R13) | delta(R4-R13) | delta(R5-R13) | delta(R6-R13) | delta(R7-R13) | delta(R8-R13) | delta(R9-R13) | delta(R10-R13) | delta(R11-R13) | delta(R12-R13) | delta(R14-R13) | delta(R15-R13) | delta(R16-R13) | delta(R17-R13) | delta(R18-R13) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.400 | 0.380 | 0.380 | 0.380 | 0.340 | 0.360 | 0.320 | 0.300 | 0.340 | 0.320 | 0.380 | 0.260 | 0.460 | 0.360 | 0.300 | 0.360 | 0.340 | 0.300 | -0.060 | -0.080 | -0.080 | -0.080 | -0.120 | -0.100 | -0.140 | -0.160 | -0.120 | -0.140 | -0.080 | -0.200 | -0.100 | -0.160 | -0.100 | -0.120 | -0.160 |
| 5 | 0.640 | 0.720 | 0.760 | 0.760 | 0.580 | 0.760 | 0.680 | 0.660 | 0.780 | 0.620 | 0.660 | 0.640 | 0.680 | 0.600 | 0.720 | 0.720 | 0.800 | 0.600 | -0.040 | +0.040 | +0.080 | +0.080 | -0.100 | +0.080 | +0.000 | -0.020 | +0.100 | -0.060 | -0.020 | -0.040 | -0.080 | +0.040 | +0.040 | +0.120 | -0.080 |
| 10 | 0.800 | 0.800 | 0.820 | 0.820 | 0.680 | 0.860 | 0.820 | 0.800 | 0.840 | 0.660 | 0.720 | 0.740 | 0.740 | 0.640 | 0.820 | 0.800 | 0.880 | 0.760 | +0.060 | +0.060 | +0.080 | +0.080 | -0.060 | +0.120 | +0.080 | +0.060 | +0.100 | -0.080 | -0.020 | +0.000 | -0.100 | +0.080 | +0.060 | +0.140 | +0.020 |
| 15 | 0.840 | 0.840 | 0.820 | 0.860 | 0.700 | 0.880 | 0.900 | 0.820 | 0.860 | 0.800 | 0.780 | 0.820 | 0.820 | 0.660 | 0.840 | 0.860 | 0.880 | 0.780 | +0.020 | +0.020 | +0.000 | +0.040 | -0.120 | +0.060 | +0.080 | +0.000 | +0.040 | -0.020 | -0.040 | +0.000 | -0.160 | +0.020 | +0.040 | +0.060 | -0.040 |
| 20 | 0.860 | 0.860 | 0.820 | 0.880 | 0.700 | 0.880 | 0.900 | 0.880 | 0.860 | 0.800 | 0.840 | 0.840 | 0.840 | 0.660 | 0.860 | 0.920 | 0.900 | 0.800 | +0.020 | +0.020 | -0.020 | +0.040 | -0.140 | +0.040 | +0.060 | +0.040 | +0.020 | -0.040 | +0.000 | +0.000 | -0.180 | +0.020 | +0.080 | +0.060 | -0.040 |
| 25 | 0.880 | 0.860 | 0.820 | 0.880 | 0.700 | 0.900 | 0.900 | 0.900 | 0.900 | 0.800 | 0.840 | 0.880 | 0.840 | 0.720 | 0.860 | 0.920 | 0.900 | 0.820 | +0.040 | +0.020 | -0.020 | +0.040 | -0.140 | +0.060 | +0.060 | +0.060 | +0.060 | -0.040 | +0.000 | +0.040 | -0.120 | +0.020 | +0.080 | +0.060 | -0.020 |
| 30 | 0.880 | 0.860 | 0.820 | 0.900 | 0.760 | 0.900 | 0.920 | 0.900 | 0.900 | 0.840 | 0.840 | 0.880 | 0.860 | 0.720 | 0.860 | 0.920 | 0.900 | 0.820 | +0.020 | +0.000 | -0.040 | +0.040 | -0.100 | +0.040 | +0.060 | +0.040 | +0.040 | -0.020 | -0.020 | +0.020 | -0.140 | +0.000 | +0.060 | +0.040 | -0.040 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_5_6_terra_r3_itr30_GH200_2026_08_25_10_41`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | best_geomean | 30 | 2.5921 | 2.6319 | +0.0398 | +1.5% |
| R1 | fast_p_best@1.0 | 30 | 0.860 | 0.880 | +0.020 | +2.3% |
| R2 | best_geomean | 30 | 2.5921 | 2.7419 | +0.1497 | +5.8% |
| R2 | fast_p_best@1.0 | 30 | 0.860 | 0.860 | +0.000 | +0.0% |
| R3 | best_geomean | 30 | 2.5921 | 2.6478 | +0.0557 | +2.1% |
| R3 | fast_p_best@1.0 | 30 | 0.860 | 0.820 | -0.040 | -4.7% |
| R4 | best_geomean | 30 | 2.5921 | 2.8414 | +0.2493 | +9.6% |
| R4 | fast_p_best@1.0 | 30 | 0.860 | 0.900 | +0.040 | +4.7% |
| R5 | best_geomean | 30 | 2.5921 | 1.8215 | -0.7706 | -29.7% |
| R5 | fast_p_best@1.0 | 30 | 0.860 | 0.760 | -0.100 | -11.6% |
| R6 | best_geomean | 30 | 2.5921 | 2.9712 | +0.3791 | +14.6% |
| R6 | fast_p_best@1.0 | 30 | 0.860 | 0.900 | +0.040 | +4.7% |
| R7 | best_geomean | 30 | 2.5921 | 2.5947 | +0.0025 | +0.1% |
| R7 | fast_p_best@1.0 | 30 | 0.860 | 0.920 | +0.060 | +7.0% |
| R8 | best_geomean | 30 | 2.5921 | 2.7309 | +0.1388 | +5.4% |
| R8 | fast_p_best@1.0 | 30 | 0.860 | 0.900 | +0.040 | +4.7% |
| R9 | best_geomean | 30 | 2.5921 | 2.4738 | -0.1183 | -4.6% |
| R9 | fast_p_best@1.0 | 30 | 0.860 | 0.900 | +0.040 | +4.7% |
| R10 | best_geomean | 30 | 2.5921 | 2.5494 | -0.0427 | -1.6% |
| R10 | fast_p_best@1.0 | 30 | 0.860 | 0.840 | -0.020 | -2.3% |
| R11 | best_geomean | 30 | 2.5921 | 2.4654 | -0.1267 | -4.9% |
| R11 | fast_p_best@1.0 | 30 | 0.860 | 0.840 | -0.020 | -2.3% |
| R12 | best_geomean | 30 | 2.5921 | 2.4031 | -0.1890 | -7.3% |
| R12 | fast_p_best@1.0 | 30 | 0.860 | 0.880 | +0.020 | +2.3% |
| R14 | best_geomean | 30 | 2.5921 | 2.1635 | -0.4287 | -16.5% |
| R14 | fast_p_best@1.0 | 30 | 0.860 | 0.720 | -0.140 | -16.3% |
| R15 | best_geomean | 30 | 2.5921 | 2.4642 | -0.1279 | -4.9% |
| R15 | fast_p_best@1.0 | 30 | 0.860 | 0.860 | +0.000 | +0.0% |
| R16 | best_geomean | 30 | 2.5921 | 2.6451 | +0.0530 | +2.0% |
| R16 | fast_p_best@1.0 | 30 | 0.860 | 0.920 | +0.060 | +7.0% |
| R17 | best_geomean | 30 | 2.5921 | 2.9101 | +0.3180 | +12.3% |
| R17 | fast_p_best@1.0 | 30 | 0.860 | 0.900 | +0.040 | +4.7% |
| R18 | best_geomean | 30 | 2.5921 | 2.7940 | +0.2019 | +7.8% |
| R18 | fast_p_best@1.0 | 30 | 0.860 | 0.820 | -0.040 | -4.7% |
