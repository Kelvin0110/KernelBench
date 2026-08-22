# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-22T17:02:21.152135+00:00`
- aggregate_generated_at_utc: `2026-08-22T17:00:02.708382+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 10
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | complete | markov_report | gpt-oss-120b | inference |
| R3 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_13_12_47` | complete | folding | gpt-oss-120b | inference |
| R4 | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` | complete | selective_retention | gpt-oss-120b | inference |
| R5 | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` | complete | compress_trigger | gpt-oss-120b | inference |
| R6 | `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_14_15_52` | complete | truncation | gpt-oss-120b | inference |
| R7 | `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_17_15_52` | complete | truncation | gpt-oss-120b | inference |
| R8 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_29` | complete | truncation | gpt-oss-120b | inference |
| R9 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_32` | complete | truncation | gpt-oss-120b | inference |
| R10 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_35` | complete | truncation | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 74.18 | 88.0 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 71.43 | 83.5 | 0 |
| R3 | folding | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 66.59 | 79.9 | 0 |
| R4 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 69.93 | 83.9 | 0 |
| R5 | compress_trigger | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.23 | 77.1 | 0 |
| R6 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 66.91 | 80.3 | 0 |
| R7 | truncation | 30 | 50 | 50 | 45 | 0.900 | total_attempted | 53.07 | 63.7 | 0 |
| R8 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.91 | 77.9 | 0 |
| R9 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 68.37 | 82.0 | 0 |
| R10 | truncation | 30 | 50 | 50 | 46 | 0.920 | total_attempted | 65.43 | 78.5 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | complete | 47/50 | 0.840 | 0.380 | 0.120 | 0.7379 | 42 | 0.940 | 0.460 | 0.180 | 0.9051 | 47 |
| R2 | markov_report | complete | 48/50 | 0.900 | 0.340 | 0.040 | 0.7145 | 45 | 0.960 | 0.460 | 0.140 | 0.9332 | 48 |
| R3 | folding | complete | 47/50 | 0.840 | 0.320 | 0.120 | 0.6744 | 42 | 0.940 | 0.420 | 0.180 | 0.8938 | 47 |
| R4 | selective_retention | complete | 48/50 | 0.900 | 0.320 | 0.080 | 0.7657 | 45 | 0.960 | 0.460 | 0.160 | 0.9541 | 48 |
| R5 | compress_trigger | complete | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 | 42 | 0.960 | 0.400 | 0.140 | 0.7265 | 48 |
| R6 | truncation+deletion | complete | 48/50 | 0.840 | 0.400 | 0.100 | 0.8691 | 42 | 0.960 | 0.540 | 0.280 | 1.2312 | 48 |
| R7 | truncation+refine | complete | 45/50 | 0.780 | 0.300 | 0.100 | 0.7107 | 39 | 0.900 | 0.340 | 0.160 | 0.7971 | 45 |
| R8 | truncation+merge@0.8 | complete | 48/50 | 0.740 | 0.260 | 0.120 | 0.7047 | 37 | 0.960 | 0.420 | 0.200 | 0.8379 | 48 |
| R9 | truncation+merge@0.8 | complete | 47/50 | 0.700 | 0.280 | 0.080 | 0.8282 | 35 | 0.940 | 0.400 | 0.140 | 0.8552 | 47 |
| R10 | truncation+merge@0.8 | complete | 46/50 | 0.800 | 0.300 | 0.140 | 0.9185 | 40 | 0.920 | 0.460 | 0.220 | 1.0919 | 46 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.5437 | 0.9394 | 0.9051 | 47 | 0.8759 | 28 | 5.4461 | 16 | 11 | 0.940 | 0.720 | 0.500 | 0.460 | 0.240 | 0.180 |
| R2 | 30 | 50 | 1.3089 | 0.9640 | 0.9332 | 48 | 0.9225 | 39 | 6.2340 | 14 | 11 | 0.960 | 0.780 | 0.600 | 0.460 | 0.160 | 0.140 |
| R3 | 30 | 50 | 1.5545 | 0.9218 | 0.8938 | 47 | 0.8176 | 26 | 6.3696 | 17 | 12 | 0.940 | 0.680 | 0.540 | 0.420 | 0.220 | 0.180 |
| R4 | 30 | 50 | 1.4787 | 0.9144 | 0.9541 | 48 | 0.9747 | 37 | 9.7594 | 17 | 12 | 0.960 | 0.740 | 0.540 | 0.460 | 0.220 | 0.160 |
| R5 | 30 | 50 | 1.2428 | 0.7582 | 0.7265 | 48 | 0.7491 | 29 | 2.9647 | 32 | 23 | 0.960 | 0.640 | 0.460 | 0.400 | 0.180 | 0.140 |
| R6 | 30 | 50 | 1.9729 | 1.0513 | 1.2312 | 48 | 1.4874 | 29 | 2.1050 | 19 | 12 | 0.960 | 0.820 | 0.720 | 0.540 | 0.300 | 0.280 |
| R7 | 30 | 50 | 1.5564 | 0.8403 | 0.7971 | 45 | 0.6148 | 32 | 7.2340 | 29 | 19 | 0.900 | 0.660 | 0.520 | 0.340 | 0.200 | 0.160 |
| R8 | 30 | 50 | 1.4754 | 0.7649 | 0.8379 | 48 | 0.6872 | 34 | 6.0788 | 21 | 16 | 0.960 | 0.700 | 0.460 | 0.420 | 0.240 | 0.200 |
| R9 | 30 | 50 | 1.3227 | 0.8354 | 0.8552 | 47 | 0.7349 | 27 | 9.0517 | 31 | 18 | 0.940 | 0.700 | 0.500 | 0.400 | 0.180 | 0.140 |
| R10 | 30 | 50 | 1.7942 | 0.9943 | 1.0919 | 46 | 1.0892 | 29 | 6.5111 | 23 | 14 | 0.920 | 0.780 | 0.540 | 0.460 | 0.240 | 0.220 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 571 | 571 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 366 | 366 | 0 | 0 | 0 | 0 | 0 |
| R3 | no | no | no | 592 | 592 | 0 | 0 | 0 | 0 | 0 |
| R4 | no | no | no | 619 | 619 | 0 | 0 | 0 | 0 | 0 |
| R5 | no | no | no | 435 | 435 | 0 | 0 | 0 | 0 | 0 |
| R6 | yes | no | no | 592 | 25 | 0 | 567 | 0 | 567 | 3 |
| R7 | no | no | yes | 703 | 626 | 0 | 0 | 83 | 0 | 1 |
| R8 | no | yes | no | 703 | 384 | 56 | 0 | 0 | 0 | 7 |
| R9 | no | yes | no | 730 | 182 | 77 | 0 | 0 | 0 | 7 |
| R10 | no | yes | no | 681 | 313 | 52 | 0 | 0 | 0 | 7 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 6.2340 | +0.7879 | +14.5% | better |
| speedup_best_mean | 1.5437 | 1.3089 | -0.2347 | -15.2% | worse |
| speedup_best_median | 0.9394 | 0.9640 | +0.0246 | +2.6% | better |
| speedup_best_geomean | 0.9051 | 0.9332 | +0.0281 | +3.1% | better |
| speedup_current_geomean | 0.8759 | 0.9225 | +0.0465 | +5.3% | better |
| hack_iteration_count | 16 | 14 | -2 | -12.5% | better |
| problems_with_hack | 11 | 11 | +0 | +0.0% | same |
| l1_entry_count | 571 | 366 | -205 | -35.9% | better |
| total_wall_time_hours | 74.177 | 71.433 | -2.745 | -3.7% | better |
| avg_wall_time_min | 88.035 | 83.506 | -4.529 | -5.1% | better |

### `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_13_12_47`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 47 | +0 | +0.0% | same |
| correct_rate | 0.9400 | 0.9400 | +0.0000 | +0.0% | same |
| best_speedup_overall | 5.4461 | 6.3696 | +0.9235 | +17.0% | better |
| speedup_best_mean | 1.5437 | 1.5545 | +0.0108 | +0.7% | better |
| speedup_best_median | 0.9394 | 0.9218 | -0.0176 | -1.9% | worse |
| speedup_best_geomean | 0.9051 | 0.8938 | -0.0113 | -1.3% | worse |
| speedup_current_geomean | 0.8759 | 0.8176 | -0.0583 | -6.7% | worse |
| hack_iteration_count | 16 | 17 | +1 | +6.2% | worse |
| problems_with_hack | 11 | 12 | +1 | +9.1% | worse |
| l1_entry_count | 571 | 592 | +21 | +3.7% | worse |
| total_wall_time_hours | 74.177 | 66.594 | -7.583 | -10.2% | better |
| avg_wall_time_min | 88.035 | 79.913 | -8.122 | -9.2% | better |

### `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 9.7594 | +4.3134 | +79.2% | better |
| speedup_best_mean | 1.5437 | 1.4787 | -0.0650 | -4.2% | worse |
| speedup_best_median | 0.9394 | 0.9144 | -0.0250 | -2.7% | worse |
| speedup_best_geomean | 0.9051 | 0.9541 | +0.0490 | +5.4% | better |
| speedup_current_geomean | 0.8759 | 0.9747 | +0.0988 | +11.3% | better |
| hack_iteration_count | 16 | 17 | +1 | +6.2% | worse |
| problems_with_hack | 11 | 12 | +1 | +9.1% | worse |
| l1_entry_count | 571 | 619 | +48 | +8.4% | worse |
| total_wall_time_hours | 74.177 | 69.928 | -4.249 | -5.7% | better |
| avg_wall_time_min | 88.035 | 83.914 | -4.121 | -4.7% | better |

### `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 2.9647 | -2.4814 | -45.6% | worse |
| speedup_best_mean | 1.5437 | 1.2428 | -0.3009 | -19.5% | worse |
| speedup_best_median | 0.9394 | 0.7582 | -0.1812 | -19.3% | worse |
| speedup_best_geomean | 0.9051 | 0.7265 | -0.1786 | -19.7% | worse |
| speedup_current_geomean | 0.8759 | 0.7491 | -0.1268 | -14.5% | worse |
| hack_iteration_count | 16 | 32 | +16 | +100.0% | worse |
| problems_with_hack | 11 | 23 | +12 | +109.1% | worse |
| l1_entry_count | 571 | 435 | -136 | -23.8% | better |
| total_wall_time_hours | 74.177 | 64.230 | -9.947 | -13.4% | better |
| avg_wall_time_min | 88.035 | 77.076 | -10.958 | -12.4% | better |

### `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_14_15_52`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 2.1050 | -3.3411 | -61.3% | worse |
| speedup_best_mean | 1.5437 | 1.9729 | +0.4292 | +27.8% | better |
| speedup_best_median | 0.9394 | 1.0513 | +0.1119 | +11.9% | better |
| speedup_best_geomean | 0.9051 | 1.2312 | +0.3261 | +36.0% | better |
| speedup_current_geomean | 0.8759 | 1.4874 | +0.6115 | +69.8% | better |
| hack_iteration_count | 16 | 19 | +3 | +18.8% | worse |
| problems_with_hack | 11 | 12 | +1 | +9.1% | worse |
| l1_entry_count | 571 | 592 | +21 | +3.7% | worse |
| total_wall_time_hours | 74.177 | 66.905 | -7.272 | -9.8% | better |
| avg_wall_time_min | 88.035 | 80.286 | -7.749 | -8.8% | better |

### `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_17_15_52`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 45 | -2 | -4.3% | worse |
| correct_rate | 0.9400 | 0.9000 | -0.0400 | -4.3% | worse |
| best_speedup_overall | 5.4461 | 7.2340 | +1.7879 | +32.8% | better |
| speedup_best_mean | 1.5437 | 1.5564 | +0.0127 | +0.8% | better |
| speedup_best_median | 0.9394 | 0.8403 | -0.0991 | -10.5% | worse |
| speedup_best_geomean | 0.9051 | 0.7971 | -0.1080 | -11.9% | worse |
| speedup_current_geomean | 0.8759 | 0.6148 | -0.2612 | -29.8% | worse |
| hack_iteration_count | 16 | 29 | +13 | +81.2% | worse |
| problems_with_hack | 11 | 19 | +8 | +72.7% | worse |
| l1_entry_count | 571 | 703 | +132 | +23.1% | worse |
| total_wall_time_hours | 74.177 | 53.070 | -21.108 | -28.5% | better |
| avg_wall_time_min | 88.035 | 63.684 | -24.351 | -27.7% | better |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_29`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 6.0788 | +0.6327 | +11.6% | better |
| speedup_best_mean | 1.5437 | 1.4754 | -0.0682 | -4.4% | worse |
| speedup_best_median | 0.9394 | 0.7649 | -0.1745 | -18.6% | worse |
| speedup_best_geomean | 0.9051 | 0.8379 | -0.0672 | -7.4% | worse |
| speedup_current_geomean | 0.8759 | 0.6872 | -0.1888 | -21.5% | worse |
| hack_iteration_count | 16 | 21 | +5 | +31.2% | worse |
| problems_with_hack | 11 | 16 | +5 | +45.5% | worse |
| l1_entry_count | 571 | 703 | +132 | +23.1% | worse |
| total_wall_time_hours | 74.177 | 64.907 | -9.270 | -12.5% | better |
| avg_wall_time_min | 88.035 | 77.889 | -10.146 | -11.5% | better |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_32`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 47 | +0 | +0.0% | same |
| correct_rate | 0.9400 | 0.9400 | +0.0000 | +0.0% | same |
| best_speedup_overall | 5.4461 | 9.0517 | +3.6056 | +66.2% | better |
| speedup_best_mean | 1.5437 | 1.3227 | -0.2210 | -14.3% | worse |
| speedup_best_median | 0.9394 | 0.8354 | -0.1040 | -11.1% | worse |
| speedup_best_geomean | 0.9051 | 0.8552 | -0.0499 | -5.5% | worse |
| speedup_current_geomean | 0.8759 | 0.7349 | -0.1410 | -16.1% | worse |
| hack_iteration_count | 16 | 31 | +15 | +93.8% | worse |
| problems_with_hack | 11 | 18 | +7 | +63.6% | worse |
| l1_entry_count | 571 | 730 | +159 | +27.8% | worse |
| total_wall_time_hours | 74.177 | 68.375 | -5.803 | -7.8% | better |
| avg_wall_time_min | 88.035 | 82.050 | -5.985 | -6.8% | better |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_35`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 46 | -1 | -2.1% | worse |
| correct_rate | 0.9400 | 0.9200 | -0.0200 | -2.1% | worse |
| best_speedup_overall | 5.4461 | 6.5111 | +1.0650 | +19.6% | better |
| speedup_best_mean | 1.5437 | 1.7942 | +0.2506 | +16.2% | better |
| speedup_best_median | 0.9394 | 0.9943 | +0.0549 | +5.8% | better |
| speedup_best_geomean | 0.9051 | 1.0919 | +0.1868 | +20.6% | better |
| speedup_current_geomean | 0.8759 | 1.0892 | +0.2133 | +24.3% | better |
| hack_iteration_count | 16 | 23 | +7 | +43.8% | worse |
| problems_with_hack | 11 | 14 | +3 | +27.3% | worse |
| l1_entry_count | 571 | 681 | +110 | +19.3% | worse |
| total_wall_time_hours | 74.177 | 65.433 | -8.745 | -11.8% | better |
| avg_wall_time_min | 88.035 | 78.519 | -9.516 | -10.8% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) | delta(R6-R1) | delta(R7-R1) | delta(R8-R1) | delta(R9-R1) | delta(R10-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4816 | 0.6979 | 0.3440 | 0.4330 | 0.3999 | 0.6971 | 0.5563 | 0.3090 | 0.4632 | 0.5761 | +0.2163 | -0.1376 | -0.0486 | -0.0817 | +0.2155 | +0.0746 | -0.1726 | -0.0184 | +0.0945 |
| 5 | 0.5552 | 0.5892 | 0.5638 | 0.5048 | 0.5168 | 0.7275 | 0.6508 | 0.5640 | 0.7862 | 0.7658 | +0.0340 | +0.0086 | -0.0504 | -0.0384 | +0.1723 | +0.0956 | +0.0088 | +0.2310 | +0.2106 |
| 10 | 0.7379 | 0.7145 | 0.6744 | 0.7657 | 0.5624 | 0.8691 | 0.7107 | 0.7047 | 0.8282 | 0.9185 | -0.0234 | -0.0635 | +0.0278 | -0.1755 | +0.1311 | -0.0273 | -0.0332 | +0.0903 | +0.1806 |
| 15 | 0.7728 | 0.8178 | 0.7570 | 0.7811 | 0.5445 | 0.9468 | 0.7666 | 0.7308 | 0.7802 | 0.8921 | +0.0450 | -0.0158 | +0.0083 | -0.2283 | +0.1740 | -0.0061 | -0.0420 | +0.0074 | +0.1193 |
| 20 | 0.8292 | 0.8441 | 0.8256 | 0.8710 | 0.6344 | 0.9518 | 0.7637 | 0.7736 | 0.7744 | 0.9611 | +0.0149 | -0.0036 | +0.0417 | -0.1948 | +0.1226 | -0.0656 | -0.0556 | -0.0548 | +0.1319 |
| 25 | 0.8806 | 0.8722 | 0.8709 | 0.9440 | 0.6856 | 1.1073 | 0.7624 | 0.7880 | 0.8797 | 1.0129 | -0.0084 | -0.0097 | +0.0634 | -0.1951 | +0.2266 | -0.1183 | -0.0926 | -0.0010 | +0.1323 |
| 30 | 0.9051 | 0.9332 | 0.8938 | 0.9541 | 0.7265 | 1.2312 | 0.7971 | 0.8379 | 0.8552 | 1.0919 | +0.0281 | -0.0113 | +0.0490 | -0.1786 | +0.3261 | -0.1080 | -0.0672 | -0.0499 | +0.1868 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) | delta(R6-R1) | delta(R7-R1) | delta(R8-R1) | delta(R9-R1) | delta(R10-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.080 | 0.040 | 0.060 | 0.040 | 0.080 | 0.120 | 0.040 | 0.000 | 0.020 | 0.060 | -0.040 | -0.020 | -0.040 | +0.000 | +0.040 | -0.040 | -0.080 | -0.060 | -0.020 |
| 5 | 0.260 | 0.280 | 0.240 | 0.200 | 0.200 | 0.300 | 0.240 | 0.100 | 0.200 | 0.200 | +0.020 | -0.020 | -0.060 | -0.060 | +0.040 | -0.020 | -0.160 | -0.060 | -0.060 |
| 10 | 0.380 | 0.340 | 0.320 | 0.320 | 0.240 | 0.400 | 0.300 | 0.260 | 0.280 | 0.300 | -0.040 | -0.060 | -0.060 | -0.140 | +0.020 | -0.080 | -0.120 | -0.100 | -0.080 |
| 15 | 0.400 | 0.380 | 0.380 | 0.380 | 0.280 | 0.400 | 0.320 | 0.300 | 0.340 | 0.380 | -0.020 | -0.020 | -0.020 | -0.120 | +0.000 | -0.080 | -0.100 | -0.060 | -0.020 |
| 20 | 0.420 | 0.400 | 0.400 | 0.420 | 0.360 | 0.420 | 0.340 | 0.320 | 0.340 | 0.380 | -0.020 | -0.020 | +0.000 | -0.060 | +0.000 | -0.080 | -0.100 | -0.080 | -0.040 |
| 25 | 0.420 | 0.420 | 0.420 | 0.460 | 0.380 | 0.480 | 0.340 | 0.380 | 0.400 | 0.400 | +0.000 | +0.000 | +0.040 | -0.040 | +0.060 | -0.080 | -0.040 | -0.020 | -0.020 |
| 30 | 0.460 | 0.460 | 0.420 | 0.460 | 0.400 | 0.540 | 0.340 | 0.420 | 0.400 | 0.460 | +0.000 | -0.040 | +0.000 | -0.060 | +0.080 | -0.120 | -0.040 | -0.060 | +0.000 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 0.9051 | 0.9332 | +0.0281 | +3.1% |
| R2 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R3 | best_geomean | 30 | 0.9051 | 0.8938 | -0.0113 | -1.3% |
| R3 | fast_p_best@1.0 | 30 | 0.460 | 0.420 | -0.040 | -8.7% |
| R4 | best_geomean | 30 | 0.9051 | 0.9541 | +0.0490 | +5.4% |
| R4 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R5 | best_geomean | 30 | 0.9051 | 0.7265 | -0.1786 | -19.7% |
| R5 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |
| R6 | best_geomean | 30 | 0.9051 | 1.2312 | +0.3261 | +36.0% |
| R6 | fast_p_best@1.0 | 30 | 0.460 | 0.540 | +0.080 | +17.4% |
| R7 | best_geomean | 30 | 0.9051 | 0.7971 | -0.1080 | -11.9% |
| R7 | fast_p_best@1.0 | 30 | 0.460 | 0.340 | -0.120 | -26.1% |
| R8 | best_geomean | 30 | 0.9051 | 0.8379 | -0.0672 | -7.4% |
| R8 | fast_p_best@1.0 | 30 | 0.460 | 0.420 | -0.040 | -8.7% |
| R9 | best_geomean | 30 | 0.9051 | 0.8552 | -0.0499 | -5.5% |
| R9 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |
| R10 | best_geomean | 30 | 0.9051 | 1.0919 | +0.1868 | +20.6% |
| R10 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
