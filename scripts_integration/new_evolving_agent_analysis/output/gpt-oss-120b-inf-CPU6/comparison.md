# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-16T03:47:00.068008+00:00`
- aggregate_generated_at_utc: `2026-08-16T03:36:44.672332+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving/inference_oss_120b`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 8
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` | complete | truncation | gpt-oss-120b | inference |
| R3 | `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48` | complete | truncation | gpt-oss-120b | inference |
| R4 | `base_agent_oss120b_folding_itr30_2026_08_09_13_47` | complete | folding | gpt-oss-120b | inference |
| R5 | `base_agent_oss120b_markov_itr30_2026_08_07_14_07` | complete | markov_report | gpt-oss-120b | inference |
| R6 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` | complete | truncation | gpt-oss-120b | inference |
| R7 | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | complete | selective_retention | gpt-oss-120b | inference |
| R8 | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` | complete | truncation | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 66.19 | 3971.3 | 0 |
| R2 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 79.55 | 4773.0 | 0 |
| R3 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 109.78 | 6586.7 | 0 |
| R4 | folding | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 90.08 | 5404.6 | 0 |
| R5 | markov_report | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 84.07 | 5044.3 | 0 |
| R6 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 90.00 | 5399.8 | 0 |
| R7 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 86.85 | 2605.6 | 0 |
| R8 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 73.14 | 4388.2 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses only correct, non-hack samples; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | complete | 49/50 | 0.880 | 0.520 | 0.140 | 1.1700 | 39 | 0.980 | 0.720 | 0.240 | 1.3855 | 41 |
| R2 | truncation+deletion | complete | 49/50 | 0.800 | 0.380 | 0.060 | 0.8606 | 34 | 0.980 | 0.640 | 0.180 | 1.2518 | 35 |
| R3 | truncation+deletion+merge@0.7+refine | complete | 48/50 | 0.820 | 0.460 | 0.200 | 1.1531 | 36 | 0.960 | 0.660 | 0.260 | 1.3966 | 36 |
| R4 | folding | complete | 48/50 | 0.940 | 0.500 | 0.160 | 1.0169 | 45 | 0.960 | 0.600 | 0.220 | 1.2243 | 38 |
| R5 | markov_report | complete | 50/50 | 0.960 | 0.460 | 0.080 | 0.8886 | 42 | 1.000 | 0.600 | 0.120 | 1.0302 | 36 |
| R6 | truncation+merge@0.7 | complete | 49/50 | 0.880 | 0.460 | 0.120 | 1.0173 | 36 | 0.980 | 0.640 | 0.160 | 1.2387 | 35 |
| R7 | selective_retention | complete | 48/50 | 0.920 | 0.540 | 0.080 | 1.0654 | 39 | 0.960 | 0.700 | 0.180 | 1.2859 | 37 |
| R8 | truncation+refine | complete | 47/50 | 0.840 | 0.520 | 0.100 | 1.1061 | 36 | 0.940 | 0.620 | 0.140 | 1.2333 | 32 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.6111 | 1.2523 | 1.3855 | 41 | 1.2805 | 32 | 1.2891 | 13 | 8 | 0.980 | 0.980 | 0.880 | 0.720 | 0.320 | 0.240 |
| R2 | 30 | 50 | 1.4124 | 1.1394 | 1.2518 | 35 | 1.1180 | 32 | 1.3636 | 16 | 14 | 0.980 | 0.960 | 0.840 | 0.640 | 0.340 | 0.180 |
| R3 | 30 | 50 | 1.8606 | 1.1701 | 1.3966 | 36 | 1.5033 | 26 | 1.3636 | 28 | 12 | 0.960 | 0.920 | 0.800 | 0.660 | 0.320 | 0.260 |
| R4 | 30 | 50 | 1.4955 | 1.0766 | 1.2243 | 38 | 1.0551 | 33 | 1.3306 | 21 | 11 | 0.960 | 0.900 | 0.760 | 0.600 | 0.260 | 0.220 |
| R5 | 30 | 50 | 1.3514 | 1.0400 | 1.0302 | 36 | 1.1555 | 37 | 1.3636 | 17 | 14 | 1.000 | 0.940 | 0.800 | 0.600 | 0.200 | 0.120 |
| R6 | 30 | 50 | 1.4341 | 1.0525 | 1.2387 | 35 | 1.0813 | 33 | 1.3636 | 21 | 15 | 0.980 | 0.980 | 0.860 | 0.640 | 0.240 | 0.160 |
| R7 | 30 | 50 | 1.4824 | 1.1080 | 1.2859 | 37 | 1.2320 | 34 | 1.3636 | 15 | 12 | 0.960 | 0.960 | 0.860 | 0.700 | 0.240 | 0.180 |
| R8 | 30 | 50 | 1.4199 | 1.0570 | 1.2333 | 32 | 0.9941 | 28 | 2.7415 | 19 | 17 | 0.940 | 0.940 | 0.860 | 0.620 | 0.180 | 0.140 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 561 | 561 | 0 | 0 | 0 | 0 | 0 |
| R2 | yes | no | no | 576 | 28 | 0 | 548 | 0 | 568 | 3 |
| R3 | yes | yes | yes | 634 | 17 | 29 | 450 | 76 | 491 | 9 |
| R4 | no | no | no | 601 | 601 | 0 | 0 | 0 | 0 | 0 |
| R5 | no | no | no | 377 | 377 | 0 | 0 | 0 | 0 | 0 |
| R6 | no | yes | no | 596 | 225 | 72 | 0 | 0 | 0 | 7 |
| R7 | no | no | no | 457 | 457 | 0 | 0 | 0 | 0 | 0 |
| R8 | no | no | yes | 653 | 569 | 0 | 0 | 90 | 0 | 1 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

### `base_agent_oss120b_deletion_itr30_2026_08_02_17_57`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 49 | +0 | +0.0% | same |
| correct_rate | 0.9800 | 0.9800 | +0.0000 | +0.0% | same |
| best_speedup_overall | 1.2891 | 1.3636 | +0.0746 | +5.8% | better |
| speedup_best_mean | 1.6111 | 1.4124 | -0.1988 | -12.3% | worse |
| speedup_best_median | 1.2523 | 1.1394 | -0.1129 | -9.0% | worse |
| speedup_best_geomean | 1.3855 | 1.2518 | -0.1337 | -9.6% | worse |
| speedup_current_geomean | 1.2805 | 1.1180 | -0.1624 | -12.7% | worse |
| hack_iteration_count | 13 | 16 | +3 | +23.1% | worse |
| problems_with_hack | 8 | 14 | +6 | +75.0% | worse |
| l1_entry_count | 561 | 576 | +15 | +2.7% | worse |
| total_wall_time_hours | 66.189 | 79.551 | +13.362 | +20.2% | worse |
| avg_wall_time_min | 3971.337 | 4773.046 | +801.709 | +20.2% | worse |

### `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 48 | -1 | -2.0% | worse |
| correct_rate | 0.9800 | 0.9600 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 1.2891 | 1.3636 | +0.0746 | +5.8% | better |
| speedup_best_mean | 1.6111 | 1.8606 | +0.2495 | +15.5% | better |
| speedup_best_median | 1.2523 | 1.1701 | -0.0821 | -6.6% | worse |
| speedup_best_geomean | 1.3855 | 1.3966 | +0.0112 | +0.8% | better |
| speedup_current_geomean | 1.2805 | 1.5033 | +0.2228 | +17.4% | better |
| hack_iteration_count | 13 | 28 | +15 | +115.4% | worse |
| problems_with_hack | 8 | 12 | +4 | +50.0% | worse |
| l1_entry_count | 561 | 634 | +73 | +13.0% | worse |
| total_wall_time_hours | 66.189 | 109.778 | +43.589 | +65.9% | worse |
| avg_wall_time_min | 3971.337 | 6586.689 | +2615.352 | +65.9% | worse |

### `base_agent_oss120b_folding_itr30_2026_08_09_13_47`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 48 | -1 | -2.0% | worse |
| correct_rate | 0.9800 | 0.9600 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 1.2891 | 1.3306 | +0.0416 | +3.2% | better |
| speedup_best_mean | 1.6111 | 1.4955 | -0.1156 | -7.2% | worse |
| speedup_best_median | 1.2523 | 1.0766 | -0.1757 | -14.0% | worse |
| speedup_best_geomean | 1.3855 | 1.2243 | -0.1612 | -11.6% | worse |
| speedup_current_geomean | 1.2805 | 1.0551 | -0.2254 | -17.6% | worse |
| hack_iteration_count | 13 | 21 | +8 | +61.5% | worse |
| problems_with_hack | 8 | 11 | +3 | +37.5% | worse |
| l1_entry_count | 561 | 601 | +40 | +7.1% | worse |
| total_wall_time_hours | 66.189 | 90.076 | +23.887 | +36.1% | worse |
| avg_wall_time_min | 3971.337 | 5404.560 | +1433.223 | +36.1% | worse |

### `base_agent_oss120b_markov_itr30_2026_08_07_14_07`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 50 | +1 | +2.0% | better |
| correct_rate | 0.9800 | 1.0000 | +0.0200 | +2.0% | better |
| best_speedup_overall | 1.2891 | 1.3636 | +0.0746 | +5.8% | better |
| speedup_best_mean | 1.6111 | 1.3514 | -0.2597 | -16.1% | worse |
| speedup_best_median | 1.2523 | 1.0400 | -0.2123 | -17.0% | worse |
| speedup_best_geomean | 1.3855 | 1.0302 | -0.3553 | -25.6% | worse |
| speedup_current_geomean | 1.2805 | 1.1555 | -0.1250 | -9.8% | worse |
| hack_iteration_count | 13 | 17 | +4 | +30.8% | worse |
| problems_with_hack | 8 | 14 | +6 | +75.0% | worse |
| l1_entry_count | 561 | 377 | -184 | -32.8% | better |
| total_wall_time_hours | 66.189 | 84.071 | +17.882 | +27.0% | worse |
| avg_wall_time_min | 3971.337 | 5044.279 | +1072.942 | +27.0% | worse |

### `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 49 | +0 | +0.0% | same |
| correct_rate | 0.9800 | 0.9800 | +0.0000 | +0.0% | same |
| best_speedup_overall | 1.2891 | 1.3636 | +0.0746 | +5.8% | better |
| speedup_best_mean | 1.6111 | 1.4341 | -0.1770 | -11.0% | worse |
| speedup_best_median | 1.2523 | 1.0525 | -0.1998 | -16.0% | worse |
| speedup_best_geomean | 1.3855 | 1.2387 | -0.1467 | -10.6% | worse |
| speedup_current_geomean | 1.2805 | 1.0813 | -0.1991 | -15.6% | worse |
| hack_iteration_count | 13 | 21 | +8 | +61.5% | worse |
| problems_with_hack | 8 | 15 | +7 | +87.5% | worse |
| l1_entry_count | 561 | 596 | +35 | +6.2% | worse |
| total_wall_time_hours | 66.189 | 89.996 | +23.807 | +36.0% | worse |
| avg_wall_time_min | 3971.337 | 5399.774 | +1428.437 | +36.0% | worse |

### `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 48 | -1 | -2.0% | worse |
| correct_rate | 0.9800 | 0.9600 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 1.2891 | 1.3636 | +0.0746 | +5.8% | better |
| speedup_best_mean | 1.6111 | 1.4824 | -0.1287 | -8.0% | worse |
| speedup_best_median | 1.2523 | 1.1080 | -0.1443 | -11.5% | worse |
| speedup_best_geomean | 1.3855 | 1.2859 | -0.0995 | -7.2% | worse |
| speedup_current_geomean | 1.2805 | 1.2320 | -0.0484 | -3.8% | worse |
| hack_iteration_count | 13 | 15 | +2 | +15.4% | worse |
| problems_with_hack | 8 | 12 | +4 | +50.0% | worse |
| l1_entry_count | 561 | 457 | -104 | -18.5% | better |
| total_wall_time_hours | 66.189 | 86.855 | +20.666 | +31.2% | worse |
| avg_wall_time_min | 3971.337 | 2605.647 | -1365.690 | -34.4% | better |

### `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 47 | -2 | -4.1% | worse |
| correct_rate | 0.9800 | 0.9400 | -0.0400 | -4.1% | worse |
| best_speedup_overall | 1.2891 | 2.7415 | +1.4524 | +112.7% | better |
| speedup_best_mean | 1.6111 | 1.4199 | -0.1912 | -11.9% | worse |
| speedup_best_median | 1.2523 | 1.0570 | -0.1953 | -15.6% | worse |
| speedup_best_geomean | 1.3855 | 1.2333 | -0.1521 | -11.0% | worse |
| speedup_current_geomean | 1.2805 | 0.9941 | -0.2863 | -22.4% | worse |
| hack_iteration_count | 13 | 19 | +6 | +46.2% | worse |
| problems_with_hack | 8 | 17 | +9 | +112.5% | worse |
| l1_entry_count | 561 | 653 | +92 | +16.4% | worse |
| total_wall_time_hours | 66.189 | 73.136 | +6.947 | +10.5% | worse |
| avg_wall_time_min | 3971.337 | 4388.164 | +416.827 | +10.5% | worse |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) | delta(R6-R1) | delta(R7-R1) | delta(R8-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7360 | 0.4934 | 0.7955 | 0.8415 | 0.3548 | 0.7761 | 0.5203 | 0.8059 | -0.2426 | +0.0595 | +0.1055 | -0.3812 | +0.0401 | -0.2157 | +0.0699 |
| 5 | 1.0063 | 0.8283 | 1.1150 | 0.9787 | 0.8234 | 1.0064 | 0.9858 | 0.7754 | -0.1781 | +0.1087 | -0.0277 | -0.1829 | +0.0001 | -0.0206 | -0.2310 |
| 10 | 1.1700 | 0.8606 | 1.1531 | 1.0169 | 0.8886 | 1.0173 | 1.0654 | 1.1061 | -0.3094 | -0.0169 | -0.1532 | -0.2814 | -0.1527 | -0.1046 | -0.0639 |
| 15 | 1.2789 | 1.0609 | 1.2063 | 1.1104 | 0.9222 | 1.0655 | 1.1446 | 1.1868 | -0.2180 | -0.0725 | -0.1684 | -0.3566 | -0.2133 | -0.1342 | -0.0921 |
| 20 | 1.3218 | 1.1428 | 1.3111 | 1.1866 | 0.9826 | 1.1548 | 1.1991 | 1.1986 | -0.1790 | -0.0107 | -0.1352 | -0.3392 | -0.1670 | -0.1227 | -0.1231 |
| 25 | 1.3986 | 1.1811 | 1.3564 | 1.2090 | 1.0130 | 1.1925 | 1.2330 | 1.2107 | -0.2175 | -0.0422 | -0.1896 | -0.3856 | -0.2061 | -0.1657 | -0.1879 |
| 30 | 1.3855 | 1.2518 | 1.3966 | 1.2243 | 1.0302 | 1.2387 | 1.2859 | 1.2333 | -0.1337 | +0.0112 | -0.1612 | -0.3553 | -0.1467 | -0.0995 | -0.1521 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) | delta(R6-R1) | delta(R7-R1) | delta(R8-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.020 | 0.040 | 0.140 | 0.100 | 0.060 | 0.040 | 0.020 | 0.040 | +0.020 | +0.120 | +0.080 | +0.040 | +0.020 | +0.000 | +0.020 |
| 5 | 0.340 | 0.280 | 0.420 | 0.420 | 0.340 | 0.320 | 0.380 | 0.260 | -0.060 | +0.080 | +0.080 | +0.000 | -0.020 | +0.040 | -0.080 |
| 10 | 0.520 | 0.380 | 0.460 | 0.500 | 0.460 | 0.460 | 0.540 | 0.520 | -0.140 | -0.060 | -0.020 | -0.060 | -0.060 | +0.020 | +0.000 |
| 15 | 0.580 | 0.520 | 0.520 | 0.540 | 0.500 | 0.560 | 0.640 | 0.580 | -0.060 | -0.060 | -0.040 | -0.080 | -0.020 | +0.060 | +0.000 |
| 20 | 0.660 | 0.560 | 0.560 | 0.580 | 0.560 | 0.600 | 0.660 | 0.600 | -0.100 | -0.100 | -0.080 | -0.100 | -0.060 | +0.000 | -0.060 |
| 25 | 0.680 | 0.600 | 0.620 | 0.600 | 0.600 | 0.620 | 0.680 | 0.620 | -0.080 | -0.060 | -0.080 | -0.080 | -0.060 | +0.000 | -0.060 |
| 30 | 0.720 | 0.640 | 0.660 | 0.600 | 0.600 | 0.640 | 0.700 | 0.620 | -0.080 | -0.060 | -0.120 | -0.120 | -0.080 | -0.020 | -0.100 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.3855 | 1.2518 | -0.1337 | -9.6% |
| R2 | fast_p_best@1.0 | 30 | 0.720 | 0.640 | -0.080 | -11.1% |
| R3 | best_geomean | 30 | 1.3855 | 1.3966 | +0.0112 | +0.8% |
| R3 | fast_p_best@1.0 | 30 | 0.720 | 0.660 | -0.060 | -8.3% |
| R4 | best_geomean | 30 | 1.3855 | 1.2243 | -0.1612 | -11.6% |
| R4 | fast_p_best@1.0 | 30 | 0.720 | 0.600 | -0.120 | -16.7% |
| R5 | best_geomean | 30 | 1.3855 | 1.0302 | -0.3553 | -25.6% |
| R5 | fast_p_best@1.0 | 30 | 0.720 | 0.600 | -0.120 | -16.7% |
| R6 | best_geomean | 30 | 1.3855 | 1.2387 | -0.1467 | -10.6% |
| R6 | fast_p_best@1.0 | 30 | 0.720 | 0.640 | -0.080 | -11.1% |
| R7 | best_geomean | 30 | 1.3855 | 1.2859 | -0.0995 | -7.2% |
| R7 | fast_p_best@1.0 | 30 | 0.720 | 0.700 | -0.020 | -2.8% |
| R8 | best_geomean | 30 | 1.3855 | 1.2333 | -0.1521 | -11.0% |
| R8 | fast_p_best@1.0 | 30 | 0.720 | 0.620 | -0.100 | -13.9% |

## Notes

- `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_deletion_itr30_2026_08_02_17_57`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_folding_itr30_2026_08_09_13_47`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_markov_itr30_2026_08_07_14_07`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
