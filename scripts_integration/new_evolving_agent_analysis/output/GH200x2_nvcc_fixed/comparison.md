# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-16T07:09:19.433972+00:00`
- aggregate_generated_at_utc: `2026-08-16T07:09:06.104942+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 4
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | complete | markov_report | gpt-oss-120b | inference |
| R3 | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` | complete | selective_retention | gpt-oss-120b | inference |
| R4 | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` | complete | compress_trigger | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 74.18 | 88.0 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 71.43 | 83.5 | 0 |
| R3 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 69.93 | 83.9 | 0 |
| R4 | compress_trigger | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.23 | 77.1 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | complete | 47/50 | 0.840 | 0.380 | 0.120 | 0.7352 | 42 | 0.940 | 0.460 | 0.180 | 0.8746 | 47 |
| R2 | markov_report | complete | 48/50 | 0.900 | 0.340 | 0.040 | 0.7373 | 45 | 0.960 | 0.460 | 0.140 | 0.9830 | 48 |
| R3 | selective_retention | complete | 48/50 | 0.900 | 0.320 | 0.080 | 0.7657 | 45 | 0.960 | 0.460 | 0.160 | 0.9541 | 48 |
| R4 | compress_trigger | complete | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 | 42 | 0.960 | 0.400 | 0.140 | 0.7265 | 48 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.3722 | 0.9704 | 0.8746 | 47 | 0.8759 | 28 | 5.4461 | 16 | 11 | 0.940 | 0.720 | 0.500 | 0.460 | 0.240 | 0.180 |
| R2 | 30 | 50 | 1.4358 | 1.0046 | 0.9830 | 48 | 0.9225 | 39 | 6.2340 | 14 | 11 | 0.960 | 0.780 | 0.600 | 0.460 | 0.160 | 0.140 |
| R3 | 30 | 50 | 1.4787 | 0.9144 | 0.9541 | 48 | 0.9747 | 37 | 9.7594 | 17 | 12 | 0.960 | 0.740 | 0.540 | 0.460 | 0.220 | 0.160 |
| R4 | 30 | 50 | 1.2428 | 0.7582 | 0.7265 | 48 | 0.7491 | 29 | 2.9647 | 32 | 23 | 0.960 | 0.640 | 0.460 | 0.400 | 0.180 | 0.140 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 571 | 571 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 366 | 366 | 0 | 0 | 0 | 0 | 0 |
| R3 | no | no | no | 619 | 619 | 0 | 0 | 0 | 0 | 0 |
| R4 | no | no | no | 435 | 435 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 6.2340 | +0.7879 | +14.5% | better |
| speedup_best_mean | 1.3722 | 1.4358 | +0.0636 | +4.6% | better |
| speedup_best_median | 0.9704 | 1.0046 | +0.0342 | +3.5% | better |
| speedup_best_geomean | 0.8746 | 0.9830 | +0.1084 | +12.4% | better |
| speedup_current_geomean | 0.8759 | 0.9225 | +0.0465 | +5.3% | better |
| hack_iteration_count | 16 | 14 | -2 | -12.5% | better |
| problems_with_hack | 11 | 11 | +0 | +0.0% | same |
| l1_entry_count | 571 | 366 | -205 | -35.9% | better |
| total_wall_time_hours | 74.177 | 71.433 | -2.745 | -3.7% | better |
| avg_wall_time_min | 88.035 | 83.506 | -4.529 | -5.1% | better |

### `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 9.7594 | +4.3134 | +79.2% | better |
| speedup_best_mean | 1.3722 | 1.4787 | +0.1065 | +7.8% | better |
| speedup_best_median | 0.9704 | 0.9144 | -0.0560 | -5.8% | worse |
| speedup_best_geomean | 0.8746 | 0.9541 | +0.0795 | +9.1% | better |
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
| speedup_best_mean | 1.3722 | 1.2428 | -0.1294 | -9.4% | worse |
| speedup_best_median | 0.9704 | 0.7582 | -0.2122 | -21.9% | worse |
| speedup_best_geomean | 0.8746 | 0.7265 | -0.1481 | -16.9% | worse |
| speedup_current_geomean | 0.8759 | 0.7491 | -0.1268 | -14.5% | worse |
| hack_iteration_count | 16 | 32 | +16 | +100.0% | worse |
| problems_with_hack | 11 | 23 | +12 | +109.1% | worse |
| l1_entry_count | 571 | 435 | -136 | -23.8% | better |
| total_wall_time_hours | 74.177 | 64.230 | -9.947 | -13.4% | better |
| avg_wall_time_min | 88.035 | 77.076 | -10.958 | -12.4% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4816 | 0.6979 | 0.4330 | 0.3999 | +0.2163 | -0.0486 | -0.0817 |
| 5 | 0.5510 | 0.5827 | 0.5048 | 0.5168 | +0.0317 | -0.0462 | -0.0342 |
| 10 | 0.7352 | 0.7373 | 0.7657 | 0.5624 | +0.0021 | +0.0305 | -0.1728 |
| 15 | 0.7724 | 0.8426 | 0.7811 | 0.5445 | +0.0702 | +0.0087 | -0.2279 |
| 20 | 0.8375 | 0.8685 | 0.8710 | 0.6344 | +0.0310 | +0.0334 | -0.2031 |
| 25 | 0.8025 | 0.9026 | 0.9440 | 0.6856 | +0.1001 | +0.1415 | -0.1170 |
| 30 | 0.8746 | 0.9830 | 0.9541 | 0.7265 | +0.1084 | +0.0795 | -0.1481 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.080 | 0.040 | 0.040 | 0.080 | -0.040 | -0.040 | +0.000 |
| 5 | 0.260 | 0.280 | 0.200 | 0.200 | +0.020 | -0.060 | -0.060 |
| 10 | 0.380 | 0.340 | 0.320 | 0.240 | -0.040 | -0.060 | -0.140 |
| 15 | 0.400 | 0.380 | 0.380 | 0.280 | -0.020 | -0.020 | -0.120 |
| 20 | 0.420 | 0.400 | 0.420 | 0.360 | -0.020 | +0.000 | -0.060 |
| 25 | 0.420 | 0.420 | 0.460 | 0.380 | +0.000 | +0.040 | -0.040 |
| 30 | 0.460 | 0.460 | 0.460 | 0.400 | +0.000 | +0.000 | -0.060 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 0.8746 | 0.9830 | +0.1084 | +12.4% |
| R2 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R3 | best_geomean | 30 | 0.8746 | 0.9541 | +0.0795 | +9.1% |
| R3 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R4 | best_geomean | 30 | 0.8746 | 0.7265 | -0.1481 | -16.9% |
| R4 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |

## Notes

- `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
