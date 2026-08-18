# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-17T15:23:11.132785+00:00`
- aggregate_generated_at_utc: `2026-08-17T15:22:57.258266+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 6
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` | complete | compress_trigger | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_14_15_52` | complete | truncation | gpt-oss-120b | inference |
| R3 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_13_12_47` | complete | folding | gpt-oss-120b | inference |
| R4 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | complete | truncation | gpt-oss-120b | inference |
| R5 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | complete | markov_report | gpt-oss-120b | inference |
| R6 | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` | complete | selective_retention | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.23 | 77.1 | 0 |
| R2 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 66.91 | 80.3 | 0 |
| R3 | folding | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 66.59 | 79.9 | 0 |
| R4 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 74.18 | 88.0 | 0 |
| R5 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 71.43 | 83.5 | 0 |
| R6 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 69.93 | 83.9 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | complete | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 | 42 | 0.960 | 0.400 | 0.140 | 0.7265 | 48 |
| R2 | truncation+deletion | complete | 48/50 | 0.840 | 0.400 | 0.100 | 0.8691 | 42 | 0.960 | 0.540 | 0.280 | 1.2312 | 48 |
| R3 | folding | complete | 47/50 | 0.840 | 0.320 | 0.120 | 0.6744 | 42 | 0.940 | 0.420 | 0.180 | 0.8938 | 47 |
| R4 | truncation | complete | 47/50 | 0.840 | 0.380 | 0.120 | 0.7379 | 42 | 0.940 | 0.460 | 0.180 | 0.9051 | 47 |
| R5 | markov_report | complete | 48/50 | 0.900 | 0.340 | 0.040 | 0.7145 | 45 | 0.960 | 0.460 | 0.140 | 0.9332 | 48 |
| R6 | selective_retention | complete | 48/50 | 0.900 | 0.320 | 0.080 | 0.7657 | 45 | 0.960 | 0.460 | 0.160 | 0.9541 | 48 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.2428 | 0.7582 | 0.7265 | 48 | 0.7491 | 29 | 2.9647 | 32 | 23 | 0.960 | 0.640 | 0.460 | 0.400 | 0.180 | 0.140 |
| R2 | 30 | 50 | 1.9729 | 1.0513 | 1.2312 | 48 | 1.4874 | 29 | 2.1050 | 19 | 12 | 0.960 | 0.820 | 0.720 | 0.540 | 0.300 | 0.280 |
| R3 | 30 | 50 | 1.5545 | 0.9218 | 0.8938 | 47 | 0.8176 | 26 | 6.3696 | 17 | 12 | 0.940 | 0.680 | 0.540 | 0.420 | 0.220 | 0.180 |
| R4 | 30 | 50 | 1.5437 | 0.9394 | 0.9051 | 47 | 0.8759 | 28 | 5.4461 | 16 | 11 | 0.940 | 0.720 | 0.500 | 0.460 | 0.240 | 0.180 |
| R5 | 30 | 50 | 1.3089 | 0.9640 | 0.9332 | 48 | 0.9225 | 39 | 6.2340 | 14 | 11 | 0.960 | 0.780 | 0.600 | 0.460 | 0.160 | 0.140 |
| R6 | 30 | 50 | 1.4787 | 0.9144 | 0.9541 | 48 | 0.9747 | 37 | 9.7594 | 17 | 12 | 0.960 | 0.740 | 0.540 | 0.460 | 0.220 | 0.160 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 435 | 435 | 0 | 0 | 0 | 0 | 0 |
| R2 | yes | no | no | 592 | 25 | 0 | 567 | 0 | 567 | 3 |
| R3 | no | no | no | 592 | 592 | 0 | 0 | 0 | 0 | 0 |
| R4 | no | no | no | 571 | 571 | 0 | 0 | 0 | 0 | 0 |
| R5 | no | no | no | 366 | 366 | 0 | 0 | 0 | 0 | 0 |
| R6 | no | no | no | 619 | 619 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

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

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3999 | 0.6971 | 0.3440 | 0.4816 | 0.6979 | 0.4330 | -0.0817 | +0.2155 | -0.1376 | +0.2163 | -0.0486 |
| 5 | 0.5168 | 0.7275 | 0.5638 | 0.5552 | 0.5892 | 0.5048 | -0.0384 | +0.1723 | +0.0086 | +0.0340 | -0.0504 |
| 10 | 0.5624 | 0.8691 | 0.6744 | 0.7379 | 0.7145 | 0.7657 | -0.1755 | +0.1311 | -0.0635 | -0.0234 | +0.0278 |
| 15 | 0.5445 | 0.9468 | 0.7570 | 0.7728 | 0.8178 | 0.7811 | -0.2283 | +0.1740 | -0.0158 | +0.0450 | +0.0083 |
| 20 | 0.6344 | 0.9518 | 0.8256 | 0.8292 | 0.8441 | 0.8710 | -0.1948 | +0.1226 | -0.0036 | +0.0149 | +0.0417 |
| 25 | 0.6856 | 1.1073 | 0.8709 | 0.8806 | 0.8722 | 0.9440 | -0.1951 | +0.2266 | -0.0097 | -0.0084 | +0.0634 |
| 30 | 0.7265 | 1.2312 | 0.8938 | 0.9051 | 0.9332 | 0.9541 | -0.1786 | +0.3261 | -0.0113 | +0.0281 | +0.0490 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.080 | 0.120 | 0.060 | 0.080 | 0.040 | 0.040 | +0.000 | +0.040 | -0.020 | -0.040 | -0.040 |
| 5 | 0.200 | 0.300 | 0.240 | 0.260 | 0.280 | 0.200 | -0.060 | +0.040 | -0.020 | +0.020 | -0.060 |
| 10 | 0.240 | 0.400 | 0.320 | 0.380 | 0.340 | 0.320 | -0.140 | +0.020 | -0.060 | -0.040 | -0.060 |
| 15 | 0.280 | 0.400 | 0.380 | 0.400 | 0.380 | 0.380 | -0.120 | +0.000 | -0.020 | -0.020 | -0.020 |
| 20 | 0.360 | 0.420 | 0.400 | 0.420 | 0.400 | 0.420 | -0.060 | +0.000 | -0.020 | -0.020 | +0.000 |
| 25 | 0.380 | 0.480 | 0.420 | 0.420 | 0.420 | 0.460 | -0.040 | +0.060 | +0.000 | +0.000 | +0.040 |
| 30 | 0.400 | 0.540 | 0.420 | 0.460 | 0.460 | 0.460 | -0.060 | +0.080 | -0.040 | +0.000 | +0.000 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | best_geomean | 30 | 0.9051 | 0.7265 | -0.1786 | -19.7% |
| R1 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |
| R2 | best_geomean | 30 | 0.9051 | 1.2312 | +0.3261 | +36.0% |
| R2 | fast_p_best@1.0 | 30 | 0.460 | 0.540 | +0.080 | +17.4% |
| R3 | best_geomean | 30 | 0.9051 | 0.8938 | -0.0113 | -1.3% |
| R3 | fast_p_best@1.0 | 30 | 0.460 | 0.420 | -0.040 | -8.7% |
| R5 | best_geomean | 30 | 0.9051 | 0.9332 | +0.0281 | +3.1% |
| R5 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R6 | best_geomean | 30 | 0.9051 | 0.9541 | +0.0490 | +5.4% |
| R6 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
