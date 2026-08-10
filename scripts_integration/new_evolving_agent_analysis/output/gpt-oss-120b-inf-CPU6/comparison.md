# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-10T06:49:36.261873+00:00`
- aggregate_generated_at_utc: `2026-08-10T06:49:06.155698+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving/inference_oss_120b`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 5

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` | complete | truncation | gpt-oss-120b | inference |
| R3 | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` | complete | truncation | gpt-oss-120b | inference |
| R4 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` | complete | truncation | gpt-oss-120b | inference |
| R5 | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | complete | selective_retention | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 65.15 | 78.2 | 0 |
| R2 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 74.71 | 89.6 | 0 |
| R3 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 69.03 | 82.8 | 0 |
| R4 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 84.68 | 101.6 | 0 |
| R5 | selective_retention | 30 | 50 | 50 | 44 | 0.880 | total_attempted | 79.39 | 95.3 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.6460 | 1.2535 | 1.4208 | 41 | 1.2910 | 31 | 6.6667 | 12 | 7 | 0.960 | 0.940 | 0.840 | 0.720 | 0.280 | 0.200 |
| R2 | 30 | 50 | 1.3956 | 1.1384 | 1.2527 | 37 | 1.1092 | 30 | 3.3680 | 12 | 11 | 0.960 | 0.940 | 0.820 | 0.620 | 0.300 | 0.180 |
| R3 | 30 | 50 | 1.2359 | 1.0214 | 1.1202 | 33 | 0.9896 | 27 | 6.6263 | 17 | 15 | 0.940 | 0.920 | 0.840 | 0.620 | 0.160 | 0.120 |
| R4 | 30 | 50 | 1.3548 | 1.0812 | 1.1885 | 36 | 1.0872 | 32 | 1.3636 | 14 | 12 | 0.940 | 0.920 | 0.780 | 0.600 | 0.200 | 0.120 |
| R5 | 30 | 50 | 1.4400 | 1.0946 | 1.2509 | 36 | 1.2033 | 29 | 1.3636 | 12 | 9 | 0.880 | 0.880 | 0.780 | 0.620 | 0.180 | 0.120 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 549 | 549 | 0 | 0 | 0 | 0 | 0 |
| R2 | yes | no | no | 566 | 31 | 0 | 535 | 0 | 535 | 3 |
| R3 | no | no | yes | 626 | 545 | 0 | 0 | 87 | 0 | 1 |
| R4 | no | yes | no | 569 | 232 | 62 | 0 | 0 | 0 | 7 |
| R5 | no | no | no | 422 | 422 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

### `base_agent_oss120b_deletion_itr30_2026_08_02_17_57`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 48 | +0 | +0.0% | same |
| correct_rate | 0.9600 | 0.9600 | +0.0000 | +0.0% | same |
| best_speedup_overall | 6.6667 | 3.3680 | -3.2987 | -49.5% | worse |
| speedup_best_mean | 1.6460 | 1.3956 | -0.2505 | -15.2% | worse |
| speedup_best_median | 1.2535 | 1.1384 | -0.1151 | -9.2% | worse |
| speedup_best_geomean | 1.4208 | 1.2527 | -0.1681 | -11.8% | worse |
| speedup_current_geomean | 1.2910 | 1.1092 | -0.1818 | -14.1% | worse |
| hack_iteration_count | 12 | 12 | +0 | +0.0% | same |
| problems_with_hack | 7 | 11 | +4 | +57.1% | worse |
| l1_entry_count | 549 | 566 | +17 | +3.1% | worse |
| total_wall_time_hours | 65.147 | 74.707 | +9.560 | +14.7% | worse |
| avg_wall_time_min | 78.177 | 89.649 | +11.472 | +14.7% | worse |

### `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 47 | -1 | -2.1% | worse |
| correct_rate | 0.9600 | 0.9400 | -0.0200 | -2.1% | worse |
| best_speedup_overall | 6.6667 | 6.6263 | -0.0403 | -0.6% | worse |
| speedup_best_mean | 1.6460 | 1.2359 | -0.4102 | -24.9% | worse |
| speedup_best_median | 1.2535 | 1.0214 | -0.2321 | -18.5% | worse |
| speedup_best_geomean | 1.4208 | 1.1202 | -0.3006 | -21.2% | worse |
| speedup_current_geomean | 1.2910 | 0.9896 | -0.3014 | -23.3% | worse |
| hack_iteration_count | 12 | 17 | +5 | +41.7% | worse |
| problems_with_hack | 7 | 15 | +8 | +114.3% | worse |
| l1_entry_count | 549 | 626 | +77 | +14.0% | worse |
| total_wall_time_hours | 65.147 | 69.031 | +3.883 | +6.0% | worse |
| avg_wall_time_min | 78.177 | 82.837 | +4.660 | +6.0% | worse |

### `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 47 | -1 | -2.1% | worse |
| correct_rate | 0.9600 | 0.9400 | -0.0200 | -2.1% | worse |
| best_speedup_overall | 6.6667 | 1.3636 | -5.3030 | -79.5% | worse |
| speedup_best_mean | 1.6460 | 1.3548 | -0.2913 | -17.7% | worse |
| speedup_best_median | 1.2535 | 1.0812 | -0.1724 | -13.8% | worse |
| speedup_best_geomean | 1.4208 | 1.1885 | -0.2323 | -16.3% | worse |
| speedup_current_geomean | 1.2910 | 1.0872 | -0.2038 | -15.8% | worse |
| hack_iteration_count | 12 | 14 | +2 | +16.7% | worse |
| problems_with_hack | 7 | 12 | +5 | +71.4% | worse |
| l1_entry_count | 549 | 569 | +20 | +3.6% | worse |
| total_wall_time_hours | 65.147 | 84.682 | +19.534 | +30.0% | worse |
| avg_wall_time_min | 78.177 | 101.618 | +23.441 | +30.0% | worse |

### `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 44 | -4 | -8.3% | worse |
| correct_rate | 0.9600 | 0.8800 | -0.0800 | -8.3% | worse |
| best_speedup_overall | 6.6667 | 1.3636 | -5.3030 | -79.5% | worse |
| speedup_best_mean | 1.6460 | 1.4400 | -0.2061 | -12.5% | worse |
| speedup_best_median | 1.2535 | 1.0946 | -0.1589 | -12.7% | worse |
| speedup_best_geomean | 1.4208 | 1.2509 | -0.1699 | -12.0% | worse |
| speedup_current_geomean | 1.2910 | 1.2033 | -0.0877 | -6.8% | worse |
| hack_iteration_count | 12 | 12 | +0 | +0.0% | same |
| problems_with_hack | 7 | 9 | +2 | +28.6% | worse |
| l1_entry_count | 549 | 422 | -127 | -23.1% | better |
| total_wall_time_hours | 65.147 | 79.391 | +14.244 | +21.9% | worse |
| avg_wall_time_min | 78.177 | 95.270 | +17.093 | +21.9% | worse |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7360 | 0.5292 | 0.8059 | 0.7669 | 0.5203 | -0.2069 | +0.0699 | +0.0309 | -0.2157 |
| 5 | 0.8924 | 0.8527 | 0.9371 | 1.0129 | 0.9752 | -0.0397 | +0.0447 | +0.1205 | +0.0828 |
| 10 | 1.1820 | 0.8748 | 1.0957 | 1.1031 | 1.0606 | -0.3072 | -0.0863 | -0.0789 | -0.1214 |
| 15 | 1.3010 | 1.0911 | 1.1187 | 1.1483 | 1.0712 | -0.2099 | -0.1823 | -0.1527 | -0.2299 |
| 20 | 1.3536 | 1.1215 | 1.1368 | 1.1273 | 1.2340 | -0.2321 | -0.2168 | -0.2263 | -0.1196 |
| 25 | 1.4390 | 1.1903 | 1.0691 | 1.1722 | 1.2499 | -0.2487 | -0.3699 | -0.2667 | -0.1890 |
| 30 | 1.4208 | 1.2527 | 1.1202 | 1.1885 | 1.2509 | -0.1681 | -0.3006 | -0.2323 | -0.1699 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | delta(R2-R1) | delta(R3-R1) | delta(R4-R1) | delta(R5-R1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.020 | 0.060 | 0.040 | 0.040 | 0.020 | +0.040 | +0.020 | +0.020 | +0.000 |
| 5 | 0.320 | 0.300 | 0.240 | 0.320 | 0.340 | -0.020 | -0.080 | +0.000 | +0.020 |
| 10 | 0.520 | 0.380 | 0.520 | 0.460 | 0.480 | -0.140 | +0.000 | -0.060 | -0.040 |
| 15 | 0.580 | 0.500 | 0.580 | 0.540 | 0.560 | -0.080 | +0.000 | -0.040 | -0.020 |
| 20 | 0.660 | 0.540 | 0.600 | 0.560 | 0.600 | -0.120 | -0.060 | -0.100 | -0.060 |
| 25 | 0.680 | 0.580 | 0.620 | 0.580 | 0.600 | -0.100 | -0.060 | -0.100 | -0.080 |
| 30 | 0.720 | 0.620 | 0.620 | 0.600 | 0.620 | -0.100 | -0.100 | -0.120 | -0.100 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.4208 | 1.2527 | -0.1681 | -11.8% |
| R2 | fast_p_best@1.0 | 30 | 0.720 | 0.620 | -0.100 | -13.9% |
| R3 | best_geomean | 30 | 1.4208 | 1.1202 | -0.3006 | -21.2% |
| R3 | fast_p_best@1.0 | 30 | 0.720 | 0.620 | -0.100 | -13.9% |
| R4 | best_geomean | 30 | 1.4208 | 1.1885 | -0.2323 | -16.3% |
| R4 | fast_p_best@1.0 | 30 | 0.720 | 0.600 | -0.120 | -16.7% |
| R5 | best_geomean | 30 | 1.4208 | 1.2509 | -0.1699 | -12.0% |
| R5 | fast_p_best@1.0 | 30 | 0.720 | 0.620 | -0.100 | -13.9% |
