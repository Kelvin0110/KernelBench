# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-10T06:49:33.907521+00:00`
- aggregate_generated_at_utc: `2026-08-10T06:49:06.155698+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving/inference_oss_120b`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 2

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | complete | selective_retention | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 65.15 | 78.2 | 0 |
| R2 | selective_retention | 30 | 50 | 50 | 44 | 0.880 | total_attempted | 79.39 | 95.3 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.6460 | 1.2535 | 1.4208 | 41 | 1.2910 | 31 | 6.6667 | 12 | 7 | 0.960 | 0.940 | 0.840 | 0.720 | 0.280 | 0.200 |
| R2 | 30 | 50 | 1.4400 | 1.0946 | 1.2509 | 36 | 1.2033 | 29 | 1.3636 | 12 | 9 | 0.880 | 0.880 | 0.780 | 0.620 | 0.180 | 0.120 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 549 | 549 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 422 | 422 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

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

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.7360 | 0.5203 | -0.2157 |
| 5 | 0.8924 | 0.9752 | +0.0828 |
| 10 | 1.1820 | 1.0606 | -0.1214 |
| 15 | 1.3010 | 1.0712 | -0.2299 |
| 20 | 1.3536 | 1.2340 | -0.1196 |
| 25 | 1.4390 | 1.2499 | -0.1890 |
| 30 | 1.4208 | 1.2509 | -0.1699 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.020 | 0.020 | +0.000 |
| 5 | 0.320 | 0.340 | +0.020 |
| 10 | 0.520 | 0.480 | -0.040 |
| 15 | 0.580 | 0.560 | -0.020 |
| 20 | 0.660 | 0.600 | -0.060 |
| 25 | 0.680 | 0.600 | -0.080 |
| 30 | 0.720 | 0.620 | -0.100 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.4208 | 1.2509 | -0.1699 | -12.0% |
| R2 | fast_p_best@1.0 | 30 | 0.720 | 0.620 | -0.100 | -13.9% |
