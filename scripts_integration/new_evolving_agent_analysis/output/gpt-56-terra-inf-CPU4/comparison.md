# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-10T06:49:35.796081+00:00`
- aggregate_generated_at_utc: `2026-08-10T06:49:07.921187+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 2

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | complete | truncation | gpt-5.6-terra | inference |
| R2 | `base_agent_terra_markov_itr30_2026_08_01_17_41` | complete | markov_report | gpt-5.6-terra | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 64.07 | 91.5 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 76.69 | 121.1 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.7173 | 1.3036 | 1.4544 | 44 | 1.3361 | 19 | 5.7294 | 50 | 6 | 0.980 | 0.940 | 0.920 | 0.780 | 0.480 | 0.260 |
| R2 | 30 | 50 | 1.6902 | 1.4956 | 1.4545 | 39 | 1.5765 | 31 | 4.8670 | 71 | 10 | 0.940 | 0.900 | 0.880 | 0.780 | 0.500 | 0.260 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 213 | 213 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 251 | 251 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`

### `base_agent_terra_markov_itr30_2026_08_01_17_41`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 47 | -2 | -4.1% | worse |
| correct_rate | 0.9800 | 0.9400 | -0.0400 | -4.1% | worse |
| best_speedup_overall | 5.7294 | 4.8670 | -0.8623 | -15.1% | worse |
| speedup_best_mean | 1.7173 | 1.6902 | -0.0272 | -1.6% | worse |
| speedup_best_median | 1.3036 | 1.4956 | +0.1920 | +14.7% | better |
| speedup_best_geomean | 1.4544 | 1.4545 | +0.0001 | +0.0% | better |
| speedup_current_geomean | 1.3361 | 1.5765 | +0.2404 | +18.0% | better |
| hack_iteration_count | 50 | 71 | +21 | +42.0% | worse |
| problems_with_hack | 6 | 10 | +4 | +66.7% | worse |
| l1_entry_count | 213 | 251 | +38 | +17.8% | worse |
| total_wall_time_hours | 64.066 | 76.691 | +12.625 | +19.7% | worse |
| avg_wall_time_min | 91.523 | 121.091 | +29.569 | +32.3% | worse |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.9950 | 1.1778 | +0.1829 |
| 5 | 1.1937 | 1.3124 | +0.1187 |
| 10 | 1.3211 | 1.4791 | +0.1580 |
| 15 | 1.3623 | 1.4086 | +0.0463 |
| 20 | 1.4332 | 1.4140 | -0.0192 |
| 25 | 1.4503 | 1.4464 | -0.0039 |
| 30 | 1.4544 | 1.4545 | +0.0001 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.280 | 0.380 | +0.100 |
| 5 | 0.600 | 0.680 | +0.080 |
| 10 | 0.660 | 0.740 | +0.080 |
| 15 | 0.740 | 0.760 | +0.020 |
| 20 | 0.760 | 0.780 | +0.020 |
| 25 | 0.780 | 0.780 | +0.000 |
| 30 | 0.780 | 0.780 | +0.000 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.4544 | 1.4545 | +0.0001 | +0.0% |
| R2 | fast_p_best@1.0 | 30 | 0.780 | 0.780 | +0.000 | +0.0% |
