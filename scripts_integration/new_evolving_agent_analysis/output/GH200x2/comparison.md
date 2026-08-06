# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-05T01:52:57.964490+00:00`
- aggregate_generated_at_utc: `2026-08-05T01:52:41.735436+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 2

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51` | complete | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52` | complete | markov_report | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 34.27 | 41.1 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 31.89 | 38.3 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.0903 | 1.1201 | 1.0767 | 3 | 1.1551 | 29 | 6.0663 | 130 | 47 | 0.980 | 0.740 | 0.660 | 0.540 | 0.380 | 0.260 |
| R2 | 30 | 50 | 1.2315 | 0.9476 | 0.7898 | 14 | 0.5928 | 38 | 4.9410 | 72 | 36 | 0.960 | 0.600 | 0.400 | 0.320 | 0.120 | 0.100 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 585 | 585 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 426 | 426 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51`

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 48 | -1 | -2.0% | worse |
| correct_rate | 0.9800 | 0.9600 | -0.0200 | -2.0% | worse |
| best_speedup_overall | 6.0663 | 4.9410 | -1.1253 | -18.5% | worse |
| speedup_best_mean | 1.0903 | 1.2315 | +0.1413 | +13.0% | better |
| speedup_best_median | 1.1201 | 0.9476 | -0.1725 | -15.4% | worse |
| speedup_best_geomean | 1.0767 | 0.7898 | -0.2868 | -26.6% | worse |
| speedup_current_geomean | 1.1551 | 0.5928 | -0.5623 | -48.7% | worse |
| hack_iteration_count | 130 | 72 | -58 | -44.6% | better |
| problems_with_hack | 47 | 36 | -11 | -23.4% | better |
| l1_entry_count | 585 | 426 | -159 | -27.2% | better |
| total_wall_time_hours | 34.273 | 31.888 | -2.385 | -7.0% | better |
| avg_wall_time_min | 41.128 | 38.266 | -2.862 | -7.0% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.5261 | 1.0508 | +0.5247 |
| 5 | 0.3889 | 0.5094 | +0.1205 |
| 10 | 0.5313 | 0.4966 | -0.0347 |
| 15 | 0.4248 | 0.5909 | +0.1661 |
| 20 | 0.4309 | 0.6425 | +0.2117 |
| 25 | 0.5858 | 0.7127 | +0.1269 |
| 30 | 1.0767 | 0.7898 | -0.2868 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | delta(R2-R1) |
| --- | --- | --- | --- |
| 1 | 0.000 | 0.020 | +0.020 |
| 5 | 0.100 | 0.140 | +0.040 |
| 10 | 0.180 | 0.160 | -0.020 |
| 15 | 0.300 | 0.220 | -0.080 |
| 20 | 0.420 | 0.280 | -0.140 |
| 25 | 0.540 | 0.280 | -0.260 |
| 30 | 0.540 | 0.320 | -0.220 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.0767 | 0.7898 | -0.2868 | -26.6% |
| R2 | fast_p_best@1.0 | 30 | 0.540 | 0.320 | -0.220 | -40.7% |
