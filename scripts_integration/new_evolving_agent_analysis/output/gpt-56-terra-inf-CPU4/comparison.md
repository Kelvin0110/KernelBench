# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-16T03:47:07.772377+00:00`
- aggregate_generated_at_utc: `2026-08-16T03:36:42.004786+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 3
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | complete | truncation | gpt-5.6-terra | inference |
| R2 | `base_agent_terra_markov_itr30_2026_08_01_17_41` | complete | markov_report | gpt-5.6-terra | inference |
| R3 | `base_agent_terra_compress_trigger_itr30_2026_08_10_15_24` | complete | compress_trigger | gpt-5.6-terra | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 104.11 | 6246.5 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 94.71 | 2841.3 | 0 |
| R3 | compress_trigger | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 73.25 | 87.9 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses only correct, non-hack samples; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | complete | 49/50 | 0.960 | 0.700 | 0.200 | 1.5023 | 45 | 0.980 | 0.820 | 0.260 | 1.7796 | 44 |
| R2 | markov_report | complete | 49/50 | 0.940 | 0.740 | 0.200 | 1.4437 | 43 | 0.980 | 0.820 | 0.300 | 1.8153 | 39 |
| R3 | compress_trigger | complete | 49/50 | 0.980 | 0.680 | 0.200 | 1.4046 | 46 | 0.980 | 0.700 | 0.300 | 1.6438 | 43 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 2.4521 | 1.4291 | 1.7796 | 44 | 1.5851 | 45 | 9.5418 | 55 | 6 | 0.980 | 0.960 | 0.940 | 0.820 | 0.460 | 0.260 |
| R2 | 30 | 50 | 2.4306 | 1.6065 | 1.8153 | 39 | 1.6506 | 39 | 4.8670 | 136 | 11 | 0.980 | 0.940 | 0.940 | 0.820 | 0.540 | 0.300 |
| R3 | 30 | 50 | 2.3215 | 1.3115 | 1.6438 | 43 | 1.4981 | 47 | 5.8842 | 65 | 7 | 0.980 | 0.940 | 0.880 | 0.700 | 0.400 | 0.300 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 345 | 345 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 293 | 293 | 0 | 0 | 0 | 0 | 0 |
| R3 | no | no | no | 280 | 280 | 0 | 0 | 0 | 0 | 0 |

## Deltas vs baseline run `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`

### `base_agent_terra_markov_itr30_2026_08_01_17_41`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 49 | +0 | +0.0% | same |
| correct_rate | 0.9800 | 0.9800 | +0.0000 | +0.0% | same |
| best_speedup_overall | 9.5418 | 4.8670 | -4.6748 | -49.0% | worse |
| speedup_best_mean | 2.4521 | 2.4306 | -0.0215 | -0.9% | worse |
| speedup_best_median | 1.4291 | 1.6065 | +0.1774 | +12.4% | better |
| speedup_best_geomean | 1.7796 | 1.8153 | +0.0357 | +2.0% | better |
| speedup_current_geomean | 1.5851 | 1.6506 | +0.0654 | +4.1% | better |
| hack_iteration_count | 55 | 136 | +81 | +147.3% | worse |
| problems_with_hack | 6 | 11 | +5 | +83.3% | worse |
| l1_entry_count | 345 | 293 | -52 | -15.1% | better |
| total_wall_time_hours | 104.109 | 94.710 | -9.399 | -9.0% | better |
| avg_wall_time_min | 6246.530 | 2841.294 | -3405.236 | -54.5% | better |

### `base_agent_terra_compress_trigger_itr30_2026_08_10_15_24`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 49 | 49 | +0 | +0.0% | same |
| correct_rate | 0.9800 | 0.9800 | +0.0000 | +0.0% | same |
| best_speedup_overall | 9.5418 | 5.8842 | -3.6575 | -38.3% | worse |
| speedup_best_mean | 2.4521 | 2.3215 | -0.1306 | -5.3% | worse |
| speedup_best_median | 1.4291 | 1.3115 | -0.1176 | -8.2% | worse |
| speedup_best_geomean | 1.7796 | 1.6438 | -0.1358 | -7.6% | worse |
| speedup_current_geomean | 1.5851 | 1.4981 | -0.0870 | -5.5% | worse |
| hack_iteration_count | 55 | 65 | +10 | +18.2% | worse |
| problems_with_hack | 6 | 7 | +1 | +16.7% | worse |
| l1_entry_count | 345 | 280 | -65 | -18.8% | better |
| total_wall_time_hours | 104.109 | 73.247 | -30.862 | -29.6% | better |
| avg_wall_time_min | 6246.530 | 87.897 | -6158.634 | -98.6% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | delta(R2-R1) | delta(R3-R1) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0442 | 1.1531 | 0.9944 | +0.1089 | -0.0498 |
| 5 | 1.3059 | 1.2867 | 1.2911 | -0.0192 | -0.0148 |
| 10 | 1.5023 | 1.4437 | 1.4046 | -0.0586 | -0.0977 |
| 15 | 1.6241 | 1.5290 | 1.4836 | -0.0951 | -0.1405 |
| 20 | 1.7194 | 1.7631 | 1.5664 | +0.0437 | -0.1530 |
| 25 | 1.7666 | 1.8064 | 1.5971 | +0.0398 | -0.1695 |
| 30 | 1.7796 | 1.8153 | 1.6438 | +0.0357 | -0.1358 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | delta(R2-R1) | delta(R3-R1) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.260 | 0.360 | 0.300 | +0.100 | +0.040 |
| 5 | 0.660 | 0.680 | 0.660 | +0.020 | +0.000 |
| 10 | 0.700 | 0.740 | 0.680 | +0.040 | -0.020 |
| 15 | 0.780 | 0.760 | 0.680 | -0.020 | -0.100 |
| 20 | 0.800 | 0.820 | 0.700 | +0.020 | -0.100 |
| 25 | 0.820 | 0.820 | 0.700 | +0.000 | -0.120 |
| 30 | 0.820 | 0.820 | 0.700 | +0.000 | -0.120 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R2 | best_geomean | 30 | 1.7796 | 1.8153 | +0.0357 | +2.0% |
| R2 | fast_p_best@1.0 | 30 | 0.820 | 0.820 | +0.000 | +0.0% |
| R3 | best_geomean | 30 | 1.7796 | 1.6438 | -0.1358 | -7.6% |
| R3 | fast_p_best@1.0 | 30 | 0.820 | 0.700 | -0.120 | -14.6% |

## Notes

- `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
- `base_agent_terra_compress_trigger_itr30_2026_08_10_15_24`: performance_stats rebuilt: run artifacts are newer than cached performance_stats.json
