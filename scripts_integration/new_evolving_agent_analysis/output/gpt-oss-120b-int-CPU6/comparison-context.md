# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-10T07:57:41.069318+00:00`
- aggregate_generated_at_utc: `2026-08-10T07:57:06.609055+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 4

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_markov_report_itr30_2026_07_21_17_11` | complete | markov_report | - | - |
| R2 | `base_agent_folding_itr30_2026_07_28_01_09` | complete | folding | - | - |
| R3 | `base_agent_selective_retention_itr30_2026_07_24_17_17` | complete | selective_retention | - | - |
| R4 | `base_agent_selective_retention_itr30_2026_07_26_15_43` | complete | selective_retention | - | - |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 50.33 | 60.4 | 0 |
| R2 | folding | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 62.41 | 79.7 | 0 |
| R3 | selective_retention | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 57.70 | 247.3 | 0 |
| R4 | selective_retention | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 59.56 | 210.2 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.2340 | 1.0775 | 1.0413 | 35 | 0.8410 | 42 | 6.6263 | 20 | 15 | 0.960 | 0.860 | 0.760 | 0.660 | 0.060 | 0.040 |
| R2 | 30 | 50 | 1.3286 | 0.9782 | 1.0670 | 39 | 1.1235 | 33 | 6.1345 | 12 | 10 | 0.980 | 0.900 | 0.680 | 0.480 | 0.200 | 0.120 |
| R3 | 30 | 50 | 1.4767 | 1.1008 | 1.2559 | 36 | 1.1757 | 43 | 6.6565 | 18 | 13 | 0.980 | 0.960 | 0.820 | 0.660 | 0.240 | 0.160 |
| R4 | 30 | 50 | 1.3226 | 1.0288 | 1.0523 | 41 | 0.9576 | 37 | 5.9918 | 11 | 8 | 0.980 | 0.900 | 0.700 | 0.520 | 0.260 | 0.140 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 375 | 375 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 570 | 570 | 0 | 0 | 0 | 0 | 0 |
| R3 | no | no | no | 412 | 412 | 0 | 0 | 0 | 0 | 0 |
| R4 | no | no | no | 531 | 531 | 0 | 0 | 0 | 0 | 0 |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 |
| --- | --- | --- | --- | --- |
| 1 | 0.7522 | 0.6842 | 0.8671 | 0.6768 |
| 5 | 0.8049 | 0.8742 | 0.8938 | 0.4921 |
| 10 | 0.8006 | 0.9700 | 1.0491 | 0.9430 |
| 15 | 1.0046 | 1.0010 | 1.1253 | 0.9943 |
| 20 | 1.0481 | 1.0096 | 1.1735 | 1.0153 |
| 25 | 1.0648 | 1.0648 | 1.2158 | 1.0526 |
| 30 | 1.0413 | 1.0670 | 1.2559 | 1.0523 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 |
| --- | --- | --- | --- | --- |
| 1 | 0.080 | 0.080 | 0.080 | 0.040 |
| 5 | 0.280 | 0.300 | 0.380 | 0.240 |
| 10 | 0.440 | 0.420 | 0.560 | 0.420 |
| 15 | 0.580 | 0.440 | 0.600 | 0.500 |
| 20 | 0.600 | 0.460 | 0.620 | 0.500 |
| 25 | 0.620 | 0.480 | 0.620 | 0.520 |
| 30 | 0.660 | 0.480 | 0.660 | 0.520 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._
