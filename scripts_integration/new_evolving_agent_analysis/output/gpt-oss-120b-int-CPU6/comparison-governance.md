# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-10T07:57:42.337907+00:00`
- aggregate_generated_at_utc: `2026-08-10T07:57:06.609055+00:00`
- runs_root: `/home/kwtamai/KernelBench/runs_evolving`
- baseline_timing_file: `/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 6

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53` | complete | - | - | - |
| R2 | `base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52` | complete | - | - | - |
| R3 | `base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13` | complete | - | - | - |
| R4 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45` | complete | truncation | - | - |
| R5 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48` | complete | truncation | - | - |
| R6 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24` | complete | truncation | - | - |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | - | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 52.21 | 62.7 | 0 |
| R2 | - | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 51.91 | 62.3 | 0 |
| R3 | - | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 57.18 | 68.6 | 0 |
| R4 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 59.45 | 713.4 | 0 |
| R5 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 59.69 | 1193.7 | 0 |
| R6 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 60.69 | 158.3 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.3007 | 1.1667 | 1.2242 | 33 | 1.2265 | 35 | 9.3293 | 30 | 17 | 0.980 | 0.960 | 0.900 | 0.680 | 0.140 | 0.120 |
| R2 | 30 | 50 | 1.2530 | 1.0851 | 1.1001 | 41 | 1.0339 | 25 | 5.5513 | 17 | 8 | 0.960 | 0.900 | 0.780 | 0.580 | 0.100 | 0.080 |
| R3 | 30 | 50 | 1.3219 | 1.0653 | 1.1324 | 42 | 1.1372 | 35 | 6.6263 | 13 | 8 | 1.000 | 0.960 | 0.840 | 0.660 | 0.220 | 0.120 |
| R4 | 30 | 50 | 1.4402 | 1.1584 | 1.2626 | 38 | 1.2990 | 38 | 6.0833 | 14 | 11 | 0.980 | 0.980 | 0.860 | 0.740 | 0.200 | 0.120 |
| R5 | 30 | 50 | 1.3776 | 1.1044 | 1.2373 | 42 | 1.3521 | 41 | 8.0526 | 11 | 8 | 0.980 | 0.980 | 0.880 | 0.680 | 0.200 | 0.120 |
| R6 | 30 | 50 | 1.3453 | 1.0850 | 1.1987 | 40 | 1.1130 | 34 | 3.8651 | 14 | 10 | 1.000 | 0.980 | 0.840 | 0.680 | 0.240 | 0.160 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | yes | - | 591 | 241 | 75 | 0 | 0 | 0 | 7 |
| R2 | no | yes | - | 554 | 448 | 39 | 0 | 0 | 0 | 7 |
| R3 | yes | no | - | 516 | 28 | 0 | 435 | 57 | 455 | 4 |
| R4 | yes | yes | no | 509 | 28 | 6 | 466 | 0 | 607 | 8 |
| R5 | yes | yes | yes | 602 | 18 | 27 | 452 | 64 | 559 | 9 |
| R6 | yes | yes | yes | 589 | 30 | 7 | 476 | 72 | 596 | 9 |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9395 | 0.7705 | 0.8766 | 0.7108 | 0.7904 | 0.9089 |
| 5 | 0.8446 | 0.8197 | 0.8066 | 1.0097 | 0.9249 | 0.8389 |
| 10 | 1.0023 | 0.9746 | 1.0053 | 1.0532 | 0.9477 | 0.9460 |
| 15 | 1.0393 | 0.9847 | 1.0586 | 1.0428 | 1.0402 | 1.1218 |
| 20 | 1.1083 | 1.0105 | 1.0844 | 1.1182 | 1.1749 | 1.1386 |
| 25 | 1.1714 | 1.0451 | 1.0856 | 1.1621 | 1.1757 | 1.1824 |
| 30 | 1.2242 | 1.1001 | 1.1324 | 1.2626 | 1.2373 | 1.1987 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.160 | 0.100 | 0.060 | 0.080 | 0.060 | 0.160 |
| 5 | 0.440 | 0.380 | 0.480 | 0.460 | 0.400 | 0.440 |
| 10 | 0.580 | 0.480 | 0.540 | 0.560 | 0.500 | 0.520 |
| 15 | 0.580 | 0.520 | 0.620 | 0.600 | 0.580 | 0.560 |
| 20 | 0.600 | 0.580 | 0.620 | 0.640 | 0.600 | 0.620 |
| 25 | 0.660 | 0.580 | 0.660 | 0.660 | 0.640 | 0.660 |
| 30 | 0.680 | 0.580 | 0.660 | 0.740 | 0.680 | 0.680 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._
