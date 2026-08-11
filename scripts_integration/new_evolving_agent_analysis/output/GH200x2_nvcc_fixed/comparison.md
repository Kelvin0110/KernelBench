# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-11T06:38:08.159359+00:00`
- aggregate_generated_at_utc: `2026-08-11T06:38:08.152856+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 2

> **Warning:** these runs are still partial (in flight or aborted) and are reported at whatever iteration/problem count they have reached: `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58`

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | partial | truncation | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | complete | markov_report | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 72.22 | 86.7 | 0 |
| R2 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 71.43 | 2143.0 | 0 |

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.3942 | 1.0013 | 0.8853 | 37 | 0.8952 | 27 | 5.4461 | 16 | 11 | 0.920 | 0.700 | 0.500 | 0.460 | 0.240 | 0.180 |
| R2 | 30 | 50 | 1.4358 | 1.0046 | 0.9830 | 38 | 0.9225 | 39 | 6.2340 | 14 | 11 | 0.960 | 0.780 | 0.600 | 0.460 | 0.160 | 0.140 |

_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. fast-p keeps the full-problem denominator so failures are penalized, and `fast_p_best` does **not** drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests were hack-flagged._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 562 | 562 | 0 | 0 | 0 | 0 | 0 |
| R2 | no | no | no | 366 | 366 | 0 | 0 | 0 | 0 | 0 |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 |
| --- | --- | --- |
| 1 | 0.4816 | 0.6979 |
| 5 | 0.5510 | 0.5827 |
| 10 | 0.7433 | 0.7373 |
| 15 | 0.7811 | 0.8426 |
| 20 | 0.8485 | 0.8685 |
| 25 | 0.8104 | 0.9026 |
| 30 | 0.8853 | 0.9830 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 |
| --- | --- | --- |
| 1 | 0.080 | 0.040 |
| 5 | 0.260 | 0.280 |
| 10 | 0.380 | 0.340 |
| 15 | 0.400 | 0.380 |
| 20 | 0.420 | 0.400 |
| 25 | 0.420 | 0.420 |
| 30 | 0.460 | 0.460 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._
