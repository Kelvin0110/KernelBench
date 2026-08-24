# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-24T14:54:05.042511+00:00`
- aggregate_generated_at_utc: `2026-08-24T14:53:45.578814+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 16
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

> **Warning:** these runs are still partial (in flight or aborted) and are reported at whatever iteration/problem count they have reached: `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_39`, `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_48`, `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_32`, `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_42`, `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_35`, `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_45`

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` | complete | compress_trigger | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_14_15_52` | complete | truncation | gpt-oss-120b | inference |
| R3 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_13_12_47` | complete | folding | gpt-oss-120b | inference |
| R4 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_39` | partial | - | - | - |
| R5 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_48` | partial | - | - | - |
| R6 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | complete | truncation | gpt-oss-120b | inference |
| R7 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_32` | partial | - | - | - |
| R8 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_42` | partial | - | - | - |
| R9 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | complete | markov_report | gpt-oss-120b | inference |
| R10 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_35` | partial | - | - | - |
| R11 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_45` | partial | - | - | - |
| R12 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_29` | complete | truncation | gpt-oss-120b | inference |
| R13 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_32` | complete | truncation | gpt-oss-120b | inference |
| R14 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_35` | complete | truncation | gpt-oss-120b | inference |
| R15 | `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_17_15_52` | complete | truncation | gpt-oss-120b | inference |
| R16 | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` | complete | selective_retention | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.23 | 77.1 | 0 |
| R2 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 66.91 | 80.3 | 0 |
| R3 | folding | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 66.59 | 79.9 | 0 |
| R4 | - | 30 | 33 | 32 | 31 | 0.969 | workspaces_finished | 50.39 | 94.5 | 0 |
| R5 | - | 30 | 34 | 33 | 28 | 0.848 | workspaces_finished | 50.65 | 92.1 | 0 |
| R6 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 74.18 | 88.0 | 0 |
| R7 | - | 30 | 36 | 35 | 35 | 1.000 | workspaces_finished | 50.96 | 87.4 | 0 |
| R8 | - | 30 | 35 | 34 | 30 | 0.882 | workspaces_finished | 50.38 | 88.9 | 0 |
| R9 | markov_report | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 71.43 | 83.5 | 0 |
| R10 | - | 30 | 35 | 34 | 33 | 0.971 | workspaces_finished | 50.09 | 88.4 | 0 |
| R11 | - | 30 | 34 | 33 | 31 | 0.939 | workspaces_finished | 50.63 | 92.1 | 0 |
| R12 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 64.91 | 77.9 | 0 |
| R13 | truncation | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 68.37 | 82.0 | 0 |
| R14 | truncation | 30 | 50 | 50 | 46 | 0.920 | total_attempted | 65.43 | 78.5 | 0 |
| R15 | truncation | 30 | 50 | 50 | 45 | 0.900 | total_attempted | 53.07 | 63.7 | 0 |
| R16 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 69.93 | 83.9 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | complete | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 | 42 | 0.960 | 0.400 | 0.140 | 0.7265 | 48 |
| R2 | truncation+deletion | complete | 48/50 | 0.840 | 0.400 | 0.100 | 0.8691 | 42 | 0.960 | 0.540 | 0.280 | 1.2312 | 48 |
| R3 | folding | complete | 47/50 | 0.840 | 0.320 | 0.120 | 0.6744 | 42 | 0.940 | 0.420 | 0.180 | 0.8938 | 47 |
| R4 | unknown | partial | 31/33 | 0.848 | 0.303 | 0.182 | 0.6124 | 28 | 0.970 | 0.424 | 0.212 | 0.7703 | 32 |
| R5 | unknown | partial | 28/34 | 0.706 | 0.382 | 0.059 | 0.6534 | 24 | 0.853 | 0.500 | 0.147 | 0.9080 | 29 |
| R6 | truncation | complete | 47/50 | 0.840 | 0.380 | 0.120 | 0.7379 | 42 | 0.940 | 0.460 | 0.180 | 0.9051 | 47 |
| R7 | unknown | partial | 35/36 | 0.667 | 0.250 | 0.083 | 0.5957 | 24 | 1.000 | 0.389 | 0.222 | 0.7754 | 36 |
| R8 | unknown | partial | 30/35 | 0.657 | 0.229 | 0.086 | 0.6343 | 23 | 0.886 | 0.429 | 0.286 | 0.9674 | 31 |
| R9 | markov_report | complete | 48/50 | 0.900 | 0.340 | 0.040 | 0.7145 | 45 | 0.960 | 0.460 | 0.140 | 0.9332 | 48 |
| R10 | unknown | partial | 33/35 | 0.829 | 0.229 | 0.000 | 0.5074 | 29 | 0.971 | 0.400 | 0.057 | 0.6180 | 34 |
| R11 | unknown | partial | 31/34 | 0.853 | 0.412 | 0.029 | 0.6068 | 29 | 0.941 | 0.500 | 0.088 | 0.8605 | 32 |
| R12 | truncation+merge@0.8 | complete | 48/50 | 0.740 | 0.260 | 0.120 | 0.7047 | 37 | 0.960 | 0.420 | 0.200 | 0.8379 | 48 |
| R13 | truncation+merge@0.8 | complete | 47/50 | 0.700 | 0.280 | 0.080 | 0.8282 | 35 | 0.940 | 0.400 | 0.140 | 0.8552 | 47 |
| R14 | truncation+merge@0.8 | complete | 46/50 | 0.800 | 0.300 | 0.140 | 0.9185 | 40 | 0.920 | 0.460 | 0.220 | 1.0919 | 46 |
| R15 | truncation+refine | complete | 45/50 | 0.780 | 0.300 | 0.100 | 0.7107 | 39 | 0.900 | 0.340 | 0.160 | 0.7971 | 45 |
| R16 | selective_retention | complete | 48/50 | 0.900 | 0.320 | 0.080 | 0.7657 | 45 | 0.960 | 0.460 | 0.160 | 0.9541 | 48 |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.2428 | 0.7582 | 0.7265 | 48 | 0.7491 | 29 | 2.9647 | 32 | 23 | 0.960 | 0.640 | 0.460 | 0.400 | 0.180 | 0.140 |
| R2 | 30 | 50 | 1.9729 | 1.0513 | 1.2312 | 48 | 1.4874 | 29 | 2.1050 | 19 | 12 | 0.960 | 0.820 | 0.720 | 0.540 | 0.300 | 0.280 |
| R3 | 30 | 50 | 1.5545 | 0.9218 | 0.8938 | 47 | 0.8176 | 26 | 6.3696 | 17 | 12 | 0.940 | 0.680 | 0.540 | 0.420 | 0.220 | 0.180 |
| R4 | 30 | 33 | 1.6483 | 0.7391 | 0.7703 | 32 | 0.6461 | 25 | - | 13 | 7 | 0.970 | 0.576 | 0.485 | 0.424 | 0.242 | 0.212 |
| R5 | 30 | 34 | 1.3948 | 1.0317 | 0.9080 | 29 | 0.9576 | 16 | - | 11 | 10 | 0.853 | 0.618 | 0.500 | 0.500 | 0.176 | 0.147 |
| R6 | 30 | 50 | 1.5437 | 0.9394 | 0.9051 | 47 | 0.8759 | 28 | 5.4461 | 16 | 11 | 0.940 | 0.720 | 0.500 | 0.460 | 0.240 | 0.180 |
| R7 | 30 | 36 | 1.4862 | 0.8335 | 0.7754 | 36 | 0.7187 | 22 | - | 13 | 12 | 1.000 | 0.667 | 0.528 | 0.389 | 0.222 | 0.222 |
| R8 | 30 | 35 | 2.0032 | 0.9917 | 0.9674 | 31 | 0.8534 | 21 | - | 23 | 13 | 0.886 | 0.600 | 0.514 | 0.429 | 0.286 | 0.286 |
| R9 | 30 | 50 | 1.3089 | 0.9640 | 0.9332 | 48 | 0.9225 | 39 | 6.2340 | 14 | 11 | 0.960 | 0.780 | 0.600 | 0.460 | 0.160 | 0.140 |
| R10 | 30 | 35 | 0.9490 | 0.8580 | 0.6180 | 34 | 0.5191 | 22 | - | 9 | 6 | 0.971 | 0.714 | 0.571 | 0.400 | 0.057 | 0.057 |
| R11 | 30 | 34 | 1.0935 | 1.0229 | 0.8605 | 32 | 0.7375 | 21 | - | 18 | 12 | 0.941 | 0.765 | 0.618 | 0.500 | 0.147 | 0.088 |
| R12 | 30 | 50 | 1.4754 | 0.7649 | 0.8379 | 48 | 0.6872 | 34 | 6.0788 | 21 | 16 | 0.960 | 0.700 | 0.460 | 0.420 | 0.240 | 0.200 |
| R13 | 30 | 50 | 1.3227 | 0.8354 | 0.8552 | 47 | 0.7349 | 27 | 9.0517 | 31 | 18 | 0.940 | 0.700 | 0.500 | 0.400 | 0.180 | 0.140 |
| R14 | 30 | 50 | 1.7942 | 0.9943 | 1.0919 | 46 | 1.0892 | 29 | 6.5111 | 23 | 14 | 0.920 | 0.780 | 0.540 | 0.460 | 0.240 | 0.220 |
| R15 | 30 | 50 | 1.5564 | 0.8403 | 0.7971 | 45 | 0.6148 | 32 | 7.2340 | 29 | 19 | 0.900 | 0.660 | 0.520 | 0.340 | 0.200 | 0.160 |
| R16 | 30 | 50 | 1.4787 | 0.9144 | 0.9541 | 48 | 0.9747 | 37 | 9.7594 | 17 | 12 | 0.960 | 0.740 | 0.540 | 0.460 | 0.220 | 0.160 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 435 | 435 | 0 | 0 | 0 | 0 | 0 |
| R2 | yes | no | no | 592 | 25 | 0 | 567 | 0 | 567 | 3 |
| R3 | no | no | no | 592 | 592 | 0 | 0 | 0 | 0 | 0 |
| R4 | - | - | - | 423 | 423 | 0 | 0 | 0 | 0 | 0 |
| R5 | - | - | - | 442 | 442 | 0 | 0 | 0 | 0 | 1 |
| R6 | no | no | no | 571 | 571 | 0 | 0 | 0 | 0 | 0 |
| R7 | - | - | - | 444 | 444 | 0 | 0 | 0 | 0 | 0 |
| R8 | - | - | - | 445 | 445 | 0 | 0 | 0 | 0 | 0 |
| R9 | no | no | no | 366 | 366 | 0 | 0 | 0 | 0 | 0 |
| R10 | - | - | - | 268 | 268 | 0 | 0 | 0 | 0 | 0 |
| R11 | - | - | - | 258 | 258 | 0 | 0 | 0 | 0 | 0 |
| R12 | no | yes | no | 703 | 384 | 56 | 0 | 0 | 0 | 7 |
| R13 | no | yes | no | 730 | 182 | 77 | 0 | 0 | 0 | 7 |
| R14 | no | yes | no | 681 | 313 | 52 | 0 | 0 | 0 | 7 |
| R15 | no | no | yes | 703 | 626 | 0 | 0 | 83 | 0 | 1 |
| R16 | no | no | no | 619 | 619 | 0 | 0 | 0 | 0 | 0 |

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

### `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_39`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 31 | -16 | -34.0% | worse |
| correct_rate | 0.9400 | 0.9688 | +0.0288 | +3.1% | better |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 1.6483 | +0.1046 | +6.8% | better |
| speedup_best_median | 0.9394 | 0.7391 | -0.2003 | -21.3% | worse |
| speedup_best_geomean | 0.9051 | 0.7703 | -0.1348 | -14.9% | worse |
| speedup_current_geomean | 0.8759 | 0.6461 | -0.2298 | -26.2% | worse |
| hack_iteration_count | 16 | 13 | -3 | -18.8% | better |
| problems_with_hack | 11 | 7 | -4 | -36.4% | better |
| l1_entry_count | 571 | 423 | -148 | -25.9% | better |
| total_wall_time_hours | 74.177 | 50.394 | -23.784 | -32.1% | better |
| avg_wall_time_min | 88.035 | 94.488 | +6.453 | +7.3% | worse |

### `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_48`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 28 | -19 | -40.4% | worse |
| correct_rate | 0.9400 | 0.8485 | -0.0915 | -9.7% | worse |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 1.3948 | -0.1489 | -9.6% | worse |
| speedup_best_median | 0.9394 | 1.0317 | +0.0923 | +9.8% | better |
| speedup_best_geomean | 0.9051 | 0.9080 | +0.0029 | +0.3% | better |
| speedup_current_geomean | 0.8759 | 0.9576 | +0.0816 | +9.3% | better |
| hack_iteration_count | 16 | 11 | -5 | -31.2% | better |
| problems_with_hack | 11 | 10 | -1 | -9.1% | better |
| l1_entry_count | 571 | 442 | -129 | -22.6% | better |
| total_wall_time_hours | 74.177 | 50.648 | -23.529 | -31.7% | better |
| avg_wall_time_min | 88.035 | 92.088 | +4.053 | +4.6% | worse |

### `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_32`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 35 | -12 | -25.5% | worse |
| correct_rate | 0.9400 | 1.0000 | +0.0600 | +6.4% | better |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 1.4862 | -0.0575 | -3.7% | worse |
| speedup_best_median | 0.9394 | 0.8335 | -0.1059 | -11.3% | worse |
| speedup_best_geomean | 0.9051 | 0.7754 | -0.1297 | -14.3% | worse |
| speedup_current_geomean | 0.8759 | 0.7187 | -0.1573 | -18.0% | worse |
| hack_iteration_count | 16 | 13 | -3 | -18.8% | better |
| problems_with_hack | 11 | 12 | +1 | +9.1% | worse |
| l1_entry_count | 571 | 444 | -127 | -22.2% | better |
| total_wall_time_hours | 74.177 | 50.964 | -23.213 | -31.3% | better |
| avg_wall_time_min | 88.035 | 87.368 | -0.667 | -0.8% | better |

### `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_42`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 30 | -17 | -36.2% | worse |
| correct_rate | 0.9400 | 0.8824 | -0.0576 | -6.1% | worse |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 2.0032 | +0.4595 | +29.8% | better |
| speedup_best_median | 0.9394 | 0.9917 | +0.0523 | +5.6% | better |
| speedup_best_geomean | 0.9051 | 0.9674 | +0.0623 | +6.9% | better |
| speedup_current_geomean | 0.8759 | 0.8534 | -0.0225 | -2.6% | worse |
| hack_iteration_count | 16 | 23 | +7 | +43.8% | worse |
| problems_with_hack | 11 | 13 | +2 | +18.2% | worse |
| l1_entry_count | 571 | 445 | -126 | -22.1% | better |
| total_wall_time_hours | 74.177 | 50.377 | -23.801 | -32.1% | better |
| avg_wall_time_min | 88.035 | 88.900 | +0.866 | +1.0% | worse |

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

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_35`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 33 | -14 | -29.8% | worse |
| correct_rate | 0.9400 | 0.9706 | +0.0306 | +3.3% | better |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 0.9490 | -0.5947 | -38.5% | worse |
| speedup_best_median | 0.9394 | 0.8580 | -0.0814 | -8.7% | worse |
| speedup_best_geomean | 0.9051 | 0.6180 | -0.2871 | -31.7% | worse |
| speedup_current_geomean | 0.8759 | 0.5191 | -0.3568 | -40.7% | worse |
| hack_iteration_count | 16 | 9 | -7 | -43.8% | better |
| problems_with_hack | 11 | 6 | -5 | -45.5% | better |
| l1_entry_count | 571 | 268 | -303 | -53.1% | better |
| total_wall_time_hours | 74.177 | 50.086 | -24.092 | -32.5% | better |
| avg_wall_time_min | 88.035 | 88.387 | +0.352 | +0.4% | worse |

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_45`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 31 | -16 | -34.0% | worse |
| correct_rate | 0.9400 | 0.9394 | -0.0006 | -0.1% | worse |
| best_speedup_overall | 5.4461 | - | - | - | - |
| speedup_best_mean | 1.5437 | 1.0935 | -0.4501 | -29.2% | worse |
| speedup_best_median | 0.9394 | 1.0229 | +0.0835 | +8.9% | better |
| speedup_best_geomean | 0.9051 | 0.8605 | -0.0446 | -4.9% | worse |
| speedup_current_geomean | 0.8759 | 0.7375 | -0.1384 | -15.8% | worse |
| hack_iteration_count | 16 | 18 | +2 | +12.5% | worse |
| problems_with_hack | 11 | 12 | +1 | +9.1% | worse |
| l1_entry_count | 571 | 258 | -313 | -54.8% | better |
| total_wall_time_hours | 74.177 | 50.630 | -23.547 | -31.7% | better |
| avg_wall_time_min | 88.035 | 92.055 | +4.021 | +4.6% | worse |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_29`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 48 | +1 | +2.1% | better |
| correct_rate | 0.9400 | 0.9600 | +0.0200 | +2.1% | better |
| best_speedup_overall | 5.4461 | 6.0788 | +0.6327 | +11.6% | better |
| speedup_best_mean | 1.5437 | 1.4754 | -0.0682 | -4.4% | worse |
| speedup_best_median | 0.9394 | 0.7649 | -0.1745 | -18.6% | worse |
| speedup_best_geomean | 0.9051 | 0.8379 | -0.0672 | -7.4% | worse |
| speedup_current_geomean | 0.8759 | 0.6872 | -0.1888 | -21.5% | worse |
| hack_iteration_count | 16 | 21 | +5 | +31.2% | worse |
| problems_with_hack | 11 | 16 | +5 | +45.5% | worse |
| l1_entry_count | 571 | 703 | +132 | +23.1% | worse |
| total_wall_time_hours | 74.177 | 64.907 | -9.270 | -12.5% | better |
| avg_wall_time_min | 88.035 | 77.889 | -10.146 | -11.5% | better |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_32`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 47 | +0 | +0.0% | same |
| correct_rate | 0.9400 | 0.9400 | +0.0000 | +0.0% | same |
| best_speedup_overall | 5.4461 | 9.0517 | +3.6056 | +66.2% | better |
| speedup_best_mean | 1.5437 | 1.3227 | -0.2210 | -14.3% | worse |
| speedup_best_median | 0.9394 | 0.8354 | -0.1040 | -11.1% | worse |
| speedup_best_geomean | 0.9051 | 0.8552 | -0.0499 | -5.5% | worse |
| speedup_current_geomean | 0.8759 | 0.7349 | -0.1410 | -16.1% | worse |
| hack_iteration_count | 16 | 31 | +15 | +93.8% | worse |
| problems_with_hack | 11 | 18 | +7 | +63.6% | worse |
| l1_entry_count | 571 | 730 | +159 | +27.8% | worse |
| total_wall_time_hours | 74.177 | 68.375 | -5.803 | -7.8% | better |
| avg_wall_time_min | 88.035 | 82.050 | -5.985 | -6.8% | better |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_19_17_35`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 46 | -1 | -2.1% | worse |
| correct_rate | 0.9400 | 0.9200 | -0.0200 | -2.1% | worse |
| best_speedup_overall | 5.4461 | 6.5111 | +1.0650 | +19.6% | better |
| speedup_best_mean | 1.5437 | 1.7942 | +0.2506 | +16.2% | better |
| speedup_best_median | 0.9394 | 0.9943 | +0.0549 | +5.8% | better |
| speedup_best_geomean | 0.9051 | 1.0919 | +0.1868 | +20.6% | better |
| speedup_current_geomean | 0.8759 | 1.0892 | +0.2133 | +24.3% | better |
| hack_iteration_count | 16 | 23 | +7 | +43.8% | worse |
| problems_with_hack | 11 | 14 | +3 | +27.3% | worse |
| l1_entry_count | 571 | 681 | +110 | +19.3% | worse |
| total_wall_time_hours | 74.177 | 65.433 | -8.745 | -11.8% | better |
| avg_wall_time_min | 88.035 | 78.519 | -9.516 | -10.8% | better |

### `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_17_15_52`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 47 | 45 | -2 | -4.3% | worse |
| correct_rate | 0.9400 | 0.9000 | -0.0400 | -4.3% | worse |
| best_speedup_overall | 5.4461 | 7.2340 | +1.7879 | +32.8% | better |
| speedup_best_mean | 1.5437 | 1.5564 | +0.0127 | +0.8% | better |
| speedup_best_median | 0.9394 | 0.8403 | -0.0991 | -10.5% | worse |
| speedup_best_geomean | 0.9051 | 0.7971 | -0.1080 | -11.9% | worse |
| speedup_current_geomean | 0.8759 | 0.6148 | -0.2612 | -29.8% | worse |
| hack_iteration_count | 16 | 29 | +13 | +81.2% | worse |
| problems_with_hack | 11 | 19 | +8 | +72.7% | worse |
| l1_entry_count | 571 | 703 | +132 | +23.1% | worse |
| total_wall_time_hours | 74.177 | 53.070 | -21.108 | -28.5% | better |
| avg_wall_time_min | 88.035 | 63.684 | -24.351 | -27.7% | better |

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

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | delta(R1-R6) | delta(R2-R6) | delta(R3-R6) | delta(R4-R6) | delta(R5-R6) | delta(R7-R6) | delta(R8-R6) | delta(R9-R6) | delta(R10-R6) | delta(R11-R6) | delta(R12-R6) | delta(R13-R6) | delta(R14-R6) | delta(R15-R6) | delta(R16-R6) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3999 | 0.6971 | 0.3440 | 0.4542 | 0.2213 | 0.4816 | 0.3722 | 0.4990 | 0.6979 | 0.2177 | 0.4144 | 0.3090 | 0.4632 | 0.5761 | 0.5563 | 0.4330 | -0.0817 | +0.2155 | -0.1376 | -0.0274 | -0.2603 | -0.1094 | +0.0174 | +0.2163 | -0.2639 | -0.0672 | -0.1726 | -0.0184 | +0.0945 | +0.0746 | -0.0486 |
| 5 | 0.5168 | 0.7275 | 0.5638 | 0.5464 | 0.6100 | 0.5552 | 0.6330 | 0.3904 | 0.5892 | 0.4112 | 0.5136 | 0.5640 | 0.7862 | 0.7658 | 0.6508 | 0.5048 | -0.0384 | +0.1723 | +0.0086 | -0.0088 | +0.0548 | +0.0778 | -0.1648 | +0.0340 | -0.1440 | -0.0416 | +0.0088 | +0.2310 | +0.2106 | +0.0956 | -0.0504 |
| 10 | 0.5624 | 0.8691 | 0.6744 | 0.6124 | 0.6534 | 0.7379 | 0.5957 | 0.6343 | 0.7145 | 0.5074 | 0.6068 | 0.7047 | 0.8282 | 0.9185 | 0.7107 | 0.7657 | -0.1755 | +0.1311 | -0.0635 | -0.1255 | -0.0846 | -0.1422 | -0.1036 | -0.0234 | -0.2305 | -0.1311 | -0.0332 | +0.0903 | +0.1806 | -0.0273 | +0.0278 |
| 15 | 0.5445 | 0.9468 | 0.7570 | 0.6841 | 0.7398 | 0.7728 | 0.5344 | 0.6386 | 0.8178 | 0.5908 | 0.5878 | 0.7308 | 0.7802 | 0.8921 | 0.7666 | 0.7811 | -0.2283 | +0.1740 | -0.0158 | -0.0887 | -0.0330 | -0.2384 | -0.1341 | +0.0450 | -0.1819 | -0.1850 | -0.0420 | +0.0074 | +0.1193 | -0.0061 | +0.0083 |
| 20 | 0.6344 | 0.9518 | 0.8256 | 0.6917 | 0.8388 | 0.8292 | 0.6307 | 0.7993 | 0.8441 | 0.5233 | 0.7132 | 0.7736 | 0.7744 | 0.9611 | 0.7637 | 0.8710 | -0.1948 | +0.1226 | -0.0036 | -0.1375 | +0.0096 | -0.1985 | -0.0299 | +0.0149 | -0.3059 | -0.1160 | -0.0556 | -0.0548 | +0.1319 | -0.0656 | +0.0417 |
| 25 | 0.6856 | 1.1073 | 0.8709 | 0.7787 | 0.8864 | 0.8806 | 0.7104 | 0.8080 | 0.8722 | 0.5679 | 0.7709 | 0.7880 | 0.8797 | 1.0129 | 0.7624 | 0.9440 | -0.1951 | +0.2266 | -0.0097 | -0.1019 | +0.0058 | -0.1702 | -0.0726 | -0.0084 | -0.3127 | -0.1097 | -0.0926 | -0.0010 | +0.1323 | -0.1183 | +0.0634 |
| 30 | 0.7265 | 1.2312 | 0.8938 | 0.7703 | 0.9080 | 0.9051 | 0.7754 | 0.9674 | 0.9332 | 0.6180 | 0.8605 | 0.8379 | 0.8552 | 1.0919 | 0.7971 | 0.9541 | -0.1786 | +0.3261 | -0.0113 | -0.1348 | +0.0029 | -0.1297 | +0.0623 | +0.0281 | -0.2871 | -0.0446 | -0.0672 | -0.0499 | +0.1868 | -0.1080 | +0.0490 |

_Matched over iterations 1..30 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | delta(R1-R6) | delta(R2-R6) | delta(R3-R6) | delta(R4-R6) | delta(R5-R6) | delta(R7-R6) | delta(R8-R6) | delta(R9-R6) | delta(R10-R6) | delta(R11-R6) | delta(R12-R6) | delta(R13-R6) | delta(R14-R6) | delta(R15-R6) | delta(R16-R6) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.080 | 0.120 | 0.060 | 0.061 | 0.029 | 0.080 | 0.056 | 0.000 | 0.040 | 0.000 | 0.000 | 0.000 | 0.020 | 0.060 | 0.040 | 0.040 | +0.000 | +0.040 | -0.020 | -0.019 | -0.051 | -0.024 | -0.080 | -0.040 | -0.080 | -0.080 | -0.080 | -0.060 | -0.020 | -0.040 | -0.040 |
| 5 | 0.200 | 0.300 | 0.240 | 0.152 | 0.294 | 0.260 | 0.139 | 0.086 | 0.280 | 0.114 | 0.206 | 0.100 | 0.200 | 0.200 | 0.240 | 0.200 | -0.060 | +0.040 | -0.020 | -0.108 | +0.034 | -0.121 | -0.174 | +0.020 | -0.146 | -0.054 | -0.160 | -0.060 | -0.060 | -0.020 | -0.060 |
| 10 | 0.240 | 0.400 | 0.320 | 0.303 | 0.382 | 0.380 | 0.250 | 0.229 | 0.340 | 0.229 | 0.412 | 0.260 | 0.280 | 0.300 | 0.300 | 0.320 | -0.140 | +0.020 | -0.060 | -0.077 | +0.002 | -0.130 | -0.151 | -0.040 | -0.151 | +0.032 | -0.120 | -0.100 | -0.080 | -0.080 | -0.060 |
| 15 | 0.280 | 0.400 | 0.380 | 0.333 | 0.471 | 0.400 | 0.278 | 0.257 | 0.380 | 0.343 | 0.412 | 0.300 | 0.340 | 0.380 | 0.320 | 0.380 | -0.120 | +0.000 | -0.020 | -0.067 | +0.071 | -0.122 | -0.143 | -0.020 | -0.057 | +0.012 | -0.100 | -0.060 | -0.020 | -0.080 | -0.020 |
| 20 | 0.360 | 0.420 | 0.400 | 0.333 | 0.500 | 0.420 | 0.333 | 0.314 | 0.400 | 0.343 | 0.471 | 0.320 | 0.340 | 0.380 | 0.340 | 0.420 | -0.060 | +0.000 | -0.020 | -0.087 | +0.080 | -0.087 | -0.106 | -0.020 | -0.077 | +0.051 | -0.100 | -0.080 | -0.040 | -0.080 | +0.000 |
| 25 | 0.380 | 0.480 | 0.420 | 0.394 | 0.500 | 0.420 | 0.333 | 0.343 | 0.420 | 0.371 | 0.471 | 0.380 | 0.400 | 0.400 | 0.340 | 0.460 | -0.040 | +0.060 | +0.000 | -0.026 | +0.080 | -0.087 | -0.077 | +0.000 | -0.049 | +0.051 | -0.040 | -0.020 | -0.020 | -0.080 | +0.040 |
| 30 | 0.400 | 0.540 | 0.420 | 0.424 | 0.500 | 0.460 | 0.389 | 0.429 | 0.460 | 0.400 | 0.500 | 0.420 | 0.400 | 0.460 | 0.340 | 0.460 | -0.060 | +0.080 | -0.040 | -0.036 | +0.040 | -0.071 | -0.031 | +0.000 | -0.060 | +0.040 | -0.040 | -0.060 | +0.000 | -0.120 | +0.000 |

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
| R4 | best_geomean | 30 | 0.9051 | 0.7703 | -0.1348 | -14.9% |
| R4 | fast_p_best@1.0 | 30 | 0.460 | 0.424 | -0.036 | -7.8% |
| R5 | best_geomean | 30 | 0.9051 | 0.9080 | +0.0029 | +0.3% |
| R5 | fast_p_best@1.0 | 30 | 0.460 | 0.500 | +0.040 | +8.7% |
| R7 | best_geomean | 30 | 0.9051 | 0.7754 | -0.1297 | -14.3% |
| R7 | fast_p_best@1.0 | 30 | 0.460 | 0.389 | -0.071 | -15.5% |
| R8 | best_geomean | 30 | 0.9051 | 0.9674 | +0.0623 | +6.9% |
| R8 | fast_p_best@1.0 | 30 | 0.460 | 0.429 | -0.031 | -6.8% |
| R9 | best_geomean | 30 | 0.9051 | 0.9332 | +0.0281 | +3.1% |
| R9 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R10 | best_geomean | 30 | 0.9051 | 0.6180 | -0.2871 | -31.7% |
| R10 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |
| R11 | best_geomean | 30 | 0.9051 | 0.8605 | -0.0446 | -4.9% |
| R11 | fast_p_best@1.0 | 30 | 0.460 | 0.500 | +0.040 | +8.7% |
| R12 | best_geomean | 30 | 0.9051 | 0.8379 | -0.0672 | -7.4% |
| R12 | fast_p_best@1.0 | 30 | 0.460 | 0.420 | -0.040 | -8.7% |
| R13 | best_geomean | 30 | 0.9051 | 0.8552 | -0.0499 | -5.5% |
| R13 | fast_p_best@1.0 | 30 | 0.460 | 0.400 | -0.060 | -13.0% |
| R14 | best_geomean | 30 | 0.9051 | 1.0919 | +0.1868 | +20.6% |
| R14 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |
| R15 | best_geomean | 30 | 0.9051 | 0.7971 | -0.1080 | -11.9% |
| R15 | fast_p_best@1.0 | 30 | 0.460 | 0.340 | -0.120 | -26.1% |
| R16 | best_geomean | 30 | 0.9051 | 0.9541 | +0.0490 | +5.4% |
| R16 | fast_p_best@1.0 | 30 | 0.460 | 0.460 | +0.000 | +0.0% |

## Notes

- `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_39`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_39`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 32 finished problems, not the full batch)
- `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_48`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_20_16_48`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 33 finished problems, not the full batch)
- `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_32`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_32`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 35 finished problems, not the full batch)
- `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_42`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_itr30_GH200_2026_08_20_16_42`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 34 finished problems, not the full batch)
- `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_35`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_35`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 34 finished problems, not the full batch)
- `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_45`: run_summary.json missing (run still in progress or aborted)
- `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_20_16_45`: total_correct derived from workspaces/*/run_finished.json (correct_rate is over 33 finished problems, not the full batch)
