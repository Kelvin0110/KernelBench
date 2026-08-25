# Evolving-agent cross-run comparison

- generated_at_utc: `2026-08-25T06:37:47.706271+00:00`
- aggregate_generated_at_utc: `2026-08-25T06:37:28.785987+00:00`
- runs_root: `/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b`
- baseline_timing_file: `/localhome/local-tianzheng/KernelBench/results/timing/NVIDIA_GH200x2_2nd/baseline_time_torch.json`
- speedup_aggregate_policy: `correct_only_exclude_hack`
- runs compared: 11
- analysis_rules: `scripts_integration/new_evolving_agent_analysis/ANALYSIS_RULES.md`
- required_checkpoints: iterations 10 and 30 with fast_p_best@0/1/2 and speedup_best geomean

## Runs

| id | run_name | status | context_mgmt | model | endpoint |
| --- | --- | --- | --- | --- | --- |
| R1 | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_22_20_31` | complete | compress_trigger | gpt-oss-120b | inference |
| R2 | `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_22_20_31` | complete | truncation | gpt-oss-120b | inference |
| R3 | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_22_20_37` | complete | folding | gpt-oss-120b | inference |
| R4 | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_22_20_24` | complete | truncation | gpt-oss-120b | inference |
| R5 | `base_agent_gpt_oss_120b_l2_itr30_GH200_2026_08_22_20_34` | complete | truncation | gpt-oss-120b | inference |
| R6 | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_22_20_25` | complete | markov_report | gpt-oss-120b | inference |
| R7 | `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_22_20_28` | complete | truncation | gpt-oss-120b | inference |
| R8 | `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_22_20_34` | complete | truncation | gpt-oss-120b | inference |
| R9 | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_22_20_28` | complete | selective_retention | gpt-oss-120b | inference |
| R10 | `smoke_truncation_p2_itr5_GH200_2026_08_22_18_20` | complete | truncation | gpt-oss-120b | inference |
| R11 | `smoke_truncation_p2_itr5_GH200v2_postfix_2026_08_22_19_41` | complete | truncation | gpt-oss-120b | inference |

## Run overview

| id | context_mgmt | itr | problems | completed | correct | correct_rate | rate_basis | wall_h | avg_min/problem | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 51.14 | 61.4 | 0 |
| R2 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 58.03 | 69.6 | 0 |
| R3 | folding | 30 | 50 | 50 | 47 | 0.940 | total_attempted | 56.72 | 68.1 | 0 |
| R4 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 50.78 | 60.9 | 0 |
| R5 | truncation | 30 | 50 | 50 | 49 | 0.980 | total_attempted | 55.33 | 66.4 | 0 |
| R6 | markov_report | 30 | 50 | 50 | 46 | 0.920 | total_attempted | 53.50 | 64.2 | 0 |
| R7 | truncation | 30 | 50 | 50 | 50 | 1.000 | total_attempted | 56.53 | 67.8 | 0 |
| R8 | truncation | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 55.97 | 67.2 | 0 |
| R9 | selective_retention | 30 | 50 | 50 | 48 | 0.960 | total_attempted | 56.43 | 67.7 | 0 |
| R10 | truncation | 5 | 2 | 2 | 0 | 0.000 | total_attempted | 0.23 | 6.8 | 0 |
| R11 | truncation | 5 | 2 | 2 | 1 | 0.500 | total_attempted | 0.23 | 6.8 | 0 |

## Required checkpoints: iterations 10 and 30

Every design variant is scored at the same two iteration budgets. `fast_p_best@0` is the correctness-like coverage (fraction of all problems whose running-best speedup is at least 0). `fast_p_best@1` and `@2` use the same full-problem denominator. `speedup_best` geomean uses every problem holding a non-hack running best, so its `n` tracks `total_correct`; read `n` next to it. Speedup is already relative to this series' native torch baseline — do not rescore one host onto another host's baseline to compare models.

| id | design | status | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | compress_trigger | complete | 50/50 | 0.920 | 0.480 | 0.140 | 1.1939 | 46 | 1.000 | 0.700 | 0.240 | 1.4368 | 50 |
| R2 | truncation+deletion | complete | 49/50 | 0.860 | 0.440 | 0.060 | 1.0255 | 43 | 0.980 | 0.620 | 0.100 | 1.2067 | 49 |
| R3 | folding | complete | 47/50 | 0.900 | 0.500 | 0.200 | 1.1700 | 45 | 0.940 | 0.660 | 0.260 | 1.5211 | 47 |
| R4 | truncation | complete | 48/50 | 0.880 | 0.420 | 0.120 | 1.0619 | 44 | 0.960 | 0.600 | 0.200 | 1.3885 | 48 |
| R5 | truncation | complete | 49/50 | 0.820 | 0.300 | 0.120 | 0.9918 | 41 | 0.980 | 0.600 | 0.200 | 1.3332 | 49 |
| R6 | markov_report | complete | 46/50 | 0.780 | 0.280 | 0.020 | 0.8141 | 39 | 0.920 | 0.440 | 0.100 | 1.0636 | 46 |
| R7 | truncation+merge@0.8 | complete | 50/50 | 0.900 | 0.400 | 0.080 | 1.0021 | 45 | 1.000 | 0.640 | 0.180 | 1.3000 | 50 |
| R8 | truncation+refine | complete | 48/50 | 0.760 | 0.480 | 0.100 | 1.1186 | 38 | 0.960 | 0.660 | 0.180 | 1.3522 | 48 |
| R9 | selective_retention | complete | 48/50 | 0.880 | 0.420 | 0.080 | 1.0143 | 44 | 0.960 | 0.660 | 0.180 | 1.2996 | 48 |
| R10 | truncation | complete | 0/2 | - | - | - | - | - | - | - | - | - | - |
| R11 | truncation | complete | 1/2 | - | - | - | - | - | - | - | - | - | - |

_`@0/@1/@2` are `fast_p_best` at thresholds 0, 1, and 2. Geomean is `speedup_best.geometric_mean`. Missing checkpoints render as `-`._

## Final-iteration performance (fast-p is `fast_p_best`)

| id | final_itr | problems | best_mean | best_median | best_geomean | best_n | cur_geomean | cur_n | best_speedup_overall | hack_itrs | problems_with_hack | fast_p@0.0 | fast_p@0.5 | fast_p@0.8 | fast_p@1.0 | fast_p@1.5 | fast_p@2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 50 | 1.9467 | 1.1448 | 1.4368 | 50 | 1.5581 | 38 | 6.4480 | 19 | 12 | 1.000 | 0.980 | 0.860 | 0.700 | 0.260 | 0.240 |
| R2 | 30 | 50 | 1.4721 | 1.0245 | 1.2067 | 49 | 1.2016 | 39 | 0.8974 | 10 | 7 | 0.980 | 0.980 | 0.820 | 0.620 | 0.180 | 0.100 |
| R3 | 30 | 50 | 2.0230 | 1.2027 | 1.5211 | 47 | 1.1606 | 34 | 3.2452 | 20 | 12 | 0.940 | 0.940 | 0.760 | 0.660 | 0.320 | 0.260 |
| R4 | 30 | 50 | 1.9505 | 1.0730 | 1.3885 | 48 | 1.5338 | 29 | 7.0546 | 19 | 15 | 0.960 | 0.940 | 0.740 | 0.600 | 0.260 | 0.200 |
| R5 | 30 | 50 | 1.8748 | 1.0952 | 1.3332 | 49 | 1.5253 | 29 | 6.5708 | 30 | 20 | 0.980 | 0.940 | 0.780 | 0.600 | 0.240 | 0.200 |
| R6 | 30 | 50 | 1.3695 | 1.0000 | 1.0636 | 46 | 1.0422 | 33 | 1.3795 | 17 | 14 | 0.920 | 0.880 | 0.700 | 0.440 | 0.140 | 0.100 |
| R7 | 30 | 50 | 1.7288 | 1.0399 | 1.3000 | 50 | 1.2170 | 32 | 5.5019 | 27 | 14 | 1.000 | 0.980 | 0.820 | 0.640 | 0.240 | 0.180 |
| R8 | 30 | 50 | 1.8053 | 1.0752 | 1.3522 | 48 | 1.3061 | 24 | 9.8249 | 23 | 13 | 0.960 | 0.960 | 0.820 | 0.660 | 0.240 | 0.180 |
| R9 | 30 | 50 | 1.6918 | 1.0569 | 1.2996 | 48 | 1.2599 | 35 | 6.4920 | 20 | 15 | 0.960 | 0.940 | 0.780 | 0.660 | 0.240 | 0.180 |
| R10 | 5 | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R11 | 5 | 2 | 0.9250 | 0.9250 | 0.9250 | 1 | 0.6932 | 1 | 0.9250 | 0 | 0 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |

_Speedup `best` aggregates use every problem with a non-hack running best (`best_correct`); `current` aggregates use `correct and not is_hack` at the last iteration. `best_n`/`cur_n` are how many of the `problems` actually entered those aggregates. Hack **iterations** never form a best, but a later hack does not revoke an earlier clean best, so `best_n` tracks `total_correct` - it is not reduced by `metrics_best.is_hack`, which is the run-level `run_had_hack` latch. fast-p keeps the full-problem denominator so failures are penalized._

## Skill governance

| id | deletion | merging | refinement | l1_entries | l1_active | merges | deleted | refined | deletion_events | sidecars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | no | no | no | 466 | 466 | 0 | 0 | 0 | 0 | 1 |
| R2 | yes | no | no | 559 | 36 | 0 | 523 | 0 | 523 | 3 |
| R3 | no | no | no | 588 | 588 | 0 | 0 | 0 | 0 | 1 |
| R4 | no | no | no | 600 | 600 | 0 | 0 | 0 | 0 | 1 |
| R5 | no | no | no | 662 | 662 | 0 | 0 | 0 | 0 | 1 |
| R6 | no | no | no | 380 | 380 | 0 | 0 | 0 | 0 | 1 |
| R7 | no | yes | no | 658 | 170 | 82 | 0 | 0 | 0 | 7 |
| R8 | no | no | yes | 692 | 588 | 0 | 0 | 112 | 0 | 2 |
| R9 | no | no | no | 550 | 550 | 0 | 0 | 0 | 0 | 1 |
| R10 | no | no | no | 5 | 5 | 0 | 0 | 0 | 0 | 1 |
| R11 | no | no | no | 4 | 4 | 0 | 0 | 0 | 0 | 1 |

## Deltas vs baseline run `base_agent_gpt_oss_120b_itr30_GH200_2026_08_22_20_24`

### `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_22_20_31`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 50 | +2 | +4.2% | better |
| correct_rate | 0.9600 | 1.0000 | +0.0400 | +4.2% | better |
| best_speedup_overall | 7.0546 | 6.4480 | -0.6067 | -8.6% | worse |
| speedup_best_mean | 1.9505 | 1.9467 | -0.0038 | -0.2% | worse |
| speedup_best_median | 1.0730 | 1.1448 | +0.0719 | +6.7% | better |
| speedup_best_geomean | 1.3885 | 1.4368 | +0.0483 | +3.5% | better |
| speedup_current_geomean | 1.5338 | 1.5581 | +0.0244 | +1.6% | better |
| hack_iteration_count | 19 | 19 | +0 | +0.0% | same |
| problems_with_hack | 15 | 12 | -3 | -20.0% | better |
| l1_entry_count | 600 | 466 | -134 | -22.3% | better |
| total_wall_time_hours | 50.785 | 51.144 | +0.359 | +0.7% | worse |
| avg_wall_time_min | 60.942 | 61.372 | +0.430 | +0.7% | worse |

### `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_22_20_31`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 49 | +1 | +2.1% | better |
| correct_rate | 0.9600 | 0.9800 | +0.0200 | +2.1% | better |
| best_speedup_overall | 7.0546 | 0.8974 | -6.1572 | -87.3% | worse |
| speedup_best_mean | 1.9505 | 1.4721 | -0.4784 | -24.5% | worse |
| speedup_best_median | 1.0730 | 1.0245 | -0.0484 | -4.5% | worse |
| speedup_best_geomean | 1.3885 | 1.2067 | -0.1818 | -13.1% | worse |
| speedup_current_geomean | 1.5338 | 1.2016 | -0.3321 | -21.7% | worse |
| hack_iteration_count | 19 | 10 | -9 | -47.4% | better |
| problems_with_hack | 15 | 7 | -8 | -53.3% | better |
| l1_entry_count | 600 | 559 | -41 | -6.8% | better |
| total_wall_time_hours | 50.785 | 58.028 | +7.243 | +14.3% | worse |
| avg_wall_time_min | 60.942 | 69.633 | +8.691 | +14.3% | worse |

### `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_22_20_37`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 47 | -1 | -2.1% | worse |
| correct_rate | 0.9600 | 0.9400 | -0.0200 | -2.1% | worse |
| best_speedup_overall | 7.0546 | 3.2452 | -3.8095 | -54.0% | worse |
| speedup_best_mean | 1.9505 | 2.0230 | +0.0725 | +3.7% | better |
| speedup_best_median | 1.0730 | 1.2027 | +0.1297 | +12.1% | better |
| speedup_best_geomean | 1.3885 | 1.5211 | +0.1326 | +9.6% | better |
| speedup_current_geomean | 1.5338 | 1.1606 | -0.3732 | -24.3% | worse |
| hack_iteration_count | 19 | 20 | +1 | +5.3% | worse |
| problems_with_hack | 15 | 12 | -3 | -20.0% | better |
| l1_entry_count | 600 | 588 | -12 | -2.0% | better |
| total_wall_time_hours | 50.785 | 56.716 | +5.931 | +11.7% | worse |
| avg_wall_time_min | 60.942 | 68.059 | +7.117 | +11.7% | worse |

### `base_agent_gpt_oss_120b_l2_itr30_GH200_2026_08_22_20_34`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 49 | +1 | +2.1% | better |
| correct_rate | 0.9600 | 0.9800 | +0.0200 | +2.1% | better |
| best_speedup_overall | 7.0546 | 6.5708 | -0.4838 | -6.9% | worse |
| speedup_best_mean | 1.9505 | 1.8748 | -0.0757 | -3.9% | worse |
| speedup_best_median | 1.0730 | 1.0952 | +0.0223 | +2.1% | better |
| speedup_best_geomean | 1.3885 | 1.3332 | -0.0553 | -4.0% | worse |
| speedup_current_geomean | 1.5338 | 1.5253 | -0.0085 | -0.6% | worse |
| hack_iteration_count | 19 | 30 | +11 | +57.9% | worse |
| problems_with_hack | 15 | 20 | +5 | +33.3% | worse |
| l1_entry_count | 600 | 662 | +62 | +10.3% | worse |
| total_wall_time_hours | 50.785 | 55.326 | +4.541 | +8.9% | worse |
| avg_wall_time_min | 60.942 | 66.391 | +5.449 | +8.9% | worse |

### `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_22_20_25`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 46 | -2 | -4.2% | worse |
| correct_rate | 0.9600 | 0.9200 | -0.0400 | -4.2% | worse |
| best_speedup_overall | 7.0546 | 1.3795 | -5.6751 | -80.4% | worse |
| speedup_best_mean | 1.9505 | 1.3695 | -0.5810 | -29.8% | worse |
| speedup_best_median | 1.0730 | 1.0000 | -0.0730 | -6.8% | worse |
| speedup_best_geomean | 1.3885 | 1.0636 | -0.3249 | -23.4% | worse |
| speedup_current_geomean | 1.5338 | 1.0422 | -0.4916 | -32.0% | worse |
| hack_iteration_count | 19 | 17 | -2 | -10.5% | better |
| problems_with_hack | 15 | 14 | -1 | -6.7% | better |
| l1_entry_count | 600 | 380 | -220 | -36.7% | better |
| total_wall_time_hours | 50.785 | 53.499 | +2.714 | +5.3% | worse |
| avg_wall_time_min | 60.942 | 64.199 | +3.257 | +5.3% | worse |

### `base_agent_gpt_oss_120b_merge_sim08_itr30_GH200_2026_08_22_20_28`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 50 | +2 | +4.2% | better |
| correct_rate | 0.9600 | 1.0000 | +0.0400 | +4.2% | better |
| best_speedup_overall | 7.0546 | 5.5019 | -1.5527 | -22.0% | worse |
| speedup_best_mean | 1.9505 | 1.7288 | -0.2217 | -11.4% | worse |
| speedup_best_median | 1.0730 | 1.0399 | -0.0331 | -3.1% | worse |
| speedup_best_geomean | 1.3885 | 1.3000 | -0.0885 | -6.4% | worse |
| speedup_current_geomean | 1.5338 | 1.2170 | -0.3168 | -20.7% | worse |
| hack_iteration_count | 19 | 27 | +8 | +42.1% | worse |
| problems_with_hack | 15 | 14 | -1 | -6.7% | better |
| l1_entry_count | 600 | 658 | +58 | +9.7% | worse |
| total_wall_time_hours | 50.785 | 56.529 | +5.745 | +11.3% | worse |
| avg_wall_time_min | 60.942 | 67.835 | +6.893 | +11.3% | worse |

### `base_agent_gpt_oss_120b_refinement_itr30_GH200_2026_08_22_20_34`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 48 | +0 | +0.0% | same |
| correct_rate | 0.9600 | 0.9600 | +0.0000 | +0.0% | same |
| best_speedup_overall | 7.0546 | 9.8249 | +2.7702 | +39.3% | better |
| speedup_best_mean | 1.9505 | 1.8053 | -0.1452 | -7.4% | worse |
| speedup_best_median | 1.0730 | 1.0752 | +0.0022 | +0.2% | better |
| speedup_best_geomean | 1.3885 | 1.3522 | -0.0363 | -2.6% | worse |
| speedup_current_geomean | 1.5338 | 1.3061 | -0.2276 | -14.8% | worse |
| hack_iteration_count | 19 | 23 | +4 | +21.1% | worse |
| problems_with_hack | 15 | 13 | -2 | -13.3% | better |
| l1_entry_count | 600 | 692 | +92 | +15.3% | worse |
| total_wall_time_hours | 50.785 | 55.975 | +5.190 | +10.2% | worse |
| avg_wall_time_min | 60.942 | 67.170 | +6.228 | +10.2% | worse |

### `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_22_20_28`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 48 | +0 | +0.0% | same |
| correct_rate | 0.9600 | 0.9600 | +0.0000 | +0.0% | same |
| best_speedup_overall | 7.0546 | 6.4920 | -0.5626 | -8.0% | worse |
| speedup_best_mean | 1.9505 | 1.6918 | -0.2587 | -13.3% | worse |
| speedup_best_median | 1.0730 | 1.0569 | -0.0161 | -1.5% | worse |
| speedup_best_geomean | 1.3885 | 1.2996 | -0.0889 | -6.4% | worse |
| speedup_current_geomean | 1.5338 | 1.2599 | -0.2739 | -17.9% | worse |
| hack_iteration_count | 19 | 20 | +1 | +5.3% | worse |
| problems_with_hack | 15 | 15 | +0 | +0.0% | same |
| l1_entry_count | 600 | 550 | -50 | -8.3% | better |
| total_wall_time_hours | 50.785 | 56.430 | +5.645 | +11.1% | worse |
| avg_wall_time_min | 60.942 | 67.716 | +6.774 | +11.1% | worse |

### `smoke_truncation_p2_itr5_GH200_2026_08_22_18_20`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 0 | -48 | -100.0% | worse |
| correct_rate | 0.9600 | 0.0000 | -0.9600 | -100.0% | worse |
| best_speedup_overall | 7.0546 | 0.0000 | -7.0546 | -100.0% | worse |
| speedup_best_mean | 1.9505 | 0.0000 | -1.9505 | -100.0% | worse |
| speedup_best_median | 1.0730 | 0.0000 | -1.0730 | -100.0% | worse |
| speedup_best_geomean | 1.3885 | 0.0000 | -1.3885 | -100.0% | worse |
| speedup_current_geomean | 1.5338 | 0.0000 | -1.5338 | -100.0% | worse |
| hack_iteration_count | 19 | 0 | -19 | -100.0% | better |
| problems_with_hack | 15 | 0 | -15 | -100.0% | better |
| l1_entry_count | 600 | 5 | -595 | -99.2% | better |
| total_wall_time_hours | 50.785 | 0.227 | -50.558 | -99.6% | better |
| avg_wall_time_min | 60.942 | 6.799 | -54.143 | -88.8% | better |

### `smoke_truncation_p2_itr5_GH200v2_postfix_2026_08_22_19_41`

| metric | baseline | run | delta | delta % | direction |
| --- | --- | --- | --- | --- | --- |
| total_correct | 48 | 1 | -47 | -97.9% | worse |
| correct_rate | 0.9600 | 0.5000 | -0.4600 | -47.9% | worse |
| best_speedup_overall | 7.0546 | 0.9250 | -6.1296 | -86.9% | worse |
| speedup_best_mean | 1.9505 | 0.9250 | -1.0255 | -52.6% | worse |
| speedup_best_median | 1.0730 | 0.9250 | -0.1480 | -13.8% | worse |
| speedup_best_geomean | 1.3885 | 0.9250 | -0.4635 | -33.4% | worse |
| speedup_current_geomean | 1.5338 | 0.6932 | -0.8406 | -54.8% | worse |
| hack_iteration_count | 19 | 0 | -19 | -100.0% | better |
| problems_with_hack | 15 | 0 | -15 | -100.0% | better |
| l1_entry_count | 600 | 4 | -596 | -99.3% | better |
| total_wall_time_hours | 50.785 | 0.227 | -50.558 | -99.6% | better |
| avg_wall_time_min | 60.942 | 6.800 | -54.142 | -88.8% | better |

## Per-iteration comparison (matched iterations)

### Best-speedup geometric mean vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) | delta(R7-R4) | delta(R8-R4) | delta(R9-R4) | delta(R10-R4) | delta(R11-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8031 | 0.7231 | 0.8107 | 0.7604 | 0.7417 | 0.5398 | 0.7020 | 0.9828 | 0.6940 | 0.0000 | 0.0000 | +0.0427 | -0.0372 | +0.0504 | -0.0187 | -0.2206 | -0.0584 | +0.2224 | -0.0664 | -0.7604 | -0.7604 |
| 5 | 0.9671 | 0.8315 | 1.0613 | 0.5868 | 0.8065 | 0.7767 | 0.8250 | 0.7496 | 0.8028 | 0.0000 | 0.9250 | +0.3802 | +0.2447 | +0.4745 | +0.2197 | +0.1899 | +0.2382 | +0.1628 | +0.2160 | -0.5868 | +0.3382 |

_Matched over iterations 1..5 (intersection of all compared runs, stride 5)._

### fast_p_best@1.0 vs iteration

| iteration | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | delta(R1-R4) | delta(R2-R4) | delta(R3-R4) | delta(R5-R4) | delta(R6-R4) | delta(R7-R4) | delta(R8-R4) | delta(R9-R4) | delta(R10-R4) | delta(R11-R4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.040 | 0.060 | 0.060 | 0.060 | 0.040 | 0.000 | 0.080 | 0.040 | 0.040 | 0.000 | 0.000 | -0.020 | +0.000 | +0.000 | -0.020 | -0.060 | +0.020 | -0.020 | -0.020 | -0.060 | -0.060 |
| 5 | 0.260 | 0.200 | 0.340 | 0.220 | 0.160 | 0.200 | 0.220 | 0.200 | 0.280 | 0.000 | 0.000 | +0.040 | -0.020 | +0.120 | -0.060 | -0.020 | +0.000 | -0.020 | +0.060 | -0.220 | -0.220 |

_Matched over iterations 1..5 (intersection of all compared runs, stride 5)._

### Aligned final-iteration deltas vs `base_agent_gpt_oss_120b_itr30_GH200_2026_08_22_20_24`

| id | metric | matched_iteration | baseline | run | delta | delta % |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | best_geomean | 30 | 1.3885 | 1.4368 | +0.0483 | +3.5% |
| R1 | fast_p_best@1.0 | 30 | 0.600 | 0.700 | +0.100 | +16.7% |
| R2 | best_geomean | 30 | 1.3885 | 1.2067 | -0.1818 | -13.1% |
| R2 | fast_p_best@1.0 | 30 | 0.600 | 0.620 | +0.020 | +3.3% |
| R3 | best_geomean | 30 | 1.3885 | 1.5211 | +0.1326 | +9.6% |
| R3 | fast_p_best@1.0 | 30 | 0.600 | 0.660 | +0.060 | +10.0% |
| R5 | best_geomean | 30 | 1.3885 | 1.3332 | -0.0553 | -4.0% |
| R5 | fast_p_best@1.0 | 30 | 0.600 | 0.600 | +0.000 | +0.0% |
| R6 | best_geomean | 30 | 1.3885 | 1.0636 | -0.3249 | -23.4% |
| R6 | fast_p_best@1.0 | 30 | 0.600 | 0.440 | -0.160 | -26.7% |
| R7 | best_geomean | 30 | 1.3885 | 1.3000 | -0.0885 | -6.4% |
| R7 | fast_p_best@1.0 | 30 | 0.600 | 0.640 | +0.040 | +6.7% |
| R8 | best_geomean | 30 | 1.3885 | 1.3522 | -0.0363 | -2.6% |
| R8 | fast_p_best@1.0 | 30 | 0.600 | 0.660 | +0.060 | +10.0% |
| R9 | best_geomean | 30 | 1.3885 | 1.2996 | -0.0889 | -6.4% |
| R9 | fast_p_best@1.0 | 30 | 0.600 | 0.660 | +0.060 | +10.0% |
| R10 | best_geomean | 5 | 0.5868 | 0.0000 | -0.5868 | -100.0% |
| R10 | fast_p_best@1.0 | 5 | 0.220 | 0.000 | -0.220 | -100.0% |
| R11 | best_geomean | 5 | 0.5868 | 0.9250 | +0.3382 | +57.6% |
| R11 | fast_p_best@1.0 | 5 | 0.220 | 0.000 | -0.220 | -100.0% |
