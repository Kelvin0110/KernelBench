# GPT-OSS-120B old NVIDIA integrate runs on CPU6/A6000

## Executive summary

This is an evidence-first, descriptive report for exactly ten completed top-level runs under `runs_evolving/`. Every run attempted and finished the same 50-problem subset for 30 iterations per problem (500 finished workspaces and nominally 15,000 iteration slots in total), reported `cuda_available=true`, and recorded `NVIDIA RTX A6000` / `SONG_CPU6_A6000x4`. Speedups use the explicit CPU6 baseline:

`results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`

The cohort contains two different experiment families:

- **Context management (4):** Markov report, folding, and two distinct selective-retention campaigns.
- **Governance/configuration (6):** merge-only at similarity 0.7 and 0.8; deletion plus refinement; deletion plus merge at 0.8; and deletion plus merge plus refinement at 0.7 and 0.8.

The strongest descriptive signals are:

1. All ten runs completed, with 48–50 of 50 final problems correct and no suspicious run-summary speedups. Deletion+refinement (G3) and deletion+merge+refinement 0.8 (G6) each reached 50/50.
2. At the headline thresholds, G3/G6 reached `fast_p_best@0.0=1.00`, deletion+merge 0.8 (G4) reached `fast_p_best@1.0=.74`, and selective campaign 1 (C3) plus G6 reached `fast_p_best@2.0=.16`. These are observations, not a ranking or treatment effect.
3. Current-state retention varied materially. At iteration 30, current geomean ranged from 0.8410 (Markov, n=42) to 1.3521 (triple 0.7, n=41). A high best profile with a weaker current profile means the run found good kernels but did not consistently finish on them.
4. Context mechanisms had large service-cost differences. Markov recorded 55.8M tokens / 6,369 calls; folding 210.9M / 7,779; and selective campaigns 172.8M / 5,781 and 198.0M / 5,830. Folding's summaries/preflight and selective retention's large retained prompts are visible in raw chat records.
5. Governance executed, rather than merely being configured. Merge-only 0.7 accepted 75 merges from 208 events and compressed 591 L1 entries to 241 active; merge-only 0.8 accepted 39/119 and retained 448/554. Deletion-bearing runs ended with only 18–30 active entries from 509–602 total entries while recording 435–476 deleted entries and, where enabled, 57–72 refined entries.

There is **no clean old-integrate truncation control at 50 problems × 30 iterations**. Consequently this report cannot estimate causal effects of Markov, folding, selective retention, merging thresholds, deletion, or refinement. The two selective runs are separately resumed campaigns, not automatically independent replicates. Launch dates, resume boundaries, accumulated L1 state, endpoint failures, and stochastic search differ.

## Scope and endpoint provenance

This report applies the authoritative directory-layout convention supplied for this analysis: **qualifying runs directly under `runs_evolving/`, rather than inside endpoint-specific subfolders, use the old NVIDIA integrate endpoint**. All ten runs satisfy that rule and are classified as old-integrate GPT-OSS-120B runs.

The older `run_summary.json` schema predates explicit `nvidia_endpoint` and model fields, so the generated aggregate correctly leaves those fields null; that schema omission does not make the endpoint classification unresolved. Independent artifacts corroborate the directory rule:

- `scripts_integration/new_evolving_agent/RUN_WITH_UV.md` requires `NVIDIA_API_KEY` for the default client.
- `RUN_WITH_UV_CONTEXT.md` describes these top-level context runs and separately directs inference-API users to the newer inference runbook.
- Sampled raw `chat_history.jsonl` records identify the chat model as `openai/gpt-oss-120b`; other old campaign records use `nvdev/openai/gpt-oss-120b`.
- The old client/runbooks route that configuration through NVIDIA's integrate service.

Accordingly, endpoint classification is fixed by the supplied scope rule and corroborated by the runbooks/chat records. This report still does not backfill fields absent from old summaries.

The `output/GH200x2/` material is unrelated invalidated inference data. None of its void metrics are quoted or reused here.

## Exact run set

### Context-management runs

| ID | Exact run name | Mode | Resume evidence |
|---|---|---|---|
| C1 | `base_agent_markov_report_itr30_2026_07_21_17_11` | markov_report | final summary says non-resume; 50 problems timed in that session |
| C2 | `base_agent_folding_itr30_2026_07_28_01_09` | folding | runbook resumes at problem 4; 47 problems timed in final session |
| C3 | `base_agent_selective_retention_itr30_2026_07_24_17_17` | selective_retention | separate resumed campaign; 14 problems timed in final session |
| C4 | `base_agent_selective_retention_itr30_2026_07_26_15_43` | selective_retention | separate resumed campaign; 17 problems timed in final session |

### Governance/configuration runs

| ID | Exact run name | Deletion | Merge | Refinement | Similarity / resume evidence |
|---|---|---:|---:|---:|---|
| G1 | `base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53` | no | yes | no | 0.7; 50 problems timed |
| G2 | `base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52` | no | yes | no | 0.8; 50 problems timed |
| G3 | `base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13` | yes | no | yes | 50 problems timed |
| G4 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45` | yes | yes | no | 0.8; resumed at problem 46; 5 timed in final session |
| G5 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48` | yes | yes | yes | 0.7; resumed at problems 48–50; 3 timed in final session |
| G6 | `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24` | yes | yes | yes | 0.8; resumed at problem 28; 23 timed in final session |

## Headline comparison

Correctness and `fast_p_best` at thresholds 0.0, 1.0, and 2.0 are the primary comparison columns, matching the original threshold style. Current fast-p is shown beside each best-state value to expose how much performance the final iteration retained. Every fast-p value uses the full 50-problem denominator.

| ID | Final correct | Correct rate | `best@0.0` | `current@0.0` | `best@1.0` | `current@1.0` | `best@2.0` | `current@2.0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 48/50 | .96 | .96 | .84 | .66 | .48 | .04 | .04 |
| C2 | 49/50 | .98 | .98 | .66 | .48 | .36 | .12 | .12 |
| C3 | 49/50 | .98 | .98 | **.86** | .66 | **.54** | **.16** | **.18** |
| C4 | 49/50 | .98 | .98 | .74 | .52 | .32 | .14 | .10 |
| G1 | 49/50 | .98 | .98 | .70 | .68 | .40 | .12 | .12 |
| G2 | 48/50 | .96 | .96 | .50 | .58 | .34 | .08 | .02 |
| G3 | **50/50** | **1.00** | **1.00** | .70 | .66 | .42 | .12 | .12 |
| G4 | 49/50 | .98 | .98 | .76 | **.74** | **.54** | .12 | .14 |
| G5 | 49/50 | .98 | .98 | .82 | .68 | .48 | .12 | **.18** |
| G6 | **50/50** | **1.00** | **1.00** | .68 | .68 | .38 | **.16** | .08 |

Bold marks the observed column maximum, including ties. `best@0.0` is close to final correctness but is derived from running-best performance state; it should not replace the explicit correctness column.

## Observed metric leaders

These are direct descriptions of this ten-run cohort—not causal winners, expected rankings, or estimates of feature effects.

- **Final correctness:** G3 (deletion+refinement) and G6 (deletion+merge+refinement, similarity 0.8), both 50/50.
- **`fast_p_best@0.0`:** G3 and G6, both 1.00.
- **`fast_p_best@1.0`:** G4 (deletion+merge, similarity 0.8), 0.74 or 37/50 problems.
- **`fast_p_best@2.0`:** C3 (first selective-retention campaign) and G6, both 0.16 or 8/50.
- **Current retention:** G5 (deletion+merge+refinement, similarity 0.7) has the highest correct/non-hack current geomean, 1.3521 with n=41. By full-denominator current fast-p, C3 leads at threshold 0.0 (.86), C3/G4 tie at 1.0 (.54), and C3/G5 tie at 2.0 (.18). “Best retention” therefore depends on the chosen threshold and sample policy.
- **Token efficiency:** C1 (Markov report) uses the fewest reported total tokens, 55.8M, and the fewest tokens per final-correct problem (about 1.16M). G4 uses the fewest calls (4,942), a different efficiency measure. Neither establishes quality-adjusted efficiency without a pre-registered utility/cost function.

Every governance cell has n=1. The selective mode has two completed campaigns, but they are separate resumed campaigns and are not automatically independent replicates. There is no clean 50×30 truncation control. The observed leaders therefore should guide hypotheses and replication priorities, not component-effect claims.

## Completion, CUDA, correctness, and timing

All rows have 50/50 finished workspaces, final iteration 30,
`cuda_available=true`, A6000 hardware, zero `cuda_home_err` rows across the
selected workspace metrics, and zero `suspicious_speedup_count`.

| ID | Completed | Final correct | Correct rate | Recorded wall hours | Final-session problems | Reported avg min/problem |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 50/50 | 48 | 0.96 | 50.33 | 50 | 60.4 |
| C2 | 50/50 | 49 | 0.98 | 62.41 | 47 | 79.7 |
| C3 | 50/50 | 49 | 0.98 | 57.70 | 14 | 247.3 |
| C4 | 50/50 | 49 | 0.98 | 59.56 | 17 | 210.2 |
| G1 | 50/50 | 49 | 0.98 | 52.21 | 50 | 62.7 |
| G2 | 50/50 | 48 | 0.96 | 51.91 | 50 | 62.3 |
| G3 | 50/50 | 50 | 1.00 | 57.18 | 50 | 68.6 |
| G4 | 50/50 | 49 | 0.98 | 59.45 | 5 | 713.4 |
| G5 | 50/50 | 49 | 0.98 | 59.69 | 3 | 1193.7 |
| G6 | 50/50 | 50 | 1.00 | 60.69 | 23 | 158.3 |

The wall-hour field aggregates timing rows retained across the campaign, while `avg_wall_time_sec` in resumed summaries divides by `problems_timed_this_session`. It therefore produces inflated “average” values for G4/G5 and other resumed campaigns. These are provenance/timing artifacts, not evidence that one problem literally required 12–20 hours. Batch start/finish fields can also describe only the latest resume session. Use wall time only as coarse service-cost context.

## Final performance profiles

### Correct, non-hack speedups

| ID | Current mean / median / geomean (n) | Best mean / median / geomean (n) |
|---|---|---|
| C1 | 1.1005 / 1.0141 / **0.8410 (42)** | 1.2340 / 1.0775 / **1.0413 (35)** |
| C2 | 1.4716 / 1.1264 / **1.1235 (33)** | 1.3286 / 0.9782 / **1.0670 (39)** |
| C3 | 1.4678 / 1.0525 / **1.1757 (43)** | 1.4767 / 1.1008 / **1.2559 (36)** |
| C4 | 1.2689 / 0.9601 / **0.9576 (37)** | 1.3226 / 1.0288 / **1.0523 (41)** |
| G1 | 1.5164 / 1.0664 / **1.2265 (35)** | 1.3007 / 1.1667 / **1.2242 (33)** |
| G2 | 1.2013 / 1.0433 / **1.0339 (25)** | 1.2530 / 1.0851 / **1.1001 (41)** |
| G3 | 1.4198 / 1.0851 / **1.1372 (35)** | 1.3219 / 1.0653 / **1.1324 (42)** |
| G4 | 1.5108 / 1.2174 / **1.2990 (38)** | 1.4402 / 1.1584 / **1.2626 (38)** |
| G5 | 1.7035 / 1.0854 / **1.3521 (41)** | 1.3776 / 1.1044 / **1.2373 (42)** |
| G6 | 1.2226 / 1.0461 / **1.1130 (34)** | 1.3453 / 1.0850 / **1.1987 (40)** |

These aggregates exclude incorrect and hack-flagged samples. Their `n` values differ, so geomeans compare different self-selected subsets. In particular, geomean alone must not be used to rank runs. `best_speedup_overall` is also not a simple maximum; the analyzer defines it as the speedup attached to the minimum non-outlier runtime after likely-hack exclusion.

### `fast_p_best` and `fast_p_current`

Values are ordered by threshold `[0.0, 0.5, 0.8, 1.0, 1.5, 2.0]` and retain the full 50-problem denominator.

| ID | `fast_p_best` | `fast_p_current` |
|---|---|---|
| C1 | `[.96, .86, .76, .66, .06, .04]` | `[.84, .72, .64, .48, .04, .04]` |
| C2 | `[.98, .90, .68, .48, .20, .12]` | `[.66, .62, .54, .36, .18, .12]` |
| C3 | `[.98, .96, .82, .66, .24, .16]` | `[.86, .80, .68, .54, .18, .18]` |
| C4 | `[.98, .90, .70, .52, .26, .14]` | `[.74, .62, .46, .32, .18, .10]` |
| G1 | `[.98, .96, .90, .68, .14, .12]` | `[.70, .70, .58, .40, .12, .12]` |
| G2 | `[.96, .90, .78, .58, .10, .08]` | `[.50, .46, .42, .34, .04, .02]` |
| G3 | `[1.00, .96, .84, .66, .22, .12]` | `[.70, .62, .60, .42, .18, .12]` |
| G4 | `[.98, .98, .86, .74, .20, .12]` | `[.76, .74, .64, .54, .18, .14]` |
| G5 | `[.98, .98, .88, .68, .20, .12]` | `[.82, .82, .72, .48, .22, .18]` |
| G6 | `[1.00, .98, .84, .68, .24, .16]` | `[.68, .66, .58, .38, .14, .08]` |

`fast_p_best` uses running-best runtime and does **not** remove a hack-flagged best, unlike the speedup aggregates. Reading fast-p together with best/current `n`, final correctness, and hack counts is mandatory.

## Trajectories

Each vector is iteration `[1, 5, 10, 15, 20, 25, 30]`.

| ID | Best-speedup geomean trajectory | `fast_p_best@1.0` trajectory |
|---|---|---|
| C1 | `.752, .805, .801, 1.005, 1.048, 1.065, 1.041` | `.08, .28, .44, .58, .60, .62, .66` |
| C2 | `.684, .874, .970, 1.001, 1.010, 1.065, 1.067` | `.08, .30, .42, .44, .46, .48, .48` |
| C3 | `.867, .894, 1.049, 1.125, 1.174, 1.216, 1.256` | `.08, .38, .56, .60, .62, .62, .66` |
| C4 | `.677, .492, .943, .994, 1.015, 1.053, 1.052` | `.04, .24, .42, .50, .50, .52, .52` |
| G1 | `.940, .845, 1.002, 1.039, 1.108, 1.171, 1.224` | `.16, .44, .58, .58, .60, .66, .68` |
| G2 | `.771, .820, .975, .985, 1.011, 1.045, 1.100` | `.10, .38, .48, .52, .58, .58, .58` |
| G3 | `.877, .807, 1.005, 1.059, 1.084, 1.086, 1.132` | `.06, .48, .54, .62, .62, .66, .66` |
| G4 | `.711, 1.010, 1.053, 1.043, 1.118, 1.162, 1.263` | `.08, .46, .56, .60, .64, .66, .74` |
| G5 | `.790, .925, .948, 1.040, 1.175, 1.176, 1.237` | `.06, .40, .50, .58, .60, .64, .68` |
| G6 | `.909, .839, .946, 1.122, 1.139, 1.182, 1.199` | `.16, .44, .52, .56, .62, .66, .68` |

Most arms improved through iteration 30, but not monotonically. C1's best geomean eased after iteration 25; C4 suffered a pronounced early dip; C2's `fast_p_best@1.0` plateaued at 0.48 by iteration 25. G4 had the largest late fast-p rise. These are aggregate trajectories over coupled, sequentially memory-sharing problems, not independent learning curves.

## Calls, tokens, iteration quality, and errors

Token totals use endpoint-reported usage. Iteration rates use rows where the corresponding boolean exists.

| ID | Calls | Prompt / completion / total tokens | Compiled rate | Correct rate | Hack rate | Error rows / observed |
|---|---:|---|---:|---:|---:|---:|
| C1 | 6,369 | 49.0M / 6.8M / **55.8M** | 84.6% | 62.5% | 1.33% | 531/1500 (35.4%) |
| C2 | 7,779 | 203.9M / 7.0M / **210.9M** | 86.8% | 57.7% | 0.80% | 613/1500 (40.9%) |
| C3 | 5,781 | 166.7M / 6.0M / **172.8M** | 86.9% | 65.1% | 1.20% | 500/1500 (33.3%) |
| C4 | 5,830 | 191.5M / 6.5M / **198.0M** | 88.9% | 56.3% | 0.73% | 627/1500 (41.8%) |
| G1 | 5,008 | 140.0M / 6.4M / **146.4M** | 87.8% | 62.9% | 2.00% | 540/1500 (36.0%) |
| G2 | 5,010 | 164.8M / 6.8M / **171.6M** | 88.3% | 60.2% | 1.13% | 582/1500 (38.8%) |
| G3 | 6,128 | 137.3M / 7.7M / **145.0M** | 90.1% | 65.7% | 0.87% | 507/1500 (33.8%) |
| G4 | 4,942 | 120.3M / 5.9M / **126.2M** | 87.7% | 65.2% | 0.93% | 488/1499 (32.6%) |
| G5 | 6,073 | 132.2M / 7.4M / **139.6M** | 89.4% | 68.0% | 0.73% | 461/1497 (30.8%) |
| G6 | 6,074 | 124.4M / 7.1M / **131.6M** | 89.5% | 64.1% | 0.93% | 517/1500 (34.5%) |

Output mismatch is the largest error category in every run (244–373 rows), followed by varying compilation, service/static-check “other,” CUDA/runtime, shape/dtype, OOM, load/link, and timeout errors. Final run correctness remains much higher than per-iteration correctness because the agent only needs one valid best during the search.

Endpoint/service errors are present and can affect campaign duration and resumes. Deterministic examples from the extracted top-error evidence include HTTP 429 `RateLimitError: Too Many Requests` at C3 `level_1_problem_58`, iteration 16 (8 occurrences of that signature), G4 `level_3_problem_4`, iteration 7 (5), and G5 `level_3_problem_32`, iteration 27 (9). These are mixed into the heuristic `other` category; that category is not a pure endpoint-failure count. Evaluation failures such as illegal/misaligned CUDA access and 300-second timeouts are separate from endpoint service errors.

## L0, L1, and governance

The on-disk L0 recorder remained complete in every mode: 30 final entries per workspace, mean growth 29, and 1,500 snapshots per run. Context management changes what is put back into prompts, not what is retained on disk.

| ID | L1 total | Active | Deleted | Refined | Accepted merges | Merge events | Compression ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 375 | 375 | 0 | 0 | 0 | 0 | 1.000 |
| C2 | 570 | 570 | 0 | 0 | 0 | 0 | 1.000 |
| C3 | 412 | 412 | 0 | 0 | 0 | 0 | 1.000 |
| C4 | 531 | 531 | 0 | 0 | 0 | 0 | 1.000 |
| G1 | 591 | 241 | 0 | 0 | 75 | 208 | 0.408 |
| G2 | 554 | 448 | 0 | 0 | 39 | 119 | 0.809 |
| G3 | 516 | 28 | 435 | 57 | 0 | 0 | 0.054 |
| G4 | 509 | 28 | 466 | 0 | 6 | 16 | 0.055 |
| G5 | 602 | 18 | 452 | 64 | 27 | 62 | 0.030 |
| G6 | 589 | 30 | 476 | 72 | 7 | 14 | 0.051 |

The merge thresholds changed both candidate volume and retention. G1 (0.7) generated 208 merge decisions and accepted 75 (36.1%), absorbing 350 sources; G2 (0.8) generated 119 and accepted 39 (32.8%), absorbing 106. In deletion-bearing arms, deletion dominates final compression. “Refined” counts include versioned entries in `shared_l1.jsonl`; only a small number remain active in final catalog sidecars.

The extractor emitted 46 warnings, all retained in `feature_evidence.json`; they principally identify absent optional sidecars when a feature was disabled or older outputs omitted that sidecar. Aggregate extraction itself emitted no run warnings or failures.

## Deterministic raw-evidence sample

To ground the qualitative interpretation without cherry-picking long chat text, this report inspected a fixed small set: workspace `level_1_problem_100`; iterations 1, 15, and 30 where available; and the first/last governance records.

- **Markov report:** `runs_evolving/base_agent_markov_report_itr30_2026_07_21_17_11/workspaces/level_1_problem_100/chat_history.jsonl`, iteration 1, records `model_id=openai/gpt-oss-120b`; the rewriter identifies the initial illegal-access cause as a non-contiguous expanded target. `iteration_snapshots.jsonl` shows 1/15/30 L0 entries at iterations 1/15/30, while `metrics_by_iteration.jsonl` moves from illegal access at iteration 1 to correct 2.223× at iteration 2 and correct 6.626× at iteration 30. This demonstrates useful diagnosis persistence in one case, but also later failed variants.
- **Folding:** the same workspace's `chat_history.jsonl` contains per-round `l0_round_summarizer` calls from iteration 1 and an unfold `preflight` at iteration 30 before the coder. Its final two metrics are correct at 6.117× and 6.134×. The raw calls verify the archive/unfold mechanism and help explain the 7,779-call, 210.9M-token cost.
- **Selective retention:** C3 `level_1_problem_100/l0_milestones.json` contains 21 retained milestones. Rules retain proposal/new-best/first-compile/first-correct rounds, while the judge retains distinct strategies; iteration 30 is not a milestone and the previous prompt totals 40,117 tokens. `chat_history.jsonl` has a milestone-judge turn at iterations 15 and 30. This verifies selective retention, but a permissive judge can still preserve many variants.
- **Merge-only 0.7:** `l1_skill_merges.jsonl` begins at global iteration 50 with both a rejected unit-test-fail cluster and an accepted unit-test-pass cluster; the final record is at global iteration 1500. This verifies recurring merge governance and shows that similarity alone did not determine acceptance.
- **Deletion+refinement:** `l1_skill_deletions.jsonl` starts at global iterations 34/38 with unit-test-fail deletions and ends at 1500. `skill_revisions.txt` records versioned diagnosis/revision entries, including an early revision that adds a required real CUDA kernel after a missing-kernel failure. This is concrete evidence that both deletion and refinement ran.
- **Triple governance:** G5/G6 `l1_skill_catalog_stats.json` show final catalogs dominated by deletion, with accepted and rejected merges and only small active refined/merged subsets. Their raw summary files explicitly record resumed ranges (G5 problems 48–50; G6 from problem 28).

These anchors are illustrative. They do not establish that the mechanism caused the run-level outcome.

## Observations versus interpretations

### Context management

**Observed**

- Markov used far fewer prompt tokens than folding or selective retention and ended at `fast_p_best@1.0=.66`, but its current profile was weaker (`.48`, current geomean .841).
- Folding generated the most calls and tokens, ended with the lowest context-arm `fast_p_best@1.0=.48`, but retained relatively more >1.5× and >2× bests than Markov.
- Selective campaign C3 ended with best/current geomeans 1.256/1.176 and `.66/.54` fast-p at 1.0; C4 ended at 1.052/.958 and `.52/.32`. The two campaigns diverged substantially despite the same mode label.

**Plausible interpretation / trade-off**

- Markov's concise evolving state can reduce prompt cost and keep a diagnosis salient; it can also over-compress alternatives or anchor later search to the report's framing.
- Folding preserves older information through summaries and can recover selected full rounds; it adds summarizer/preflight calls, large prompt volume, latency, and possible summary distortion.
- Selective retention preserves high-value attempts at full fidelity and omits routine history; its judge and rule set can retain many milestones, making prompts large and potentially overweighting “novel” but unproductive variants.
- The C3/C4 spread is a warning against treating one selective campaign as a stable effect estimate. They are separate resumed campaigns with different retained histories and service interruptions, not automatic independent replicates.

### Merge thresholds

**Observed**

- Similarity 0.7 produced more merge events, accepted merges, and catalog compression than 0.8. G1 also had higher final fast-p/geomean than G2 on several measures, while having more hack iterations (30 vs 17).

**Plausible interpretation / trade-off**

- A lower threshold creates more consolidation opportunities and a smaller retrieval catalog, potentially reducing redundancy.
- It also raises the risk of merging superficially similar but semantically distinct skills. Unit-test rejection (133 of 208 G1 events) is direct evidence that candidate similarity did not guarantee a valid merge.
- Because G1 and G2 are single campaigns with different trajectories and hack incidence, the observed performance gap cannot be attributed to the threshold.

### Deletion and refinement combinations

**Observed**

- Deletion-bearing runs compressed active catalogs to 18–30 entries, versus 241/448 for merge-only and 375–570 for context runs.
- G3 and G6 achieved 50/50 final correctness. G4 had the cohort-high `.74` `fast_p_best@1.0`; G5 had the highest final current geomean (1.352, n=41) and iteration correctness rate (68.0%).
- Refinement added 57–72 versioned entries and roughly 1,100 refinement-phase calls in the combined arms, but no clean control isolates that extra work.

**Plausible interpretation / trade-off**

- Deletion keeps retrieval state small and removes repeatedly unused or unit-test-failing knowledge; aggressive pruning can also discard niche skills needed by later problems.
- Refinement can repair concrete failure modes while preserving lineage; it consumes calls/iterations and may elaborate a weak parent instead of exploring a new strategy.
- Combining deletion, merge, and refinement offers complementary cleanup, consolidation, and repair, but interactions make attribution harder. A small final active catalog may reflect effective curation, over-pruning, or both.

## Limits and next experiment

1. No old-integrate truncation run matches 50×30, so no causal component delta is available.
2. There is one campaign per governance cell and two non-equivalent resumed selective campaigns.
3. Problems are not independent within a run: later workspaces consume a sequentially evolving shared L1 catalog.
4. Launch dates, endpoint load/rate limits, resume boundaries, and final-session timing differ.
5. Best-state hack flags are sticky, and fast-p-best does not filter hacks while speedup geomeans do.
6. Geomeans use different correct/non-hack subsets and must always be read with `n`, correctness, and fast-p.
7. The error taxonomy is heuristic; `other` mixes endpoint errors, static-check failures, worker exits, and framework issues.

A defensible follow-up would run a contemporaneous randomized/block-balanced matrix with an explicit truncation control, fixed endpoint/model metadata in every summary, multiple fresh seeds per cell, identical resume policy, and pre-registered primary metrics (`fast_p_best@1.0`, `fast_p_current@1.0`, final correctness, token/call budget, and governance event counts).

## Evidence sources and semantics

- Generated run aggregate: `aggregate_runs.json`, `aggregate_runs.csv`
- Generated comparisons: `comparison.md`, `comparison-context.md`, `comparison-governance.md`
- Generated behavioral evidence: `feature_evidence.json`, `feature_evidence.csv`
- Metric semantics: `scripts_integration/new_evolving_agent_analysis/README.md`
- Old integrate and context runbooks: `scripts_integration/new_evolving_agent/RUN_WITH_UV.md`, `RUN_WITH_UV_CONTEXT.md`
- Raw summaries, timing, chat, snapshots, metrics, L1 journals, and governance ledgers under each exact `runs_evolving/<run_name>/`
- Reproducible commands and exact inventory: `MANIFEST.md`
