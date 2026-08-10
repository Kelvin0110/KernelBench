# OSS-120B inference experiment report — CPU6 / RTX A6000

Status: evidence frozen from `aggregate_runs.json` generated **2026-08-10 06:49:06 UTC** and `feature_evidence.json` generated **2026-08-10 07:06:17 UTC**. This report covers only the five completed `gpt-oss-120b` inference runs listed below. It does not use GH200 or Terra results.

## Executive summary

- The truncation control had the highest final `fast_p_best@1.0`: **0.72 (36/50)**. Deletion-only, refinement-only, and selective retention each reached **0.62 (31/50)**; merge-only reached **0.60 (30/50)**. These are descriptions of one run per configuration, not estimates of feature effects.
- Final total correctness was **48/50** for the control and deletion, **47/50** for refinement and merge-only, and **44/50** for selective retention. No candidate improved the primary fast-p result over the control; deletion alone matched its total correctness.
- The governance mechanisms demonstrably executed: deletion removed **535** entries and left **31** active; refinement recorded **87** refinements and left **545** active; merge-only accepted **62/187** merge events, absorbed **337** skills, and left **232** active.
- Selective retention used fewer action-selector calls and ended with fewer L1 entries (**422** versus **549** for the control), but it consumed the most reported tokens (**152,258,588**) and had the lowest final correctness. A deterministic raw case also shows endpoint 404s ending one selective workspace after five iterations, so context-policy and service-drift effects cannot be separated.
- Wall time ranged from **65.15 h** to **84.68 h**, but the runs were launched in concurrent waves and used a shared remote endpoint. Wall time is therefore operational evidence, not a clean estimate of feature overhead.
- The strongest caveats are **n=1 per cell**, sequential shared-L1 coupling across problems, staggered launch dates and endpoint drift, and the sticky best-hack flag. The `speedup_best` geomeans use different correct/non-hack subsets (`n=33` to `41`) and must not be used alone to rank features.

## 1. Experimental design and run aliases

All five runs used:

- problem set `subset_selection/selected_problems_50.csv` (10 L1, 15 L2, 25 L3);
- 30 requested iterations per problem;
- `gpt-oss-120b` for coder, summarizer, extractor, and action selector;
- endpoint `inference`;
- static checking enabled;
- RTX A6000 evaluation artifacts and the explicit baseline
  `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`;
- FP32 baseline timings;
- one shared, sequentially growing L1 catalog per run.

The aliases below are design labels, not performance ranks.

| alias | run name | started UTC | context | deletion | refinement | merging |
|---|---|---:|---|:---:|:---:|:---:|
| T0 | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | 2026-08-02 17:58 | truncation | no | no | no |
| D1 | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` | 2026-08-02 17:57 | truncation | yes | no | no |
| R1 | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` | 2026-08-02 17:57 | truncation | no | yes | no |
| M1 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` | 2026-08-05 15:49 | truncation | no | no | yes |
| S1 | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | 2026-08-05 15:56 | selective retention | no | no | no |

T0 is the shared control. D1, R1, and M1 are governance comparisons against T0 with truncation held fixed. S1 is the context-management comparison against T0 with governance disabled.

## 2. Metric semantics

The headline metric is `fast_p_best@1.0`: the fraction of all 50 problems whose running-best speedup is at least 1.0. It retains the full-problem denominator, so incorrect or unfinished solutions do not disappear from the denominator.

Other columns are interpreted as follows:

- **total correct** comes from each completed `run_summary.json`;
- **current fast-p** is final `fast_p_current@1.0`, measuring the current rather than running-best point;
- **wall time** is the sum of recorded per-problem batch timing;
- **best geomean (`n`)** is `speedup_best.geometric_mean` over correct, non-hack samples only, with `n` showing the actual contributing subset.

Two pipeline asymmetries are important:

1. `fast_p_best` does not drop hack-flagged bests, while `speedup_best` does.
2. The recorded best-hack field is sticky: after a problem has any hack-flagged iteration, the best record can remain flagged. A correct best can therefore be excluded from the geomean because of an earlier iteration. This makes `n` mandatory and prevents feature ranking from `speedup_best` geomean alone.

## 3. Validity checks and exclusions

### Included-run checks

| check | result |
|---|---|
| Completed runs | 5/5 `status=complete` |
| Completion | every run 50 attempted, 50 completed, 50 finished workspaces |
| Final alignment | every run has 50 problems at final iteration 30 |
| Aggregate discovery | 5 discovered, 5 aggregated, 0 failures, 0 requested names missing |
| Problem set | identical 50-problem subset in every run |
| Model and endpoint | identical four model roles, `gpt-oss-120b`, endpoint `inference` |
| Baseline | one explicit CPU6/A6000 baseline file for all speedups |
| Static checking | enabled in every run |
| CUDA toolchain health | 0 `cuda_home_err`/CUDA-home error rows across selected workspace metrics |
| Performance cache | all five rows report `performance_stats_source=cached`; no regeneration was done for this report |
| Aggregate warnings | none for the five included rows |

The aggregate records evaluation hardware as `NVIDIA RTX A6000`, matching the baseline contents. T0, D1, and R1 record `hardware_server: SONG_CPU6_A6000x4` in their summaries. **M1 and S1 omit `hardware_server` from `run_summary.json`.** Their CPU6 server/baseline association is inferred from this run series, the explicit baseline path, and their A6000 evaluation artifacts; it is not directly asserted by those two summaries.

### Exclusions

- Three runs under the same root are partial and excluded from every result table: markov report, folding, and the combined deletion+merge+refinement run. Their progress snapshot is in §10.
- GH200 outputs are outside this CPU6/A6000 run set and are not quoted or used.
- Terra outputs differ in model and experimental series and are not quoted or used.
- No partial-run performance, correctness rate, fast-p, or geomean is treated as a result.

## 4. Headline outcomes

| alias | `fast_p_best@1.0` | total correct | `fast_p_current@1.0` | wall time (h) | best geomean (`n`) |
|---|---:|---:|---:|---:|---:|
| T0 | **0.72** | **48/50** | **0.46** | **65.15** | 1.4208 (`n=41`) |
| D1 | 0.62 | **48/50** | 0.34 | 74.71 | 1.2527 (`n=37`) |
| R1 | 0.62 | 47/50 | 0.28 | 69.03 | 1.1202 (`n=33`) |
| M1 | 0.60 | 47/50 | 0.38 | 84.68 | 1.1885 (`n=36`) |
| S1 | 0.62 | 44/50 | 0.38 | 79.39 | 1.2509 (`n=36`) |

At the 1.0 threshold, T0 has 36 qualifying problems, D1/R1/S1 have 31 each, and M1 has 30. The candidate deficits versus T0 are therefore 5 or 6 problems, not differences induced by a changing denominator.

Correctness by level further localizes the run-specific differences:

| alias | L1 correct /10 | L2 correct /15 | L3 correct /25 |
|---|---:|---:|---:|
| T0 | 9 | 15 | 24 |
| D1 | 10 | 15 | 23 |
| R1 | 10 | 14 | 23 |
| M1 | 7 | 15 | 25 |
| S1 | 8 | 15 | 21 |

M1's perfect L3 count coexists with three L1 failures, while S1's four-problem deficit versus T0 is concentrated in L1 and L3. This heterogeneity is another reason not to collapse the experiment into one feature ranking.

## 5. Matched trajectories and problem transitions

### 5.1 `fast_p_best@1.0` at matched iterations

All runs reached the same 30 iterations, so these points are directly aligned.

| iteration | T0 | D1 | R1 | M1 | S1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.02 | 0.06 | 0.04 | 0.04 | 0.02 |
| 5 | 0.32 | 0.30 | 0.24 | 0.32 | 0.34 |
| 10 | 0.52 | 0.38 | 0.52 | 0.46 | 0.48 |
| 15 | 0.58 | 0.50 | 0.58 | 0.54 | 0.56 |
| 20 | 0.66 | 0.54 | 0.60 | 0.56 | 0.60 |
| 25 | 0.68 | 0.58 | 0.62 | 0.58 | 0.60 |
| 30 | **0.72** | 0.62 | 0.62 | 0.60 | 0.62 |

Observed trajectory facts:

- S1 is 0.02 above T0 at iteration 5, then trails it from iteration 10 onward.
- R1 matches T0 at iterations 10 and 15, then ends 0.10 lower.
- M1 matches T0 at iteration 5 and ends 0.12 lower.
- D1 is already 0.14 lower at iteration 10 and ends 0.10 lower.
- The control's separation is therefore a late-run pattern, not an iteration-1 advantage.

The companion best-geomean trajectories are not used to rank arms. Their contributing subset changes as correctness and sticky hack flags change, so even an arithmetically higher point can represent a different set of problems.

### 5.2 Matched best-correctness transitions versus T0

`feature_evidence.json` compares all 50 workspace names using each problem's best result.

| alias | gains | losses | same correct | same incorrect |
|---|---:|---:|---:|---:|
| D1 | 2 | 2 | 46 | 0 |
| R1 | 1 | 2 | 46 | 1 |
| M1 | 2 | 3 | 45 | 0 |
| S1 | 1 | 5 | 43 | 1 |

These transitions show that equal totals can hide swaps: D1's 48/50 is not the same set of correct problems as T0's 48/50.

## 6. Calls, tokens, actions, memory, errors, and hacks

### 6.1 Chat calls and tokens

Token totals sum endpoint-reported usage. M1 has 3 calls with missing usage and S1 has 2, so their totals are lower bounds for those recorded chat rows.

| alias | chat turns | prompt tokens | completion tokens | reported total tokens | feature-specific phases |
|---|---:|---:|---:|---:|---|
| T0 | 4,963 | 120,923,031 | 5,348,695 | 126,271,726 | none |
| D1 | 4,968 | 118,584,854 | 5,456,901 | 124,041,755 | none |
| R1 | 6,113 | 126,308,629 | 6,770,948 | 133,079,577 | 1,185 `skill_diagnosis`; 87 `skill_revision` |
| M1 | 4,673 | 121,202,215 | 5,023,297 | 126,225,512 | merge work is represented in governance sidecars, not a named chat phase |
| S1 | 5,055 | 147,687,027 | 4,571,561 | 152,258,588 | 792 `milestone_judge` |

Standard phase call counts were:

| alias | action selector | coder | extractor | summarizer |
|---|---:|---:|---:|---:|
| T0 | 1,469 | 1,471 | 1,474 | 549 |
| D1 | 1,460 | 1,469 | 1,473 | 566 |
| R1 | 1,461 | 1,459 | 1,382 | 539 |
| M1 | 1,367 | 1,412 | 1,387 | 507 |
| S1 | 1,270 | 1,341 | 1,230 | 422 |

S1's reported token total is highest despite the fewest standard calls. Directly, this means its average recorded prompt burden was larger; attributing that to the retention policy is a hypothesis because endpoint behavior and early workspace termination also changed.

### 6.2 Action mix and L0/L1 state

All action-selector rows in the extractor were parsable; parse errors were 0 in every run.

| alias | debug | propose new | refine current | mean final L0 entries | L1 entries | L1 active |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 385 | 553 | 531 | 29.56 | 549 | 549 |
| D1 | 357 | 567 | 536 | 29.54 | 566 | 31 |
| R1 | 368 | 542 | 551 | 29.36 | 626 | 545 |
| M1 | 334 | 507 | 526 | 28.56 | 569 | 232 |
| S1 | 315 | 422 | 533 | 27.20 | 422 | 422 |

Governance evidence:

- D1: 535 deletion events, comprising 297 `unit_test_fail` and 238 `consecutive_unused`; catalog compression ratio 0.054770.
- R1: 87 refinement records, 81 superseded entries, catalog compression ratio 0.870607.
- M1: 28 merge passes; 187 events, 62 accepted, 125 rejected, 0 skipped; 337 skills absorbed; 185 unit-test runs; catalog compression ratio 0.407733.
- T0 and S1: no governance sidecars and no deletion, refinement, or merge events.

### 6.3 Iteration outcomes, errors, and hack flags

The rates below use observed `metrics_iteration` rows, not 50 final problems. Error categories are heuristic labels from `feature_evidence.json`.

| alias | metric rows | compiled | correct | hack-flagged | compilation errors | output mismatches | timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 1,478 | 1,198 (0.8106) | 798 (0.5399) | 12 (0.0081) | 99 | 295 | 24 |
| D1 | 1,477 | 1,182 (0.8003) | 811 (0.5491) | 12 (0.0081) | 116 | 273 | 31 |
| R1 | 1,468 | 1,222 (0.8324) | 831 (0.5661) | 17 (0.0116) | 70 | 317 | 25 |
| M1 | 1,428 | 1,211 (0.8480) | 798 (0.5588) | 14 (0.0098) | 66 | 308 | 15 |
| S1 | 1,360 | 1,172 (0.8618) | 728 (0.5353) | 12 (0.0088) | 65 | 321 | 17 |

Aggregate problem-level `problems_with_hack` counts are T0 7, D1 11, R1 15, M1 12, and S1 9. They should not be read as independent behavioral outcomes: the best-hack state is sticky, and a static-check trip can latch the problem even when a later best is clean. Every run has `suspicious_speedup_count=0`; the hack flags here are static-check/evaluator signals rather than post-hoc extreme-speedup detections.

## 7. Feature-by-feature evidence

### 7.1 T0 — truncation control

Direct observations:

- T0 produced 48/50 correct, `fast_p_best@1.0=0.72`, and `fast_p_current@1.0=0.46`.
- It had no governance events, leaving all 549 L1 entries active.
- It had the lowest recorded wall time, 65.15 h, and the highest headline fast-p.

Advantages in this run: simple policy, no governance-call machinery, highest best/current fast-p, and lowest wall time.

Disadvantages in this run: no L1 compression or validation, 549 active entries, and no protection against stale or redundant promoted skills.

Hypothesis, not cause: retaining the full sequential L1 may have preserved useful problem-family details that aggressive governance discarded. Counterevidence is that this is one stochastic run; the common L1P56 case below was solved by every candidate but not T0.

### 7.2 D1 — deletion-only

Direct observations:

- D1 matched T0's 48/50 total correctness but had `fast_p_best@1.0=0.62` and current fast-p 0.34.
- Deletion executed 535 times and left 31 of 566 entries active.
- Its matched problem set contains 2 gains and 2 losses versus T0.
- Wall time was 74.71 h.

Advantage: deletion materially constrained the active catalog while preserving the total number of correct problems in this run.

Disadvantages: it did not preserve the same correct-problem set, its headline fast-p was 0.10 below T0, and it used 9.56 more recorded wall hours.

Hypothesis, not cause: unit-test and unused-skill deletion may reduce retrieval noise, but a 31-entry active catalog may also remove useful specialized knowledge. The observed result does not determine which effect dominated.

### 7.3 R1 — refinement-only

Direct observations:

- R1 produced 47/50 correct, best/current fast-p 0.62/0.28, and wall time 69.03 h.
- It added 1,185 diagnosis calls and 87 revision calls; 87 refinements were recorded.
- It had the most chat turns (6,113) and 133,079,577 reported tokens.
- Its iteration-level correct rate, 0.5661 over 1,468 observed rows, was the highest of the five, while its final total correctness was not.

Advantage: refinement generated explicit diagnosis/revision evidence and more correct iteration rows.

Disadvantages: it added call and token burden, grew L1 to 626 entries, and did not improve final fast-p or correctness over T0.

Hypothesis, not cause: diagnosis and revision can repair reusable skills but can also amplify a mistaken abstraction through sequential L1. The final aggregate cannot distinguish useful revisions from propagated errors.

Deterministic positive candidate: `feature_evidence.json` selects `level_3_problem_49` as R1's largest valid improvement: T0 best 1.3352 at iteration 10 versus R1 best 3.7125 at iteration 29. Raw R1 metrics confirm iteration 29 was compiled, correct, and recorded non-hack, with runtime 6.33 against a 23.5 reference. The snapshot and coder chat at iteration 29 show a `propose_new` round. They also record static warnings (`workload_shrink`, PyTorch operations/wrapping), so the evaluator's non-hack value should not be interpreted as independent proof that the approach was semantically clean. This is direct case evidence, not evidence that refinement caused the improvement.

### 7.4 M1 — merge-only

Direct observations:

- M1 produced 47/50 correct, best/current fast-p 0.60/0.38, and the longest wall time, 84.68 h.
- It accepted 62 of 187 merge events, absorbed 337 skills, and reduced 569 total entries to 232 active.
- Its matched set contains 2 gains and 3 losses versus T0.

Advantage: merge-only achieved substantial catalog compression without deletion and retained perfect L3 final correctness (25/25) in this run.

Disadvantages: it had the lowest best fast-p, three L1 correctness failures, 28 merge passes plus 185 unit-test runs, and the highest recorded wall time.

Hypothesis, not cause: semantic merging may reduce redundant retrieval, but merged entries can blur specialized preconditions. Merge computation may also contribute overhead, though concurrent scheduling prevents assigning the wall-time difference to merging.

Deterministic regression candidate: `level_3_problem_34` is M1's largest valid regression, T0 3.0247 versus M1 0.6255. Raw M1 metrics show the first/best correct point at iteration 9, runtime 23.5 against 14.7. The iteration-9 action-selector and coder chat exist; the snapshot records `propose_new` after repeated unsuccessful debugging. This directly shows a slower correct solution after search instability, not that merging caused the regression.

### 7.5 S1 — selective retention

Direct observations:

- S1 produced 44/50 correct, best/current fast-p 0.62/0.38, and wall time 79.39 h.
- It ended with 422 L1 entries and used 1,270 action-selector, 1,341 coder, and 1,230 extractor calls, all fewer than T0.
- It added 792 milestone-judge calls and reported 152,258,588 tokens, the highest total.
- Its matched set has 1 gain, 5 losses, 43 same-correct, and 1 same-incorrect problem.

Advantages: fewer standard agent calls, fewer final L1 entries, and a milestone record that makes retention decisions observable.

Disadvantages: lowest total correctness, highest reported token burden, and no fast-p improvement.

Hypothesis, not cause: retaining selected milestones may reduce repetitive turns while expanding the context carried by the remaining calls, consistent with fewer calls but more prompt tokens. Endpoint failures and later launch dates prevent isolating that mechanism.

Deterministic endpoint-drift case: `feature_evidence.json` selects `level_1_problem_54` as S1's highest-baseline correctness loss. T0 had a correct 2.9591 best at iteration 24. S1's workspace ended after iteration 5 with no correct best; raw metrics and snapshots record model-group 404 errors at iterations 1, 2, and 5. A coder chat exists at iteration 3, but the failed iteration-5 call has no successful chat row and is represented in the snapshot/metrics error. This is direct service-failure evidence. It makes the S1 correctness deficit partly contaminated; it does not establish how many other losses share that mechanism.

## 8. Deterministic raw case study across all arms

To avoid selecting a different favorable example for each feature, one common candidate category was inspected across all arms: `case_study_candidates.correctness_gain`. It deterministically selects `level_1_problem_56` for every candidate because T0's best remained incorrect and each candidate found a correct best.

| alias | cited iteration | action in snapshot | compiled | correct | recorded hack | speedup | runtime / reference |
|---|---:|---|:---:|:---:|:---:|---:|---:|
| T0 | 30 | debug current | yes | no | no | 0.0000 | n/a / 13.3 |
| D1 | 30 | refine current | yes | yes | no | 1.1565 | 11.5 / 13.3 |
| R1 | 17 | refine current | yes | yes | no | 1.1875 | 11.2 / 13.3 |
| M1 | 17 | propose new | yes | yes | no | 1.0153 | 13.1 / 13.3 |
| S1 | 10 | refine current | yes | yes | no | 1.1770 | 11.3 / 13.3 |

Direct artifact observations:

- T0 `workspaces/level_1_problem_56/metrics_by_iteration.jsonl`, iteration 30, records `Output mismatch`; its snapshot says the action was `debug_current`.
- D1 iteration 30, R1 iteration 17, M1 iteration 17, and S1 iteration 10 each have a corresponding coder row in `chat_history.jsonl`, a snapshot row, and a compiled/correct metrics row.
- M1's row includes static warnings for PyTorch wrapping and stream injection; S1's row includes a PyTorch-wrap warning. The recorded `is_hack` value is nevertheless false for both rows.

Interpretation: all four candidate policies fixed the same control failure through different action paths. That pattern is consistent with stochastic search and sequential-memory divergence and is not specific evidence for deletion, refinement, merging, or selective retention.

Only these deterministic anchors and the three feature-specific cases in §7 were inspected in raw chat/snapshot/metrics. Long model text and generated code are intentionally not reproduced.

## 9. Advantages, disadvantages, and plausible mechanisms

| arm | observed advantage | observed disadvantage | plausible hypothesis — not a cause |
|---|---|---|---|
| T0 | highest fast-p; lowest wall time | no L1 pruning or validation | broad memory preserved specialized skills |
| D1 | 31 active entries with unchanged total correctness | lower fast-p; different correct set | aggressive pruning reduced noise and useful coverage simultaneously |
| R1 | explicit diagnosis/revision; highest iteration correct rate | extra calls/tokens; no final gain | revised skills sometimes repair patterns but can propagate bad abstractions |
| M1 | 337 skills absorbed; 232 active; perfect L3 total | lowest best fast-p; longest wall time | compression reduced redundancy but blurred specialized conditions |
| S1 | fewer standard calls and L1 entries | lowest correctness; highest token total | retained milestones made fewer prompts larger |

The data support these as mechanism candidates for future ablation or repeated-run testing. They do not support causal wording because each cell has one run and the waves were not simultaneous.

## 10. Pending partial runs

Snapshot from existing `batch_timing.jsonl` and `run_finished.json` files around **2026-08-10 07:10 UTC**:

| pending run | intended arm | finished markers /50 | batch status rows | treatment as evidence |
|---|---|---:|---|---|
| `base_agent_oss120b_markov_itr30_2026_08_07_14_07` | markov report | 40 | 38 ok, 2 error | excluded; no result quoted |
| `base_agent_oss120b_folding_itr30_2026_08_09_13_47` | folding | 10 | 10 ok | excluded; no result quoted |
| `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48` | deletion + merge + refinement | 9 | 8 ok, 1 error | excluded; no result quoted |

None has `run_summary.json`; all remain partial. Their prefixes are not comparable with 50-problem completed runs.

## 11. Threats and limitations

1. **One run per configuration (`n=1`).** There is no run-to-run variance estimate, no controlled repeat, and no basis for statistical feature ranking.
2. **Sequential L1 coupling.** Problems are not independent: later workspaces consume an L1 catalog produced by earlier workspaces in the same run. An early stochastic difference can alter every later prompt and governance event.
3. **Staggered dates and endpoint drift.** T0/D1/R1 began together on August 2; M1/S1 began on August 5. The remote endpoint could change between waves. The S1 L1P54 404s are direct evidence of service instability.
4. **Unseeded model sampling.** The endpoint does not provide a controlled replicate here, so matched problems can diverge even without treatment differences.
5. **Wall-clock confounding.** Runs used concurrent waves, shared host resources, compilation infrastructure, and the remote endpoint. Wall time combines treatment work with contention and service latency.
6. **Hardware-server metadata gap.** M1 and S1 lack `hardware_server` in their summaries. CPU6 membership is inferred from series placement, explicit baseline selection, and A6000 evaluation metadata.
7. **Baseline dependence.** Every fast-p and speedup uses one baseline timing file. Replacing it can move threshold classifications near 1.0.
8. **Sticky best-hack issue.** `speedup_best` can exclude a clean best because an earlier hack flag latched. `fast_p_best` does not apply the same exclusion. Best geomeans therefore compare different subsets and are secondary only.
9. **Static-check semantics.** `is_hack` is a policy/evaluator signal, not proof of cheating. All runs have zero suspicious extreme-speedup problems.
10. **Extractor limits.** Token totals depend on reported usage; error taxonomy is heuristic; action counts count parsable call rows; deterministic cases are descriptive selections; optional-artifact warnings often reflect intentionally disabled features.
11. **Case-study limits.** Raw cases establish what artifacts record at named iterations. Model rationales are model-generated interpretations, not ground truth about why a kernel succeeded or failed.

## 12. Reproducibility links

- [Aggregate JSON](aggregate_runs.json) and [CSV](aggregate_runs.csv)
- [Full comparison](comparison.md)
- [Governance comparison](comparison-governance.md)
- [Context comparison](comparison-context.md)
- [Feature evidence JSON](feature_evidence.json) and [CSV](feature_evidence.csv)
- [Metric and extractor semantics](../../README.md)
- [Exact run manifest and commands](MANIFEST.md)

Raw evidence is rooted at `runs_evolving/inference_oss_120b/<run_name>/`. The cited case-study files are:

- `workspaces/level_1_problem_56/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl}` in all five runs;
- R1 `workspaces/level_3_problem_49/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl}`;
- M1 `workspaces/level_3_problem_34/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl}`;
- S1 and T0 `workspaces/level_1_problem_54/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl}`.

The exact read sources, caveats, and commands are recorded in `MANIFEST.md`.
