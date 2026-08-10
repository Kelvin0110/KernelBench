# GPT-5.6 Terra inference — truncation versus Markov report on CPU4/A6000

Status: **2026-08-10 07:08 UTC**. This report covers only the two completed
GPT-5.6 Terra inference runs scored against the CPU4/A6000 baseline. It does not
pool or quote GH200 or GPT-OSS results.

## Executive summary

- The headline result is a tie: both runs reached
  `fast_p_best@1.0 = 0.78`, or 39 of 50 problems. The tie does **not** establish
  equivalence. Markov led at matched early iterations, the runs differed on
  correctness, final-current quality, per-problem winners, hack exposure,
  token/call use, and wall time.
- Truncation finished with 49/50 problems correct; Markov finished with 47/50.
  At the level breakdown, truncation was 10/10, 14/15, 25/25 and Markov was
  9/10, 13/15, 25/25 for Levels 1, 2, and 3.
- Markov had the much stronger final current state:
  `fast_p_current@1.0 = 0.50` (25/50) versus 0.26 (13/50). Its current
  speedup geomean was 1.576454 over `n=31`, versus 1.336064 over `n=19`.
- The final best-speedup geomeans were numerically almost identical:
  1.454369 (`n=44`) for truncation and 1.454476 (`n=39`) for Markov.
  These are differently selected subsets, and the sticky best-hack flag makes
  the sample sizes path-dependent. They are not evidence of equal kernel
  quality.
- Markov used more calls (5,349 versus 3,169) but fewer reported total tokens
  (47,399,320 versus 60,927,079). It added 1,295 evolving-report calls consuming
  5,658,262 reported tokens. Four Markov calls lack complete token totals, so
  its token total is a lower bound over reported usage.
- Markov accumulated 76.6912 hours of problem wall time versus 64.0659 hours
  for truncation, a reported increase of 12.6253 hours. This is not a clean
  mode-cost estimate: both runs were resumed, their resumed windows overlapped,
  the remote endpoint was shared and time-varying, and the recorded
  minutes/problem divides cumulative time by only the resumed-session problem
  count.
- There is one run per mode (`n=1`). The evidence describes these two runs; it
  does not estimate a repeatable treatment effect.

## 1. Design, aliases, and held-constant settings

| Alias | Context mode | Exact run name | Status |
|---|---|---|---|
| T / R1 | `truncation` | `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | complete, 50/50 |
| M / R2 | `markov_report` | `base_agent_terra_markov_itr30_2026_08_01_17_41` | complete, 50/50 |

Held constant in the recorded configuration:

- problem set: `subset_selection/selected_problems_50.csv` (10 Level 1,
  15 Level 2, 25 Level 3);
- nominal budget: 30 iterations per problem;
- all model roles: `gpt-5.6-terra`;
- endpoint: `inference`;
- GPU: NVIDIA RTX A6000; backend/precision: CUDA/FP32;
- scoring baseline:
  `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`;
- static checking enabled;
- skill deletion, merging, refinement, and L1 unit-test GC disabled;
- per-run shared L1 memory enabled.

The intended treatment difference is L0 context management. T records an
`evolving_report_max_tokens` setting of 1,536 but makes no evolving-report
calls; M uses `markov_report` with a 65,536-token cap. The inert
`skill_merge_similarity` metadata differs (0.7 for T, 0.9 for M), but merging
is disabled in both.

“CPU4” names the baseline/server folder (`SONG_CPU4_A6000x2`); the measured
accelerator in the baseline and runs is an RTX A6000, not a CPU.

## 2. Validity, completion, and exclusions

The aggregate contains exactly two discovered and aggregated runs:
`complete_runs=2`, `partial_runs=0`, no failures, and no missing requested
runs. Both have `run_summary.json`, 50 finished workspaces, and 50 completed
problems. Both cached performance-stat files were accepted, and the aggregate
recorded no hardware/baseline warning. A direct scan of the selected workspace
metrics found zero `cuda_home_err` or CUDA-home error rows.

Important qualifications:

1. **One replicate per mode.** There is no run-to-run variance estimate.
2. **Both final summaries are resumptions.** T records `resume=true`,
   `start_problem=9`, and 42 problems timed in the final session. M records
   `resume=true`, `start_problem=13`, and 38 problems timed in the final
   session. M retained the first 12 completed workspaces; its
   `level_2_problem_13` ended when the endpoint budget was exceeded and was not
   rerun.
3. **Sequential L1 coupling.** Each run's L1 catalog accumulates across the
   ordered problem stream. Later problems are conditioned on lessons produced
   by earlier problems, so the 50 problems are not independent observations.
   The two runs use separate catalogs, but early run-specific divergence can
   propagate within each run.
4. **Hardware/model exclusions.** GH200 data are invalid for this CPU4/A6000
   comparison, and GPT-OSS data use a different model and hardware baseline.
   Neither is quoted or pooled here.
5. **Pending folding is excluded.** At the 07:08 UTC status check,
   `base_agent_terra_folding_itr30_2026_08_09_15_11` had no
   `run_summary.json`; 12 workspaces had `run_finished.json`, and the next
   workspace (`level_2_problem_19`) had reached iteration 6 without a finish
   marker. It is pending, not a third result.

## 3. Metric semantics

- `fast_p_best@p` uses the full 50-problem denominator, so problems without a
  qualifying best count against the score. It does not apply the sticky
  `metrics_best.is_hack` exclusion used by the speedup aggregates.
- `speedup_best.*` and `speedup_current.*` include correct, non-hack samples
  only. Every aggregate must therefore be read with its `n`.
- `fast_p_current` describes the last observed submission for each problem;
  it can differ sharply from the running best when later attempts fail.
- `best_speedup_overall` from `run_summary.json` is not a literal maximum; it
  is the selected non-outlier best-runtime summary.
- The feature extractor defines a valid matched speedup as finite and positive
  with `best_correct=true` and `best_is_hack=false`. Its deterministic case
  candidates are descriptive selections, not causal estimates.

### Sticky best-hack issue

`metrics_best.is_hack` is a run-within-problem latch: after any hack-flagged
iteration it can remain true even when the retained best came from a clean
iteration. The final best geomean then drops the whole problem. Here T has six
problems with a hack and `best_n=44`; M has ten such problems and
`best_n=39`, with correctness removing an additional eligible Markov problem.
The two geomeans therefore average different, path-selected sets.

The raw `level_2_problem_13` trace makes the defect concrete. M iteration 11
is correct and non-hack, but a hack at iteration 14 makes
`metrics_best.is_hack=true`; it remains true on later non-hack iterations and
at the budget failure on iteration 25. T has 27 hack-flagged rows; its only
three non-hack rows are a timeout, a compilation failure, and an output
mismatch. Thus all 27 correct T rows are hack-flagged, and the final correct
iteration 30 still leaves `metrics_best` incorrect/empty. “Best is hack” is
not a reliable verdict on the retained best kernel.

## 4. Headline outcomes

| Metric | T: truncation | M: Markov | Direct reading |
|---|---:|---:|---|
| `fast_p_best@1.0` | 0.78 (39/50) | 0.78 (39/50) | tie |
| total correct | 49/50 | 47/50 | M −2 |
| `fast_p_current@1.0` | 0.26 (13/50) | 0.50 (25/50) | M +12 problems |
| current geomean (`n`) | 1.336064 (19) | 1.576454 (31) | M higher, different `n` |
| best geomean (`n`) | 1.454369 (44) | 1.454476 (39) | numerically equal, different subset |
| `best_speedup_overall` | 5.729350 | 4.867021 | T higher |
| total wall hours | 64.0659 | 76.6912 | M +12.6253 h; confounded |
| hack iterations / observed metric rows | 50/1,025 (4.8780%) | 71/1,295 (5.4826%) | M +21 flags and +0.6046 pp |
| problems with any hack | 6/50 | 10/50 | affected by exposure and latching |
| suspicious-speedup count | 0 | 0 | tie |

The threshold profile shows why one equal headline does not imply
equivalence:

| `p` | T best | M best | T current | M current |
|---:|---:|---:|---:|---:|
| 0.0 | 0.98 | 0.94 | 0.38 | 0.62 |
| 0.5 | 0.94 | 0.90 | 0.36 | 0.58 |
| 0.8 | 0.92 | 0.88 | 0.32 | 0.58 |
| 1.0 | 0.78 | 0.78 | 0.26 | 0.50 |
| 1.5 | 0.48 | 0.50 | 0.12 | 0.34 |
| 2.0 | 0.26 | 0.26 | 0.04 | 0.14 |

T has more ever-correct problems and slightly better best coverage below
1.0; M retains far more qualifying kernels at the final observed attempt and
is slightly ahead at best `p=1.5`.

## 5. Matched trajectories

These are iteration-aligned, not wall-time-aligned. They use the common
iteration range and a stride of five.

| Iteration | T best geomean | M best geomean | T `fast_p_best@1.0` | M `fast_p_best@1.0` |
|---:|---:|---:|---:|---:|
| 1 | 0.9950 | 1.1778 | 0.280 | 0.380 |
| 5 | 1.1937 | 1.3124 | 0.600 | 0.680 |
| 10 | 1.3211 | 1.4791 | 0.660 | 0.740 |
| 15 | 1.3623 | 1.4086 | 0.740 | 0.760 |
| 20 | 1.4332 | 1.4140 | 0.760 | 0.780 |
| 25 | 1.4503 | 1.4464 | 0.780 | 0.780 |
| 30 | 1.4544 | 1.4545 | 0.780 | 0.780 |

Direct observation: M leads `fast_p_best@1.0` through iteration 20, then T
catches up at iteration 25. The best-geomean trajectories also converge, with
their ordering reversing around iteration 20.

Interpretation: the Markov run found threshold-crossing candidates earlier,
but T eventually reached the same count. The geomean path is weaker evidence
because its sticky-hack/correctness-selected `n` changes with iteration.

## 6. Feature-level evidence

### Calls, tokens, and evolving-report overhead

| Evidence | T | M |
|---|---:|---:|
| chat turns / calls | 3,169 | 5,349 |
| prompt tokens | 55,047,736 | 38,702,022 |
| completion tokens | 5,879,343 | 8,697,298 |
| reported total tokens | 60,927,079 | 47,399,320 |
| action-selector calls | 1,025 | 1,292 |
| coder calls | 921 | 1,226 |
| extractor calls | 1,010 | 1,285 |
| summarizer calls | 213 | 251 |
| evolving-report calls | 0 | 1,295 |
| evolving-report reported tokens | 0 | 5,658,262 |
| evolving-report output characters | 0 | 4,499,135 |

M made 2,180 more calls while reporting 13,527,759 fewer total tokens. Its
prompts used 16,345,714 fewer tokens, while completions used 2,817,955 more.
The Markov total omits four calls with missing total-token usage (three
evolving-report calls and one summarizer call), whereas T has no missing token
usage. The extractor therefore supports a prompt-compression observation, not
an exact cost comparison.

For M, evolving reports averaged 3,474.235521 output characters over 1,295
iterations. The final report in each workspace averaged 3,779.58 characters
over 50 workspaces.

### Actions, L0, and L1

| Evidence | T | M |
|---|---:|---:|
| `refine_current` | 592 | 845 |
| `propose_new` | 254 | 279 |
| `debug_current` | 179 | 168 |
| action parse errors | 0 | 0 |
| final L0 entries/workspace, mean | 20.5 | 25.9 |
| final L0 entries, sum | 1,025 | 1,295 |
| active L1 entries | 213 | 251 |
| refinement / deletion / merge events | 0 / 0 / 0 | 0 / 0 / 0 |

The L0 totals equal the observed metric-row totals: they primarily measure how
many attempts survived, not independent memory quality. M is more
refinement-heavy in both count and share. Its L1 catalog has 38 more entries,
but all governance features are disabled and every entry remains active.
Because L1 is sequentially shared within a run, catalog size is both an
outcome and a source of downstream coupling.

### Compilation, correctness, errors, and hacks by observed attempt

| Evidence | T | M |
|---|---:|---:|
| observed metric rows | 1,025 | 1,295 |
| compiled | 897 (0.875122) | 1,187 (0.916602) |
| correct | 808 (0.788293) | 1,081 (0.834749) |
| error rows | 211 | 205 |
| timeout | 104 | 69 |
| output mismatch | 82 | 91 |
| compilation error | 10 | 18 |
| CUDA runtime error | 6 | 5 |
| other | 6 | 20 |
| out of memory | 1 | 1 |
| shape/dtype error | 2 | 0 |
| load/link error | 0 | 1 |
| hack rows | 50 (0.048780) | 71 (0.054826) |

M generated 270 more metric-bearing attempts, with higher observed compile
and correctness rates and fewer timeout rows. It also had more output
mismatches, compilation/other errors, and hack flags. The error categories are
the extractor's heuristic taxonomy. A row-level correctness advantage can
coexist with lower run-level correctness because run-level correctness asks
whether each problem ever acquired an accepted best, while endpoint failures
can terminate individual workspaces early.

## 7. Deterministic matched case studies

The extractor matched all 50 workspaces: one Markov correctness gain, three
losses, and 46 unchanged. The five candidates below are selected by embedded
deterministic rules, not by manual cherry-picking.

| Extractor category | Workspace | T best | M best | Direct result |
|---|---|---:|---:|---|
| largest valid improvement | `level_3_problem_3` | 1.173184 at i14 | 3.360000 at i16 | M +2.186816 |
| largest valid regression | `level_3_problem_24` | 5.729350 at i26 | 0.964497 at i29 | M −4.764853 |
| correctness gain | `level_2_problem_13` | no accepted best; sticky hack | correct best, sticky hack | M gain, no valid speedup delta |
| correctness loss | `level_1_problem_56` | 1.243243 at i19 | no compiled attempt | M loss |
| representative no-change | `level_2_problem_19` | 1.569405 at i25 | 1.569405 at i13 | exact recorded tie |

### 7.1 Largest improvement: `level_3_problem_3`

Direct observations:

- T's `metrics_by_iteration.jsonl` iteration 14 records a correct, non-hack
  1.173184 result at 3.58 ms. Its snapshot says three prior rounds had stalled
  and the action selector chose a fresh GEMM/ReLU strategy.
- M's iteration 16 records a correct, non-hack 3.36 result at 1.25 ms.
  `chat_history.jsonl` iteration 16 shows the action selector consuming the
  evolving report, choosing `refine_current`, and preserving a 1.30 ms
  cuBLASLt path. The iteration-16 evolving-report call records that expanding
  zero-workspace heuristic coverage and repeated candidate timing improved the
  best to 1.25 ms.
- The corresponding snapshots are
  `workspaces/level_3_problem_3/iteration_snapshots.jsonl`, T iteration 14 and
  M iterations 15–16; the metric locators are the same workspace's
  `metrics_by_iteration.jsonl`.

Interpretation — **hypothesis:** the bounded report preserved a multi-round
cuBLASLt optimization state well enough to support a productive refinement.
This is a plausible mechanism for this problem, not proof of a general mode
effect.

### 7.2 Largest regression: `level_3_problem_24`

Direct observations:

- T iteration 26 is correct and non-hack at 0.569 ms, a 5.729350 speedup. It
  carries a non-fatal `stream_injection` static warning. The snapshot says the
  action retained a CUDA-graph/native-semantics core.
- M iteration 29 is correct and non-hack at 3.38 ms, a 0.964497 speedup.
  Its action selector uses the evolving report to debug an incomplete
  manually expanded MBConv path; the iteration-29 report correctly records
  that restoring omitted layers fixed correctness and produced the run's best.
  It never reaches T's graph-level result.
- Sources are
  `workspaces/level_3_problem_24/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl}`,
  at T iteration 26 and M iteration 29.

Interpretation — **hypothesis:** Markov's report was useful for local debugging
but may also have anchored later work to a slower strategy family. The
alternative explanation is ordinary model/timing randomness; one case cannot
separate them.

### 7.3 Correctness boundary and resumed budget:
`level_2_problem_13`

Direct observations:

- T iteration 30 is compiled and numerically correct at 1.84 ms, but it is
  hack-flagged for PyTorch wrapper/computation use. The workspace has 27
  correct rows and all 27 are hack-flagged; its three non-hack rows fail.
  Consequently, `run_finished.json` reports no accepted best.
- M iteration 11 is correct and non-hack at 4.35 ms. The historical snapshot
  and `run_finished.json` record 8.735632 using a 38.0 ms reference; the
  current raw metrics row records 8.689655 using the CPU4 baseline value
  37.8 ms. The extractor case uses the `run_finished` value.
- M iteration 14 is hack-flagged. Later clean iterations retain
  `metrics_best.is_hack=true`, demonstrating the sticky flag.
- M iteration 25 ends with the endpoint's budget-exceeded error. The workspace
  remains finished with its earlier correct best and is part of the first 12
  workspaces retained when the run resumes at problem index 13.
- Sources are
  `workspaces/level_2_problem_13/{chat_history.jsonl,iteration_snapshots.jsonl,metrics_by_iteration.jsonl,run_finished.json}`,
  especially M iterations 11, 14–15, and 25, plus T iteration 30.

Interpretation: the extractor's “correctness gain” is real under
`run_finished.metadata.best_correct`, but it is entangled with hack policy,
reference-runtime provenance, and the resume boundary. It is not clean causal
evidence for Markov.

### 7.4 Correctness loss and no-change controls

- `level_1_problem_56`: M's only three metric rows are all coder API timeouts;
  no code compiles. T finds a correct non-hack 1.243243 best at iteration 19.
  This selected correctness loss is directly attributable to observed endpoint
  failure, not an observed bad kernel strategy.
- `level_2_problem_19`: both modes record the same 35.3 ms runtime and
  1.569405 speedup, at T iteration 25 and M iteration 13. Both snapshots
  describe lowered convolution plus fused GELU/GroupNorm work. This is a useful
  local tie, but it does not erase the opposite extremes above.

## 8. Advantages and disadvantages observed in these runs

### Markov report

Advantages:

- reached the final `fast_p_best@1.0` level earlier;
- ended with much stronger current fast-p across every reported threshold;
- produced more observed attempts with higher row-level compile/correct rates;
- used fewer reported prompt and total tokens despite more calls;
- supplied a compact, explicit optimization state in inspected chats;
- reduced timeout rows and enabled the strongest selected improvement.

Disadvantages:

- finished two fewer problems correct;
- did not improve the final headline best fast-p;
- incurred 1,295 extra report calls and 5.66 million reported report tokens;
- accumulated more wall time, L0/L1 entries, hack rows, and hack-affected
  problems;
- suffered a budget-exceeded stop before resume;
- produced the largest selected per-problem regression as well as the largest
  improvement.

### Truncation

Advantages:

- higher final correctness and low-threshold best coverage;
- lower cumulative wall time and substantially fewer calls;
- fewer hack rows and a smaller sequential L1 catalog;
- found the strongest selected result on `level_3_problem_24`.

Disadvantages:

- much larger prompt-token use;
- fewer observed iterations and more timeout rows;
- weak final-current retention: only 13/50 remain above 1.0 at the last
  observed attempt;
- the selected `level_2_problem_13` never acquires an accepted best because
  all correct attempts trip hack policy.

## 9. Plausible mechanisms — hypotheses, not findings

1. **Compression/continuity hypothesis.** Markov's report replaces repeated
   long history with a compact state, explaining fewer prompt tokens and
   allowing more calls/iterations before endpoint failure. The extra attempts
   could improve final-current robustness.
2. **Refinement-bias hypothesis.** M's action mix is more refinement-heavy.
   The report may preserve productive state, as in `level_3_problem_3`, but may
   also entrench a locally coherent, slower family, as plausibly seen in
   `level_3_problem_24`.
3. **Endpoint-survival hypothesis.** M's current-fast-p advantage may partly
   reflect fewer timeout rows and more complete trajectories, not intrinsically
   faster generated kernels.
4. **Exposure hypothesis.** More surviving attempts create more opportunities
   both to find a good kernel and to trigger static/hack rules. M's higher hack
   count is therefore partly exposure-dependent.
5. **Sequential-memory hypothesis.** M's 38 additional L1 entries may help
   later problems, pollute them, or simply reflect more completed rounds.
   Because problem order and L1 are coupled, the observed difference cannot be
   assigned uniquely to L0 context management.
6. **Threshold-saturation hypothesis.** Both runs saturate at 39/50 above
   1.0, concealing different problem identities and margins around the
   threshold. Equal fast-p can coexist with different current state and
   per-problem extremes.

## 10. Wall-time interpretation

The reported cumulative totals are useful scheduling facts but poor treatment
effects:

- T: 230,637.137 seconds = 64.0659 hours.
- M: 276,088.216 seconds = 76.6912 hours.
- Recorded `avg_wall_time_min` is 91.5227 for T and 121.0913 for M, but these
  are exactly cumulative total divided by `problems_timed_this_session`
  (42 and 38), not by all 50 completed problems.
- `batch_timing.jsonl` has 92 rows for T and 88 for M because resumed attempts
  coexist with prior-session rows. Row status counts are not completion counts.
- The final resumed windows overlap from 2026-08-07 13:52 UTC until
  2026-08-08 17:49 UTC. Host/GPU contention is not controlled in the artifacts.
- Both modes call the same remote inference endpoint; request latency, budget,
  serving drift, and rate limits are not controlled. M also deliberately adds
  one report call per observed iteration.

A mode-cost claim requires serialized repeated runs on an idle, fixed endpoint
with fresh timing accounting.

## 11. Limitations

- `n=1` per mode; no confidence interval for run-to-run variability.
- No endpoint or LLM sampling seed is recorded.
- The runs are resumed and contain different endpoint-failure histories.
- Problem order is fixed and sequential L1 makes observations dependent.
- Fast-p values depend on one CPU4/A6000 baseline file.
- Historical snapshots can retain a different reference runtime from the
  current baseline. The 38.0 versus 37.8 discrepancy in M
  `level_2_problem_13` is directly observed.
- The sticky best-hack flag changes aggregate inclusion after any flagged
  iteration; best geomean is not a fixed-sample estimator.
- Hack flags are policy/static-check outcomes, not proof of malicious behavior.
- The feature error taxonomy is heuristic, and token totals depend on endpoint
  usage fields; four M calls have missing totals.
- The extractor emits nine missing-optional-artifact warnings per run because
  all governance sidecars are absent. Governance is disabled, so these are
  expected availability warnings, not evidence of malformed core artifacts.
- Deterministic case candidates are post-hoc descriptive examples. They bound
  cherry-picking but do not make case-level mechanisms causal.
- The pending folding run is not comparable until complete and re-extracted.

## 12. Reproducibility and provenance

Primary generated evidence:

- `aggregate_runs.json` and `aggregate_runs.csv`, generated
  2026-08-10 06:49:07 UTC;
- `comparison.md`, generated 2026-08-10 06:49:35 UTC;
- `feature_evidence.json` and `feature_evidence.csv`, generated
  2026-08-10 07:05:54 UTC.

Result evidence comes only from the two completed run directories under
`runs_evolving/inference_gpt_56_terra`, including `run_summary.json`,
`batch_timing.jsonl`, `shared_l1.jsonl`, cached performance statistics, and
per-workspace chat, metric, snapshot, and finish artifacts. The folding
directory was inspected only for the explicitly labeled pending-status
snapshot. Exact paths, extractor caveats, and reproducible commands with
explicit run root and baseline are in [MANIFEST.md](MANIFEST.md).

Reproduction must preserve the exact baseline file and run names. Rebuilding a
baseline, mixing hardware, or substituting GPT-OSS changes the estimand.
