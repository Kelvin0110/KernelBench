# GPT-OSS-120B inference — design variants on CPU6/A6000

Status: **2026-08-16**. Eight completed inference runs under
`runs_evolving/inference_oss_120b/`, scored against the native CPU6 baseline.
Rules: [ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

## Decision summary

- **Headline (`fast_p_best@1.0` at iteration 30):** truncation (T0) still
  leads at **0.72**. Selective retention is closest at **0.70**. Every
  governance arm and both other L0 modes finish below T0 on this metric.
- **Correctness (`fast_p_best@0` / `total_correct`):** Markov is the only
  50/50 run (`1.00`). T0, deletion, and merge-only are 49/50 (`0.98`).
  Refinement is weakest at 47/50 (`0.94`).
- **High bar (`fast_p_best@2.0` at iteration 30):** combined
  deletion+merge+refine leads at **0.26**, then T0 **0.24**, folding **0.22**.
  Markov is last at **0.12**.
- **Best-speedup geomean at iteration 30:** combined governance **1.3966
  (n=36)** slightly above T0 **1.3855 (n=41)**. These are different selected
  subsets; T0 still wins the full-denominator fast-p@1 ranking.
- **Current retention:** Markov has the strongest `fast_p_current@1.0`
  (**0.50**). T0 and selective tie at 0.46. Refinement is weakest at 0.30.
- There is **no metric-independent winner**. Truncation is the speed-coverage
  control; Markov trades speed for perfect correctness and better current
  kernels; combined governance is the only arm that beats T0 at the 2.0 bar.

## 1. Required checkpoints: iterations 10 and 30

Native baseline:
`results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`.
`@0/@1/@2` are `fast_p_best`. Geomean is `speedup_best.geometric_mean`.

| design | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean (n) | I30 @0 | I30 @1 | I30 @2 | I30 geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (T0) | 49/50 | 0.880 | **0.520** | 0.140 | 1.1700 (39) | 0.980 | **0.720** | 0.240 | 1.3855 (41) |
| selective_retention | 48/50 | 0.920 | **0.540** | 0.080 | 1.0654 (39) | 0.960 | 0.700 | 0.180 | 1.2859 (37) |
| truncation+deletion+merge@0.7+refine | 48/50 | 0.820 | 0.460 | **0.200** | 1.1531 (36) | 0.960 | 0.660 | **0.260** | **1.3966 (36)** |
| truncation+deletion | 49/50 | 0.800 | 0.380 | 0.060 | 0.8606 (34) | 0.980 | 0.640 | 0.180 | 1.2518 (35) |
| truncation+merge@0.7 | 49/50 | 0.880 | 0.460 | 0.120 | 1.0173 (36) | 0.980 | 0.640 | 0.160 | 1.2387 (35) |
| truncation+refine | 47/50 | 0.840 | 0.520 | 0.100 | 1.1061 (36) | 0.940 | 0.620 | 0.140 | 1.2333 (32) |
| folding | 48/50 | **0.940** | 0.500 | 0.160 | 1.0169 (45) | 0.960 | 0.600 | 0.220 | 1.2243 (38) |
| markov_report | **50/50** | **0.960** | 0.460 | 0.080 | 0.8886 (42) | **1.000** | 0.600 | 0.120 | 1.0302 (36) |

The same table is in [comparison.md](comparison.md).

`fast_p_best@1.0` trajectory (iterations 1 / 5 / 10 / 15 / 20 / 25 / 30):

| design | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | 0.02 | 0.34 | 0.52 | 0.58 | 0.66 | 0.68 | **0.72** |
| selective_retention | 0.02 | 0.38 | **0.54** | **0.64** | 0.66 | 0.68 | 0.70 |
| all-gov | 0.14 | **0.42** | 0.46 | 0.52 | 0.56 | 0.62 | 0.66 |
| deletion | 0.04 | 0.28 | 0.38 | 0.52 | 0.56 | 0.60 | 0.64 |
| merge@0.7 | 0.04 | 0.32 | 0.46 | 0.56 | 0.60 | 0.62 | 0.64 |
| refine | 0.04 | 0.26 | 0.52 | 0.58 | 0.60 | 0.62 | 0.62 |
| folding | 0.10 | 0.42 | 0.50 | 0.54 | 0.58 | 0.60 | 0.60 |
| markov_report | 0.06 | 0.34 | 0.46 | 0.50 | 0.56 | 0.60 | 0.60 |

Observation: selective leads T0 at iterations 5–15, then T0 overtakes after
iteration 20. Markov and folding saturate at 0.60.

## 2. Design, aliases, and held-constant settings

| Alias | Design | Exact run name |
|---|---|---|
| T0 | truncation | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` |
| Del | truncation+deletion | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` |
| Ref | truncation+refine | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` |
| Merge | truncation+merge@0.7 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` |
| Sel | selective_retention | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` |
| Markov | markov_report | `base_agent_oss120b_markov_itr30_2026_08_07_14_07` |
| Fold | folding | `base_agent_oss120b_folding_itr30_2026_08_09_13_47` |
| AllGov | truncation+deletion+merge@0.7+refine | `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48` |

Held constant: 50-problem subset, 30 iterations, `gpt-oss-120b`, inference
endpoint, RTX A6000 eval, CPU6 torch baseline, static checking on, per-run
shared L1. All eight summaries record `resume=true`.

## 3. Reason for each variant

| Design | What actually changed |
|---|---|
| truncation | Control. Raw L0 history, truncated to the context window. No skill GC. |
| selective_retention | Keep a recent L0 window; 886 `milestone_judge` calls. Prompt tokens are the highest in the series (172.4M). |
| folding | Pack L0 by folding older rounds; 1,500 `l0_round_summarizer` and 1,209 `preflight` calls. Highest total tokens (189.2M) and calls (7,727). |
| markov_report | Replace raw history with an evolving report. 1,500 report calls / 6.62M report tokens. Lowest total tokens (51.7M). |
| truncation+deletion | Delete unused L1 skills. Active catalog 576→28 (548 deleted). |
| truncation+merge@0.7 | Merge similar L1 skills (72 merges). Active catalog 596→225. |
| truncation+refine | Diagnose/revise L1 skills (90 refinements; 1,189 diagnosis + 87 revision calls). |
| all-gov | Deletion + merge@0.7 + refine together. Active catalog 634→17 (450 deleted, 29 merges, 76 refinements). Longest wall time (109.8 h). |

## 4. Possible root causes (hypotheses)

1. **Truncation keeps late-run search context.** T0 continues to pick up
   `fast_p_best@1` after iteration 20 (0.66→0.72) while folding and Markov
   flatten. Hypothesis: discarding or compressing history removes the traces
   that later `propose_new` / `refine_current` steps need for a 1.0-bar
   kernel.
2. **Selective is sample-efficient early, expensive later.** It leads T0 at
   iteration 10 (0.54 vs 0.52) and iteration 15 (0.64 vs 0.58), then stalls.
   Hypothesis: the recent window helps while the useful context still fits;
   once the window rotates, specialized earlier work is gone. Highest prompt
   tokens are consistent with carrying large remaining chunks.
3. **Markov preserves a current kernel, not a fast one.** Current@1 = 0.50 is
   the series best, but best@1 = 0.60 and geomean = 1.03 are the worst speed
   profile. Hypothesis: the report biases toward `debug_current` (492 vs T0's
   390) and a locally coherent slower family. The L1P54 case (below) is the
   extreme.
4. **Deletion/merge compress L1 but drop specialized kernels.** Deletion
   matches T0 correctness (49/50) while cutting active L1 to 28, and loses
   0.08 on best@1. Merge is similar. Hypothesis: GC removes both junk and
   the rare high-speed skill. The L1P100 regression (6.67→1.95) is consistent
   with losing a specialized path.
5. **Combined governance helps the 2.0 tail, not the 1.0 bulk.** AllGov is
   the only arm above T0 at best@2 (0.26 vs 0.24) and geomean (1.3966 vs
   1.3855, different `n`). It is below T0 at best@1 (0.66). Hypothesis:
   aggressive catalog rewrite occasionally unlocks an outlier kernel
   (L3P49) while disrupting the median 1.0-bar search.
6. **Refinement adds work without an aggregate 1.0 gain.** 90 refinements,
   extra diagnosis calls, 47/50 correct, best@1 = 0.62, current@1 = 0.30.
   Hypothesis: revision churn increases hack exposure (17 problems, highest)
   and final-current collapse.

These are hypotheses. `n=1`, sequential L1, and resumes prevent causal
attribution.

## 5. Key insights

1. On OSS-120B inference, **turning on a context or governance feature did
   not beat truncation on `fast_p_best@1.0`.** The closest is selective
   (−0.02).
2. **Iteration 10 is not a sufficient proxy for iteration 30.** Selective
   wins I10 @1 (0.54) and loses I30 @1 (0.70 vs 0.72). Folding looks strong
   on I10 @0 (0.94) and finishes tied for last on I30 @1.
3. **Perfect correctness can coexist with weak speed.** Markov’s 50/50 is
   real; its 1.03 geomean and 0.12 best@2 say the accepted kernels are
   mostly modest relative to the CPU6 torch reference.
4. **Current vs best is a retention metric, not a scoring bug.** Markov
   keeps 0.50 of problems above 1.0 at the last attempt; refinement keeps
   0.30. Ranking on best@1 alone hides that split.
5. **Geomean without `n` would mis-rank AllGov over T0.** AllGov’s 1.3966
   uses 36 problems; T0’s 1.3855 uses 41. Fast-p keeps the 50-problem
   denominator and still prefers T0 at 1.0.

## 6. Case studies (deterministic extractor vs T0)

Matched 50/50 workspaces for every arm. Candidates are descriptive, not
causal. Artifact locators are `workspaces/<ws>/metrics_by_iteration.jsonl`.

### 6.1 Markov vs T0 — the speed/correctness split

| category | workspace | T0 | Markov | Direct result |
|---|---|---:|---:|---|
| correctness gain | `level_1_problem_56` | no accepted best; final output mismatch | 1.167 at i22 (11.4 ms) | Markov gain; T0 never clears correctness |
| largest valid improvement | `level_2_problem_51` | 1.048 at i30 (14.6 ms) | **4.857** at i29 (3.15 ms) | Markov +3.809 |
| largest valid regression | `level_1_problem_54` | **2.959** at i24 (2.2 ms) | 0.157 at i30 (41.4 ms) | Markov −2.802 |
| representative tie | `level_1_problem_100` | 6.667 at i30 | 6.667 at i28 | exact recorded tie |

**Insight from this pair.** Markov can find a much faster kernel
(`level_2_problem_51`) and can also collapse a 3× kernel to 0.16×
(`level_1_problem_54`). The report is not uniformly helpful. The 50/50
correctness gain is a single T0 failure (`level_1_problem_56`) that every
other completed arm also recovered — so Markov’s correctness lead is that
one T0 miss plus no additional losses.

**Possible root cause (hypothesis):** on `level_1_problem_54` the evolving
report anchored later debug/refine work on a slow path (41.4 ms vs T0’s
2.2 ms). On `level_2_problem_51` the same continuity preserved a productive
refinement. One problem each way is the expected extractor pair, not a
treatment estimate.

### 6.2 Selective vs T0 — early lead, late stall

| category | workspace | T0 | Selective | Direct result |
|---|---|---:|---:|---|
| correctness gain | `level_1_problem_56` | no accepted best | 1.177 at i10 | Selective gain |
| correctness loss | `level_3_problem_50` | **2.691** at i27 | no accepted best; final compile error (`mask_relu_ext`) | Selective loss of a 2.7× kernel |
| largest improvement | `level_3_problem_49` | 1.335 at i10 | **3.766** at i22 | Selective +2.431 |
| largest regression | `level_1_problem_54` | 2.959 at i24 | 1.240 at i12 | Selective −1.719 |

**Insight.** Selective both creates a 3.8× kernel and drops T0’s 2.7×
`level_3_problem_50` to a compile failure. Net best@1 is only 0.02 behind
T0 because these extremes mostly cancel in the 50-problem count.

### 6.3 Combined governance vs T0 — 2.0-bar tail

| category | workspace | T0 | AllGov | Direct result |
|---|---|---:|---:|---|
| correctness gain | `level_1_problem_56` | no accepted best | sticky-hack best at i20 | gain on `run_finished`, not a valid geomean sample |
| correctness loss | `level_3_problem_46` | 1.216 at i22 | no accepted best; output mismatch | AllGov loss |
| largest improvement | `level_3_problem_49` | 1.335 at i10 | 2.530 at i10 | AllGov +1.194 |
| largest regression | `level_3_problem_50` | 2.691 at i27 | 1.174 at i26 | AllGov −1.517 |

**Insight.** The 2.0-bar lead is compatible with losing a 2.7× kernel and
gaining a 2.5× one. Combined GC rewrites the catalog (active L1 17) and
does not dominate the 1.0 bulk.

### 6.4 Deletion vs T0 — catalog compression with a specialized-kernel loss

| category | workspace | T0 | Deletion | Direct result |
|---|---|---:|---:|---|
| largest regression | `level_1_problem_100` | **6.667** at i30 (6.57 ms) | 1.947 at i27 (22.5 ms) | Deletion −4.72 |
| largest improvement | `level_2_problem_32` | 1.286 at i8 | 1.791 at i27 | Deletion +0.505 |
| correctness swap | `level_1_problem_56` / `level_3_problem_28` | miss / 0.755 | 1.157 / miss | one-for-one |

**Insight.** Deletion’s headline 0.64 vs 0.72 is not a small uniform tax.
The largest selected regression is a 6.7× kernel falling to 1.9×. That is
the kind of specialized L1 skill GC can delete.

## 7. Operational profile (not treatment effects)

| design | wall h | calls | total tokens | current@1 | active L1 | problems with hack |
|---|---:|---:|---:|---:|---:|---:|
| truncation | 66.19 | 5,042 | 129.2M | 0.46 | 561 | 8 |
| deletion | 79.55 | 5,055 | 127.7M | 0.38 | 28 | 14 |
| refine | 73.14 | 6,214 | 138.2M | 0.30 | 569 | 17 |
| merge@0.7 | 90.00 | 4,941 | 134.7M | 0.38 | 225 | 15 |
| selective | 86.85 | 5,610 | 177.6M | 0.46 | 457 | 12 |
| markov | 84.07 | 6,316 | **51.7M** | **0.50** | 377 | 14 |
| folding | 90.08 | **7,727** | **189.2M** | 0.36 | 601 | 11 |
| all-gov | **109.78** | 6,076 | 131.6M | 0.34 | **17** | 12 |

All eight runs were resumed. Wall time includes endpoint latency and host
contention. Tokens are endpoint-reported.

T0 error rows are dominated by output mismatch (302) and compilation (107).
Feature arms do not remove that pattern; selective has even more mismatches
(352).

## 8. Limitations

- `n=1` per design. No confidence interval.
- Sequential shared L1 couples problems inside a run.
- Every run is a resume.
- Sticky `metrics_best.is_hack` changes geomean `n`.
- Fast-p@0 is coverage of speedup ≥ 0, which tracks `total_correct` at
  iteration 30 here but is not identical to `run_summary.total_correct` at
  earlier checkpoints.
- Case studies are deterministic extractor picks.

## 9. Provenance

See [MANIFEST.md](MANIFEST.md). Aggregates, `comparison.md`, and
`feature_evidence.{json,csv}` were regenerated 2026-08-16 against the CPU6
baseline. Source run caches were rebuilt only because artifacts were newer
than the previous `performance_stats.json` (same CPU6 file, not a foreign
baseline).
