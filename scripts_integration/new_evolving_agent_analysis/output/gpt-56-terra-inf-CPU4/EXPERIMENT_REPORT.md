# GPT-5.6 Terra inference — truncation, Markov, compress-trigger on CPU4/A6000

Status: **2026-08-16**. Three completed inference runs under
`runs_evolving/inference_gpt_56_terra/`, scored against the native CPU4
baseline. Folding is partial (15/50) and excluded. Rules:
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

This refresh includes a later truncation/Markov resume relative to the
2026-08-10 report: truncation `fast_p_best@1.0` moved 0.78→**0.82**,
`fast_p_current@1.0` moved 0.26→**0.70**, and Markov correctness moved
47→**49**.

## Decision summary

- **Headline (`fast_p_best@1.0` at iteration 30):** truncation and Markov
  **tie at 0.82** (41/50). Compress-trigger is behind at **0.70**.
- **Correctness:** all three complete runs are **49/50** (`fast_p_best@0 =
  0.98`).
- **High bar (`fast_p_best@2.0`):** Markov and compress-trigger **0.30**,
  truncation **0.26**.
- **Best-speedup geomean at iteration 30:** Markov **1.8153 (n=49)**,
  truncation **1.7796 (n=49)**, compress-trigger **1.6438 (n=49)**. All three
  are the **same 49 problems**, so Markov’s +0.036 edge over truncation is a
  like-for-like comparison, not a smaller-sample artifact.
- **Current retention:** truncation now leads `fast_p_current@1.0` at
  **0.70**, then compress-trigger 0.66, Markov 0.64. This reverses the
  2026-08-10 story, where Markov led current@1 0.50 vs truncation 0.26.
- Compress-trigger is the only **non-resumed** complete Terra run, the
  cheapest in wall time (73.2 h) and L1 size (280), and the weakest on the
  1.0 headline.

## 1. Required checkpoints: iterations 10 and 30

Native baseline:
`results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`.

| design | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean (n) | I30 @0 | I30 @1 | I30 @2 | I30 geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation | 49/50 | 0.960 | 0.700 | 0.200 | **1.5023 (48)** | 0.980 | **0.820** | 0.260 | 1.7796 (49) |
| markov_report | 49/50 | 0.940 | **0.740** | 0.200 | 1.4437 (47) | 0.980 | **0.820** | **0.300** | **1.8153 (49)** |
| compress_trigger | 49/50 | **0.980** | 0.680 | 0.200 | 1.4046 (49) | 0.980 | 0.700 | **0.300** | 1.6438 (49) |

The same table is in [comparison.md](comparison.md). Folding is omitted
because it is partial.

`fast_p_best@1.0` trajectory:

| design | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | 0.26 | 0.66 | 0.70 | **0.78** | 0.80 | **0.82** | **0.82** |
| markov_report | **0.36** | **0.68** | **0.74** | 0.76 | **0.82** | **0.82** | **0.82** |
| compress_trigger | 0.30 | 0.66 | 0.68 | 0.68 | 0.70 | 0.70 | 0.70 |

Observation: Markov leads through iteration 10 and reaches the 0.82 plateau
at iteration 20; truncation catches at iteration 25. Compress-trigger is
essentially done by iteration 10 (0.68) and only adds one more 1.0-bar
problem.

## 2. Design, aliases, and held-constant settings

| Alias | Design | Exact run name | Resume |
|---|---|---|---|
| T | truncation | `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | true |
| M | markov_report | `base_agent_terra_markov_itr30_2026_08_01_17_41` | true |
| C | compress_trigger | `base_agent_terra_compress_trigger_itr30_2026_08_10_15_24` | **false** |

Held constant: 50-problem subset, 30 iterations, `gpt-5.6-terra`, inference
endpoint, RTX A6000 eval, CPU4 torch baseline, static checking on, skill
deletion/merge/refine off, per-run shared L1.

Excluded: `base_agent_terra_folding_itr30_2026_08_09_15_11` (15/50 completed,
18/19 workspaces finished).

## 3. Reason for each variant

| Design | What actually changed |
|---|---|
| truncation | Control. Raw L0 history truncated to the window. 4,776 calls, 114.9M tokens, 0 evolving-report calls. Action mix 952 refine / 356 propose / 190 debug. |
| markov_report | Bounded evolving report replaces raw history. 1,500 report calls / 6.76M report tokens. Total tokens **55.0M** despite 6,220 calls. Action mix 978 refine / 309 propose / 212 debug. |
| compress_trigger | Keep the latest hot L0 rounds in full and microcompact older rounds when packed prompt tokens hit 85% or every 15 iterations. This run used **`compress_hot_rounds=3`** (governor default), not the runbook’s comparison recipe of 15. 100 compression-event rows across 50 workspaces. 4,756 calls, 64.9M tokens. Most refine-heavy mix (1,036 refine / 280 propose / 183 debug). |

## 4. Possible root causes (hypotheses)

1. **Markov is sample-efficient to the 1.0 bar, truncation needs the long
   tail.** Markov is at 0.74 by iteration 10 and 0.82 by iteration 20.
   Truncation is at 0.70 / 0.80 at those same points and only ties at
   iteration 25. Hypothesis: the report carries a usable optimization state
   earlier; truncation eventually reconstructs it from raw history after
   more iterations (including the post-Aug-10 resume, which also repaired
   current@1).
2. **The Aug-10 truncation current-gap was unfinished search, not an
   intrinsic mode defect.** Current@1 0.26→0.70 after resume is too large
   to treat as a stable truncation property. Hypothesis: later iterations
   recovered last-attempt quality that the first session had abandoned.
3. **Compress-trigger with hot=3 over-compacts before the 1.0 search
   finishes.** I10 @1 is already 0.68 and I30 is 0.70. Hypothesis: keeping
   only three hot rounds deletes the traces needed to convert a 0.8-bar
   kernel into a 1.0-bar kernel. The 2.0 bar is not hurt (0.30, tied with
   Markov) — so the mode is not “generally slower,” it is specifically
   thinner at the 1.0 bulk. This run is also the only fresh (non-resume)
   Terra cell, so resume-vs-mode is confounded with truncation/Markov.
4. **`level_3_problem_24` is a shared failure mode of compressed context.**
   Truncation’s valid best is 5.729 at 0.569 ms. Markov’s is 0.964 at
   3.38 ms; compress-trigger’s is 0.803 at 4.06 ms. Hypothesis: once the
   CUDA-graph / native-semantics path is not in the prompt, later debug
   work rebuilds a slower expanded MBConv-style kernel and never returns.
5. **Hack policy still creates fake correctness swaps.** Both Markov and
   compress-trigger “gain” `level_2_problem_13` because truncation’s
   correct attempts are all sticky-hack-flagged, so `run_finished` records
   no accepted best. The extractor gain is real under that predicate and
   is not a clean L0-mode effect.

## 5. Key insights

1. **On Terra, Markov does not lose the headline 1.0 bar to truncation**
   (tie 0.82). That is the opposite of OSS, where Markov finishes at 0.60
   vs truncation 0.72. Cross-model component trend is not the same.
2. **Compress-trigger as launched (hot=3) is not a truncation substitute
   on best@1** (−0.12). It is cheaper (tokens, wall, L1) and matches Markov
   at best@2.
3. **Read iteration 10 and 30 together.** Markov’s I10 @1 lead (0.74 vs
   0.70) is real and gone by I30. Compress-trigger’s I10 @0 lead (0.98)
   does not predict I30 @1.
4. **Geomean and fast-p disagree, and here the geomean is the sharper
   instrument.** Markov 1.815 vs truncation 1.780 is measured over the **same
   49 problems** (both n=49), so it *is* a shared-sample statement: Markov’s
   retained bests are marginally faster in aggregate. Fast-p@1 is tied at 0.82
   because the count of problems crossing the 1.0 bar is identical — the two
   metrics are not in conflict, they answer different questions (how many
   clear the bar vs how fast the kept kernels are).
5. **Do not quote the 2026-08-10 Terra current@1 ranking.** Resume changed
   it.

## 6. Case studies (deterministic extractor vs truncation)

Matched 50/50 workspaces. Locators:
`workspaces/<ws>/metrics_by_iteration.jsonl`.

### 6.1 Markov vs truncation

| category | workspace | T | M | Direct result |
|---|---|---:|---:|---|
| largest valid improvement | `level_3_problem_3` | 1.221 at i16 (3.44 ms) | **3.360** at i16 (1.25 ms) | M +2.139 |
| largest valid regression | `level_3_problem_24` | **5.729** at i26 (0.569 ms) | 0.964 at i29 (3.38 ms) | M −4.765 |
| correctness gain | `level_2_problem_13` | no accepted best; sticky hack; final i30 correct | sticky-hack best 5.642 at i17 | M gain under `run_finished`, not a valid geomean delta |
| correctness loss | `level_2_problem_51` | sticky-hack best 1.124 at i8 | no accepted best; final i30 correct | M loss under the same predicate |
| representative tie | `level_1_problem_56` | 1.232 at i1 (11.2 ms) | 1.232 at i20 (11.2 ms) | exact recorded tie |

**Insight.** The improvement and regression are the same two problems as
the 2026-08-10 audit, with the same runtimes. Markov still both enables a
cuBLASLt-scale GEMM/ReLU win on `level_3_problem_3` and misses truncation’s
graph-level 0.569 ms kernel on `level_3_problem_24`. Correctness is now
49–49 except for this hack-entangled swap (`level_2_problem_13` vs
`level_2_problem_51`).

**Possible root cause (hypothesis):** report continuity helps refinement
when the current family is already fast, and anchors search when it is
not. The `level_3_problem_24` pair is the strongest single illustration in
this series.

### 6.2 Compress-trigger vs truncation

| category | workspace | T | C | Direct result |
|---|---|---:|---:|---|
| largest valid improvement | `level_3_problem_4` | 4.442 at i16 (0.412 ms); final compile error | **5.884** at i26 (0.311 ms); final still correct | C +1.442 |
| largest valid regression | `level_3_problem_24` | **5.729** at i26 | 0.803 at i12 (4.06 ms) | C −4.926 |
| correctness gain | `level_2_problem_13` | no accepted best; sticky hack | sticky-hack best 8.326 at i18 | same predicate issue as Markov |
| correctness loss | `level_2_problem_42` | sticky-hack best 1.937 at i20 | no accepted best; final i30 correct | C loss under the predicate |
| representative tie | `level_1_problem_34` | 1.429 at i22 (27.5 ms) | 1.429 at i18 (27.5 ms) | exact recorded tie |

**Insight.** Compress-trigger can beat truncation on a fused LeNet-style
kernel (`level_3_problem_4`, 0.311 ms) and still lose the same
`level_3_problem_24` 5.7× kernel that Markov lost, by an even larger
margin. The 1.0-bar deficit is not “no fast kernels exist”; it is missing
the bulk of 1.0 conversions, including this one extreme.

## 7. Operational profile (not treatment effects)

| design | wall h | calls | total tokens | report tokens | current@1 | L1 entries | hack problems |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | 104.11 | 4,776 | 114.9M | 0 | **0.70** | 345 | 6 |
| markov_report | 94.71 | 6,220 | **55.0M** | 6.76M | 0.64 | 293 | 11 |
| compress_trigger | **73.25** | 4,756 | 64.9M | 0 | 0.66 | **280** | 7 |

Error rows (heuristic): truncation 21 timeouts / 116 mismatches / 18
compile; Markov 38 timeouts / 141 mismatches / 23 compile; compress-trigger
**2 timeouts** / 131 mismatches / 24 compile. Compress-trigger’s timeout
collapse is an operational observation, not a kernel-quality result.

## 8. Limitations

- `n=1` per design. Compress-trigger is also the only non-resumed cell.
- `compress_hot_rounds=3` is not the hot=15 recipe in
  `RUN_WITH_UV_INFER.md`; do not generalize to that configuration.
- Sequential L1 coupling; CPU4-native speedup only.
- The sticky `run_had_hack` latch still governs the `run_finished`
  accepted-best predicate used by the §6 case studies (the
  `level_2_problem_13` / `level_2_problem_51` swaps). It does **not** gate
  geomean eligibility — see the `n` correction in Provenance.
- Folding is excluded until complete.
- Do not rescore these numbers onto CPU6 for a cross-model ranking. See
  the synthesis report.

## 9. Provenance

See [MANIFEST.md](MANIFEST.md). Aggregates, `comparison.md`, and
`feature_evidence.{json,csv}` were regenerated 2026-08-16 against the CPU4
baseline. Truncation/compress-trigger stats were rebuilt because run
artifacts were newer than the cache. Markov’s cache was still current and
was reused (still CPU4).

**Correction (2026-08-16, second pass).** `aggregate_runs.py:552,601` had been
ANDing `metrics_best.is_hack` into the geomean sample count. That field is the
run-level `run_had_hack` latch, not "this best is a hack", and
`generate_run_performance_stats.py` (module docstring, line 369) forbids using
it as an eligibility gate. Every `n` in this report was therefore understated
by roughly `problems_with_hack`; the corrected values are above. **No geomean,
fast-p, or correctness value changed** — only `n`, and the reasoning that had
been resting on it. `avg_wall_time_min` was also corrected: it had fallen
through to the `total / problems_timed_this_session` fallback the code warns
against.
