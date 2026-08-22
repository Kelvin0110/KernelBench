# GPT-OSS-120B on GH200x2 — design variants, post-NVCC-fix

Status: **2026-08-22**. **Ten completed runs** covering **eight settings** under
`runs_evolving/gpt-oss-120b/`, scored against the native GH200 baseline. Six
further replicate runs (launched 2026-08-20) are still in flight and excluded.
Rules: [ANALYSIS_RULES.md](../../ANALYSIS_RULES.md). Provenance:
[MANIFEST.md](MANIFEST.md).

All `visualizations/performance_stats.json` were regenerated on 2026-08-22 with
`Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py`
(`--all-runs`, native GH200 baseline). See §8 for the two values this moved.

## Decision summary

- **`truncation+deletion` wins on three of four metrics** at iteration 30:
  `fast_p_best@1.0` **0.54**, `fast_p_best@2.0` **0.28**, geomean **1.2312
  (n=48)**. It is also the only setting whose geomean clears **1.0** — i.e. the
  only one whose median retained kernel beats GH200 torch.
- **Read that win with the confound in §5.** This arm is really
  *deletion + unit-test GC*: 326 of its 567 deletions (58%) are `unit_test_fail`,
  a path that is always on regardless of the flag. The two mechanisms cannot be
  separated from this run.
- **Correctness (`fast_p_best@0`) is a four-way tie at 48/50 (0.96)** —
  markov_report, selective_retention, compress_trigger, deletion.
  `trunc+refinement` is last at **45/50 (0.90)**.
- **Among pure context-management arms the field is flat**: truncation, markov,
  and selective all land at **0.46** on `fast_p_best@1.0`; folding 0.42,
  compress 0.40. Their geomeans span 0.7265–0.9541, all **below 1.0**.
- **Replicate spread is large and bounds every single-run claim.** The three
  `merge@0.8` replicates — identical config, same host — return I30 geomeans of
  **0.8379 / 0.8552 / 1.0919** (a 30% spread) and `fast_p_best@1.0` of
  0.42 / 0.40 / 0.46. Only deletion's margin over the control clearly exceeds
  that band.
- **Iteration 10 does not predict iteration 30.** markov leads correctness at
  I10 and ties at I30; selective is joint-last on `fast_p_best@1.0` at I10
  (0.32) and joint-first at I30 (0.46).

## 1. Required checkpoints: iterations 10 and 30

Native baseline: `results/timing/NVIDIA_GH200x2/baseline_time_torch.json`.
`@0/@1/@2` are `fast_p_best`. Geomean is `speedup_best.geometric_mean`; `n`
equals `total_correct` (ANALYSIS_RULES §4). `r` is the number of completed
replicates; multi-replicate rows are **averages** (fast-p arithmetic, geomean in
log space).

| design | r | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean (n) | I30 @0 | I30 @1 | I30 @2 | I30 geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (control) | 1 | 47/50 | 0.840 | 0.380 | **0.120** | 0.7379 (42) | 0.940 | 0.460 | 0.180 | 0.9051 (47) |
| markov_report | 1 | 48/50 | **0.900** | 0.340 | 0.040 | 0.7145 (45) | **0.960** | 0.460 | 0.140 | 0.9332 (48) |
| folding | 1 | 47/50 | 0.840 | 0.320 | **0.120** | 0.6744 (42) | 0.940 | 0.420 | 0.180 | 0.8938 (47) |
| selective_retention | 1 | 48/50 | **0.900** | 0.320 | 0.080 | 0.7657 (45) | **0.960** | 0.460 | 0.160 | 0.9541 (48) |
| compress_trigger | 1 | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 (42) | **0.960** | 0.400 | 0.140 | 0.7265 (48) |
| **truncation+deletion** | 1 | 48/50 | 0.840 | **0.400** | 0.100 | **0.8691 (42)** | **0.960** | **0.540** | **0.280** | **1.2312 (48)** |
| truncation+refinement | 1 | 45/50 | 0.780 | 0.300 | 0.100 | 0.7107 (39) | 0.900 | 0.340 | 0.160 | 0.7971 (45) |
| truncation+merge@0.8 | 3 | 47.0/50 | 0.747 | 0.280 | 0.113 | 0.8123 (37.3) | 0.940 | 0.427 | 0.187 | 0.9215 (47.0) |

`truncation+merge@0.8` per replicate:

| replicate | correct | I10 @0/@1/@2 | I10 geomean (n) | I30 @0/@1/@2 | I30 geomean (n) |
|---|---:|---|---:|---|---:|
| `..._17_29` | 48/50 | 0.740 / 0.260 / 0.120 | 0.7047 (37) | 0.960 / 0.420 / 0.200 | 0.8379 (48) |
| `..._17_32` | 47/50 | 0.700 / 0.280 / 0.080 | 0.8282 (35) | 0.940 / 0.400 / 0.140 | 0.8552 (47) |
| `..._17_35` | 46/50 | 0.800 / 0.300 / 0.140 | 0.9185 (40) | 0.920 / 0.460 / 0.220 | 1.0919 (46) |

The same table is in [comparison.md](comparison.md).

### Per-metric leaders

| metric | iteration 10 | iteration 30 |
|---|---|---|
| correctness (`@0`) | markov, selective — 0.900 | markov, selective, compress, deletion — 0.960 |
| `fast_p_best@1.0` | **deletion 0.400** | **deletion 0.540** |
| `fast_p_best@2.0` | truncation, folding — 0.120 | **deletion 0.280** |
| geomean (best) | **deletion 0.8691** | **deletion 1.2312** |

## 2. Design, aliases, held-constant settings

| Alias | Design | Runs |
|---|---|---|
| G0 | truncation (control) | `..._itr30_GH200_2026_08_07_13_58` |
| GM | markov_report | `..._markov_itr30_GH200_2026_08_07_13_58` |
| GF | folding | `..._folding_itr30_GH200_2026_08_13_12_47` |
| GS | selective_retention (5 rounds) | `..._selective_r5_itr30_GH200_2026_08_11_14_09` |
| GC | compress_trigger | `..._compress_itr30_GH200_2026_08_10_15_22` |
| GD | truncation+deletion | `..._deletion_itr30_GH200_2026_08_14_15_52` |
| GR | truncation+refinement | `..._refinement_itr30_GH200_2026_08_17_15_52` |
| GG×3 | truncation+merge@0.8 | `..._merge_sim08_itr30_GH200_2026_08_19_{17_29,17_32,17_35}` |

Held constant: 50-problem subset, 30 iterations, `gpt-oss-120b`, inference
endpoint, GH200 144G HBM3e eval, native GH200 torch baseline, per-run shared L1.
The five context arms hold governance **off**; the three governance arms hold
context at **truncation**, so the two axes stay separable.

## 3. Reason for each variant

| Design | What actually changed |
|---|---|
| truncation | Control. Raw L0 history truncated to the context window. |
| markov_report | Bounded evolving report replaces raw history. Smallest L1 (366). |
| folding | L0 folding. L1 592. |
| selective_retention | Keep a 5-round recent L0 window. L1 619. |
| compress_trigger | Microcompact older L0 rounds on a token/iteration trigger. L1 435. |
| truncation+deletion | Skill deletion — **plus always-on unit-test GC** (§5). L1 592. |
| truncation+refinement | LLM rewrites/refines L1 skills. L1 703. |
| truncation+merge@0.8 | DBSCAN-cluster + LLM-merge near-duplicate skills at sim 0.8. L1 681–730. |

## 4. Observations at iteration 10 vs 30

`fast_p_best@1.0` trajectory (stride 5; merge row is the 3-replicate mean):

| design | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | 0.08 | 0.26 | 0.38 | 0.40 | 0.42 | 0.42 | 0.46 |
| markov_report | 0.04 | 0.28 | 0.34 | 0.38 | 0.40 | 0.42 | 0.46 |
| selective_retention | 0.04 | 0.20 | 0.32 | 0.38 | 0.42 | 0.46 | 0.46 |
| **truncation+deletion** | 0.06 | 0.30 | **0.40** | **0.44** | **0.48** | **0.52** | **0.54** |

Deletion is ahead from iteration 5 onward and never gives the lead back — the
only arm in this series with a monotone separation from the control. Every other
design converges into the 0.40–0.46 band.

## 5. Possible root causes (hypotheses)

1. **The deletion arm is confounded with unit-test GC, and GC is the dominant
   path.** `l1_skill_deletions.jsonl` records 567 deletions: **326
   `unit_test_fail`** and 241 `consecutive_unused`. Per project memory, the
   `--enable-l1-skill-unit-test-gc` flag is a no-op — `gen3_stages.py` gates GC
   on `cfg.enable_l1_skill_unit_tests` (default `True`) while the batch runner
   only ever sets `enable_l1_skill_unit_test_gc`. `run_summary.json` therefore
   records the flag as `False` while GC ran throughout. **Hypothesis:** the win
   comes mostly from evicting skills that fail their own unit tests — i.e. from
   *catalog quality*, not from usage-based deletion. This is not separable from
   this run and the gate must be fixed before more deletion cells are run.
2. **The GH200 torch baseline is the binding constraint for the context arms.**
   All five context-only geomeans sit at 0.56–0.95 and their `fast_p_best@1.0`
   values span 0.06. **Hypothesis:** on Hopper the cuBLAS/cuDNN paths the
   reference calls are strong enough that hand-written CUDA rarely wins, leaving
   context management little headroom to express itself. Governance, which
   changes *what knowledge is available* rather than *how history is packed*,
   still moves the needle.
3. **Refinement actively hurts.** GR is last on correctness (45/50), last on
   `fast_p_best@1.0` (0.34), and has the second-worst geomean (0.7971), while
   producing the joint-largest catalog (703). **Hypothesis:** LLM rewriting
   inflates the catalog with paraphrases and drifts skills away from what
   actually compiled — the opposite of deletion's effect. Its elevated hack
   exposure (29 iterations over 19 problems) is consistent with a degraded
   catalog pushing the search toward shortcuts.
4. **compress_trigger is the weakest context arm and the most hack-exposed.**
   Worst geomean of the context group (0.7265) and 32 hack iterations across 23
   of 50 problems, roughly double every other arm. **Hypothesis:** over-compaction
   destroys the traces needed to recover from a failed optimization.
5. **Merge is a wash at this threshold.** The 3-replicate mean (0.427 @1,
   geomean 0.9215) straddles the control (0.460, 0.9051) and its spread swallows
   the difference. All three replicates did real work (679–728 embedded skills,
   124–180 merges), so this is a null result, not a silent failure.

These are hypotheses. Apart from merge, `n=1` per configuration prevents causal
attribution.

## 6. Key insights

1. **Skill governance separates on this host where context management does not.**
   The five context arms span 0.06 on the headline metric; deletion alone clears
   the control by 0.08 and is the only run above geomean 1.0. If a single design
   is to be carried forward for OSS-120B on GH200, it is deletion — with the §5
   caveat that what is being carried forward is *deletion + unit-test GC*.
2. **Sub-1.0 geomeans remain the norm.** Seven of eight settings finish below
   1.0: the median retained kernel is *slower* than GH200 torch. Correctness
   stays at 0.90–0.96 throughout. The two metrics are decoupled — high
   correctness on this host says nothing about speed.
3. **The replicate spread is the story for every non-deletion comparison.**
   Identical-config merge replicates differ by 30% in geomean and 0.06 in
   `fast_p_best@1.0`. That band covers *every* pairwise gap among the context
   arms. Those rankings are not resolvable at n=1.
4. **More skills is not better.** Ordering the eight settings by final catalog
   size against geomean: markov (366) → 0.9332, compress (435) → 0.7265,
   truncation (571) → 0.9051, deletion (592) → **1.2312**, selective (619) →
   0.9541, merge (~705) → 0.9215, refinement (703) → 0.7971. The best and worst
   results both come from mid-to-large catalogs; what distinguishes deletion is
   that its catalog is *pruned by evidence*, not that it is small.
5. **Iteration 10 mis-ranks the field.** Deletion is the only setting whose I10
   rank survives to I30. Judging any other arm at I10 inverts its final order.

## 7. Operational profile (not treatment effects)

| design | correct | wall h | L1 active | hack itrs | problems w/ hack | best_overall |
|---|---:|---:|---:|---:|---:|---:|
| truncation | 47/50 | 74.2 | 571 | 16 | 11 | 5.45 |
| markov_report | 48/50 | 71.4 | **366** | **14** | **11** | 6.23 |
| folding | 47/50 | 66.6 | 592 | 17 | 12 | 6.37 |
| selective_retention | 48/50 | 69.9 | 619 | 17 | 12 | **9.76** |
| compress_trigger | 48/50 | 64.2 | 435 | **32** | **23** | 2.96 |
| truncation+deletion | 48/50 | 66.9 | 592 | 19 | 12 | 2.10 |
| truncation+refinement | 45/50 | **53.1** | 703 | 29 | 19 | 7.23 |
| merge@0.8 r1 | 48/50 | 64.9 | 703 | 21 | 16 | 6.08 |
| merge@0.8 r2 | 47/50 | 68.4 | 730 | 31 | 18 | 9.05 |
| merge@0.8 r3 | 46/50 | 65.4 | 681 | 23 | 14 | 6.51 |

Wall time includes endpoint latency and host contention; per CLAUDE.md §7 it
drifts up to 26% across dates and is not comparable across arms as a treatment
effect. `best_speedup_overall` is a selected non-outlier summary, not a maximum.

## 8. Correction versus the 2026-08-16 report

The previous edition of this file reported I30 geomeans of **0.8746** for
truncation and **0.9830** for markov_report. Regeneration with the current
generator gives **0.9051 (n=47)** and **0.9332 (n=48)**. Both new values were
independently recomputed from raw `metrics_by_iteration.jsonl` against the
native GH200 baseline and matched to 1e-6.

The run artifacts did not change (no workspace file is newer than 2026-08-16)
and neither did the baseline file (unmodified since 2026-08-03); the generator
was updated on 2026-08-16. `selective_retention` (0.9541) and `compress_trigger`
(0.7265) are unchanged, so the shift is confined to those two runs. **The
2026-08-16 truncation and markov geomeans should not be quoted.**

That edition also listed folding (49/50) and deletion (30/31) as partial. Both
have since completed at 50/50 and are included here in full.

## 9. Limitations

- `n=1` for seven of eight settings. Only `merge@0.8` has replicates, and its
  spread (§6.3) is wide enough to caution against every other single-run gap.
- **The deletion arm cannot isolate deletion from unit-test GC** (§5.1). Its
  headline win is a win for the *combination*.
- Sequential shared L1 couples problems within a run.
- Six replicate runs (2 × truncation, 2 × markov, 2 × folding, launched
  2026-08-20) are still in flight. When they land, the control and two context
  arms gain replicates and these rankings should be recomputed.
- Native GH200 speedup only. Do **not** rescore onto an A6000 vector to compare
  hosts; compare native relative scores instead
  (see `../model-endpoint-comparison/`).
- The hack detector is heuristic; elevated counts are observations, not
  validated mechanisms.
