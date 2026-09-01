# Shipped L2 gate vs `--redesign-l2` — measured on `runs_evolving/gpt-oss-120b/median`

Generated 2026-09-01. Baseline `results/timing/NVIDIA_GH200x2_median`, uniform
`is_hack` at 30x (`aggregate_runs.py --hack-threshold 30 --regenerate-stats`).

Arms read (all gpt-oss-120b, 50 problems x 30 iterations, `truncation` L0):

| arm | dir | wave | gate |
|---|---|---|---|
| `truncation` (control) | `..._itr30_GH200_2026_08_22_21_07` | Aug-22 | no L2 |
| `l2` | `..._l2_itr30_GH200_2026_08_22_21_32` | Aug-22 | shipped |
| `l2redesign_r1/r2/r3` | `..._l2redesign_r{1,2,3}_..._2026_08_29_17_58` | Aug-29 | `--redesign-l2` |

## 1. The two designs

`L2_REDESIGN_PRESET` (`l2_promotion.py:150`) changes exactly two things. Everything
else — `min_tasks 3`, `min_selections 50`, `min_new_bests 0`, `verbatim` render,
one-way promotion, task-boundary firing, no standing cap — is identical.

| | shipped `--enable-l2` | `--enable-l2 --redesign-l2` |
|---|---|---|
| rate metric | `selection_rate = selections / (global_iter - created_at)` | `hit_rate = selections / total_offers` |
| floor on it | **0.70** | **0.60** |
| redundancy check | none | greedy cosine >= **0.80**, candidate vs standing then vs higher-ranked candidate |
| standing cap | none (`max_entries` is PER-PASS) | none (`-1`, deliberately excluded from the preset) |
| judge | off | off (excluded: worst on quality, one LLM call/pass, cannot dedup) |

`hit_rate` **replaces** `selection_rate` (`passes_floors:305`) rather than adding to
it. `total_offers` is a new ledger counter: the Aug-22 `l2` arm's
`l1_skill_usage.json` has it on **0 of 557** skills, the redesign arms on **538/538**,
so the redesign gate could not have run on a pre-counter ledger at all — it fails
closed to `hit_rate = 0`.

## 2. What each arm actually promoted

| arm | rules | standing text | coder system prompt: mean | terminal | calls with no L2 text |
|---|---|---|---|---|---|
| `truncation` | — | 0 | 4,190 ch (1.00x) | 1.00x | 100% |
| `l2` (shipped) | **0** | 0 | 4,190 ch (**1.00x**) | 1.00x | 100% |
| `l2redesign_r1` | 12 | 20,838 ch | 14,852 ch (3.54x) | **6.20x** | 16% |
| `l2redesign_r2` | 9 | 15,386 ch | 14,405 ch (3.44x) | 4.84x | 8% |
| `l2redesign_r3` | 7 | 12,490 ch | 12,508 ch (2.99x) | 4.13x | 16% |

The shipped arm promoted nothing, so its coder system prompt is a constant 4,190
characters — **byte-identical in size to the control's**. Mechanically it is a
truncation arm. Any `l2`-vs-control difference below is therefore a null contrast.

For scale, CLAUDE.md 8.5's 9-rule arm reached 4.79x terminal / 2.21x run-mean. The
redesign preset is **heavier**, not lighter: it promotes earlier and promotes more.
Dedup removes duplicates; the 0.70 -> 0.60 floor more than adds them back.

## 3. Gate replay — the promotion counts, decomposed

Boundary-by-boundary replay over the exact per-iteration candidate/selected sets
(`l2_redesign/build_visibility_cache.py` + `sweep_gates.py`; validated to reproduce
`l2_promotions.jsonl` exactly). Dedup is not modelled, so these are pre-dedup counts.

| arm | shipped gate (rate>=0.70) | redesign gate (hit>=0.60) | actually promoted |
|---|---|---|---|
| `l2` (Aug-22) | **0** | **19** | 0 |
| `l2redesign_r1` | **7** | 13 | 12 |
| `l2redesign_r2` | **7** | 15 | 9 |
| `l2redesign_r3` | **4** | 10 | 7 |

Two readings, and the second is the one CLAUDE.md 8.13 does not yet record:

- **The zero-promotion failure is arm-specific, not gate-specific.** The shipped gate
  replayed on the three redesign arms promotes 7/7/4, not 0. What produced 0 on the
  Aug-22 arm was that arm's `selection_rate` ceiling (0.7059, CLAUDE.md 8.11), not the
  gate design.
- **On these arms `hit_rate` and `selection_rate` are nearly the same number.** Swept
  at a common floor of 0.70 they give 7/7/4 either way; `total_offers == opportunity`
  for every promoted candidate but two, because the redesign arms promote while the
  skill is still inside the newest-50 extractor tail. The metric swap only bites on
  arms that promote late — which is exactly the Aug-22 arm (0 at rate>=0.70,
  4 at rate>=0.60, 19 at hit>=0.60). **On the redesign arms the working knob is the
  floor value 0.70 -> 0.60, not the change of metric.**

## 4. Dedup — it fires, and it does not deliver the invariant that was claimed

Distinct skills blocked (not pass-events; the same candidate is re-tested every
boundary):

| arm | distinct blocked | families |
|---|---|---|
| `l2redesign_r1` | **0** | — |
| `l2redesign_r2` | 6 | ids 89/90/91/92 vs standing 88 (cuDNN warm-up kernel); 369, 386 vs 104 (avoid trivial kernels) |
| `l2redesign_r3` | 3 | 54, 57 vs 55 (conv2d correctness); 420 vs 418 |

Direct evidence it matters: the shipped-gate replay of `r2` promotes ids **369 and
386** alongside 104 — *"Avoid Trivial Custom Kernels that Degrade Performance"*,
*"Avoid Trivial Custom CUDA Kernels that Undermine Performance"*, *"Avoid No-Op CUDA
Kernels as Performance Hacks"*. That is CLAUDE.md 8.4's duplicate-family defect
reproduced live, and dedup blocked it.

**But the threshold is applied to a different text than the one that reaches the
coder.** `_entry_dedup_text` (`l2_promotion.py:519`) embeds `title + description +
trigger`; the prompt carries `render_verbatim`. Embedding both representations of the
same final standing sets:

| arm | rules | pairs >= 0.80 on **gate text** | max | pairs >= 0.80 on **rendered text** | max |
|---|---|---|---|---|---|
| `l2redesign_r1` | 12 | **0** / 66 | 0.773 | **1** / 66 | 0.805 |
| `l2redesign_r2` | 9 | **0** / 36 | 0.787 | **7** / 36 | 0.842 |
| `l2redesign_r3` | 7 | **0** / 21 | 0.797 | **4** / 21 | 0.863 |

The gate enforces its invariant exactly — and every arm lands 0.773-0.797, i.e.
*packed against* the threshold. On the rendered text the same sets carry 1/7/4
duplicate pairs, because the shared verbatim scaffolding inflates cosine by ~0.03-0.07.

CLAUDE.md 8.12's headline dedup result — `l2_redesign` the sole arm with **0** pairs
>= 0.80, against 2-4 for every other arm — was measured on the rendered text
(`standing_diversity.py` embeds `row["text"]`). **It does not replicate.** Two of
these three replicates sit inside the un-deduped band (19% of pairs vs 13-30% there).
The r1 result (1/66) is the only one that looks like the Aug-27 arm, and r1 is also
the arm where dedup never fired at all.

Fix is cheap and needs no GPU: dedup on the text you render, or drop the threshold to
~0.75 on the current text.

## 5. Quality — a null, in both directions

Lottery problems identified treatment-agnostically (max/min clean best speedup >= 4x
across all **12** completed 50-problem arms in this directory): **12 of 50** —
L1P100, L2P3/19/37/41/42/51/56/94/97, L3P2, L3P34.

| arm | geo (all) | n | geo (adj) | n | fast_p@1.0 | adj | per-problem median (adj) |
|---|---|---|---|---|---|---|---|
| `truncation` (ctl) | 1.325 | 45 | 1.059 | 33 | 0.660 | 0.553 | 1.047 |
| `l2` (shipped, 0 rules) | 1.471 | 48 | 1.144 | 36 | 0.720 | 0.632 | 1.111 |
| `l2redesign_r1` | 1.568 | 48 | 1.164 | 36 | 0.700 | 0.632 | 1.116 |
| `l2redesign_r2` | 1.403 | 47 | 1.195 | 35 | 0.680 | 0.632 | 1.159 |
| `l2redesign_r3` | 1.259 | 46 | 1.229 | 34 | 0.680 | 0.632 | 1.159 |

Paired per-problem log-ratio, 95% CI, on the adjusted set:

| contrast | ratio (all) | ratio (adj) | 95% CI (adj) | n | fast_p win/loss (p) |
|---|---|---|---|---|---|
| `l2` / ctl **(null: 0 rules both sides)** | 1.084 | 1.021 | [0.945, 1.103] | 32 | 6/3 (p=0.51) |
| `r1` / ctl | 1.152 | 1.027 | [0.947, 1.114] | 32 | 7/4 (p=0.55) |
| `r2` / ctl | 1.032 | 1.081 | [1.009, 1.159] | 32 | 6/3 (p=0.51) |
| `r3` / ctl | 0.906 | 1.114 | [1.011, 1.229] | 29 | 8/5 (p=0.58) |
| `r1` / `l2` | 1.070 | 1.021 | [0.952, 1.096] | 35 | 3/3 (p=1.00) |
| `r2` / `l2` | 0.948 | 1.047 | [0.979, 1.121] | 34 | 2/2 (p=1.00) |
| `r3` / `l2` | 0.838 | 1.066 | [0.966, 1.177] | 33 | 2/2 (p=1.00) |

Cell test, redesign (n=3) vs the single shipped arm:

- raw: **0.947x**, replicate SD 0.123, CI [0.698, 1.285]
- adjusted: **1.045x**, replicate SD 0.021, CI [0.990, 1.102]

`fast_p_best@1.0` — the headline metric — is **0.632 for all four L2 arms** on the
adjusted denominator. Zero separation.

Three cautions on those starred CIs. (a) The r2/r3-vs-control CIs that exclude 1.0 are
*paired per problem*, which treats problems as the replication unit and overstates
confidence about a cell (CLAUDE.md 4). (b) The null contrast `l2`/ctl — two arms whose
coder prompts are byte-identical in size and whose only difference is trajectory —
still reads 1.084x raw / 1.021x adjusted with a +0.079 fast_p gap. That is the floor
for any single arm-vs-arm claim here. (c) `r3`'s adjusted ratio is the *highest* of the
three while its raw ratio is the *lowest*, which is the lottery effect, not a signal.

**Replicate noise, a fourth gpt-oss measurement.** The three identical-config redesign
arms: raw geomeans 1.568 / 1.403 / 1.259 (max/min 1.246, log-SD **0.110**, in line with
the 0.147 of open item 10); adjusted 1.164 / 1.195 / 1.229 (max/min 1.056, log-SD
**0.027**). The lottery adjustment cuts replicate noise ~4x, on fresh data.

## 6. What is NOT comparable here

`l2` is from the Aug-22 wave and the redesign arms from Aug-29. Three seams cross that
boundary, all favouring the redesign arms:

1. **Token budgets (CLAUDE.md 3.7, 2026-08-27).** Every LLM completion budget went
   uniform 65,536. `run_summary.json` shows `evolving_report_max_tokens` 1536 (Aug-22)
   vs 65536 (Aug-29); the coder, preflight, extractor and summarizer budgets moved with
   it. Parent-side, so it applies to new runs only.
2. **Contention.** Aug-22 ran 9 gpt-oss arms on GPU0; Aug-29 ran 3 gpt-oss + 3 terra.
   Contention only ever deflates speedup.
3. **Endpoint latency drift** (open item 11).

Also, the Aug-29 wave shipped **no control on its own GPU**, which CLAUDE.md 8.10 makes
a requirement for an L2 batch. The control used above is Aug-22's.

The only same-wave, same-GPU comparison of the two gate designs remains the Aug-27
11-arm wave (`output/GH200x2_l2redesign/`, run dirs since deleted), at n=1 per cell:
`l2_redesign` 1.057 adj [0.967, 1.155] vs `l2` (shipped) 0.986 adj [0.928, 1.048]
against a common control — a ratio of ~1.07 between them, same direction and magnitude
as the 1.045 measured here, and equally non-significant.

## 7. Bottom line

- **Mechanism:** the redesign does what it says — it un-sticks the zero-promotion
  failure (0 -> 19 in replay on the arm that produced it) and it demonstrably blocks
  duplicate families (6 and 3 distinct skills on r2/r3). But on arms that promote
  early, `hit_rate` and `selection_rate` are the same number, so the effective knob is
  the 0.70 -> 0.60 floor; and the dedup invariant is enforced on a text the coder never
  sees, so 2 of 3 replicates still carry near-duplicate standing rules at >= 0.80.
- **Cost:** the redesign preset triples the coder system prompt (2.99-3.54x mean,
  4.13-6.20x terminal) against the shipped arm's 1.00x.
- **Quality:** no difference, either design vs control or against each other, to within
  about +/-6%. `fast_p_best@1.0` is identical (0.632) for all four L2 arms. The
  zero-mechanism null contrast in this same directory reads 1.021x adjusted with a
  +0.079 fast_p gap, which bounds what any of these contrasts can mean.
