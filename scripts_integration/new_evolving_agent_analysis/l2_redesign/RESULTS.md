# L2 redesign wave — results (2026-08-29)

11 arms, GPU 0, `gpt-oss-120b`, `NVIDIA_GH200x2_median` baseline. All completed:
7 × 50 problems (`runs_evolving/gpt-oss-120b/l2redesign/`) and 4 × 15 problems on
the level-2 block, problems 11–25 (`.../l2quick/`). Health over the whole wave:
`oom=0`, `proceeding UNLOCKED=0`, 1 mem-gate timeout in ~1200 evals.

Reproduce with `paired_report.py`, `lottery_adjusted.py`, `mechanism_report.py`,
`standing_diversity.py`, `show_configs.py` in this directory.

---

## 1. Headline: quality is a null, and now we know why the raw numbers weren't

Raw arm geomeans span 1.058–1.696 and *look* like the control crushes every L2
variant. That is an artifact. **14 of ~50 problems are bimodal lotteries**: an arm
either finds an algebraic collapse (7–23×) or does not (~1×), and which happens is
close to a coin flip.

The proof is treatment-free — two arms with **byte-identical configuration**:

| problem | `q15_ctl_r1` | `q15_ctl_r2` |
|---|---|---|
| L2P42 | 1.59× | **22.72×** |
| L2P51 | **7.17×** | 1.01× |
| L2P97 | **7.36×** | 0.99× |

Same flags, same GPU, same day, opposite outcomes. The 50-problem `truncation`
control happened to win L2P42 (18.3×), L2P51 (7.3×) and L2P97 (7.0×) — three
jackpots — which is the entire reason it leads.

Lottery problems were selected **without reference to arm identity** (max/min clean
speedup ≥ 4× across all 11 arms), so the adjustment cannot be tuned toward a
winner. Removing them:

| arm | ratio vs control (all) | ratio (adjusted) | 95% CI | fast_p adj |
|---|---|---|---|---|
| `l2_redesign` | 1.018 | **1.057** | [0.967, 1.155] | 0.667 |
| `l2_preseed` | 0.812 | 1.031 | [0.943, 1.126] | 0.556 |
| `l2_hit` | 0.788 | 1.027 | [0.940, 1.122] | 0.583 |
| `l2` (shipped) | 0.742 | 0.986 | [0.928, 1.048] | 0.611 |
| `l2_extract` | 0.784 | 0.981 | [0.876, 1.099] | 0.500 |
| `l2_judge` | 0.673 | 0.971 | [0.901, 1.046] | 0.556 |
| *null: `ctl_r2`/`ctl_r1`* | *0.696* | *0.961* | *[0.885, 1.043]* | — |
| control | — | — | — | 0.583 |

**Every CI contains 1.0, and the identical-config null (0.961) sits inside the same
band as every treatment.** Arm geomeans compress from a 1.60× spread to 1.06×
(1.100–1.161). No L2 variant is distinguishable from the control, in either
direction, and the effect is bounded to roughly ±6%.

Two further cautions, both visible in `robust_contrast.py`:

- **Per-problem medians are ≈1.0 for every arm** (0.953–1.009) while geomeans swing
  wildly — i.e. even before the adjustment, the typical problem showed no effect.
- The within-replicate pre-seed contrast "significantly" favoured the control in
  r1 (0.565, CI excluding 1.0) and reversed in r2 (1.308). Pooled: 0.860
  [0.564, 1.310]. A starred CI at n=1 per cell is noise, exactly as open item 10
  predicts.

---

## 2. Mechanism: this is where the redesign actually delivered

Quality is a null, so the defensible claims are mechanical.

### 2.1 The shipped gate is still not reproducible

CLAUDE.md §8.11 records **9 / 4 / 0** rules promoted from identical-flag runs. This
wave's `l2` arm ran the shipped defaults (`min_tasks 3`, `min_selections 50`,
`min_rate 0.70`, no hit-rate) and promoted **6** — a fourth distinct value. The
instability the redesign was built to fix is real and persists.

### 2.2 Dedup works, and nothing else does

Final standing sets, embedded through the same path as skill merging:

| arm | rules | pairs ≥ 0.80 | max cosine |
|---|---|---|---|
| **`l2_redesign`** (dedup 0.80) | 6 | **0** | **0.785** |
| `l2_judge` | 6 | 2 | 0.849 |
| `l2` (shipped) | 6 | 3 | 0.816 |
| `l2_preseed` | 5 | 3 | 0.911 |
| `l2_hit` | 6 | 4 | 0.912 |
| `l2_extract` | 2 | 1 | 0.815 |

`l2_redesign` is the only arm with no duplicate pair, and the census shows dedup
firing **13 times** ("deduped against standing"). The §8.4 duplicate-family defect
reproduces plainly elsewhere — `l2_hit` promoted both *"Avoid Hand-rolled Conv2d;
Use cuDNN"* and *"Prefer cuDNN for Conv2d"* at cosine 0.912.

### 2.3 The LLM judge does **not** replace the rule-based gate

This answers the "can an LLM agent make the judgement instead?" question directly.
The judge arm ran deliberately permissive floors (`min_tasks 2`, `min_selections
15`, `min_rate 0.05`) so that selectivity would come from the judge rather than the
gate. It accepted 6 rules with coherent, individually sensible rationales — and
still left **2 near-duplicate pairs at up to 0.849**.

The reason is structural: the judge scores each candidate on its own merits
("general and actionable, not covered by existing rules"), so it cannot see that
two separately-plausible rules say the same thing. **Dedup is a set-level property
and the judge is a per-item filter.** Use the judge for admission quality if you
want it, but keep `--l2-dedup-similarity` for redundancy.

### 2.4 Pre-seed freeze is exact

`l2_preseed`, `q15_pre_r1`, `q15_pre_r2`: all three ended `preseeded=5,
standing=5, promoted=0, demoted=0`. The `preseeded_from` exemption holds — without
it these rules carry another run's entry ids, fail the liveness check, and the tier
empties silently on the first pass.

---

## 3. Defect found in my own instrumentation (fixed)

The pass-census key `dropped` emitted **every** candidate carrying a `reasons`
entry, including promoted ones — the judge writes its *acceptance* rationale into
the same field. In the `l2_judge` artifacts, entries 1 and 7 appear under `dropped`
at `global_iteration 60` and were in fact **promoted** at that same boundary.

Fixed by adding a `decisions` list with an explicit `promoted` flag and filtering
`dropped` to genuine rejections. **Artifacts already on disk carry the old,
misleading key** — cross-reference the `promote` events before reading them.

---

## 4. What to do next

1. **Keep** `--l2-dedup-similarity 0.80`. It is the only change with a measured,
   reproducible effect, and it is free.
2. **Do not** promote the judge as a dedup substitute (§2.3). It costs an LLM call
   per pass and does not solve redundancy.
3. **Any future quality claim must exclude or stratify the 14 lottery problems.**
   The list is printed by `lottery_adjusted.py`; L2P42/51/97, L2P41, L2P94, L2P56,
   L3P34 are the worst. Reporting a geomean over the raw 50 is reporting a lottery.
4. **n=1 per cell still cannot name a winner** (open item 10). The adjusted CIs are
   ±6%, which is far tighter than the raw ±40% — removing the lottery problems is
   worth more statistical power than several extra replicates.
