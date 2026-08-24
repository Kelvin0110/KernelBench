# L2 Skill Promotion — Design Report

Third memory tier for the self-evolving agent. Shared by KernelBench and MLE-Bench.
**Opt-in** via `--enable-l2`; when off, prompts and ledger are byte-identical to before.

---

## 1. Concept

- **L0** — per-problem working memory (rounds: code, terminal, metrics).
- **L1** — cross-task skills, **retrieved** per iteration by the extractor (≤10 of N).
- **L2** — skills promoted to **standing instructions**, injected into *every* coder
  system prompt with **no retrieval gate**.

**Why it exists:** with `--no-skill-deletion` the extractor catalog is tail-capped at 50
entries. At ~0.38 new skills per call, a skill is visible ~130 calls (~4 problems) and
then scrolls out **permanently**. L2 is the only mechanism that keeps a proven skill alive.

---

## 2. Data model

- `tier` field on the L1 chain: `l1` | `l2` — **orthogonal to `status`**
  (`active` | `superseded` | `deleted`).
- Refinement / merge / deletion mutate `status`; promotion only moves `tier`. Those
  lifecycles are untouched. Legacy rows without `tier` read back as `l1`.
- Per-run artifacts, beside `shared_l1.jsonl`:
  - `l2_standing.jsonl` — rendered standing set (**source of truth** for injection)
  - `l2_promotions.jsonl` — append-only promote/demote audit with frozen metrics
  - `l1_skill_usage.json` — evidence ledger

---

## 3. Evidence ledger

Per skill, in `l1_skill_usage.json`:

- `created_at_global_iter` — birth, for the rate denominator
- `total_selections` — cumulative extractor picks
- `tasks_used[]` — distinct task keys (breadth)
- `new_best_attributions` — iterations that became a new best with the skill in play

Key properties:

- Recorded whenever the **L1 extractor runs** — deliberately decoupled from
  skill deletion/merging. Previously the ledger only existed under skill governance, so
  the standard `--no-skill-deletion` arms wrote **no usage data at all**.
- `record_l2_selection_evidence` **never touches the consecutive-unused streak**, so
  enabling L2 cannot perturb deletion behavior.

---

## 4. Promotion gate

**Conjunction — a skill must clear ALL four floors.** Universal: one set for both
benchmarks and every catalog regime.

| Floor | Default | Blocks |
|---|---|---|
| `min_tasks` | **3** | task-specific skills that never generalize |
| `min_selections` | **50** | small-sample noise (3-of-5 picks = rate 0.6, means nothing) |
| `min_rate` | **0.70** | "old and occasionally useful" — high count purely from age |
| `min_new_bests` | **0 (disabled)** | popular-but-inert skills |

```
rate = total_selections / max(1, global_iteration − created_at_global_iter)
         ↑ times picked          ↑ chances it had to be picked
```

- **Why a rate, not a count.** A count measures *age*. A skill born at iteration 1,400
  gets ~100 chances; one born at iteration 4 gets ~1,500. Count-based floors promote
  every early-born skill and permanently lock out later discoveries.
- **Why AND, not OR.** Each floor blocks a *different* failure mode. Under OR the weakest
  filter defines the gate.
- **Scale for 0.70:** the extractor picks ~7 skills per call, so a *random* skill scores
  ~0.14 against a 50-entry catalog. 0.70 is ~5× chance.
- **Why one set works everywhere.** End-of-run rates diverge by regime (the deletion-on
  catalog grows, so its random baseline decays 0.16 → 0.066 → 0.029; tail-capped stays
  ~0.14). But promotions fire at boundaries ~4–12, when the visible catalog is ~40–100
  entries in *both* regimes. A regime-split profile was tried and **removed**: calibrated
  on decayed end-of-run rates — the wrong statistic for a gate that fires early — it
  over-promoted 3× (25–55 rules instead of 5–16).
- **`min_new_bests` disabled:** ~7 skills are in play on any new-best iteration, so credit
  is shared and **correlational, not causal**. Still recorded, still feeds the ranking
  score and the `l2_meta` audit; re-arm with `--l2-min-new-bests N`.

**Ranking (dormant):** survivors sort by `rate × log1p(tasks) × log1p(new_bests)`, then
truncate to `max_entries`. Default `max_entries=0` (unlimited), so **the score is computed
but never changes the outcome** — the floors alone decide.

---

## 5. Promotion lifecycle

- **Runs at task boundaries** — end of one governor run (KB: one problem; MLE: one
  competition). The gate needs cross-task breadth, so per-iteration checks are wasted work.
- **Standing set is immutable within a task** ⇒ the coder system message is byte-identical
  across that task's iterations ⇒ free prompt-cache reuse.
- **Promotion is permanent and cumulative.** The set is a **union over all boundaries**, not
  a snapshot — a skill only has to clear the bar once, at its personal peak.
  *(Sizing this gate from an end-of-run snapshot undercounts it ~3×.)*
- **Evidence frozen at promotion** into `l2_meta` and never re-derived. L2 skills leave the
  extractor catalog, so a recomputed rate would grow its denominator against a frozen
  numerator and demote every standing rule on a delay.
- **Demotion only** when the underlying entry stops being `active` (superseded by
  refinement/merge, or deleted). No metric-based demotion — it would be unsound.

---

## 6. Prompt surfaces

| Stage | L2 content |
|---|---|
| **Coder** | **Full standing block appended to the system prompt** (directive, not context) |
| **L1 extractor** | Titles only, marked *"do NOT select these"* |
| **Action selector** | None — writes no code, so it would be pure token cost |

- Promoted ids are **filtered out of the picker catalog** — otherwise they are billed twice
  and waste selection slots.

---

## 7. Rendering (pluggable)

| Mode | Behavior |
|---|---|
| `verbatim` *(default)* | Full L1 content unchanged |
| `extract` | Deterministic: keeps only `Generalizable Rule` / `Anti-Pattern to Avoid`; drops retrospective `Root Cause` / `Pivot`. Falls back to full content if structure is absent |
| `distill` | LLM rewrite into imperative form; **fails soft to `extract`** |

- `extract` invents nothing, so it keeps an A/B clean. `distill` is the only mode that puts
  a generative mutation upstream of an always-on prompt surface — hence opt-in.

---

## 8. Guards

| Risk | Guard |
|---|---|
| Unused-streak GC deletes L2 rules (they are never "selected") | `tier == l2` skipped in `evaluate_consecutive_unused_deletions` |
| Merge silently rewrites a rule already in every prompt | L2 excluded from merge clustering |
| Extractor bills a standing rule twice | L2 ids filtered from the picker catalog |
| Resume leaks future knowledge as standing law | `rollback_l2_for_resume` drops rules whose source was purged or whose provenance is ≥ `--start-problem` |
| Standing rules dilute the integrity / no-reward-hacking block | Track `is_hack` rate as a guardrail metric |

- `l1_allowed_entry_ids` (the causal resume filter) only constrains the **extractor**, so it
  cannot cover L2 — hence the separate rollback.

---

## 9. Cross-benchmark design

- All logic in `evolving_common/governor/l2_promotion.py`; both governors call the same pass.
- Only benchmark-specific input is the **task key**: KernelBench `L{level}P{problem_id}`
  (`cfg.l2_task_key`), MLE-Bench falls back to `cfg.kaggle_competition_id`.
- **Metric-direction agnostic** — it counts *new-best events*, which each governor already
  computes under its own `lower_is_better` policy.
- Identical flags on both CLIs (`evolve_kb_batch.py` and `Self-Evolving-Agent/cli.py`);
  all recorded in `run_summary.json` and checked on resume.
- Startup warning when `enable_l2` is set without a competition id (MLE would otherwise
  accumulate no breadth and silently never promote).

---

## 10. Measured behavior

Boundary-by-boundary replay of the exact gate over five 50-problem × 30-iteration
`gpt-oss-120b` runs:

| Run | Standing rules | Verbatim tokens | First promotion |
|---|---|---|---|
| no-deletion A | 8 | ~3.6K | boundary 8 |
| no-deletion B | 7 | ~3.3K | boundary 12 |
| deletion | 5 | ~2.1K | boundary 5 |
| merge-only | 2 | ~1.5K | boundary 4 |
| deletion+merge+refine | 10 | ~6.1K | boundary 4 |

- L1 content averages ~450–500 tokens ⇒ always-on cost ≈ **1–5% of a 128K window**.

---

## 11. Known limitations

- **Shared credit.** `new_best_attributions` is correlational (~7 skills share credit per
  new-best iteration). Good noise filter (elite 43–118 vs noise 0–2); not causal evidence.
- **No formal bound.** `max_entries=0` means the floors alone limit size. Measured 2–10, but
  nothing structurally prevents more — watch `is_hack` if the set grows.
- **Start-up dilution.** `min_tasks=3` means L2 cannot exist for the first ~6% of a
  50-problem KB run, ~25% of a 12-competition MLE run. Any A/B **understates** steady state.
- **MLE + `--no-skill-deletion`** leaves only ~2.6 tasks of achievable breadth (50 iters per
  competition vs the ~132-call visibility window). Keep skill deletion on (the MLE default)
  or pass `--l2-min-tasks 2`.
- **Score annihilation.** The ranking multiplies by `log1p(new_bests)`, so skills with zero
  new-best credit score 0. Harmless while `max_entries=0`; if a cap is set they tie at the
  bottom and sort by entry id.
- **Thin margin at the floor.** Promoted skills historically cleared `min_rate` by little;
  once a skill scrolls out of a tail-capped catalog its rate only decays, so a near-miss is
  permanent.

---

## 12. Status

- Implemented and tested; **no experiment has been run**.
- 28 L2 unit tests + batch CLI dry-run coverage; no regressions against the HEAD baseline.
- Design spec:
  `Self-Evolving-Agent/docs/superpowers/specs/2026-08-20-l2-standing-instructions-design.md`

---

# Appendix A — Completed `gpt-oss-120b` GH200 runs: aggregated results

Added 2026-08-24. Scope: `runs_evolving/gpt-oss-120b/` **excluding the `median/`
subfolder** (all 9 arms there are live, at 3–25 of 50 problems completed). Regenerated with
`--regenerate-stats`; artifacts in
`scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed/`.

## A.1 Inclusion

Complete = `run_summary.json` present, `total_completed == total_attempted == 50`,
`run_finished.json` in all 50 workspaces (ANALYSIS_RULES §3). **10 of 16** non-median
runs qualify; all 10 reach iteration 30 on every problem, with one exception noted in A.6.

| Excluded (partial) | problems done |
|---|---|
| `..._itr30_GH200_2026_08_20_16_32` / `..._16_42` (truncation reps) | 36 / 35 |
| `..._markov_itr30_GH200_2026_08_20_16_35` / `..._16_45` | 35 / 34 |
| `..._folding_itr30_GH200_2026_08_20_16_39` / `..._16_48` | 33 / 34 |

Those six were killed, not finished; they are in `aggregate_runs.{json,csv}` with warnings
and appear in no table below.

## A.2 Headline — iterations 10 and 30, native baseline

Baseline `results/timing/NVIDIA_GH200x2/baseline_time_torch.json` — the file each of these
runs was actually scored against (`hardware_server: NVIDIA_GH200x2` in every
`run_summary.json`). `@0` is running-best coverage on the full 50-problem denominator
(correctness); `@1` / `@2` are `fast_p_best` at 1.0× and 2.0× on the same denominator;
geomean is `speedup_best.geometric_mean` over the `n` problems holding a non-hack running
best. At iteration 30, `@0 × 50 == total_correct`.

| design | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I10 n | I30 @0 | I30 @1 | I30 @2 | I30 geomean | I30 n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **truncation** (control) | 47/50 | 0.840 | 0.380 | 0.120 | 0.7379 | 42 | 0.940 | 0.460 | 0.180 | 0.9051 | 47 |
| markov_report | 48/50 | 0.900 | 0.340 | 0.040 | 0.7145 | 45 | 0.960 | 0.460 | 0.140 | 0.9332 | 48 |
| folding | 47/50 | 0.840 | 0.320 | 0.120 | 0.6744 | 42 | 0.940 | 0.420 | 0.180 | 0.8938 | 47 |
| selective_retention (r=5) | 48/50 | 0.900 | 0.320 | 0.080 | 0.7657 | 45 | 0.960 | 0.460 | 0.160 | 0.9541 | 48 |
| compress_trigger | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 | 42 | 0.960 | 0.400 | 0.140 | 0.7265 | 48 |
| truncation+deletion † | 48/50 | 0.840 | 0.400 | 0.100 | 0.8691 | 42 | 0.960 | **0.540** | **0.280** | **1.2312** | 48 |
| truncation+refine | 45/50 | 0.780 | 0.300 | 0.100 | 0.7107 | 39 | 0.900 | 0.340 | 0.160 | 0.7971 | 45 |
| truncation+merge@0.8 rep1 | 48/50 | 0.740 | 0.260 | 0.120 | 0.7047 | 37 | 0.960 | 0.420 | 0.200 | 0.8379 | 48 |
| truncation+merge@0.8 rep2 | 47/50 | 0.700 | 0.280 | 0.080 | 0.8282 | 35 | 0.940 | 0.400 | 0.140 | 0.8552 | 47 |
| truncation+merge@0.8 rep3 | 46/50 | 0.800 | 0.300 | 0.140 | 0.9185 | 40 | 0.920 | 0.460 | 0.220 | 1.0919 | 46 |
| *merge@0.8, 3-rep average* | *47/50* | *0.747* | *0.280* | *0.113* | *0.8123* | *37.3* | *0.940* | *0.427* | *0.187* | *0.9215* | *47.0* |

Replicate row: fast-p averaged arithmetically, geomean averaged in log space.

## A.3 Sensitivity — corrected `NVIDIA_GH200x2_median` baseline

The as-run baseline is the Aug-3 file, which is measurably contended on the level-1
problems: it records 11.7 ms for `22_Tanh` and 11.8 ms for `26_GELU` where the corrected
median baseline records 2.96 ms for both (`L1P34` 18.4 → 8.03, `L1P33` 14.1 → 8.27).
Median ratio over all 50 subset problems is 0.992, but 18 problems differ by >5%. Rescored
identically for every arm:

| design | I10 @0 | I10 @1 | I10 @2 | I10 geomean | I30 @0 | I30 @1 | I30 @2 | I30 geomean |
|---|---|---|---|---|---|---|---|---|
| **truncation** (control) | 0.840 | 0.320 | 0.080 | 0.6637 | 0.940 | 0.420 | 0.140 | 0.8180 |
| markov_report | 0.900 | 0.380 | 0.040 | 0.6421 | 0.960 | 0.460 | 0.140 | 0.8427 |
| folding | 0.840 | 0.260 | 0.100 | 0.5909 | 0.940 | 0.400 | 0.140 | 0.8057 |
| selective_retention (r=5) | 0.900 | 0.300 | 0.080 | 0.6871 | 0.960 | 0.420 | 0.140 | 0.8541 |
| compress_trigger | 0.840 | 0.240 | 0.080 | 0.5096 | 0.960 | 0.440 | 0.140 | 0.6578 |
| truncation+deletion † | 0.840 | 0.440 | 0.060 | 0.7897 | 0.960 | 0.480 | 0.200 | **1.1026** |
| truncation+refine | 0.780 | 0.260 | 0.080 | 0.6333 | 0.900 | 0.320 | 0.140 | 0.7141 |
| truncation+merge@0.8 rep1 | 0.740 | 0.260 | 0.100 | 0.6373 | 0.960 | 0.440 | 0.180 | 0.7591 |
| truncation+merge@0.8 rep2 | 0.700 | 0.240 | 0.040 | 0.7248 | 0.940 | 0.380 | 0.100 | 0.7726 |
| truncation+merge@0.8 rep3 | 0.800 | 0.260 | 0.100 | 0.8441 | 0.920 | 0.440 | 0.180 | 0.9945 |

Every geomean drops ~10%; `n` and correctness are unchanged (both baselines cover all 50
problems). **The ordering is preserved** — deletion first, compress last, everything else
inside one band. These stats were written to a scratch directory; no run's
`visualizations/performance_stats.json` was overwritten with the non-native baseline.

## A.4 Paired per-problem comparison against the control

Geometric mean of the per-problem ratio `best_speedup(arm) / best_speedup(truncation)`,
over problems where **both** hold a non-hack running best. The baseline cancels in the
ratio, so these numbers are **identical under both baselines** — they are the
baseline-independent form of the comparison.

| arm | I10 ratio (95% CI) | I10 n | I30 ratio (95% CI) | I30 n |
|---|---|---|---|---|
| markov_report | 0.926 [0.705, 1.215] | 39 | 0.993 [0.763, 1.293] | 45 |
| folding | 0.927 [0.640, 1.345] | 39 | 0.980 [0.807, 1.190] | 46 |
| selective_retention | 0.986 [0.769, 1.265] | 38 | 1.058 [0.850, 1.316] | 45 |
| compress_trigger | **0.732 [0.542, 0.988]** | 37 | 0.774 [0.584, 1.027] | 45 |
| truncation+deletion † | 1.158 [0.867, 1.545] | 38 | **1.371 [1.060, 1.773]** | 46 |
| truncation+refine | 0.999 [0.749, 1.332] | 34 | 0.842 [0.658, 1.078] | 42 |
| truncation+merge@0.8 rep1 | 0.812 [0.587, 1.124] | 35 | 0.911 [0.734, 1.132] | 46 |
| truncation+merge@0.8 rep2 | 1.059 [0.830, 1.350] | 31 | 0.933 [0.714, 1.220] | 46 |
| truncation+merge@0.8 rep3 | **1.235 [1.010, 1.510]** | 35 | 1.179 [0.986, 1.410] | 43 |

Only `truncation+deletion` at iteration 30 clears the control with a CI excluding 1.0.
Note that merge rep3 does so at iteration 10 while its two identical-config siblings do
not — a direct demonstration of the noise floor.

## A.5 Run-level context

| design | correct | wall h | min/problem | hack itrs | problems w/ hack | L1 entries | L1 active | merges | deletions | refines |
|---|---|---|---|---|---|---|---|---|---|---|
| truncation | 47/50 | 74.18 | 88.0 | 16 | 11 | 571 | 571 | 0 | 0 | 0 |
| markov_report | 48/50 | 71.43 | 83.5 | 14 | 11 | 366 | 366 | 0 | 0 | 0 |
| folding | 47/50 | 66.59 | 79.9 | 17 | 12 | 592 | 592 | 0 | 0 | 0 |
| selective_retention | 48/50 | 69.93 | 83.9 | 17 | 12 | 619 | 619 | 0 | 0 | 0 |
| compress_trigger | 48/50 | 64.23 | 77.1 | **32** | **23** | 435 | 435 | 0 | 0 | 0 |
| truncation+deletion † | 48/50 | 66.91 | 80.3 | 19 | 12 | 592 | **25** | 0 | 567 | 0 |
| truncation+refine | 45/50 | 53.07 | 63.7 | 29 | 19 | 703 | 626 | 0 | 0 | 83 |
| truncation+merge@0.8 rep1 | 48/50 | 64.91 | 77.9 | 21 | 16 | 703 | 384 | 56 | 0 | 0 |
| truncation+merge@0.8 rep2 | 47/50 | 68.37 | 82.0 | 31 | 18 | 730 | 182 | 77 | 0 | 0 |
| truncation+merge@0.8 rep3 | 46/50 | 65.43 | 78.5 | 23 | 14 | 681 | 313 | 52 | 0 | 0 |

Wall time is operational (endpoint latency, contention, resumes), not a treatment effect —
solo baselines on this host drifted 26% across August. The merge arms did real work
(`l1_skill_embeddings.json` 700/728/679 skills, `l1_skill_merges.jsonl` 124/171/180 lines),
so no silent embedding failure.

## A.6 Caveats

- **† `truncation+deletion` is deletion + unit-test GC.** `l1_skill_deletions.jsonl` records
  326 `unit_test_fail` against 241 `consecutive_unused` while `run_summary.json` says
  `enable_l1_skill_unit_test_gc: false`. This run predates the gate fix (submodule
  `bd92795`), so the two mechanisms are not separable in it — and it is the only arm whose
  advantage is significant.
- **`n = 1` per configuration except merge.** The three identical merge replicates span
  **1.30×** in I30 geomean (0.8379 / 0.8552 / 1.0919), and CLAUDE.md open item 6 records a
  22% gap between two identical truncation runs. Every I30 geomean delta in A.2 outside
  deletion is inside that band.
- **One truncated problem.** `truncation+refine` stopped `level_3_problem_10` at iteration 4
  of 30; all other 499 problem-runs reach 30.
- `metrics_best.is_hack` is the run-level `run_had_hack` latch, so it never gates geomean
  eligibility — `n` tracks `total_correct` (ANALYSIS_RULES §4).
- Problems within a run are coupled through sequential shared L1 memory; the `median/`
  series and these runs use different baselines and are not directly comparable.
