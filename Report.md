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
