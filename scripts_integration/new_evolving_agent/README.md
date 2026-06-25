# KernelBench Integration: New Evolving Agent

This directory contains the Gen3 KernelBench integration: staged prompts, L1 skill catalog, and L0 compaction. It uses `Self-Evolving-Agent/evolving_common`.

## 1. Key Files & Responsibilities

### Core Integration
- **[evolve_kb_batch.py](evolve_kb_batch.py)**: **Outer loop** — batch orchestrator (subset CSV, resume, result aggregation).
- **Governor (inner loop)**: [`Self-Evolving-Agent/kernelbench_integration/`](../../Self-Evolving-Agent/kernelbench_integration/) — see [developer README](../../Self-Evolving-Agent/kernelbench_integration/README.md).
  - `KBGovernor` runs Gen3 staged prompts + KernelBench subprocess eval for one problem.
  - Deprecated shim: [kb_governor.py](kb_governor.py) re-exports `kernelbench_integration` with a warning.

- **[RUN_WITH_UV.md](RUN_WITH_UV.md)**: Standardized execution guide using the `uv` package manager for reproducible environments and dependency management.

### Testing & Verification
- Governor unit tests: [`Self-Evolving-Agent/tests/test_kb_governor.py`](../../Self-Evolving-Agent/tests/test_kb_governor.py)
- **[tests/test_evolve_kb_batch.py](tests/test_evolve_kb_batch.py)**: Batch orchestrator integration tests.

---

## 2. Interaction with `Self-Evolving-Agent`

This integration relies on shared components located in the `Self-Evolving-Agent/` submodule:

### Shared Models
- **[Self-Evolving-Agent/kernelbench_integration/config.py](../../Self-Evolving-Agent/kernelbench_integration/config.py)**: `KBGovernorConfig`
- **[Self-Evolving-Agent/kernelbench_integration/schemas.py](../../Self-Evolving-Agent/kernelbench_integration/schemas.py)**: `KBEvalResult`, `KBGovernorResult`

### `evolving_common` Helpers
The `KBGovernor` leverages these modules from `Self-Evolving-Agent/evolving_common/`:
- **`llm_client`**: Handles structured NVIDIA/OpenAI-compatible API calls with retry logic.
  - Exposes assistant `content` and `reasoning` in metadata for downstream logging.
  - Supports a dedicated extractor-model path (`call_extractor_with_meta`) used to retrieve relevant L1 memories before each coder call.
- **`memory_manager`**: Manages the two-tier hierarchical memory:
    - **L0 (round-native)**: Each iteration is one `L0Round` (`round_id`, `action`, `metrics`, `code`, `terminal[]`, `round_summary`). Written once per iteration via `finalize_l0_round`; compact views use LLM `l0_round_summarizer` (not truncation heuristics).
    - **L1 (Journal-level)**: Long-term cross-problem skills from `SUMMARIZER_SYSTEM_PROMPT` on `format_l0_for_l1_promotion(rounds)`; promotion triggers every `promote_every_n_rounds` (default 2) or token budget.
      - Persists to both `shared_l1.txt` (human-readable journal) and `shared_l1.jsonl` (structured entries with `description`).
- **`metrics_holder`**: A thread-safe container (`BestMetricsHolder`) that tracks the "Current Best" performance (speedup/runtime/correctness).
- **`run_recorder`**: Handles filesystem logging and writes the canonical `metrics_by_time.jsonl` traces found in result folders.
  - `chat_history.jsonl` now records `assistant_reasoning` keys when present.
  - `evaluation_terminal_output.jsonl` stores per-iteration code-evaluation terminal output.

---

## 3. Workflow & Data Flow (Gen3 per iteration)

1. **Initialization**: `evolve_kb_batch.py` reads the problem subset and initializes a `KBGovernor` for each task.
2. **Action selection** (`action_selector`): Task + L0 summary + last 3 attempts in full detail. No L1. Returns `propose_new` / `debug_current` / `refine_current`.
3. **L1 skill picker** (`l1_skill_picker`): Catalog of all skill titles; load full content for selected IDs only.
4. **Action coder prompt**: Task, iteration context, latest eval, L0 last 5–10 full + archived summaries, selected L1 skills, action-specific guidelines.
5. **L0 unfold preflight** (`coder_preflight`, up to `l0_unfold_max_attempts`): Coder model may request full archived rounds via JSON; prompt is updated before the final code pass.
6. **Final coder** (`coder`): One fenced Python `ModelNew` implementation. Backbone reasoning is stored from `assistant_reasoning` in LLM metadata (not `<reasoning>` tags).
7. **Evaluation**, **`finalize_l0_round`**, optional **`l0_round_summarizer`** (per-attempt compact line), then **L0→L1 promotion** when round threshold is met.
8. **Aggregation**: Batch-level `eval_results_level_X.json` as before.

### Prompt roles (Gen3 default: `enable_action_selector=True`)

| Phase | System prompt constant | User prompt builder |
|-------|------------------------|---------------------|
| `action_selector` | `ACTION_SELECTOR_SYSTEM_PROMPT` | `build_action_selector_user_message` — L0 only, no L1 |
| `l1_skill_picker` | `DEFAULT_EXTRACTOR_SYSTEM_PROMPT` | `build_extractor_user_message` — skill IDs by title, description, trigger |
| `coder_preflight` | `CODER_PREFLIGHT_SYSTEM_PROMPT` | draft + `build_coder_preflight_user_appendix` |
| `coder` | `CODER_SYSTEM_PROMPT` (= KernelBench rules + `GEN3_CODER_FINAL_SYSTEM_PROMPT` + `GEN3_MEMORY_PRIMER`) | `build_action_coder_user_prompt` + `build_coder_final_user_appendix` |
| `l0_round_summarizer` | `L0_ROUND_SUMMARIZER_SYSTEM_PROMPT` | `build_l0_round_summarizer_user_message` — one attempt, prompt compaction only |
| L0→L1 | `SUMMARIZER_SYSTEM_PROMPT` | `build_summarizer_user_message` on `format_l0_for_l1_promotion(rounds)` |

Memory primer (`GEN3_MEMORY_PRIMER`) is included in both the final coder **system** prompt and the coder **user** prompt under `## Memory roles`.

### Legacy path (`enable_action_selector=False`)

Uses `LEGACY_CODER_SYSTEM_PROMPT` (`BASE_EVOLVING_CODER_SYSTEM_PROMPT` with required `<diagnosis>`, `<hypothesis>`, `<action>`, `<reasoning>` tags) and `build_user_prompt_with_memory`. Action and reasoning come from coder output tags; tests disable the staged loop for simpler mocking.

### Gen3 config flags (`KBGovernorConfig`)

| Flag | Default | Meaning |
|------|---------|---------|
| `enable_action_selector` | `true` | Staged loop vs legacy single prompt |
| `l1_catalog_max_skills` | `50` | Most recent L1 skills shown in extractor catalog (metadata only) |
| `action_selector_recent_l0_full` | `15` | Full L0 attempts shown to action selector |
| `action_coder_l0_full_recent` | `15` | Full L0 attempts in coder prompt |
| `context_management` | `truncation` | `truncation`: latest N raw L0 rounds only; `folding`: archived summaries, round summaries, and unfold preflight |
| `enable_l0_unfold` | `true` | Coder preflight unfold passes (folding mode only) |
| `l0_unfold_max_attempts` | `1` | Max preflight rounds per iteration |
| `enable_l0_round_summary` | `true` | LLM summary after each round (folding mode only; fallback: deterministic line) |
| `l0_round_summary_max_tokens` | `512` | Max tokens for round summarizer |
| `promote_every_n_rounds` | `2` | L1 promotion after this many new rounds since last promotion |
| `promote_token_budget` | `4000` | Alternate promotion trigger on serialized round payload size |
| `enable_skill_refinement` | `false` | Opt-in SkillRevise-style skill refinement add-on (see §3.1). When off, the loop is unchanged |
| `skill_refinement_max_rounds` | `3` | Max inline refinement rounds per trigger |
| `skill_refinement_model` | `None` | Model spec for diagnosis/revision (defaults to the summarizer model) |
| `skill_refinement_max_tokens` | `8192` | Max tokens for the revision call (diagnosis uses `min(this, 4096)`) |
| `skill_refinement_timeout_sec` | `90.0` | Per-call timeout for diagnosis/revision |

### 3.1 Skill Refinement add-on (opt-in, SkillRevise-style)

Enable with `--enable-skill-refinement` (CLI) or `enable_skill_refinement=True` (config).
**Disabled by default**; when off, the controller is a no-op and the loop behaves
exactly as before. Refinement only runs on the Gen3 staged path with L1 enabled.

**Trigger.** After an evaluated iteration that *used* one or more L1 skills but made
**no progress** (it neither became a new best nor improved over the prior best —
subsuming both "failed to fix the bug" and "failed to improve"), refinement begins.
It will not start on the final iteration (no room to use the refined skills next).

**Diagnosis (V/A/K).** An LLM produces a structured `Diagnosis`:
- *Verification Specification* — the observable requirements (output contract,
  shapes, correctness assertions, target metric + direction).
- *Failure Attribution* — failed checks, root causes, and which skill(s) contributed.
- *Preservation Constraints* — what already works and must not be broken.

The diagnosis also returns `skill_defect` and `blamed_skill_ids`. If no provided
skill is at fault, the controller **abstains** (no revision, loop continues normally).

**Revision + pinning.** Each *blamed* skill is revised individually into a new
version. The refined set (blamed skills replaced, unblamed kept) is **pinned**: the
next iteration reuses it and **skips the L1 picker** (`forced_l1_entries`). This
repeats inline for up to `skill_refinement_max_rounds` (default 3), each round
consuming one main iteration (counts within `max_iterations`).

**Finalize (utility-gated lazy deletion).** The window ends when the attempt becomes
a new best, the diagnosis abstains, or the round budget is exhausted. The best
version per skill chain is kept **active**; the rest are marked **superseded**.
Selection is correctness-first, then metric, honoring `lower_is_better` so it
generalizes to MLE-Bench metrics (KernelBench uses speedup, higher-better). Ties
keep the original, so a refinement only wins by strictly improving.

**Versioned catalog.** Refined versions are appended to `shared_l1.jsonl` with
`parent_id`, `version`, `status` (`active|superseded|deleted`), `refinement_round`,
`refinement_meta` (`bug_solved`, `metric_before/after`, `metric_delta_pct`,
`lower_is_better`), and a `revision_trace`. `read_recent_l1_jsonl` returns only
`active` versions, so superseded skills are never re-selected (lazy deletion: kept
on disk for auditing). A human-readable mirror of each revision is appended to
`skill_revisions.txt` next to the journal.

Prompts: `SKILL_DIAGNOSIS_SYSTEM_PROMPT`, `SKILL_REVISION_SYSTEM_PROMPT`; engine in
[`evolving_common/governor/skill_refinement.py`](../../Self-Evolving-Agent/evolving_common/governor/skill_refinement.py)
(`RefinementController`). Tests: [`test_skill_refinement.py`](../../Self-Evolving-Agent/tests/test_skill_refinement.py).

### L0 round schema (Gen3)

```python
L0Round = {
  "round_id": "round_1",
  "attempt": 1,
  "action": "debug_current",
  "action_selector_rationale": "...",
  "reasoning_full": "...",  # from assistant_reasoning metadata
  "code": "...",
  "terminal": ["KERNEL_BENCH_CORRECT: ...", ...],
  "metrics": {"compiled": true, "correct": false, "speedup": 0.0, ...},
  "round_summary": "LLM or fallback one-liner for archived catalog",
}
```

Archived L0 in coder/selector prompts uses `round_summary` only; recent rounds use full code/terminal/reasoning (with size caps on raw blobs).

### Environment (`Self-Evolving-Agent/.env.example`)

- `NVIDIA_ACTION_SELECTOR_MODEL` — action selection stage
- `NVIDIA_EXTRACTOR_MODEL` — L1 skill picker
- `NVIDIA_CODER_MODEL` — preflight + final code

---

## 4. Implementation Notes

### Pydantic & `sys.modules` Caching
To prevent `ValidationError` or "Model already defined" errors during dynamic loading (common when scripts and libraries share Pydantic schemas), the `KBGovernor` implements a custom module caching mechanism. It ensures that `sys.modules` keeps a single, stable instance of the shared schemas across the process lifecycle.

### CUDA Context Safety
Since multiple iterations run CUDA code, the system is designed to handle potential GPU memory leakage or kernel crashes by isolating evaluations where possible, though the main loop is currently sequential to preserve the evolutionary chain state.
