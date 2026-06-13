# KernelBench Integration: New Evolving Agent

This directory contains the Gen3 KernelBench integration: staged prompts, L1 skill catalog, and L0 compaction. It uses `Self-Evolving-Agent/evolving_common`.

## 1. Key Files & Responsibilities

### Core Integration
- **[kb_governor.py](kb_governor.py)**: Implements the **Inner Loop** logic.
  - Defines `KBGovernor`, which orchestrates the `Prompt -> Generate -> Evaluate -> Reflect` cycle for a *single* kernel optimization task.
  - Interfaces with `kernelbench.eval` to run generated CUDA code and extract correctness/speedup metrics.
  - Captures and persists per-iteration evaluation terminal output (stdout/stderr) into recorder artifacts.
  - Manages the interaction with `llm_client`, `memory_manager`, and `run_recorder`.

- **[evolve_kb_batch.py](evolve_kb_batch.py)**: Implements the **Outer Loop** (Batch Orchestrator).
  - Processes a subset of KernelBench problems (from a CSV file).
  - Manages per-level execution (e.g., Level 1, Level 2) and aggregates results into standard KernelBench JSON formats (`eval_results_level_X.json`).
  - Uses exception-safe JSON serialization so evaluation metadata containing runtime exceptions cannot crash artifact persistence.
  - Supports `--resume` with the full timestamped `--run-name` and `--start-problem` (1-based subset index): reuses shared L1, keeps earlier problems, replaces results from the start index onward (including failed entries such as rate-limit errors), and clears per-problem workspaces before re-run.

- **[RUN_WITH_UV.md](RUN_WITH_UV.md)**: Standardized execution guide using the `uv` package manager for reproducible environments and dependency management.

### Testing & Verification
- **[test_kb_governor.py](test_kb_governor.py)**: Unit tests for `KBGovernor` logic, including dry-run modes and evaluation mocking.
- **[test_evolve_kb_batch.py](test_evolve_kb_batch.py)**: Integration tests for the batch orchestrator, verifying CSV parsing and result aggregation.

---

## 2. Interaction with `Self-Evolving-Agent`

This integration relies on shared components located in the `Self-Evolving-Agent/` submodule:

### Shared Models
- **[Self-Evolving-Agent/kernelbench/config.py](../../Self-Evolving-Agent/kernelbench/config.py)**: Defines `KBGovernorConfig`, which controls hyperparameters like `max_iterations`, `temperature`, and `promotion` thresholds.
- **[Self-Evolving-Agent/kernelbench/schemas.py](../../Self-Evolving-Agent/kernelbench/schemas.py)**: Defines `KBEvalResult` and `KBGovernorResult` for structured data exchange between the governor and the orchestrator.

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
