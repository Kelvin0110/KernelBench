# KernelBench Integration: New Evolving Agent

This directory contains the Gen3 KernelBench integration: staged prompts, L1 skill catalog, and L0 compaction. It uses `Self-Evolving-Agent/evolving_common_gen3` (not Gen2 `evolving_common`).

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
  - Handles resume-on-failure logic by tracking progress in `evolving_runs.json`.

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

### `evolving_common_gen3` Helpers
The `KBGovernor` leverages these modules from `Self-Evolving-Agent/evolving_common_gen3/`:
- **`llm_client`**: Handles structured NVIDIA/OpenAI-compatible API calls with retry logic.
  - Exposes assistant `content` and `reasoning` in metadata for downstream logging.
  - Supports a dedicated extractor-model path (`call_extractor_with_meta`) used to retrieve relevant L1 memories before each coder call.
- **`memory_manager`**: Manages the two-tier hierarchical memory:
    - **L0 (Iteration-level)**: Short-term logs of attempts, failures, and intermediate metrics.
    - **L1 (Journal-level)**: Long-term extracted insights (meta-learning) that persist across iterations.
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
7. **Evaluation** and **L0→L1 promotion** (unchanged semantics).
8. **Aggregation**: Batch-level `eval_results_level_X.json` as before.

### Gen3 config flags (`KBGovernorConfig`)

| Flag | Default | Meaning |
|------|---------|---------|
| `enable_action_selector` | `true` | Staged loop vs legacy single prompt |
| `action_selector_recent_l0_full` | `3` | Full L0 attempts shown to action selector |
| `action_coder_l0_full_recent` | `8` | Full L0 attempts in coder prompt |
| `enable_l0_unfold` | `true` | Coder preflight unfold passes |
| `l0_unfold_max_attempts` | `3` | Max preflight rounds per iteration |

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
