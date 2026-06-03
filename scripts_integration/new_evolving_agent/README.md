# KernelBench Integration: New Evolving Agent

This directory containing the "Inner Loop" and "Outer Loop" implementation for integrating KernelBench with the `Self-Evolving-Agent` framework. It leverages the generic evolution abstractions in `evolving_common` while specializing them for CUDA kernel optimization.

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
The `KBGovernor` leverages these reusable modules from `Self-Evolving-Agent/evolving_common/`:
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

## 3. Workflow & Data Flow

1. **Initialization**: `evolve_kb_batch.py` reads the problem subset and initializes a `KBGovernor` for each task.
2. **L1 Selection**: If structured L1 entries exist, the governor runs an extractor model stage to select the most relevant entries (configurable max count).
3. **Context Building**: The governor prepares one system prompt + one user prompt using selected L1 entries and structured L0 history.
4. **Generation**: `llm_client` receives the prompt and generates a candidate `ModelNew` class.
5. **Evaluation**: `KBGovernor` executes the code via KernelBench's evaluation engine.
6. **Reflection**: The iteration's results are summarized, updating **L1** while retaining L0 history for future rounds.
7. **Persistence**: `run_recorder` saves snapshots of the code and metrics to `results/evolving_logs/<run_name>/`.
  - Includes iteration-level terminal output logs for evaluation/execution results.
8. **Aggregation**: Once the batch finishes, `evolve_kb_batch.py` flattens the results into a level-first `eval_results.json` for standard analysis.
  - Run summaries include both `best_speedup_overall` and `best_runtime_overall`.

---

## 4. Implementation Notes

### Pydantic & `sys.modules` Caching
To prevent `ValidationError` or "Model already defined" errors during dynamic loading (common when scripts and libraries share Pydantic schemas), the `KBGovernor` implements a custom module caching mechanism. It ensures that `sys.modules` keeps a single, stable instance of the shared schemas across the process lifecycle.

### CUDA Context Safety
Since multiple iterations run CUDA code, the system is designed to handle potential GPU memory leakage or kernel crashes by isolating evaluations where possible, though the main loop is currently sequential to preserve the evolutionary chain state.
