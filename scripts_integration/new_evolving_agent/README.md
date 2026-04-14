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

### `evolving_common` Helpers
The `KBGovernor` leverages these reusable modules from `Self-Evolving-Agent/evolving_common/`:
- **`llm_client`**: Handles structured NVIDIA/OpenAI-compatible API calls with retry logic.
  - Exposes assistant `content` and `reasoning` in metadata for downstream logging.
- **`memory_manager`**: Manages the two-tier hierarchical memory:
    - **L0 (Iteration-level)**: Short-term logs of attempts, failures, and intermediate metrics.
    - **L1 (Journal-level)**: Long-term extracted insights (meta-learning) that persist across iterations.
- **`metrics_holder`**: A thread-safe container (`BestMetricsHolder`) that tracking the "Current Best" performance (speedup/correctness).
- **`run_recorder`**: Handles filesystem logging and writes the canonical `metrics_by_time.jsonl` traces found in result folders.
  - `chat_history.jsonl` now records `assistant_reasoning` / `assistant_reasoning_content` keys when present.
  - `evaluation_terminal_output.jsonl` stores per-iteration code-evaluation terminal output.

---

## 3. Workflow & Data Flow

1. **Initialization**: `evolve_kb_batch.py` reads the problem subset and initializes a `KBGovernor` for each task.
2. **Context Building**: The governor fetches insights from the **L1 Journal** to prep the LLM with "learned" CUDA best practices.
3. **Generation**: `llm_client` receives a prompt (augmented with **L0** history) and generates a candidate `ModelNew` class.
4. **Evaluation**: `KBGovernor` executes the code via KernelBench's evaluation engine.
5. **Reflection**: The iteration's results are summarized, updating the **L0** memory and potentially promoting insights to **L1**.
6. **Persistence**: `run_recorder` saves snapshots of the code and metrics to `results/evolving_logs/<run_name>/`.
  - Includes iteration-level terminal output logs for evaluation/execution results.
7. **Aggregation**: Once the batch finishes, `evolve_kb_batch.py` flattens the results into a level-first `eval_results.json` for standard analysis.

---

## 4. Implementation Notes

### Pydantic & `sys.modules` Caching
To prevent `ValidationError` or "Model already defined" errors during dynamic loading (common when scripts and libraries share Pydantic schemas), the `KBGovernor` implements a custom module caching mechanism. It ensures that `sys.modules` keeps a single, stable instance of the shared schemas across the process lifecycle.

### CUDA Context Safety
Since multiple iterations run CUDA code, the system is designed to handle potential GPU memory leakage or kernel crashes by isolating evaluations where possible, though the main loop is currently sequential to preserve the evolutionary chain state.
