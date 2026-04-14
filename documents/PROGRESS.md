# Project Progress & Development Ledger

## Current Phase: Scoped Implementation
- **Active Goal**: Complete the new `scripts_integration/new_evolving_agent` integration path with evolving_common-driven governor loop, recorder-backed metrics, and subset batch orchestration.
- **Critical Reminders**:
  - Keep `eval_results.json` in level-first shape: `{level: {problem_id: [entries]}}`.
  - Preserve runtime fields (`runtime`, `runtime_stats`) for downstream analysis.
  - Handle optional integration dependencies gracefully (LLM/memory helpers may be unavailable locally).
  - Ensure new integrations use `evolving_common` helpers (`BestMetricsHolder`, `BenchmarkRunRecorder`, memory/prompt helpers) instead of duplicating local logic.

---

## Development Log

### 2026-04-01 - GitHub Copilot
- **Feature**: Subset-driven metrics schema + runtime propagation for evolving agent integration.
- **Implementation**:
  - Modified `scripts_integration/evolving_agent/evolve_kb_batch.py`.
  - Updated `_to_kernelbench_eval_entry` to include runtime/runtime stats and hardware/device metadata passthrough.
  - Added `_normalize_level_first_eval_doc` to migrate legacy flat eval payloads into level-first shape.
  - Updated batch write path to persist `eval_results.json` as `{level -> problem_id -> [entries]}`.
  - Modified `scripts_integration/evolving_agent/kb_evolving_governor.py` to harden import fallbacks, normalize error handling, and propagate runtime/metadata from isolated eval workers.
- **Impact**: Batch outputs now match the requested schema and carry richer timing metadata for analysis; governor behavior is safer under missing optional dependencies and non-string errors.
- **Status**: Completed

### 2026-04-01 - GitHub Copilot
- **Feature**: Test coverage for schema and governor robustness.
- **Implementation**:
  - Added `scripts_integration/evolving_agent/test_evolve_kb_batch.py`.
  - Extended `scripts_integration/evolving_agent/test_kb_evolving_governor.py`.
  - Tests cover level-first output shape, runtime propagation, and `_is_fatal_cuda_error` handling of exception objects.
- **Impact**: Prevents regressions for the new result format and known runtime-error edge cases.
- **Status**: Completed

### 2026-04-01 - GitHub Copilot
- **Feature**: Added a second integration script set inside the `Self-Evolving-Agent` module tree.
- **Implementation**:
  - Created `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/environment.py` for KernelBench dataset/prompt/eval adaptation.
  - Created `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py` implementing a `SelfEvolvingAgent` subclass for iterative KernelBench solving.
  - Created `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/reflection.py` with a minimal reflection engine to promote distilled traces.
  - Created `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/batch_runner.py` for subset-driven batch execution and level-first JSON output.
  - Added `Self-Evolving-Agent/tests/test_kernelbench_batch_runner.py` with coverage for subset parsing, runtime metadata mapping, and level-first persistence schema.
- **Impact**: The integration now exists in both the legacy prototype path and the canonical `self_evolving_agent` package path requested for modular architecture.
- **Status**: Completed

### 2026-04-01 - GitHub Copilot
- **Feature**: Plan-aligned script entrypoints created under `scripts_integration/self_evolving_agent` with verification and run guide.
- **Implementation**:
  - Added `scripts_integration/self_evolving_agent/kb_environment.py` and `scripts_integration/self_evolving_agent/kb_agent.py` as compatibility entrypoints to package-core logic.
  - Added `scripts_integration/self_evolving_agent/run_batch.py` as subset-driven runner with dry-run mode and level-first metrics output.
  - Added `scripts_integration/self_evolving_agent/RUN_WITH_UV.md` documenting setup and execution commands.
  - Added `scripts_integration/self_evolving_agent/test_run_batch.py`.
  - Improved `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py` reflection behavior to execute immediately when supported.
  - Improved `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/batch_runner.py` to continue after per-task failure and persist failure entries.
  - Added batch-resilience test in `Self-Evolving-Agent/tests/test_kernelbench_batch_runner.py`.
- **Impact**: Implementation now follows the plan’s expected script location while preserving modular core code; failure handling and reflection execution are stronger and covered by tests.
- **Status**: Completed

### 2026-04-01 - GitHub Copilot
- **Feature**: Resolved location decision after plan-review: keep both layers, with scripts as canonical run surface.
- **Implementation**:
  - `scripts_integration/self_evolving_agent/*.py` now contains concrete, runnable orchestration logic and no longer depends on package-only helper modules for subset formatting.
  - `scripts_integration/self_evolving_agent/run_batch.py` now includes local subset parsing, level-first result shaping, and per-task failure continuation.
  - Kept `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/*` as reusable package-core for future direct library usage.
- **Impact**: Plan expectation is satisfied (scripts in `scripts_integration/self_evolving_agent`), while modular architecture remains available for programmatic reuse.
- **Status**: Completed

### 2026-04-02 - GitHub Copilot
- **Feature**: Refactored SEA Integration for Persistent Memory and Iterative Optimization.
- **Implementation**:
  - **Memory Persistence**: Switched from in-memory mocks to `JSONLinesLocalMemory` (per-task JSONL traces) and `ChromaGlobalMemory` (persistent Vector DB for cross-task wisdom) in `scripts_integration/self_evolving_agent/run_batch.py`.
  - **Iterative Optimization**: Updated `KernelBenchEvolvingAgent` in `agent.py` to default `stop_on_first_correct=False`, ensuring the agent continues to refine kernels for speed even after passing correctness checks.
  - **Structured Logging**: Enhanced `batch_runner.py` and `agent.py` to capture and save full iteration traces (`iteration_logs.json`), source code (`kernels/`), and prompts. Folder structure now aligns with integration standards: `logs/level_X_problem_Y/`.
  - **Code Reuse**: Refactored `run_batch.py` to utilize the centralized `run_subset` function from `batch_runner.py`, employing an `agent_factory` pattern to handle fresh local memory per task while maintaining a shared global memory.
- **Impact**: Enables "true self-evolution" where the agent learns strategies across different problems and persists its knowledge base to disk.
- **Status**: Completed

### 2026-04-13 - GitHub Copilot
- **Feature**: Implemented the new KernelBench integration loop under `scripts_integration/new_evolving_agent` with shared evolving_common primitives.
- **Implementation**:
  - Expanded `Self-Evolving-Agent/kernelbench/config.py` with run-level fields used by batch orchestration and recorder/promotion controls.
  - Expanded `Self-Evolving-Agent/kernelbench/schemas.py` with runtime/metadata fields and a new `KBGovernorResult` schema.
  - Rewrote `scripts_integration/new_evolving_agent/kb_governor.py` to implement the full iteration loop and integrate:
    - `evolving_common.prompt_context` for coder/summarizer memory-aware prompts,
    - `evolving_common.llm_client` for coder calls,
    - `evolving_common.benchmark_memory` + `memory_manager` for L0/L1 handling,
    - `BestMetricsHolder` + `BenchmarkRunRecorder` for per-iteration and time-sampled metrics.
  - Added `scripts_integration/new_evolving_agent/evolve_kb_batch.py` to execute subset runs, write level-first `eval_results.json`, per-level result files, and aggregated `run_summary.json`.
  - Added `scripts_integration/new_evolving_agent/RUN_WITH_UV.md` with environment setup, dependency checks, dry-run, and CUDA run commands.
  - Added tests:
    - `scripts_integration/new_evolving_agent/test_kb_governor.py`
    - `scripts_integration/new_evolving_agent/test_evolve_kb_batch.py`
- **Impact**: New integration path now supports full iterative evaluation with common memory/prompt/metric infrastructure and generates both per-problem and aggregated metrics artifacts in the expected format.
- **Status**: Completed

### 2026-04-14 - GitHub Copilot
- **Feature**: Systematic debugging and stabilization of `new_evolving_agent` batch execution for multi-iteration runs.
- **Implementation**:
  - Investigated a stuck run using the exact integration command:
    - `CUDA_VISIBLE_DEVICES=1 uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name new_evolving_gpu_latest --max-problems 2 --max-iterations 3`
  - Verified that earlier hangs were tied to torch extension compile state (`torch.utils.cpp_extension` / lock contention scenarios) and confirmed the run now progresses through all inner-loop iterations.
  - Fixed a hard failure in `scripts_integration/new_evolving_agent/kb_governor.py` where invalid negative runtime values from failed evals propagated into `KBEvalResult(runtime>=0)` and raised `pydantic.ValidationError`.
  - Normalized invalid/negative `runtime` and `ref_runtime` to `None` before constructing `KBEvalResult`, allowing failed candidates to be recorded without aborting governor progress.
  - Re-ran the batch from a clean run directory state and verified full completion for both selected problems with per-iteration artifacts emitted.
- **Impact**:
  - Batch execution no longer stops at iteration recording due to schema validation crashes.
  - `evolving_runs.json` now records `iterations_run=3` entries with structured per-attempt error traces instead of `iterations_run=0` aborted records.
  - Current remaining failures are model-level CUDA correctness/runtime issues (illegal memory access), not orchestrator control-flow failures.
- **Status**: Completed

### 2026-04-14 - GitHub Copilot
- **Feature**: Added per-evaluation process isolation and kernel export artifacts for `scripts_integration/new_evolving_agent`, then validated with a full 3-problem run.
- **Implementation**:
  - Updated `Self-Evolving-Agent/kernelbench/config.py` with `isolate_evaluation_process` and `evaluation_start_method` controls (default spawn isolation on).
  - Reworked `scripts_integration/new_evolving_agent/kb_governor.py` to execute each candidate eval in a dedicated subprocess with timeout handling and structured payload/result marshaling.
  - Added per-attempt unique build directories (`runs_evolving/<run>/builds/l<level>_p<problem>_iter<k>_<id>`) to avoid cross-attempt extension build interference.
  - Added kernel export support in `scripts_integration/new_evolving_agent/evolve_kb_batch.py`:
    - Exports per-problem best/fallback kernel to `runs_evolving/<run>/kernels/level_<level>_problem_<id>_sample_0_kernel.py`.
    - Stores `exported_kernel_path` in `evolving_runs.json` entries.
  - Updated tests in `scripts_integration/new_evolving_agent/test_kb_governor.py` and `scripts_integration/new_evolving_agent/test_evolve_kb_batch.py` for isolation-aware behavior and export helpers.
  - Executed the exact command requested for debugging:
    - `CUDA_VISIBLE_DEVICES=1 uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name debug_agent --max-problems 3 --max-iterations 5`
- **Impact**:
  - Run completed end-to-end (`total_completed=3`) with mixed outcomes: one problem achieved correctness/speedup, while others failed with OOM, demonstrating failures are no longer a universal illegal-memory cascade.
  - Verified exported kernels for all three problems under the run `kernels/` directory.
  - Confirms subprocess isolation mitigates process-wide CUDA poisoning after illegal-address failures while preserving continued batch progress.
- **Status**: Completed
