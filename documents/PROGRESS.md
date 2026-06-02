# Project Progress & Development Ledger

## Current Phase: Scoped Implementation
- **Active Goal**: Complete the `scripts_integration/new_evolving_agent` integration path with evolving_common-driven governor loop, recorder-backed metrics, subset batch orchestration, and GPU memory reservation around evaluation.
- **Critical Reminders**:
  - Keep `eval_results.json` in level-first shape: `{level: {problem_id: [entries]}}`.
  - Preserve runtime fields (`runtime`, `runtime_stats`) for downstream analysis.
  - Handle optional integration dependencies gracefully (LLM/memory helpers may be unavailable locally).
  - Ensure new integrations use `evolving_common` helpers (`BestMetricsHolder`, `BenchmarkRunRecorder`, memory/prompt helpers) instead of duplicating local logic.
  - Hoard and release GPU memory only through `evolving_common.governor.gpu_reserver.GPUMemoryReserver` when wrapping eval execution.

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

### 2026-04-14 - GitHub Copilot
- **Feature**: Added reasoning field propagation and per-iteration evaluation terminal-output logging for the new evolving-agent integration.
- **Implementation**:
  - Updated `Self-Evolving-Agent/evolving_common/llm_client.py` to extract reasoning from NVIDIA/OpenAI-compatible responses (handling both `reasoning` and `reasoning_content` keys).
  - Updated `Self-Evolving-Agent/evolving_common/run_recorder.py` so `chat_history.jsonl` stores explicit `assistant_reasoning` keys.
  - Added `evaluation_terminal_output.jsonl` recorder stream for per-iteration code evaluation/execution terminal logs.
  - Updated `scripts_integration/new_evolving_agent/kb_governor.py` to capture KernelBench eval stdout/stderr, propagate it into `KBEvalResult.terminal_output`, and persist iteration terminal logs.
  - Updated `Self-Evolving-Agent/kernelbench/schemas.py` with `terminal_output` on `KBEvalResult`.
  - Added tests:
    - `Self-Evolving-Agent/tests/test_kernelbench_adapter.py`
    - Extended `scripts_integration/new_evolving_agent/test_kb_governor.py` to verify terminal-output capture/persistence.
- **Impact**:
  - LLM calls now preserve reasoning payloads for downstream analysis.
  - Iteration artifacts now include structured terminal output for each code evaluation/execution step, improving debuggability.
- **Status**: Completed

### 2026-04-14 - GitHub Copilot
- **Feature**: Fixed non-serializable evaluation metadata crash in batch JSON dumping and added runtime-focused metrics logging.
- **Implementation**:
  - Reproduced and traced batch failure (`TypeError: Object of type RuntimeError is not JSON serializable`) to raw exception objects carried in nested run metadata during `_write_json(...)`.
  - Updated `scripts_integration/new_evolving_agent/evolve_kb_batch.py` JSON persistence to use a safe `default` encoder that converts exception/path objects to strings instead of crashing writes.
  - Extended `scripts_integration/new_evolving_agent/kb_governor.py` iteration logging to include runtime and reference runtime in:
    - terminal logs (`KERNEL_BENCH_RUNTIME`, `KERNEL_BENCH_REF_RUNTIME`),
    - `evaluation_terminal_output.jsonl` `extra` payload,
    - `metrics_iteration` / `metrics_best` snapshots.
  - Extended batch summaries in `evolve_kb_batch.py` to report runtime aggregation (`best_runtime_overall`, per-level `best_runtime`).
  - Added regression coverage:
    - `scripts_integration/new_evolving_agent/test_evolve_kb_batch.py::test_write_json_serializes_exception_objects`
    - updated `scripts_integration/new_evolving_agent/test_kb_governor.py` to assert runtime appears in evaluation terminal-output logs.
- **Impact**:
  - Batch runs no longer fail mid-run when evaluation metadata includes exception objects.
  - Runtime is now extracted and logged alongside speedup, improving observability for performance analysis.
- **Status**: Completed

### 2026-04-15 - GitHub Copilot
- **Feature**: Implemented structured L0/L1 prompt updates with extractor-driven L1 selection for `new_evolving_agent`.
- **Implementation**:
  - Added design and implementation artifacts:
    - `docs/superpowers/specs/2026-04-15-kb-l0-l1-prompt-memory-selection-design.md`
    - `docs/superpowers/plans/2026-04-15-kb-l0-l1-prompt-memory-selection.md`
  - Updated shared memory/prompt stack:
    - `Self-Evolving-Agent/evolving_common/memory_manager.py`
      - Added structured L1 JSONL support (`append_l1_jsonl`, `read_l1_jsonl`, `resolve_l1_jsonl_path`).
      - Promotion now writes both `.txt` and `.jsonl` outputs.
      - Added optional `clear_l0_after_promotion` behavior to support L0 retention.
    - `Self-Evolving-Agent/evolving_common/prompt_context.py`
      - Added structured user-prompt sections for selected L1 entries, allowed actions, iteration context, and latest eval feedback.
      - Updated summarizer prompts to require `Description:` and `Details:` format for L1 memories.
    - `Self-Evolving-Agent/evolving_common/governor/promotion.py`
      - Propagates `clear_l0_after_promotion` to memory promotion.
    - `Self-Evolving-Agent/evolving_common/llm_client.py`
      - Added dedicated extractor model path (`get_tri_llm_model_ids`, `call_extractor`, `call_extractor_with_meta`).
  - Updated KernelBench governor/config integration:
    - `Self-Evolving-Agent/kernelbench/config.py`
      - Added extractor controls (`extractor_model`, `extractor_max_memories`, `extractor_max_tokens`, `extractor_timeout_sec`, `enable_l1_extractor`).
    - `scripts_integration/new_evolving_agent/kb_governor.py`
      - Added extractor stage to select relevant L1 entries (configurable top-k, default 10).
      - Updated coder system prompt to include action-policy and memory semantics.
      - Preserved one-system/one-user message shape while enriching user prompt structure.
      - Added optional `<action>`/`<reasoning>` tag parsing and L0 logging.
      - Disabled L0 clearing on promotion for this governor path.
      - Restricted coder prompt L1 context to extractor-selected entries (full L1 text is no longer injected when selection is available).
      - Added richer iteration context including best metrics so far.
      - Added structured L0 rendering for coder prompts (action/reasoning/code/terminal grouped by attempt history).
  - Added/updated tests:
    - `Self-Evolving-Agent/tests/test_memory_manager.py`
    - `Self-Evolving-Agent/tests/test_prompt_context.py`
    - `scripts_integration/new_evolving_agent/tests/test_kb_governor.py`
    - `scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py` (fixed missing `sys` import for `sys.argv` monkeypatch).
    - Added malformed JSONL resilience coverage in `Self-Evolving-Agent/tests/test_memory_manager.py`.
- **Impact**:
  - Coder prompts now consume structured L0 + selectively retrieved L1 memories without switching to full chat-history replay.
  - L1 memory is now machine-readable and queryable while preserving existing journal text output.
  - Iterative runs can retain richer L0 context across promotions.
  - Runs are now resilient to malformed lines in `shared_l1.jsonl` (invalid lines are skipped rather than aborting the run).
- **Status**: Completed

### 2026-04-15 - GitHub Copilot
- **Feature**: Refined memory-promotion defaults, description constraints, and extractor-model override semantics based on follow-up review.
- **Implementation**:
  - Updated `Self-Evolving-Agent/evolving_common/memory_manager.py`:
    - Added `DEFAULT_L1_DESCRIPTION_MAX_TOKENS = 512`.
    - Changed `promote_l0_to_l1(..., clear_l0_after_promotion=False)` default so L0 is retained unless explicitly cleared.
    - Kept robust malformed JSONL handling in `read_l1_jsonl`.
  - Updated `Self-Evolving-Agent/evolving_common/governor/promotion.py`:
    - Changed `maybe_promote_l0_to_l1(..., clear_l0_after_promotion=False)` default to align with retained-L0 design.
  - Updated `Self-Evolving-Agent/evolving_common/prompt_context.py`:
    - Summarizer prompt now explicitly states description max length (`512` chars) for deterministic extraction behavior.
  - Updated `Self-Evolving-Agent/kernelbench/config.py`:
    - Changed `extractor_model` to optional (`str | None`), allowing env-driven default model resolution when unset.
  - Updated `scripts_integration/new_evolving_agent/kb_governor.py`:
    - Extractor call now only passes explicit model override when configured; otherwise relies on llm client env/default selection.
  - Updated tests:
    - `Self-Evolving-Agent/tests/test_memory_manager.py`
    - `scripts_integration/new_evolving_agent/tests/test_kb_governor.py`
- **Impact**:
  - Default memory behavior now matches retained-L0 strategy across both direct promotion and governor promotion helpers.
  - Description extraction is no longer silently constrained to 200 chars and is now contract-aligned with prompt guidance.
  - Extractor model config no longer duplicates env settings by force; per-run override remains available.
- **Status**: Completed

## Refinement of L1 Summarization and L0 Retention Policy (2026-04-15)
- **Goal**: Transition L1 summarization from generic summaries to high-density engineering logs and enforce L0 retention by default.
- **Actions**:
  - Updated `Self-Evolving-Agent/evolving_common/prompt_context.py`:
    - Redesigned `SUMMARIZER_SYSTEM_PROMPT` to enforce a structured, technical schema: "Description", "Lesson Learned", "What Failed", "What Worked", and "Strategies for Success".
    - Updated `build_summarizer_user_message` to provide explicit formatting examples and emphasize technical specificity.
  - Updated `scripts_integration/new_evolving_agent/kb_governor.py`:
    - Modified `maybe_promote_l0_to_l1` call to explicitly set `clear_l0_after_promotion=False`, ensuring L0 history remains available for the coder model even after summarization.
- **Impact**:
  - L1 entries now capture deep CUDA/kernel insights, following the detailed pattern requested for better cross-problem learning.
  - L0 history is now fully preserved across promotions, allowing the coder to see the immediate context of recent attempts alongside the broader L1 journal.
- **Status**: Completed

## Generalization and Optimization of L1 Summarization Loop (2026-04-15)
- **Goal**: Generalize summarizer prompts for all ML benchmaks and prevent redundant per-iteration summarization when L0 is retained.
- **Actions**:
  - Updated `Self-Evolving-Agent/evolving_common/prompt_context.py`:
    - Refined `SUMMARIZER_SYSTEM_PROMPT` to be benchmark-agnostic, replacing CUDA-specific examples with general ML failure modes (e.g., "vanishing gradients", "OOM").
  - Updated `Self-Evolving-Agent/evolving_common/memory_manager.py`:
    - Modified `should_promote_l0` to accept `last_promoted_count`. It now triggers based on *new* entries since the last promotion, preventing redundant summaries when L0 is not cleared.
  - Updated `Self-Evolving-Agent/evolving_common/governor/promotion.py`:
    - Updated `maybe_promote_l0_to_l1` to track and return the incremented `last_promoted_count`.
  - Updated `scripts_integration/new_evolving_agent/kb_governor.py`:
    - Applied state tracking for `self._last_promoted_count` to the main execution loop.
- **Impact**:
  - The framework is now cleaner and more applicable to a wider range of ML tasks.
  - Promotion logic is now efficient: even if L0 is retained for the coder's benefit, the summarizer is only called when a significant batch of *new* work has been accumulated.
- **Status**: Completed

### Generalized Common Components and Code Migration (2026-04-16)

- **Prompt Generalization:** 
    - Moved `ALLOWED_CODER_ACTIONS` and `DEFAULT_EXTRACTOR_SYSTEM_PROMPT` to `evolving_common/prompt_context.py`.
    - Introduced `BASE_EVOLVING_CODER_SYSTEM_PROMPT` as a template for evolving agent coder prompts.
    - Standardized `format_l0_for_coder_prompt` in common utilities to handle structured L0 attempts (action/reasoning/code/terminal).
- **Core Utility Migration:**
    - Created `evolving_common/governor/util.py` for generic tag extraction (`extract_optional_tag`) and fallback action selection logic.
    - Moved extractor parsing logic to `evolving_common/governor/code_extract.py` via `parse_selected_entry_ids`.
    - Migrated subprocess-based evaluation orchestration to `evolving_common/execution.py` (`evaluate_in_subprocess`).
- **KernelBench Adaptation:**
    - Refactored `kb_governor.py` to import and utilize all shared components from `evolving_common`.
    - Simplified `KBGovernor` by removing redundant local implementations of parsing and formatting.

### 2026-04-16 - GitHub Copilot
- **Feature**: Added cached run-level performance visualization pipeline for the KernelBench web visualizer.
- **Implementation**:
  - Added `Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py`.
    - Reads each run's `workspaces/*/metrics_by_iteration.jsonl`.
    - Resolves baseline runtimes using `results/timing/<hardware>/<baseline>.json` + `kernelbench.dataset` mapping.
    - Computes per-iteration per-problem `current_speedup` / `best_speedup` (50 dots/iteration expected for full subset runs).
    - Computes per-iteration fast-p curves using default thresholds `[0.0, 0.5, 0.8, 1.0, 1.5, 2.0]`.
    - Caches output to `runs_evolving/<run_name>/visualizations/performance_stats.json`.
  - Extended `Self-Evolving-Agent/visualizations/kernelbench/server/app.py` with:
    - `GET /api/runs/{run_name}/performance-stats` to serve cached chart stats without scanning workspace directories on each UI request.
  - Extended `Self-Evolving-Agent/visualizations/kernelbench/index.html` with:
    - Sidebar button flow: `Load Run Charts` / `Refresh Run Charts` / `Show Charts`.
    - Plot controls: `current|best` speedup, `mean|median` aggregate line, and log-scale toggle.
    - Two charts: speedup scatter+aggregate line and multi-line fast-p plot.
  - Updated `Self-Evolving-Agent/visualizations/README.md` with cached-chart generation and usage instructions.
- **Impact**:
  - Visualization is now scalable for run-level trend analysis and does not re-walk all problem folders during frontend chart rendering.
  - Users can compare per-iteration speedup distributions and fast-p progression directly in the UI for selected runs.
- **Status**: Completed

### 2026-04-16 - GitHub Copilot
- **Feature**: Added reusable performance-stats helpers, AIDE checkpoint fast-p extraction, PDF export, and web AIDE-vs-evolving comparison.
- **Implementation**:
  - Added shared helper module `src/kernelbench/performance_stats.py` for reusable parsing/stat aggregation utilities:
    - fast-p threshold parsing and score computation,
    - subset CSV loading,
    - baseline lookup construction,
    - checkpoint/sample extraction helpers,
    - autoscaling and series-alignment helpers for charting.
  - Refactored `Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py` to use shared helpers and changed fast-p to use **best runtime** (while keeping compatibility aliases in payload keys).
  - Added `Self-Evolving-Agent/visualizations/kernelbench/server/generate_aide_integration_stats.py`:
    - Scans `run_integration/docker_level_x_<model>_step<step>` (supports optional `inte_` prefix),
    - reads subset problems from `subset_selection/selected_problems_50.csv`,
    - extracts node checkpoint runtimes from `checkpoints/node_*/eval_results.json`,
    - computes fast-p curves across levels and writes `run_integration/analysis/aide_subset_<model>_step<step>.json`.
  - Added `Self-Evolving-Agent/visualizations/kernelbench/server/export_performance_plots_pdf.py` for PDF export of:
    - speedup scatter+aggregate,
    - evolving fast-p curves,
    - selected-threshold AIDE-vs-evolving fast-p comparison (overlapping iterations only).
  - Extended `Self-Evolving-Agent/visualizations/kernelbench/server/app.py`:
    - fixed refresh command flag (`--runs-root`),
    - added AIDE stats endpoints:
      - `GET /api/integration/aide-fastp?model=...&step=...`
      - `POST /api/integration/aide-fastp/refresh?model=...&step=...`.
  - Extended `Self-Evolving-Agent/visualizations/kernelbench/index.html`:
    - dynamic fast-p autoscaling,
    - AIDE model/step controls and refresh/load actions,
    - p-threshold selector,
    - AIDE-vs-evolving fast-p comparison chart.
  - Added tests in `scripts_integration/new_evolving_agent/tests/test_performance_stats.py` for:
    - threshold parsing,
    - runtime-mode-dependent fast-p,
    - comparison alignment,
    - best-runtime fast-p behavior in evolving stats generation.
  - Updated `Self-Evolving-Agent/visualizations/README.md` and added `matplotlib` to `pyproject.toml` `vis` dependency group for PDF plotting.
- **Impact**:
  - Web dashboard and exported PDFs now support direct AIDE-vs-evolving fast-p comparisons.
  - fast-p semantics for evolving runs now match best-runtime intent.
  - Performance-stat logic is centralized and reusable across scripts/API flows.
- **Status**: Completed

### 2026-05-07 - GitHub Copilot
- **Feature**: Refactored extractor core to common prompts and enhanced L1 promotion with source-problem tagging.
- **Implementation**:
  - **Prompt Generalization**: Extracted  to [`evolving_common/prompt_context.py`](Self-Evolving-Agent/evolving_common/prompt_context.py), standardizing how the agent selects relevant refined knowledge entries.
  - **Memory Evolution**: Updated [`evolving_common/governor/promotion.py`](Self-Evolving-Agent/evolving_common/governor/promotion.py) to accept an optional `source` field during L0 -> L1 promotion.
  - **KernelBench Wiring**: Refactored [`scripts_integration/new_evolving_agent/kb_governor.py`](scripts_integration/new_evolving_agent/kb_governor.py) to:
    - Use the generalized extractor prompt builder.
    - Pass problem-specific source tags (e.g., `Level x problem y`) into the L1 journal, improving cross-task searchability and audit trails.
- **Impact**: Standardizes the "Two-Loop" architecture's extraction phase and provides better provenance for refined engineering insights in the persistent L1 knowledge base.
- **Status**: Completed

### 2026-05-07 - GitHub Copilot (Repository Updates)
- **Feature**: Stabilized visualization logic and solidified KernelBench fast-p consistency.
- **Implementation** (Self-Evolving-Agent [`98ece63`](https://github.com/Kelvin0110/Self-Evolving-Agent/commit/98ece639f49715222a8ffbbe5738e861862d956a)):
  - Refactored web dashboard specifically at [`visualizations/kernelbench/index.html`](Self-Evolving-Agent/visualizations/kernelbench/index.html) to support interactive threshold toggling (legend clicking) and improved line-dot markers for fast-p progression.
  - Hardened fast-p data lookup logic to handle numeric key mismatches between iteration snapshots and pre-aggregated series.
  - Simplified [`generate_run_performance_stats.py`](Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py) by removing redundant compatibility logic for `fast_p_current` and standardizing on the `fast_p_best` metric.
- **Implementation** (KernelBench [`9704fc5`](https://github.com/ScalingIntelligence/KernelBench/commit/9704fc561d579607a44c0186eb2598ef38da3e92)):
  - Updated [`Self-Evolving-Agent`](Self-Evolving-Agent) submodule pointer to incorporate the visualization and logic refinements.
  - Refined [`visualizations/kernelbench/README.md`](visualizations/kernelbench/README.md) usage instructions for PDF export.
- **Impact**: Provides a more robust and interactive visualization suite for comparing Gen 2 Evolving Agent performance against baselines and AIDE results.
- **Status**: Completed

### 2026-05-07 - Consolidated Progress: Extractor, L1 Provenance, and Debugging
- **Scope**: Consolidate several May 7 updates into one cohesive summary covering extractor prompt generalization, L0/L1 enhancements, provenance tagging, and targeted runtime/debug improvements.
- **Key Changes**:
  - Moved extractor prompt builder into the common prompt utilities and standardized the extractor user message format to include iteration context and a candidate code preview.
  - Enhanced L0 records to capture `coder_diagnosis` and `coder_hypothesis` tags alongside existing `coder_action` and `coder_reasoning` entries, and rendered them in `format_l0_for_coder_prompt`.
  - Added `source` provenance when promoting L0 -> L1 so L1 entries record the originating problem (e.g., "Level X problem Y") for better cross-task retrieval and audit.
  - Improved `maybe_promote_l0_to_l1`/promotion flow to accept and forward `source`, and made L1 writes include structured JSONL plus the human-readable journal block.
  - Updated `kb_governor.py` to use the generalized extractor prompt builder, include candidate code preview in latest evaluation feedback, append diagnosis/hypothesis into L0, and log configured iteration budgets for improved local debugging.
  - Added concise debug prints in batch and governor runners to expose `max_iterations` and iteration-loop entry when `verbose=True` to help trace early loop exits in real runs.
  - Fixed various resiliency issues around evaluation metadata serialization, subprocess isolation for builds, and invalid runtime normalization so multi-iteration runs do not abort prematurely.
- **Files touched (representative)**:
  - Self-Evolving-Agent/evolving_common/prompt_context.py
  - Self-Evolving-Agent/evolving_common/memory_manager.py
  - Self-Evolving-Agent/evolving_common/governor/promotion.py
  - scripts_integration/new_evolving_agent/kb_governor.py
  - scripts_integration/new_evolving_agent/evolve_kb_batch.py
- **Impact**: Standardizes the Two-Loop extraction stage, improves L1 provenance and searchability, enriches L0 context for coder decisions, and provides better runtime debug traces to diagnose early iteration exits or build hangs.
- **Status**: Completed

### 2026-05-07 - GitHub Copilot
- **Feature**: Refactored extractor core to common prompts and enhanced L1 promotion with source-problem tagging.
- **Implementation**:
  - **Prompt Generalization**: Extracted `build_extractor_user_message` to [`evolving_common/prompt_context.py`](Self-Evolving-Agent/evolving_common/prompt_context.py), standardizing how the agent selects relevant refined knowledge entries.
  - **Memory Evolution**: Updated [`evolving_common/governor/promotion.py`](Self-Evolving-Agent/evolving_common/governor/promotion.py) to accept an optional `source` field during L0 -> L1 promotion.
  - **KernelBench Wiring**: Refactored [`scripts_integration/new_evolving_agent/kb_governor.py`](scripts_integration/new_evolving_agent/kb_governor.py) to:
    - Use the generalized extractor prompt builder.
    - Pass problem-specific source tags (e.g., `Level x problem y`) into the L1 journal, improving cross-task searchability and audit trails.
- **Impact**: Standardizes the "Two-Loop" architecture's extraction phase and provides better provenance for refined engineering insights in the persistent L1 knowledge base.
- **Status**: Completed

### 2026-05-07 - GitHub Copilot (Repository Updates)
- **Feature**: Stabilized visualization logic and solidified KernelBench fast-p consistency.
- **Implementation** (Self-Evolving-Agent [`98ece63`](https://github.com/Kelvin0110/Self-Evolving-Agent/commit/98ece639f49715222a8ffbbe5738e861862d956a)):
  - Refactored web dashboard specifically at [`visualizations/kernelbench/index.html`](Self-Evolving-Agent/visualizations/kernelbench/index.html) to support interactive threshold toggling (legend clicking) and improved line-dot markers for fast-p progression.
  - Hardened fast-p data lookup logic to handle numeric key mismatches between iteration snapshots and pre-aggregated series.
  - Simplified [`generate_run_performance_stats.py`](Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py) by removing redundant compatibility logic for `fast_p_current` and standardizing on the `fast_p_best` metric.
- **Implementation** (KernelBench [`9704fc5`](https://github.com/ScalingIntelligence/KernelBench/commit/9704fc561d579607a44c0186eb2598ef38da3e92)):
  - Updated [`Self-Evolving-Agent`](Self-Evolving-Agent) submodule pointer to incorporate the visualization and logic refinements.
  - Refined [`visualizations/kernelbench/README.md`](visualizations/kernelbench/README.md) usage instructions for PDF export.
- **Impact**: Provides a more robust and interactive visualization suite for comparing Gen 2 Evolving Agent performance against baselines and AIDE results.
- **Status**: Completed

### 2026-05-08 - GitHub Copilot
- **Feature**: Added GPU memory reservation for KernelBench governor evaluation windows.
- **Implementation**:
  - Created `Self-Evolving-Agent/evolving_common/governor/gpu_reserver.py` with `GPUMemoryReserver`, which allocates a large `torch.uint8` CUDA tensor on acquire and calls `torch.cuda.empty_cache()` on release.
  - Updated `scripts_integration/new_evolving_agent/kb_governor.py` to instantiate `GPUMemoryReserver(reserve_gb=12.0)` during governor initialization.
  - Wrapped `_evaluate_candidate(...)` so GPU memory is released immediately before KernelBench eval and reacquired in a `finally` block.
- **Impact**: The governor now frees reserved VRAM while isolated CUDA evaluation is running, reducing contention and avoiding avoidable OOM pressure on shared GPUs.
- **Status**: Completed

### 2026-06-02 - GitHub Copilot
- **Feature**: Refined Docker batch orchestration and single-run GPU reservation behavior for AIDE + KernelBench.
- **Implementation**:
  - Updated [`scripts_integration/docker/docker_batch_run.py`](scripts_integration/docker/docker_batch_run.py) to parse CLI-friendly `problem_ids` / `subset` values and expose a `max_problems` cap for subset-style batch runs.
  - Updated [`scripts_integration/docker/docker_single_run.py`](scripts_integration/docker/docker_single_run.py) to use a held `GPUMemoryReserver` tensor during active runs and release it during cleanup instead of only warming the cache.
  - Added [`scripts_integration/docker/RUN_WITH_UV.md`](scripts_integration/docker/RUN_WITH_UV.md) with the current `uv run` batch commands for GPT-OSS, Kimi Thinking, checkpointed runs, and subset examples.
- **Impact**: Docker runs now preserve file ownership on mounted volumes, can be limited to a smaller problem slice without editing code, and keep GPU reservation semantics aligned with the shared evolving-agent helper.
- **Status**: Completed