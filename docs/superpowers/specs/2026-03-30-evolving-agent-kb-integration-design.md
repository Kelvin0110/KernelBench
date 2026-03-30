# Design Spec: Evolving Agent + KernelBench Integration (Shared L1 Batch)

**Date:** 2026-03-30
**Topic:** Integrating Self-Evolving Agent with KernelBench for batch optimization.

## 1. Goal
Integrate the `Self-Evolving-Agent` memory loop into `KernelBench` to run a small batch of selected problems on a GPU. The agent will maximize `KERNEL_BENCH_SPEEDUP` and share `L1 (Refined Knowledge)` across all problems in the batch to facilitate cross-problem learning of GPU kernel optimization techniques.

## 2. Architecture

### 2.1 File Structure
All new files will be located in `d:\HKUST_program\PG\LLM4Sci\KernelBench\scripts_integration\evolving_agent\`:
- `evolve_kb_batch.py`: Orchestrator that reads the subset CSV and manages the batch loop.
- `kb_evolving_governor.py`: Specialized governor logic adapted from `Self-Evolving-Agent/governor.py`.
- `kb_harness_template.py`: Template helper for a harness-style evaluation bridge.

Prototype note: v1 executes KernelBench evaluation directly inside `kb_evolving_governor.py`
using `kernelbench.eval.eval_kernel_against_ref` to reduce moving parts. The harness
template is kept for future parity with docker-style harness flows.

### 2.2 Shared Memory Strategy
- **L1 Journal:** A single `shared_l1.txt` file stored in `results/evolving_logs/<run_name>/`.
- **L0 Buffer:** Initialized fresh for each `(level, problem_id)` to keep trial-specific logs isolated.
- **Promotion:** When L0 exceeds thresholds, the summarizer appends generalizable optimization insights to the shared L1.

## 3. Workflow
1. **Selection:** Load `subset_selection/selected_problems_50.csv`.
2. **Initialization:** Prepare `results/evolving_logs/<run_name>/` and a shared L1 file.
3. **Problem Loop:**
    - Fetch problem details using `KernelBench` dataset utilities.
    - Initialize a fresh L0 buffer.
    - Run the `Governor` loop (up to `max_iterations`, default 20).
    - **Execution:**
        - Coder generates a kernel (CUDA/Triton).
        - Governor evaluates directly via `kernelbench.eval.eval_kernel_against_ref`.
        - Results (speedup/correctness/errors) are captured in L0.
    - **Promotion:** Update shared L1 if criteria met.
4. **Aggregation:** Write final metrics (best speedup per problem) to `eval_results.json`.

## 4. Technical Constraints
- **GPU Presence:** Requires `torch.cuda.is_available()`.
- **Backend:** Defaults to `cuda`, supports Triton/HIP if specified in dataset.
- **Metric:** `Ref_Time / Kernel_Time` (Speedup).
- **Environment:** Runs within the `KernelBench` `uv` environment.

## 5. Success Criteria
- [ ] Successfully iterate through a subset of 10 problems.
- [ ] L1 file contains cross-problem optimization insights (e.g., "Use syncwarp").
- [ ] Final results match the `KernelBench` evaluation schema (`eval_results.json` keyed by problem_id).
