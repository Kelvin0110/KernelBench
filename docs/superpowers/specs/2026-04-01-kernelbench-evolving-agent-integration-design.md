# KernelBench Self-Evolving-Agent (SEA) Integration Design

## 1. Objective
Integrate the newly developed `Self-Evolving-Agent` memory framework with `KernelBench` to evaluate its ability to write optimal GPU kernels iteratively. The new architecture should abandon legacy AIDE scripts to become a standalone evaluation track, leveraging `KernelBench`'s exact evaluation harness while enabling both local (within-problem) and global (cross-problem) memory-augmented learning.

## 2. Architecture & Components
This integration will strictly follow the **"Harness Wrapper"** approach, keeping the `SelfEvolvingAgent` abstracted from `eval.py` details.

All code will reside in `scripts_integration/self_evolving_agent/`:

- **`kb_environment.py`**: An adapter for KernelBench internals.
  - Formats problem prompts via `prompt_constructor_toml.py`.
  - Executes generated code (reuse `eval.run_and_check_code`).
  - Standardizes the response feedback payload strictly parsing correctness, speedup, and error traces.
- **`kb_agent.py`**: Extends `self_evolving_agent.agent.core.SelfEvolvingAgent`.
  - Uses `IterationController` to drive code improvement on compilation or performance failures (Local Memory).
  - Fetches contextual kernel optimization strategies prior to the generation step (Global Memory).
- **`run_batch.py`**: The main execution harness for batched problem subsets (e.g., `subset_selection/selected_problems_50.csv`).
  - Sets up the `LocalMemory` and `GlobalMemory` backends.
  - Iterates cleanly through levels and tasks.

## 3. Metrics and Evaluation Payload Map
To stay completely backwards-compatible with `benchmark_eval_analysis.py` and prior benchmarks, logs must be recorded exactly like `eval_from_generations.py` outputs. 

Specifically, `eval_results.json` will be nested natively by problem_id arrays over sample outputs (with the file structure separating levels, or keying directly by problem_id nested inside level):

```json
{
  "100": [
    {
      "sample_id": 0,
      "compiled": false,
      "correctness": false,
      "metadata": {
        "hardware": "NVIDIA RTX A6000",
        "device": "cuda:0",
        "correctness_trials": "(5 / 5)",
        "source": "evolving_agent_prototype",
        "level": 1,
        "problem_id": 100,
        "best_speedup": 0.0,
        "backend": "cuda",
        "precision": "fp32",
        "iterations_run": 0,
        "error": "Compiler stacktrace here..."
      },
      "runtime": 25.1,
      "runtime_stats": {
        "mean": 25.1,
        "std": 26.1,
        "min": 22.2,
        "max": 285.0,
        "num_trials": 100,
        "hardware": "NVIDIA RTX A6000",
        "device": "cuda:0"
      }
    }
  ]
}
```
*(If correct, `runtime` and `runtime_stats` expand out mean/std/min/max from trials).*

## 4. Execution Lifecycle per Problem
1. **Init**: The global loop provides problem metadata to `kb_agent.py`.
2. **Strategy Fetch**: The agent checks the chroma-backend DB for overarching strategies (e.g. Triton tuning rules).
3. **Execution Loop**:
    - Agent generates a kernel.
    - `kb_environment.py` executes it, providing numerical mismatch output or `torch.OutOfMemoryError` strings.
    - If `correctness` is false, or `speedup < target`, the error and code are logged to `local_memory`.
    - Iteration continues up to `max_steps`.
4. **Reflection**: Output trajectory is appended to `global_memory` with an extraction prompt pulling insights.
5. **Output**: Result dict inserted seamlessly into the JSON dictionary architecture defined above.
