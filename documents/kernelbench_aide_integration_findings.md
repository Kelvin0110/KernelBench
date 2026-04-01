# KernelBench and AIDE Integration Findings

## 1. Structure of `src/kernelbench`
The core engine files handle the end-to-end evaluation pipeline:
- **`dataset.py`**: Fetches the problem specs (from HuggingFace or locally) to be solved.
- **`prompts/` & `prompt_constructor_toml.py`**: Assemble the structured context sent to the LLM, including code templates, few-shot examples, and chain-of-thought steps.
- **`compile.py` / `kernel_static_checker.py`**: Validates the syntax, statically analyzes the generated code, and handles JIT compiling of the raw string to a runtime module.
- **`eval.py`**: The "judge". Takes the generated string, compiles it (CUDA/Triton), runs it against random inputs to check correctness (`n_correctness` runs), and measures the execution time to compare against the PyTorch baseline (`n_trial` runs).
- **`score.py` & `timing.py`**: Provide functions for profiling and determining the overall benchmark metric, usually denoted as `fast_p` (fraction of correct kernels faster than the baseline multiplier `p`).

## 2. Normal Evaluation Flow 
According to `README.md` and `TUTORIAL.md`, the standard pipeline is:
1. **Baseline Creation (`scripts/generate_baseline_time.py`)**: Generate profiling baselines natively on the exact GPU hardware running the test.
2. **Generation (`scripts/generate_samples.py`)**: Iterate over the dataset to let the LLM output raw CUDA/Triton kernels into an output directory.
3. **Evaluation (`scripts/eval_from_generations.py`)**: Iterate through the generated `.py` assets, run the `eval.py` pipeline to ensure they compile, are numerically correct, and log their runtime.
4. **Analysis (`scripts/benchmark_eval_analysis.py`)**: Cross-reference the evaluation results against the initial baselines to establish `fast_p` scores.

## 3. AIDE Framework Integration & The 1st Evolving Agent Prototype
The AIDE (Self-Evolving-Agent) integration lives in `scripts_integration/evolving_agent/` and `scripts_integration/docker/`. 

Rather than standard one-shot generation, the AIDE integration wraps KernelBench to form a reinforcing, iteratively improving tree-search loop:
- **AIDE Tree Search**: AIDE manages a tree of prospective solutions. A node (candidate code) is evaluated, and the numerical/textual feedback dictates how the model iterates on it.
- **`kb_evolving_governor.py`**: The glue between AIDE and KernelBench. It orchestrates a `KernelBenchGovernor` wrapper. 
- **`kb_harness_template.py`**: During AIDE's evaluation step, the candidate kernel isn't blindly merged into the host space. The governor drops the code into a sandbox alongside an evaluation harness template. The isolated harness executes `eval.py` strictly on the generated code.
- **Feedback Loop**: When the harness executes, it doesn't just return pass/fail. It captures verbose PyTorch profiler traces, compiler stack traces, memory faults, and precise numerical mismatch deltas (if `torch.allclose` fails). The `governor` maps this raw data—along with the measured `speedup` penalty logic—directly back into AIDE as an observation. AIDE then prompts the model (via its Feedback/Code actors) with the compilation errors or poor performance logs to fix or optimize the kernel in the next iteration step.
- **Docker Isolation (`docker_single_run.py`)**: To prevent iterative code generation from permanently destroying the system context or exhausting system resources, the loop is wrapped inside a managed Docker container with explicit SIGTERM/SIGKILL bounds and GPU VRAM reservation mapping constraints.
