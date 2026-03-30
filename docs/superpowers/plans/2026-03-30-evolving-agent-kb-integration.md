# Evolving Agent + KernelBench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a bridge between the Self-Evolving-Agent (L0/L1 memory loop) and KernelBench to run shared-memory optimization batches on GPU.

**Architecture:** A new integration directory containing a batch runner, a specialized governor for KernelBench metrics, and a harness for evaluation. The system uses a persistent L1 journal for cross-problem learning.

**Tech Stack:** Python, PyTorch (CUDA), OpenAI-compatible API (NVIDIA NIM), KernelBench Dataset/Eval APIs.

---

### Task 1: Environment & Directory Setup

**Files:**
- Create: `scripts_integration/evolving_agent/__init__.py`
- Create: `scripts_integration/evolving_agent/kb_harness_template.py`

- [ ] **Step 1: Create the integration directory and init file**
Run: `mkdir -p scripts_integration/evolving_agent; touch scripts_integration/evolving_agent/__init__.py`

- [ ] **Step 2: Create the kb_harness_template.py**
This template will be injected into the agent's workspace to provide the `run_benchmark` function.
```python
import torch
import sys
import os
from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset

def run_benchmark(kernel_source_code, level, problem_id, backend, precision):
    try:
        dataset = construct_kernelbench_dataset(level=level, source="local")
        problem = dataset.get_problem_by_id(problem_id)
        
        result = kb_eval.eval_kernel_against_ref(
            problem.code,
            kernel_source_code,
            backend=backend,
            precision=kb_eval.get_torch_dtype_from_string(precision),
            measure_performance=True
        )

        print(f"KERNEL_BENCH_CORRECT: {result.correctness}")
        if result.correctness and result.runtime > 0:
            speedup = result.ref_runtime / result.runtime
            print(f"KERNEL_BENCH_SPEEDUP: {speedup:.4f}")
            return speedup
        else:
            print(f"KERNEL_BENCH_SPEEDUP: 0.0")
            error_info = result.metadata.get("compilation_error") or result.metadata.get("runtime_error") or "Unknown error"
            print(f"KERNEL_BENCH_ERROR: {error_info}")
            return 0.0
    except Exception as e:
        print(f"Harness Error: {str(e)}")
        return 0.0
```

- [ ] **Step 3: Commit**
Run: `git add scripts_integration/evolving_agent/; git commit -m "feat(evolving): initialize integration directory and harness template"`

### Task 2: Implement specialized KernelBench Governor

**Files:**
- Create: `scripts_integration/evolving_agent/kb_evolving_governor.py`

- [ ] **Step 1: Implement the KB Governor**
Adapter for `governor.py` that handles `KERNEL_BENCH_SPEEDUP` and problem-specific setup logic.
```python
import os
import re
import sys
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend

# Import from Self-Evolving-Agent (assuming it's in the path or linked)
sys.path.append(os.path.abspath("Self-Evolving-Agent"))
from execution import extract_python_code, run_solution, write_solution_py
from llm_client import call_coder, call_summarizer
from memory_manager import (
    new_l0, append_l0, format_l0_for_prompt, 
    should_promote_l0, promote_l0_to_l1, read_l1
)

@dataclass
class KBGovernorConfig:
    run_name: str
    level: int
    problem_id: int
    backend: str = "cuda"
    precision: str = "fp32"
    max_iterations: int = 20
    shared_l1_path: Path = None
    workspace_base: Path = Path("run_integration/evolving/workspaces")

def run_kb_governor(cfg: KBGovernorConfig):
    # Setup workspace and harness...
    # (Implementation details using the logic from Task 1 and 2)
    pass
```
*Note: The actual implementation will be developed in Task 2's sub-steps to ensure completeness.*

- [ ] **Step 2: Commit**
Run: `git add scripts_integration/evolving_agent/kb_evolving_governor.py; git commit -m "feat(evolving): add KB-specific governor script"`

### Task 3: Implement Batch Orchestrator

**Files:**
- Create: `scripts_integration/evolving_agent/evolve_kb_batch.py`

- [ ] **Step 1: Implement the batch runner**
This script reads `subset_selection/selected_problems_50.csv` and runs the `KBGovernor`.
```python
import csv
import argparse
from pathlib import Path
from scripts_integration.evolving_agent.kb_evolving_governor import KBGovernorConfig, run_kb_governor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", type=str, default="subset_selection/selected_problems_50.csv")
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    
    # Load problems, initialize shared L1, loop and run governor...
    pass
```

- [ ] **Step 2: Commit**
Run: `git add scripts_integration/evolving_agent/evolve_kb_batch.py; git commit -m "feat(evolving): add batch orchestrator"`

### Task 4: Verification on GPU

- [ ] **Step 1: Run a single-problem smoke test**
Run: `python scripts_integration/evolving_agent/evolve_kb_batch.py --run-name smoke_test --subset-csv <test_csv_with_1_row>`
Expected: Kernel compiled, Speedup > 0, L1 updated.

- [ ] **Step 2: Final Commit**
Run: `git commit -m "test(evolving): verify smoke test on GPU"`
