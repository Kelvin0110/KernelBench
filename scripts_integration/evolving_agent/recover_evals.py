#!/usr/bin/env python3
"""
Recovery script to update eval_results.json based on files in the workspaces directory.
This repairs runs that were interrupted by script bugs but managed to save code.
"""

import json
import re
from pathlib import Path
import torch
from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset

# Configuration - adjust if your paths differ
RESULTS_ROOT = Path("results/evolving_logs/evolving_proto_gpu_isolated")
WORKSPACES_DIR = RESULTS_ROOT / "workspaces"
EVAL_RESULTS_PATH = RESULTS_ROOT / "eval_results.json"

def get_best_code(problem_dir: Path):
    """Find the best_iter_X.py with the highest iteration number."""
    files = list(problem_dir.glob("best_iter_*.py"))
    if not files:
        return None, -1
    
    # Sort by iteration number
    def get_iter(p):
        match = re.search(r"best_iter_(\d+)\.py", p.name)
        return int(match.group(1)) if match else -1
    
    files.sort(key=get_iter, reverse=True)
    return files[0].read_text(), get_iter(files[0])

def main():
    if not EVAL_RESULTS_PATH.exists():
        print(f"Error: {EVAL_RESULTS_PATH} not found.")
        return

    print(f"Loading {EVAL_RESULTS_PATH}...")
    with open(EVAL_RESULTS_PATH, "r") as f:
        eval_data = json.load(f)

    # We assume Level 1 for this recovery based on your logs
    level = "1"
    if level not in eval_data:
        eval_data[level] = {}

    dataset = construct_kernelbench_dataset(level=int(level), source="local")

    updated_count = 0
    for problem_dir in WORKSPACES_DIR.iterdir():
        if not problem_dir.is_dir():
            continue
        
        match = re.match(r"level_(\d+)_problem_(\d+)", problem_dir.name)
        if not match:
            continue
        
        prob_id = match.group(2)
        
        # Check if we have code but missing/failed eval
        code, iters = get_best_code(problem_dir)
        if not code:
            continue

        existing_entries = eval_data[level].get(prob_id, [])
        needs_update = True
        
        if existing_entries:
            # If the best sample is already correct and has a speedup, maybe skip?
            # But if it crashed with AttributeError, it usually has speedup 0.0 or iterations 0.
            sample = existing_entries[0]
            if sample.get("correctness") is True and sample.get("runtime", -1) > 0:
                needs_update = False

        if needs_update:
            print(f"Repairing Problem {prob_id} (found iteration {iters})...")
            try:
                problem = dataset.get_problem_by_id(int(prob_id))
                
                # Use the existing evaluator
                dtype = torch.float32 # Default
                result = kb_eval.eval_kernel_against_ref(
                    problem.code,
                    code,
                    backend="cuda",
                    precision=dtype,
                    measure_performance=True,
                )
                
                speedup = 0.0
                if result.correctness and result.runtime and result.runtime > 0 and result.ref_runtime and result.ref_runtime > 0:
                    speedup = float(result.ref_runtime / result.runtime)

                # Construct entry
                entry = {
                    "sample_id": 0,
                    "compiled": bool(result.compiled),
                    "correctness": bool(result.correctness),
                    "metadata": {
                        "source": "evolving_agent_recovery",
                        "level": int(level),
                        "problem_id": int(prob_id),
                        "best_speedup": speedup,
                        "backend": "cuda",
                        "precision": "fp32",
                        "iterations_run": iters,
                        "error": str(result.metadata.get("runtime_error") or result.metadata.get("compilation_error") or "") if not result.correctness else None
                    },
                    "runtime": float(result.runtime) if result.runtime else -1.0,
                    "runtime_stats": dict(result.runtime_stats or {})
                }
                
                eval_data[level][prob_id] = [entry]
                updated_count += 1
            except Exception as e:
                print(f"  [Skip] Problem {prob_id} failed during eval: {e}")
                continue

    if updated_count > 0:
        print(f"Saving changes to {EVAL_RESULTS_PATH} ({updated_count} problems updated)...")
        with open(EVAL_RESULTS_PATH, "w") as f:
            json.dump(eval_data, f, indent=2)
    else:
        print("No problems needed repair.")

if __name__ == "__main__":
    main()
