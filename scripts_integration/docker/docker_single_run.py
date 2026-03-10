"""
AIDE + KernelBench single-problem runner designed for Docker containers.

Ported from scripts_integration/integration_test_aide_kb.py with changes for
container execution:
- Results written to --results_dir (mounted volume) instead of relative paths
- No workspace cleanup on exit (container is ephemeral)
- Error/status marker files for host orchestrator
- Always uses cuda:0 (NVIDIA toolkit remaps the assigned GPU)
"""

import os
import sys
import aide
import logging
import shutil
import time
import argparse
import signal
import atexit
import json
import traceback
import fcntl
from collections import defaultdict
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend


def add_to_eval_results_file(problem_id, sample_id, eval_result, eval_file_path):
    """Add evaluation result to eval results file with timeout-protected file locking.

    Handles stale lock detection and cleanup to prevent permanent deadlock if container
    crashes while holding fcntl.flock(). Uses non-blocking lock with retry and timeout.
    """
    from kernelbench.eval import check_metadata_serializable_all_types

    # Ensure directory exists
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)

    lock_file = eval_file_path + ".lock"
    max_wait_secs = 30  # Total timeout for acquiring lock

    # Stale lock detection: if lock file is >10 min old, it's from a crashed container
    if os.path.exists(lock_file):
        try:
            lock_age_secs = time.time() - os.path.getmtime(lock_file)
            if lock_age_secs > 600:  # 10 minutes
                print(f"WARNING: Stale lock file detected (age: {lock_age_secs}s), removing.")
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        except Exception:
            pass

    # Retry loop with timeout for non-blocking lock acquisition
    start_time = time.time()
    while True:
        try:
            with open(lock_file, "w") as lock_fh:
                # Use LOCK_NB (non-blocking) to detect if already locked
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                try:
                    # Read existing results
                    if os.path.exists(eval_file_path):
                        with open(eval_file_path, "r") as f:
                            eval_results = json.load(f)
                            eval_results = defaultdict(lambda: [], eval_results)
                    else:
                        eval_results = defaultdict(lambda: [])

                    # Append new result
                    eval_results[str(problem_id)].append(
                        {
                            "sample_id": sample_id,
                            "compiled": eval_result.compiled,
                            "correctness": eval_result.correctness,
                            "metadata": check_metadata_serializable_all_types(eval_result.metadata),
                            "runtime": eval_result.runtime,
                            "runtime_stats": eval_result.runtime_stats,
                        }
                    )

                    # Write updated results (sorted by numeric key)
                    sorted_results = dict(sorted(eval_results.items(), key=lambda x: int(x[0])))
                    with open(eval_file_path, "w") as f:
                        json.dump(sorted_results, f, indent=4)

                    break  # Success
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)  # Release lock

        except (IOError, BlockingIOError) as e:
            # Lock already held by another process (expected with LOCK_NB)
            elapsed = time.time() - start_time
            if elapsed > max_wait_secs:
                print(f"ERROR: Lock acquisition timeout after {elapsed:.1f}s")
                raise RuntimeError(
                    f"Could not acquire eval_results.json lock after {max_wait_secs}s. "
                    f"This likely means another container crashed while holding the lock."
                )
            time.sleep(0.5)  # Retry after 0.5s

        except Exception as e:
            print(f"ERROR: Failed to write eval_results: {e}")
            raise

        finally:
            # Clean up lock file
            try:
                os.remove(lock_file)
            except OSError:
                pass


def reserve_gpu_memory(device, fraction):
    """Pre-warm PyTorch's GPU memory cache to defend against other processes.

    How it works:
      1. Allocate a dummy tensor claiming `fraction` of total GPU memory.
      2. Delete it — memory goes to PyTorch's caching allocator, NOT back to the OS.
      3. Subsequent allocations (AIDE kernels, eval) reuse from the cache.
      4. Other processes on the server cannot claim this cached memory.
    """
    import torch
    props = torch.cuda.get_device_properties(device)
    total_bytes = props.total_memory
    already_reserved = torch.cuda.memory_reserved(device)
    target_bytes = int(total_bytes * fraction) - already_reserved

    if target_bytes <= 0:
        print(f"GPU memory already reserved: {already_reserved / 1e9:.1f} GB >= target {total_bytes * fraction / 1e9:.1f} GB")
        return

    print(f"Reserving GPU memory: {target_bytes / 1e9:.1f} GB ({fraction*100:.0f}% of {total_bytes / 1e9:.1f} GB total)")
    try:
        dummy = torch.empty(target_bytes, dtype=torch.uint8, device=device)
        del dummy
        # Do NOT call torch.cuda.empty_cache() — that would return memory to the OS
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.1f} GB in PyTorch cache")
    except RuntimeError as e:
        print(f"WARNING: GPU memory reservation failed (fraction={fraction}): {e}")
        print("Continuing without reservation — other processes may steal GPU memory mid-run.")


def setup_integration_env(task_dir, level=1, problem_id=1, backend="cuda", precision="fp32", mock_eval=False):
    """Sets up a directory with a harness that uses kernelbench utilities."""
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir)

    # Fetch real problem data
    dataset = construct_kernelbench_dataset(level=level, source="local")
    problem = dataset.get_problem_by_id(problem_id)
    ref_arch_src = problem.code

    # Generate the official prompt
    task_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend=backend,
        option="one_shot",
        precision=precision
    )

    # Build the run_benchmark function body.
    # In mock mode: return a fake random speedup so AIDE's search loop can iterate
    # without a real GPU. All file I/O, orchestration, and AIDE logic still runs.
    if mock_eval:
        run_benchmark_body = (
            "    import random\n"
            "    speedup = random.uniform(0.5, 3.0)\n"
            '    print(f"KERNEL_BENCH_CORRECT: True")\n'
            '    print(f"KERNEL_BENCH_SPEEDUP: {speedup:.4f}")\n'
            "    return speedup\n"
        )
    else:
        run_benchmark_body = f"""\
    try:
        dataset = construct_kernelbench_dataset(level={level}, source="local")
        problem = dataset.get_problem_by_id({problem_id})

        result = kb_eval.eval_kernel_against_ref(
            problem.code,
            kernel_source_code,
            backend="{backend}",
            precision=kb_eval.get_torch_dtype_from_string("{precision}"),
            measure_performance=True
        )

        print(f"KERNEL_BENCH_CORRECT: {{result.correctness}}")
        if result.correctness and result.runtime > 0:
            speedup = result.ref_runtime / result.runtime
            print(f"KERNEL_BENCH_SPEEDUP: {{speedup:.4f}}")
            return speedup
        else:
            print(f"KERNEL_BENCH_SPEEDUP: 0.0")
            error_info = result.metadata.get("compilation_error") or result.metadata.get("runtime_error") or "Unknown error"
            print(f"KERNEL_BENCH_ERROR: {{error_info}}")
            return 0.0

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Harness Error: {{str(e)}}")
        return 0.0"""

    # Create the harness file (concatenate so run_benchmark_body is not re-parsed
    # as an f-string, which would misinterpret its own {var} placeholders)
    harness_code = (
        f"""
import torch
import sys
import os
from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset

REFERENCE_CODE = r'''{ref_arch_src}'''

def run_benchmark(kernel_source_code):
"""
        + run_benchmark_body
        + "\n"
    )
    with open(os.path.join(task_dir, "kb_harness.py"), "w") as f:
        f.write(harness_code)

    return task_prompt


def run_checkpoint_eval(
    node_count: int,
    best_node,
    problem_code: str,
    prev_checkpoint_code,
    args,
    task_dir: str,
    elapsed_hours: float,
    prev_checkpoint_eval_path: str = None,
):
    """Evaluate the current best kernel at a checkpoint.

    Stores kernel in kernels/ subdirectory and appends result to aggregated
    eval_results.json (keyed by problem_id). Supports forward-copying skipped
    results from previous checkpoints.

    Returns:
        (new_prev_code, problem_result_dict) — new_prev_code is the best kernel
        code at this checkpoint, problem_result_dict is the entry to append to
        eval_results.json[str(problem_id)].
    """
    from kernelbench.eval import check_metadata_serializable_all_types

    # Node-first path organization
    checkpoint_node_dir = os.path.join(
        args.results_dir, "checkpoints",
        f"node_{node_count:04d}",
    )
    kernels_dir = os.path.join(checkpoint_node_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)

    # Kernel path in kernels/ subdirectory
    kernel_filename = f"level_{args.level}_problem_{args.problem_id}_kernel.py"
    kernel_path = os.path.join(kernels_dir, kernel_filename)

    code_changed = (best_node is not None and best_node.code != prev_checkpoint_code)
    aide_metric = (best_node.metric.value
                   if best_node and best_node.metric else None)

    # Determine skip conditions
    skip_eval = False
    skip_reason = None

    if best_node is None:
        skip_eval = True
        skip_reason = "no_solution_found"
    elif not code_changed and prev_checkpoint_code is not None:
        skip_eval = True
        skip_reason = "code_unchanged_since_last_checkpoint"

    # Build result entry (schema matches final eval_results.json)
    result_entry = {
        "sample_id": 0,
        "compiled": None,
        "correctness": None,
        "runtime": None,
        "runtime_stats": None,
        "aide_metric": aide_metric,
        "code_changed_since_last_checkpoint": code_changed,
        "eval_skipped": skip_eval,
        "skip_reason": skip_reason,
        "metadata": None,
    }

    # Handle skipped evaluation: try to forward-copy from previous checkpoint
    if skip_eval:
        if best_node:
            with open(kernel_path, "w") as f:
                f.write(best_node.code)

        # Try to copy result from previous checkpoint
        if prev_checkpoint_eval_path and os.path.exists(prev_checkpoint_eval_path):
            try:
                with open(prev_checkpoint_eval_path) as f:
                    prev_eval_results = json.load(f)
                    problem_id_str = str(args.problem_id)
                    if problem_id_str in prev_eval_results and prev_eval_results[problem_id_str]:
                        prev_result = prev_eval_results[problem_id_str][0]
                        result_entry.update({
                            "compiled": prev_result.get("compiled"),
                            "correctness": prev_result.get("correctness"),
                            "runtime": prev_result.get("runtime"),
                            "runtime_stats": prev_result.get("runtime_stats"),
                            "metadata": prev_result.get("metadata"),
                        })
            except Exception as e:
                print(f"[Checkpoint node {node_count}] Failed to forward-copy result: {e}")

        print(f"[Checkpoint node {node_count}] L{args.level}P{args.problem_id} skipped ({skip_reason})")
        return prev_checkpoint_code, result_entry

    # Save kernel
    with open(kernel_path, "w") as f:
        f.write(best_node.code)

    # Run eval (using safe wrapper that records errors)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)

    eval_result = safe_eval_kernel_against_ref(
        original_model_src=problem_code,
        custom_model_src=best_node.code,
        build_dir=os.path.join("/tmp/cache", str(args.problem_id),
                               f"ckpt_{node_count}"),
        device=torch.device("cuda:0"),
        backend=args.backend,
        measure_performance=True,
        timing_method="cuda_event",
        verbose=False,
        num_correct_trials=5,
        num_perf_trials=100,
    )

    # Always record the result (even if compilation failed)
    result_entry.update({
        "compiled": eval_result.compiled,
        "correctness": eval_result.correctness,
        "runtime": eval_result.runtime,
        "ref_runtime": getattr(eval_result, 'ref_runtime', None),
        "runtime_stats": eval_result.runtime_stats,
        "aide_metric": aide_metric,
        "eval_skipped": False,
        "skip_reason": None,
        "metadata": check_metadata_serializable_all_types(eval_result.metadata),
    })

    if eval_result.compiled and eval_result.correctness:
        print(f"[Checkpoint node {node_count}] L{args.level}P{args.problem_id}: compiled={eval_result.compiled} correct={eval_result.correctness} metric={aide_metric}")
    else:
        # Log errors but don't treat as skip
        error_info = eval_result.metadata.get("cuda_error") or eval_result.metadata.get("other_error") or "Unknown error"
        print(f"[Checkpoint node {node_count}] L{args.level}P{args.problem_id}: compilation/runtime failed: {error_info}")

    new_prev_code = best_node.code

    return new_prev_code, result_entry


def safe_eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    build_dir: str,
    device,
    backend: str,
    measure_performance: bool = True,
    timing_method: str = "cuda_event",
    verbose: bool = False,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
):
    """
    Safely evaluate a kernel against reference with proper error handling.

    Records compilation errors, CUDA errors, and runtime failures as failed
    evaluations instead of exceptions. Follows the pattern from
    eval_from_generations.py:evaluate_single_sample().

    Returns:
        eval_result: Object with compiled, correctness, runtime, metadata fields.
                    On error, returns failed result with error details in metadata.
    """
    from kernelbench.eval import eval_kernel_against_ref
    import torch

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            measure_performance=measure_performance,
            timing_method=timing_method,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=backend,
            precision=torch.float32,
        )

        if eval_result is None:
            # Handle case where eval_kernel_against_ref returns None
            metadata = {
                "other_error": "eval_kernel_against_ref returned None",
                "other_error_name": "NoneEvalResult",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            return type('obj', (object,), {
                'compiled': False,
                'correctness': False,
                'runtime': None,
                'ref_runtime': None,
                'speedup': None,
                'runtime_stats': None,
                'metadata': metadata,
            })()

        return eval_result

    except Exception as e:
        # Handle errors during kernel execution
        print(f"[WARNING] Evaluation error: {e}")
        error_str = str(e)

        if "CUDA error" in error_str:
            # CUDA errors (illegal memory access, kernel launch failures)
            metadata = {
                "cuda_error": f"CUDA Error: {error_str}",
                "cuda_error_name": type(e).__name__,
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
        else:
            # Other errors (compilation, runtime, etc.)
            metadata = {
                "other_error": f"Evaluation error: {error_str}",
                "other_error_name": type(e).__name__,
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }

        # Return failed result object (not an exception)
        return type('obj', (object,), {
            'compiled': False,
            'correctness': False,
            'runtime': None,
            'ref_runtime': None,
            'speedup': None,
            'runtime_stats': None,
            'metadata': metadata,
        })()


def main():
    parser = argparse.ArgumentParser(description="AIDE + KernelBench Docker Runner")
    parser.add_argument("-l", "--level", type=int, required=True, help="KernelBench problem level (1-4)")
    parser.add_argument("-i", "--problem_id", type=int, required=True, help="Problem ID within the level")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Maximum search nodes/steps")
    parser.add_argument("-t", "--hours", type=float, default=24.0, help="Maximum execution time in hours")
    parser.add_argument("-r", "--run_name", type=str, default="docker_run", help="Name of the run")
    parser.add_argument("-c", "--code_model", type=str, default="openai/gpt-oss-120b", help="Code generation model")
    parser.add_argument("-f", "--feedback_model", type=str, default="openai/gpt-oss-120b", help="Feedback model")
    parser.add_argument("--backend", type=str, default="cuda", help="Backend: cuda, triton, tilelang, cute")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision: fp32, fp16, bf16")
    parser.add_argument(
        "--mock-eval", action="store_true",
        default=os.environ.get("MOCK_EVAL", "0") == "1",
        help="Skip real CUDA eval; return fake scores (for M1/CPU testing)",
    )
    parser.add_argument(
        "--gpu-memory-fraction", type=float,
        default=float(os.environ.get("GPU_MEMORY_FRACTION", "0.90")),
        help="Fraction of GPU memory to pre-warm into PyTorch cache (0.0 to disable, 0.90 default)",
    )
    parser.add_argument(
        "--max-debug-depth", type=int,
        default=int(os.environ.get("MAX_DEBUG_DEPTH", "5")),
        help="Max debug chain depth for AIDE search (AIDE default: 3)",
    )
    parser.add_argument(
        "--debug-prob", type=float,
        default=float(os.environ.get("DEBUG_PROB", "0.5")),
        help="Probability of debugging vs new draft in AIDE search (AIDE default: 0.5)",
    )
    parser.add_argument(
        "--num-drafts", type=int,
        default=int(os.environ.get("NUM_DRAFTS", "3")),
        help="Number of initial draft solutions in AIDE search tree (AIDE default: 5)",
    )
    parser.add_argument(
        "--checkpoint-distance", type=int,
        default=int(os.environ.get("CHECKPOINT_DISTANCE", "0")),
        help="Evaluate best kernel every N nodes; 0 = disabled",
    )
    # results_dir comes from  RESULTS_DIR env var or /app/run default (set by docker_batch_run.py)
    import os as os_module
    default_results_dir = os_module.environ.get("RESULTS_DIR", "/app/run")
    parser.add_argument("--results_dir", type=str, default=default_results_dir, help="Directory for results (mounted volume)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("aide")
    logger.setLevel(logging.INFO)

    # Task dir is ephemeral inside the container
    task_dir = f"/tmp/aide_task/L{args.level}/P{args.problem_id}"

    # Setup environment and get prompt
    prompt_desc = setup_integration_env(
        task_dir, level=args.level, problem_id=args.problem_id,
        backend=args.backend, precision=args.precision,
        mock_eval=args.mock_eval,
    )

    print(f"--- Starting AIDE + KernelBench (Level {args.level}, ID {args.problem_id}) ---")

    # Pre-warm GPU memory cache to prevent other server processes from taking it mid-run.
    # Skip in mock mode (no real GPU) or if disabled (fraction=0).
    if not args.mock_eval and args.gpu_memory_fraction > 0:
        import torch
        if torch.cuda.is_available():
            reserve_gpu_memory(torch.device("cuda:0"), args.gpu_memory_fraction)
        else:
            print("WARNING: GPU memory reservation requested but no CUDA GPU found, skipping.")

    # Set PyTorch build cache to /tmp (writable in container)
    # This prevents "Permission denied: /.cache" when load_inline compiles CUDA code
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
    os.makedirs("/tmp/torch_extensions", exist_ok=True)

    # Goal for AIDE agent
    goal = f"""
{prompt_desc}

INSTRUCTIONS FOR AGENT:
1. You are writing a high-performance kernel based on the description above.
2. The environment has a utility `kb_harness` to check your work.
3. Your solution code MUST contain the full kernel implementation AND a call to the harness at the end.
4. Structure your code like this:

   import torch
   import kb_harness

   # ... your kernel code ...

   # At the very end, read your own source code (or pass it as string) to the validator
   if __name__ == "__main__":
       with open(__file__, 'r') as f:
           my_code = f.read()
       kb_harness.run_benchmark(my_code)

5. Maximise the metric 'KERNEL_BENCH_SPEEDUP'. If it is 0, fix the correctness bugs.
"""

    # AIDE workspace and logs: stored per-problem to avoid naming conflicts
    workspace_dir = os.path.join(args.results_dir, "workspaces", f"level_{args.level}_problem_{args.problem_id}")
    log_dir = os.path.join(args.results_dir, "logs", f"level_{args.level}_problem_{args.problem_id}")

    exp = aide.Experiment(
        data_dir=task_dir,
        goal=goal,
        eval="KERNEL_BENCH_SPEEDUP",
        exp_name=f"level_{args.level}_problem_{args.problem_id}",
        workspace_dir=workspace_dir,
        log_dir=log_dir,
    )

    # Cleanup handler
    def cleanup():
        try:
            exp.interpreter.cleanup_session()
        except Exception:
            pass

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Configure models
    exp.cfg.agent.code.model = args.code_model
    exp.cfg.agent.feedback.model = args.feedback_model

    # AIDE search hyperparameters
    exp.cfg.agent.search.max_debug_depth = args.max_debug_depth
    exp.cfg.agent.search.debug_prob      = args.debug_prob
    exp.cfg.agent.search.num_drafts      = args.num_drafts

    max_steps = args.steps
    max_hours = args.hours
    start_time = time.time()

    # Fetch problem data once (used for checkpoint eval and final eval)
    dataset = construct_kernelbench_dataset(level=args.level, source="local")
    problem = dataset.get_problem_by_id(args.problem_id)

    # Checkpoint state (tracks best code and previous eval results for forward-copying)
    prev_checkpoint_code = None
    prev_checkpoint_eval_path = None  # Path to previous checkpoint's eval_results.json

    print(f"Running AIDE (Max {max_steps} nodes/steps, {max_hours} hours)...")
    try:
        from aide.utils.config import save_run

        for i in range(max_steps):
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= max_hours:
                print(f"Time limit reached ({max_hours} hours). Stopping.")
                break

            print(f"\n--- Node {i+1}/{max_steps} (Elapsed: {elapsed_hours:.2f}h) ---")
            try:
                exp.agent.step(exec_callback=exp.interpreter.run)
                save_run(exp.cfg, exp.journal)
            except Exception as e:
                print(f"Node execution or save failed: {e}. Skipping this node...")
                traceback.print_exc()
                try:
                    save_run(exp.cfg, exp.journal)
                except Exception:
                    pass
                continue

            # Checkpoint evaluation (triggered every N nodes)
            if args.checkpoint_distance > 0 and (i + 1) % args.checkpoint_distance == 0:
                checkpoint_node_dir = os.path.join(
                    args.results_dir, "checkpoints",
                    f"node_{i+1:04d}",
                )
                eval_results_path = os.path.join(checkpoint_node_dir, "eval_results.json")

                # Run checkpoint eval (handles forward-copying of skipped results)
                ckpt_best = exp.journal.get_best_node(only_good=False)
                elapsed_h = (time.time() - start_time) / 3600
                prev_checkpoint_code, result_entry = run_checkpoint_eval(
                    node_count=i + 1,
                    best_node=ckpt_best,
                    problem_code=problem.code,
                    prev_checkpoint_code=prev_checkpoint_code,
                    args=args,
                    task_dir=task_dir,
                    elapsed_hours=elapsed_h,
                    prev_checkpoint_eval_path=prev_checkpoint_eval_path,
                )

                # Append result to aggregated eval_results.json (keyed by problem_id)
                if not os.path.exists(eval_results_path):
                    eval_results = {}
                else:
                    with open(eval_results_path) as f:
                        eval_results = json.load(f)

                problem_id_str = str(args.problem_id)
                if problem_id_str not in eval_results:
                    eval_results[problem_id_str] = []
                eval_results[problem_id_str].append(result_entry)

                with open(eval_results_path, "w") as f:
                    json.dump(eval_results, f, indent=2)

                # Build checkpoint_summary.json from all problem entries in eval_results.json
                problems_list = []
                for pid_str, results in eval_results.items():
                    if results:
                        r = results[-1]  # Latest entry for this problem_id
                        problems_list.append({
                            "level": args.level,
                            "problem_id": int(pid_str),
                            "eval_skipped": r.get("eval_skipped", False),
                            "skip_reason": r.get("skip_reason"),
                            "code_changed": r.get("code_changed_since_last_checkpoint", False),
                            "compiled": r.get("compiled"),
                            "correct": r.get("correctness"),
                            "aide_metric": r.get("aide_metric"),
                            "runtime_secs": r.get("runtime"),
                        })

                summary_doc = {
                    "checkpoint_node": i + 1,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "elapsed_hours": round(elapsed_h, 4),
                    "problems_evaluated": problems_list,
                    "summary": {
                        "total_problems": len(problems_list),
                        "problems_evaluated": sum(1 for p in problems_list if not p.get("eval_skipped", False)),
                        "problems_skipped": sum(1 for p in problems_list if p.get("eval_skipped", False)),
                        "avg_aide_metric": sum(p.get("aide_metric") or 0 for p in problems_list) / len(problems_list) if problems_list else None,
                    },
                }

                with open(os.path.join(checkpoint_node_dir, "checkpoint_summary.json"), "w") as f:
                    json.dump(summary_doc, f, indent=2)

                # Update path for forward-copying in next checkpoint
                prev_checkpoint_eval_path = eval_results_path

        # Final cleanup and result extraction
        exp.interpreter.cleanup_session()

        best_node = exp.journal.get_best_node(only_good=False)

        print("\n--- Run Complete ---")
        if best_node and best_node.metric:
            print(f"Best Speedup: {best_node.metric.value}")
            print("Best Code Snippet (Head):")
            print(best_node.code[:200] + "...")

            # Save the best kernel
            best_kernel_dir = os.path.join(args.results_dir, "kernels")
            os.makedirs(best_kernel_dir, exist_ok=True)  # Create directory if it doesn't exist
            kernel_file_path = os.path.join(
                best_kernel_dir,
                f"level_{args.level}_problem_{args.problem_id}_sample_0_kernel.py",
            )
            with open(kernel_file_path, "w") as f:
                f.write(best_node.code)
            print(f"Saved best kernel to {kernel_file_path}")

            # Final evaluation
            print("Evaluating best kernel against reference...")

            # Ensure task_dir is in sys.path so generated code can import kb_harness
            if task_dir not in sys.path:
                sys.path.insert(0, task_dir)

            if args.mock_eval:
                # M1/CPU testing: return a fake result so the pipeline completes
                import types
                eval_result = types.SimpleNamespace(
                    compiled=True,
                    correctness=True,
                    metadata={"mock": True},
                    runtime=1.0,
                    runtime_stats={"mean": 1.0, "std": 0.0},
                )
                eval_file_path = os.path.join(args.results_dir, "eval_results.json")
                add_to_eval_results_file(args.problem_id, 0, eval_result, eval_file_path)
                print(f"Saved mock evaluation results to {eval_file_path}")
            else:
                # Use safe evaluation wrapper (records errors instead of raising)
                eval_result = safe_eval_kernel_against_ref(
                    original_model_src=problem.code,
                    custom_model_src=best_node.code,
                    build_dir=os.path.join("/tmp/cache", str(args.problem_id), "0"),
                    device=torch.device("cuda:0"),
                    backend=args.backend,
                    measure_performance=True,
                    timing_method="cuda_event",
                    verbose=False,
                    num_correct_trials=5,
                    num_perf_trials=100,
                )

                if eval_result:
                    eval_file_path = os.path.join(args.results_dir, "eval_results.json")
                    add_to_eval_results_file(args.problem_id, 0, eval_result, eval_file_path)
                    print(f"Saved evaluation results to {eval_file_path}")
                    if eval_result.compiled and eval_result.correctness:
                        print(f"✓ Final evaluation: PASSED (speedup={getattr(eval_result, 'speedup', 'N/A')})")
                    else:
                        error_info = eval_result.metadata.get("cuda_error") or eval_result.metadata.get("other_error") or "Unknown error"
                        print(f"✗ Final evaluation: FAILED ({error_info})")
        else:
            print("No solutions found.")

        # Clean up ephemeral /tmp/aide_task workspace AFTER final evaluation
        # so that kb_harness module is still available during final evaluation
        try:
            if os.path.exists(task_dir):
                print(f"Cleaning up ephemeral workspace: {task_dir}")
                shutil.rmtree(task_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean ephemeral workspace: {e}")

    except Exception as e:
        traceback.print_exc()
        print(f"Experiment failed: {e}")

    # Final cleanup: remove AIDE workspace directory to free space
    workspace_dir = os.path.join(args.results_dir, "workspaces", f"level_{args.level}_problem_{args.problem_id}")
    try:
        if os.path.exists(workspace_dir):
            print(f"Cleaning up AIDE workspace: {workspace_dir}")
            shutil.rmtree(workspace_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not clean workspace: {e}")


if __name__ == "__main__":
    main()
