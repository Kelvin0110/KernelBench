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
    """Add evaluation result to eval results file with file locking for concurrent access."""
    from kernelbench.eval import check_metadata_serializable_all_types

    # Ensure directory exists
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)

    # Use file locking to prevent concurrent write corruption
    lock_file = eval_file_path + ".lock"
    try:
        with open(lock_file, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)  # Exclusive lock

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

            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)  # Release lock
    except Exception as e:
        print(f"ERROR: Failed to write eval_results with locking: {e}")
        raise
    finally:
        # Clean up lock file if it exists
        try:
            os.remove(lock_file)
        except OSError:
            pass


def write_problem_status(results_dir, problem_id, status):
    """Write structured status.json and DONE marker to per-problem directory.

    Adopted from Caesar's DONE marker pattern for atomic completion signaling.
    Each problem gets P{id}/status.json with structured outcome data, and
    P{id}/DONE (empty file) on success for reliable resume detection.
    """
    from pathlib import Path
    problem_dir = os.path.join(results_dir, f"P{problem_id}")
    os.makedirs(problem_dir, exist_ok=True)

    status_file = os.path.join(problem_dir, "status.json")
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    if status.get("outcome") == "success":
        Path(os.path.join(problem_dir, "DONE")).touch()


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

    max_steps = args.steps
    max_hours = args.hours
    start_time = time.time()

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

        # Final cleanup and result extraction
        exp.interpreter.cleanup_session()
        best_node = exp.journal.get_best_node(only_good=False)

        print("\n--- Run Complete ---")
        if best_node and best_node.metric:
            print(f"Best Speedup: {best_node.metric.value}")
            print("Best Code Snippet (Head):")
            print(best_node.code[:200] + "...")

            # Save the best kernel
            kernel_file_path = os.path.join(
                args.results_dir,
                f"level_{args.level}_problem_{args.problem_id}_sample_0_kernel.py",
            )
            with open(kernel_file_path, "w") as f:
                f.write(best_node.code)
            print(f"Saved best kernel to {kernel_file_path}")

            # Final evaluation
            print("Evaluating best kernel against reference...")

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
                write_problem_status(args.results_dir, args.problem_id, {
                    "outcome": "success",
                    "compiled": eval_result.compiled,
                    "correctness": eval_result.correctness,
                    "runtime_ms": eval_result.runtime,
                    "speedup": best_node.metric.value if best_node.metric else None,
                    "mock": True,
                })
            else:
                import torch
                from kernelbench.eval import eval_kernel_against_ref

                dataset = construct_kernelbench_dataset(level=args.level, source="local")
                problem = dataset.get_problem_by_id(args.problem_id)

                try:
                    eval_result = eval_kernel_against_ref(
                        original_model_src=problem.code,
                        custom_model_src=best_node.code,
                        measure_performance=True,
                        timing_method="cuda_event",
                        verbose=False,
                        num_correct_trials=5,
                        num_perf_trials=100,
                        build_dir=os.path.join("/tmp/cache", str(args.problem_id), "0"),
                        device=torch.device("cuda:0"),
                        backend=args.backend,
                        precision=torch.float32,
                    )

                    if eval_result:
                        eval_file_path = os.path.join(args.results_dir, "eval_results.json")
                        add_to_eval_results_file(args.problem_id, 0, eval_result, eval_file_path)
                        print(f"Saved evaluation results to {eval_file_path}")
                        write_problem_status(args.results_dir, args.problem_id, {
                            "outcome": "success",
                            "compiled": eval_result.compiled,
                            "correctness": eval_result.correctness,
                            "runtime_ms": eval_result.runtime,
                            "speedup": best_node.metric.value if best_node.metric else None,
                        })
                except Exception as e:
                    print(f"Final evaluation failed: {e}")
                    traceback.print_exc()
                    write_problem_status(args.results_dir, args.problem_id, {
                        "outcome": "eval_error",
                        "error": str(e),
                    })
        else:
            print("No solutions found.")
            write_problem_status(args.results_dir, args.problem_id, {
                "outcome": "no_solution",
                "steps_completed": max_steps,
            })

    except Exception as e:
        traceback.print_exc()
        print(f"Experiment failed: {e}")
        write_problem_status(args.results_dir, args.problem_id, {
            "outcome": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    main()
