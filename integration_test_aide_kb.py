import os
import aide
import logging
import shutil
import time
import argparse
import signal
import sys
import atexit
import json
import torch
from collections import defaultdict
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.eval import check_metadata_serializable_all_types, eval_kernel_against_ref, KernelExecResult

def add_to_eval_results_file(
    problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str
):
    """
    Add evaluation result to eval results file
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            eval_results = json.load(f)
            eval_results = defaultdict(lambda: [], eval_results)
    else:
        eval_results = defaultdict(lambda: [])

    # Add new result
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

    # Write updated results back to file (sorted by numeric key)
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)

    sorted_results = dict(sorted(eval_results.items(), key=lambda x: int(x[0])))
    with open(eval_file_path, "w") as f:
        json.dump(sorted_results, f, indent=4)

def setup_integration_env(task_dir, level=1, problem_id=1, backend="cuda", precision="fp32"):
    """Sets up a directory with a harness that uses kernelbench utilities."""
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir)

    # 1. Fetch real problem data (Reference Architecture)
    dataset = construct_kernelbench_dataset(level=level, source="local") # or "huggingface"
    problem = dataset.get_problem_by_id(problem_id)
    ref_arch_src = problem.code
    
    # 2. Generate the official Prompt
    task_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend=backend,
        option="one_shot",  # or "zero_shot"
        precision=precision
    )

    # 3. Create the harness file
    # This harness imports the generated kernel, compiles it, and evaluates it against the reference
    harness_code = f"""
import torch
import sys
import os
from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset

# We need to expose the Reference Model so the evaluation logic can find it
# The reference code is usually expected to be in a file or string
REFERENCE_CODE = r'''{ref_arch_src}'''

def run_benchmark(kernel_source_code):
    try:
        # Load dataset to get the reference problem context
        dataset = construct_kernelbench_dataset(level={level}, source="local")
        problem = dataset.get_problem_by_id({problem_id})
        
        # Use KernelBench's robust evaluation function
        # eval_kernel_against_ref compiles the string, runs correctness checks, and times it
        result = kb_eval.eval_kernel_against_ref(
            problem.code,
            kernel_source_code,
            backend="{backend}",
            precision="{precision}",
            measure_performance=True
        )

        print(f"KERNEL_BENCH_CORRECT: {{result.correctness}}")
        if result.correctness and result.runtime > 0:
            speedup = result.ref_runtime / result.runtime
            print(f"KERNEL_BENCH_SPEEDUP: {{speedup:.4f}}")
            # We want to maximize speedup
            return speedup
        else:
            print(f"KERNEL_BENCH_SPEEDUP: 0.0")
            # results object uses 'metadata' to store error info
            error_info = result.metadata.get("compilation_error") or result.metadata.get("runtime_error") or "Unknown error"
            print(f"KERNEL_BENCH_ERROR: {{error_info}}")
            return 0.0

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Harness Error: {{str(e)}}")
        return 0.0
"""
    with open(os.path.join(task_dir, "kb_harness.py"), "w") as f:
        f.write(harness_code)
    
    return task_prompt

def main():
    parser = argparse.ArgumentParser(description="AIDE + KernelBench Integration")
    parser.add_argument("-l", "--level", type=int, default=1, help="KernelBench problem level (1-4)")
    parser.add_argument("-i", "--problem_id", type=int, default=1, help="Problem ID within the level")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Maximum search nodes/steps")
    parser.add_argument("-t", "--hours", type=float, default=24.0, help="Maximum execution time in hours")
    parser.add_argument("-r", "--run_name", type=str, default="default_run", help="Name of the run for saving results")
    args = parser.parse_args()

    # Setup logging as recommended by AIDE
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("aide")
    logger.setLevel(logging.INFO)

    task_dir = f"integration_test_tasks/L{args.level}_P{args.problem_id}"
    
    # Setup environment and get the official prompt
    prompt_desc = setup_integration_env(task_dir, level=args.level, problem_id=args.problem_id)

    print(f"--- Starting AIDE + KernelBench Integration Test (Level {args.level}, ID {args.problem_id}) ---")
    
    # Define a goal that points to the harness
    # We augment the KernelBench prompt with specific instructions for the Agent
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
   # optimization: pass the current file's content
   if __name__ == "__main__":
       with open(__file__, 'r') as f:
           my_code = f.read()
       kb_harness.run_benchmark(my_code)

5. Maximise the metric 'KERNEL_BENCH_SPEEDUP'. If it is 0, fix the correctness bugs.
"""

    # Initialize the experiment
    # 'eval' tells the agent what to look for in the output logs
    exp = aide.Experiment(
        data_dir=task_dir,
        goal=goal,
        eval="KERNEL_BENCH_SPEEDUP",
        exp_name=args.run_name,
        workspace_dir="run_integration"
    )

    # Ensure all sub-processes (Interpreter) are killed when the main script exits/crashes
    def cleanup():
        print("\n--- Cleaning up Interpreter session... ---")
        try:
            exp.interpreter.cleanup_session()
        except:
            pass
    
    atexit.register(cleanup)
    # Signal handlers to allow graceful exit on Ctrl+C or kill
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    exp.cfg.agent.code.model = "openai/gpt-oss-120b"
    exp.cfg.agent.feedback.model = "openai/gpt-oss-120b"

    # Settings for the test run
    max_steps = args.steps
    max_hours = args.hours
    start_time = time.time()

    print(f"Running AIDE (Max {max_steps} nodes/steps, {max_hours} hours)...")
    try:
        from aide.utils.config import save_run
        # Loop iteration to respect both time and node limits
        for i in range(max_steps):
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= max_hours:
                print(f"Time limit reached ({max_hours} hours). Stopping.")
                break
            
            print(f"\n--- Node {i+1}/{max_steps} (Elapsed: {elapsed_hours:.2f}h) ---")
            try:
                exp.agent.step(exec_callback=exp.interpreter.run)
                # Save progress after EACH successful node
                save_run(exp.cfg, exp.journal)
            except Exception as e:
                print(f"Node execution or save failed: {e}. Skipping code generation for this node..")
                import traceback
                traceback.print_exc()
                # We still try to save if it's an agent error, maybe it was a transient API bug
                try:
                    save_run(exp.cfg, exp.journal)
                except:
                    pass
                continue
        
        # Final cleanup and retrieval of findings
        exp.interpreter.cleanup_session()
        best_node = exp.journal.get_best_node(only_good=False)

        print("\n--- Integration Test Complete ---")
        if best_node and best_node.metric:
            print(f"Best Speedup: {best_node.metric.value}")
            print("Best Code Snippet (Head):")
            print(best_node.code[:200] + "...")
            
            # Save the best solution
            run_dir = os.path.join("run_integration", args.run_name)
            os.makedirs(run_dir, exist_ok=True)
            kernel_file_path = os.path.join(run_dir, f"level_{args.level}_problem_{args.problem_id}_sample_0_kernel.py")
            with open(kernel_file_path, "w") as f:
                f.write(best_node.code)
            print(f"Saved best kernel to {kernel_file_path}")
            
            # Evaluate the best solution
            print("Evaluating best kernel against reference...")
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
                    build_dir=os.path.join("cache", args.run_name, str(args.problem_id), "0"),
                    device=torch.device("cuda:0"),
                    backend="cuda",
                    precision=torch.float32
                )
                
                if eval_result:
                    eval_file_path = os.path.join(run_dir, "eval_results.json")
                    add_to_eval_results_file(args.problem_id, 0, eval_result, eval_file_path)
                    print(f"Saved evaluation results to {eval_file_path}")
            except Exception as e:
                print(f"Final evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No solutions found.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
