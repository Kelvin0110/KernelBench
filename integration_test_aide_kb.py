import os
import aide
import logging
import shutil
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend

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
    # Setup logging as recommended by AIDE
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("aide")
    logger.setLevel(logging.INFO)

    task_dir = "integration_test_task"
    level = 1
    problem_id = 1
    
    # Setup environment and get the official prompt
    prompt_desc = setup_integration_env(task_dir, level=level, problem_id=problem_id)

    print("--- Starting AIDE + KernelBench Integration Test ---")
    
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
        eval="KERNEL_BENCH_SPEEDUP" 
    )

    exp.cfg.agent.code.model = "nvdev/openai/gpt-oss-120b"
    exp.cfg.agent.feedback.model = "nvdev/openai/gpt-oss-120b"

    # Run for steps
    print("Running AIDE...")
    try:
        best_solution = exp.run(steps=3)
        print("\n--- Integration Test Complete ---")
        print(f"Best Speedup: {best_solution.valid_metric}")
        print("Best Code Snippet (Head):")
        print(best_solution.code[:200] + "...")
    except Exception as e:
        print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
