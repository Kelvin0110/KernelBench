import os
import json
import subprocess
import concurrent.futures
from pathlib import Path
from pydra import Config, REQUIRED
import pydra
from kernelbench.dataset import construct_kernelbench_dataset
from tqdm import tqdm

class BatchAideConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED
        self.level = REQUIRED
        self.num_workers = 4
        self.gpus = "0" # Comma separated GPU IDs
        self.subset = (None, None) # (start_id, end_id)
        self.problem_ids = None # List of specific problem IDs
        self.steps = 500 # Max steps for AIDE
        self.hours = 24.0 # Max hours for AIDE

def get_completed_problems(run_name):
    eval_results_file = Path(f"run_integration/{run_name}/eval_results.json")
    if not eval_results_file.exists():
        return set()
    
    completed = set()
    try:
        with open(eval_results_file, "r") as f:
            eval_results = json.load(f)
            # eval_results is a dict: {"1": [...], "2": [...]}
            for prob_id_str in eval_results.keys():
                completed.add(int(prob_id_str))
    except Exception as e:
        print(f"Error reading {eval_results_file}: {e}")
    return completed

def run_single_problem(problem_id, level, run_name, gpu_id, steps, hours):
    print(f"Starting Level {level} Problem {problem_id} on GPU {gpu_id}")
    
    # Create log directory
    log_dir = Path(f"run_integration/{run_name}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"L{level}_P{problem_id}.log"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "uv", "run",
        "python", "scripts_integration/integration_test_aide_kb.py",
        "--level", str(level),
        "--problem_id", str(problem_id),
        "--run_name", run_name,
        "--steps", str(steps),
        "--hours", str(hours)
    ]
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if process.returncode == 0:
            print(f"Successfully completed Level {level} Problem {problem_id}")
        else:
            print(f"Failed Level {level} Problem {problem_id} with return code {process.returncode}")
            
    except Exception as e:
        print(f"Error running Level {level} Problem {problem_id}: {e}")

@pydra.main(base=BatchAideConfig)
def main(config: BatchAideConfig):
    print(f"Starting Batch AIDE Run with config: {config}")
    
    # Determine problems to run
    if config.problem_ids is not None:
        problems_to_run = config.problem_ids
    else:
        dataset = construct_kernelbench_dataset(level=config.level, source="local")
        all_problem_ids = dataset.get_problem_ids()
        
        start_id, end_id = config.subset
        if start_id is None and end_id is None:
            problems_to_run = all_problem_ids
        else:
            start = start_id if start_id is not None else min(all_problem_ids)
            end = end_id if end_id is not None else max(all_problem_ids)
            problems_to_run = [pid for pid in all_problem_ids if start <= pid <= end]
        
    completed_problems = get_completed_problems(config.run_name)
    print(f"Found {len(completed_problems)} completed problems: {completed_problems}")
    
    pending_problems = [p for p in problems_to_run if p not in completed_problems]
    print(f"Pending problems to run: {len(pending_problems)}")
    
    if not pending_problems:
        print("All problems completed!")
        return
        
    gpus = [g.strip() for g in str(config.gpus).split(",")]
    if not gpus:
        gpus = ["0"]
        
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []
        for i, problem_id in enumerate(pending_problems):
            gpu_id = gpus[i % len(gpus)]
            futures.append(
                executor.submit(
                    run_single_problem,
                    problem_id,
                    config.level,
                    config.run_name,
                    gpu_id,
                    config.steps,
                    config.hours
                )
            )
            
        # Wait for all to complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Level {config.level} Batch Run ({config.run_name})"):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with exception: {e}")

if __name__ == "__main__":
    main()
