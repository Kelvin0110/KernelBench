import os
import time
import json
import subprocess
import threading
import queue
from pathlib import Path
from pydra import Config, REQUIRED
import pydra
from kernelbench.dataset import construct_kernelbench_dataset
from tqdm import tqdm
import signal
import sys
import atexit

# Global list to keep track of active subprocesses
active_processes = []
is_shutting_down = False

def cleanup_subprocesses():
    """Kill all active subprocesses when the main script exits."""
    # Use a local copy to avoid "list changed during iteration" errors
    to_kill = list(active_processes)
    for p in to_kill:
        try:
            if p.poll() is None:  # Process is still running
                print(f"Terminating child PID {p.pid}...")
                p.terminate()
        except Exception:
            pass

    # Give them a short time to exit gracefully
    if to_kill:
        time.sleep(0.5)

    for p in to_kill:
        try:
            if p.poll() is None:
                print(f"Killing child PID {p.pid}...")
                p.kill()
        except:
            pass

atexit.register(cleanup_subprocesses)

def signal_handler(sig, frame):
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True
    print("\nReceived termination signal. Cleaning up subprocesses...")
    cleanup_subprocesses()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# Only handle SIGHUP if it's not already being ignored (e.g. by nohup)
if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
    signal.signal(signal.SIGHUP, signal_handler)

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
        # Model selection to pass through to worker processes
        self.code_model = "openai/gpt-oss-120b"
        self.feedback_model = "openai/gpt-oss-120b"

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

def run_single_problem(problem_id, level, run_name, gpu_id, steps, hours, code_model=None, feedback_model=None, pbar=None):
    if is_shutting_down:
        if pbar:
            pbar.update(1)
        return
    print(f"Starting Level {level} Problem {problem_id} on GPU {gpu_id}")
    
    # Create log directory
    log_dir = Path(f"run_integration/{run_name}/logs/level_{level}_problem_{problem_id}")
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
    # Pass model selection through
    if code_model:
        cmd.extend(["--code_model", str(code_model)])
    if feedback_model:
        cmd.extend(["--feedback_model", str(feedback_model)])
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            active_processes.append(process)
            
            # Wait for the process to complete
            returncode = process.wait()
            
            if process in active_processes:
                active_processes.remove(process)
        
        if returncode == 0:
            print(f"Successfully completed Level {level} Problem {problem_id}")
        else:
            print(f"Failed Level {level} Problem {problem_id} with return code {returncode}")
            
    except Exception as e:
        print(f"Error running Level {level} Problem {problem_id}: {e}")
    finally:
        if pbar:
            pbar.update(1)

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
    # temp
    pending_problems = pending_problems[:10]
    print(f"Pending problems to run: {len(pending_problems)}")
    

    if not pending_problems:
        print("All problems completed!")
        return

    gpus = [g.strip() for g in str(config.gpus).split(",")]
    if not gpus:
        gpus = ["0"]

    # Fill the work queue with all pending problems
    problem_queue = queue.Queue()
    for pid in pending_problems:
        problem_queue.put(pid)

    pbar = tqdm(total=len(pending_problems), desc=f"Level {config.level} Batch Run ({config.run_name})")

    # Rate limiter: ensures at least stagger_secs between any two process starts,
    # regardless of how many workers finish at the same time.
    stagger_secs = 5
    start_rate_lock = threading.Lock()
    last_start_time = [0.0]  # list so the closure can mutate it

    def worker_loop(worker_id):
        gpu_id = gpus[worker_id % len(gpus)]
        while not is_shutting_down:
            try:
                problem_id = problem_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                break  # No more problems; this worker is done

            # --- Rate limiter: only one process may start at a time,
            #     and each start must be at least stagger_secs apart. ---
            with start_rate_lock:
                gap = time.time() - last_start_time[0]
                wait = stagger_secs - gap
                if wait > 0:
                    # Sleep in small chunks so shutdown is detected quickly
                    slept = 0.0
                    while slept < wait:
                        if is_shutting_down:
                            break
                        time.sleep(0.1)
                        slept += 0.1
                last_start_time[0] = time.time()
            # Lock released: the start time slot is now reserved for this
            # worker. Other workers wait until their own 3s window arrives.

            run_single_problem(
                problem_id,
                config.level,
                config.run_name,
                gpu_id,
                config.steps,
                config.hours,
                config.code_model,
                config.feedback_model,
                pbar=pbar,
            )
            problem_queue.task_done()

    # Start exactly num_workers long-lived threads (never more)
    threads = []
    for i in range(config.num_workers):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads to finish, checking shutdown flag every second
    while any(t.is_alive() for t in threads):
        if is_shutting_down:
            break
        time.sleep(1.0)

    pbar.close()

if __name__ == "__main__":
    main()
