"""
Docker-based batch orchestrator for AIDE + KernelBench.

Replaces batch_run_aide_kb.py with Docker containers for process/resource isolation.
Each problem runs in an ephemeral container with hard memory, I/O, and PID limits.

Usage:
    python scripts_integration/docker/docker_batch_run.py \
        run_name=my_run level=1 num_workers=4 gpus="0,1" steps=500 hours=24.0
"""

import os
import sys
import json
import time
import platform
import subprocess
import threading
import queue
import signal
import atexit
from pathlib import Path
from collections import defaultdict
from pydra import Config, REQUIRED
import pydra
from kernelbench.dataset import construct_kernelbench_dataset
from tqdm import tqdm

# Docker image name and Dockerfile path (relative to repo root)
IMAGE_NAME = "kernelbench-aide"
DOCKERFILE_PATH = "scripts_integration/docker/Dockerfile.kernelbench"

# Globals for cleanup
active_containers = []
active_containers_lock = threading.Lock()
is_shutting_down = False


class DockerBatchConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED
        self.level = REQUIRED
        self.num_workers = 4
        self.gpus = "0"  # Comma-separated GPU IDs
        self.subset = (None, None)  # (start_id, end_id)
        self.problem_ids = None  # List of specific problem IDs
        self.steps = 500
        self.hours = 24.0
        self.code_model = "openai/gpt-oss-120b"
        self.feedback_model = "openai/gpt-oss-120b"
        self.backend = "cuda"
        self.precision = "fp32"
        # Docker resource limits
        self.memory_limit = "32g"
        self.pids_limit = 256
        self.io_read_bps = "100mb"  # Per container read throughput cap
        self.io_write_bps = "100mb"  # Per container write throughput cap
        self.io_device = "/dev/sda"  # Block device for I/O limits (Linux only)
        # Docker control
        self.build_image = True  # Whether to build image before running
        self.stagger_secs = 10  # Seconds between container starts


def cleanup_containers():
    """Kill and remove all active containers on exit."""
    with active_containers_lock:
        to_kill = list(active_containers)
    for cid in to_kill:
        try:
            subprocess.run(["docker", "kill", cid], capture_output=True, timeout=10)
        except Exception:
            pass
    if to_kill:
        time.sleep(1)
    for cid in to_kill:
        try:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)
        except Exception:
            pass


atexit.register(cleanup_containers)


def signal_handler(sig, frame):
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True
    print("\nReceived termination signal. Killing containers...")
    cleanup_containers()
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
    signal.signal(signal.SIGHUP, signal_handler)


def build_docker_image():
    """Build the Docker image from the repo root."""
    print(f"Building Docker image '{IMAGE_NAME}'...")
    result = subprocess.run(
        ["docker", "build", "-f", DOCKERFILE_PATH, "-t", IMAGE_NAME, "."],
        timeout=3600,  # 1 hour max for build
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed with exit code {result.returncode}")
    print(f"Image '{IMAGE_NAME}' built successfully.")


def check_docker_gpu_support():
    """Verify that NVIDIA Container Toolkit is available."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("Docker GPU support: OK")
            return True
        else:
            print(f"Docker GPU support: FAILED\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Docker GPU support check failed: {e}")
        return False


def get_completed_problems(run_dir):
    """Check which problems already have eval results (for resume support)."""
    completed = set()
    results_base = Path(run_dir)
    if not results_base.exists():
        return completed

    for d in results_base.iterdir():
        if d.is_dir() and d.name.startswith("P"):
            eval_file = d / "eval_results.json"
            if eval_file.exists():
                try:
                    pid = int(d.name[1:])
                    completed.add(pid)
                except ValueError:
                    pass
    return completed


def run_container(problem_id, level, config, gpu_id, run_dir, pbar=None):
    """Launch a Docker container for a single problem and wait for completion."""
    if is_shutting_down:
        if pbar:
            pbar.update(1)
        return

    problem_dir = os.path.abspath(os.path.join(run_dir, f"P{problem_id}"))
    os.makedirs(problem_dir, exist_ok=True)
    log_file = os.path.join(problem_dir, "container.log")

    # Time limit with grace period for cleanup
    time_limit_secs = int(config.hours * 3600) + 300

    container_name = f"kb-L{level}-P{problem_id}-{config.run_name}"

    # Build docker run command
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        # GPU assignment via NVIDIA Container Toolkit
        "--gpus", f"device={gpu_id}",
        # Resource limits
        "--memory", config.memory_limit,
        "--memory-swap", config.memory_limit,  # Same as memory = no swap
        "--pids-limit", str(config.pids_limit),
    ]

    # I/O limits (only work on Linux with cgroups v1; silently ignored elsewhere)
    is_linux = platform.system() == "Linux"
    if is_linux and config.io_device and config.io_read_bps:
        cmd.extend([
            "--device-read-bps", f"{config.io_device}:{config.io_read_bps}",
            "--device-write-bps", f"{config.io_device}:{config.io_write_bps}",
        ])

    # Mount results volume
    cmd.extend(["-v", f"{problem_dir}:/results"])

    # Environment variables
    env_vars = {
        "LEVEL": str(level),
        "PROBLEM_ID": str(problem_id),
        "STEPS": str(config.steps),
        "HOURS": str(config.hours),
        "TIME_LIMIT_SECS": str(time_limit_secs),
        "CODE_MODEL": config.code_model,
        "FEEDBACK_MODEL": config.feedback_model,
        "RUN_NAME": config.run_name,
        "BACKEND": config.backend,
        "PRECISION": config.precision,
    }
    # Pass through API keys from host environment
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "GEMINI_API_KEY", "SGLANG_API_KEY", "OPENAI_BASE_URL",
    ]:
        if key in os.environ:
            env_vars[key] = os.environ[key]

    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])

    # Image name
    cmd.append(IMAGE_NAME)

    print(f"[GPU {gpu_id}] Starting L{level} P{problem_id}")

    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
            )

        with active_containers_lock:
            active_containers.append(container_name)

        returncode = process.wait()

        with active_containers_lock:
            if container_name in active_containers:
                active_containers.remove(container_name)

        if returncode == 0:
            print(f"[GPU {gpu_id}] Completed L{level} P{problem_id}")
        else:
            print(f"[GPU {gpu_id}] Failed L{level} P{problem_id} (exit={returncode})")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error L{level} P{problem_id}: {e}")
        with active_containers_lock:
            if container_name in active_containers:
                active_containers.remove(container_name)
    finally:
        if pbar:
            pbar.update(1)


def aggregate_results(run_dir, level):
    """Merge per-problem eval_results.json into a single aggregated file."""
    aggregated = defaultdict(list)
    results_base = Path(run_dir)
    errors = []
    no_solutions = []

    for d in sorted(results_base.iterdir()):
        if not d.is_dir() or not d.name.startswith("P"):
            continue

        pid_str = d.name[1:]

        # Check for error/no-solution markers
        if (d / "ERROR.txt").exists():
            errors.append(pid_str)
        if (d / "NO_SOLUTION.txt").exists():
            no_solutions.append(pid_str)

        eval_file = d / "eval_results.json"
        if eval_file.exists():
            try:
                with open(eval_file) as f:
                    data = json.load(f)
                for pid_key, results in data.items():
                    aggregated[pid_key].extend(results)
            except Exception as e:
                print(f"Warning: Failed to read {eval_file}: {e}")

    # Write aggregated results
    output_file = results_base / "eval_results_aggregated.json"
    sorted_results = dict(sorted(aggregated.items(), key=lambda x: int(x[0])))
    with open(output_file, "w") as f:
        json.dump(sorted_results, f, indent=4)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Aggregated Results: {output_file}")
    print(f"  Problems with results: {len(sorted_results)}")
    print(f"  Problems with errors:  {len(errors)}")
    print(f"  No solution found:     {len(no_solutions)}")
    print(f"{'='*60}")

    correct_count = 0
    for pid_str, results in sorted_results.items():
        for r in results:
            status = "PASS" if r.get("correctness") else "FAIL"
            compiled = "compiled" if r.get("compiled") else "compile_fail"
            runtime = r.get("runtime", -1)
            runtime_str = f"{runtime:.1f}us" if runtime > 0 else "N/A"
            print(f"  P{pid_str}: {status} ({compiled}, {runtime_str})")
            if r.get("correctness"):
                correct_count += 1

    if sorted_results:
        print(f"\n  Correctness rate: {correct_count}/{len(sorted_results)}")


@pydra.main(base=DockerBatchConfig)
def main(config: DockerBatchConfig):
    print(f"Docker Batch Run Config: {config}")

    # Platform check for I/O limits
    if platform.system() != "Linux":
        print("WARNING: I/O rate limits (--device-read-bps, --device-write-bps) only "
              "work on Linux with cgroups v1. Other resource limits still apply.")

    # Build image if requested
    if config.build_image:
        build_docker_image()

    # Determine which problems to run
    if config.problem_ids is not None:
        problems_to_run = config.problem_ids
    else:
        dataset = construct_kernelbench_dataset(level=config.level, source="local")
        all_ids = dataset.get_problem_ids()

        start_id, end_id = config.subset
        if start_id is None and end_id is None:
            problems_to_run = all_ids
        else:
            start = start_id if start_id is not None else min(all_ids)
            end = end_id if end_id is not None else max(all_ids)
            problems_to_run = [p for p in all_ids if start <= p <= end]

    run_dir = os.path.join("run_integration", config.run_name, "results")
    os.makedirs(run_dir, exist_ok=True)

    completed = get_completed_problems(run_dir)
    print(f"Already completed: {len(completed)} problems: {sorted(completed)}")

    pending = [p for p in problems_to_run if p not in completed]
    print(f"Pending: {len(pending)} problems")

    if not pending:
        print("All problems completed!")
        aggregate_results(run_dir, config.level)
        return

    gpus = [g.strip() for g in str(config.gpus).split(",")]
    if not gpus:
        gpus = ["0"]

    # Fill the work queue
    problem_queue = queue.Queue()
    for pid in pending:
        problem_queue.put(pid)

    pbar = tqdm(
        total=len(pending),
        desc=f"L{config.level} Docker Batch ({config.run_name})",
    )

    # Rate limiter: ensures stagger_secs between container starts
    start_rate_lock = threading.Lock()
    last_start_time = [0.0]

    def worker_loop(worker_id):
        gpu_id = gpus[worker_id % len(gpus)]
        while not is_shutting_down:
            try:
                problem_id = problem_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                break  # No more problems

            # Rate limiter
            with start_rate_lock:
                gap = time.time() - last_start_time[0]
                wait = config.stagger_secs - gap
                if wait > 0:
                    slept = 0.0
                    while slept < wait and not is_shutting_down:
                        time.sleep(0.1)
                        slept += 0.1
                last_start_time[0] = time.time()

            run_container(
                problem_id, config.level, config, gpu_id, run_dir, pbar,
            )
            problem_queue.task_done()

    # Start worker threads
    threads = []
    for i in range(config.num_workers):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads, checking shutdown flag
    while any(t.is_alive() for t in threads):
        if is_shutting_down:
            break
        time.sleep(1.0)

    pbar.close()

    # Aggregate results from all problem directories
    aggregate_results(run_dir, config.level)


if __name__ == "__main__":
    main()
