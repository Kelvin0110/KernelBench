"""
Docker-based batch orchestrator for AIDE + KernelBench.

Replaces batch_run_aide_kb.py with Docker containers for process/resource isolation.
Each problem runs in an ephemeral container with hard memory, I/O, and PID limits.

Usage:
    python scripts_integration/docker/docker_batch_run.py \
        run_name=my_run level=1 num_workers=4 gpus="0,1" steps=500 hours=24.0
"""

import os
import re
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

# Docker image names and Dockerfile paths (relative to repo root)
IMAGE_NAME = "kernelbench-aide"
IMAGE_NAME_CPU = "kernelbench-aide-cpu"
DOCKERFILE_PATH = "scripts_integration/docker/Dockerfile.kernelbench"
DOCKERFILE_CPU_PATH = "scripts_integration/docker/Dockerfile.cpu"
# Final evaluation grace period (seconds) used when computing container shell timeout
FINAL_EVAL_GRACE_SECS = 3600

# Globals for cleanup
active_containers = []
active_containers_lock = threading.Lock()
is_shutting_down = False


class DockerBatchConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED
        self.level = REQUIRED
        self.num_workers = 2
        self.gpus = "0"  # Comma-separated GPU IDs
        self.subset = (None, None)  # (start_id, end_id)
        self.problem_ids = None  # List of specific problem IDs
        self.steps = 50
        self.hours = 2.0
        self.code_model = "openai/gpt-oss-120b"
        self.feedback_model = "openai/gpt-oss-120b"
        self.backend = "cuda"
        self.precision = "fp32"
        # Docker resource limits
        self.memory_limit = "32g"
        self.pids_limit = 256
        self.io_read_bps = "200mb"   # Per container read throughput cap
        self.io_write_bps = "200mb"  # Per container write throughput cap
        self.io_read_iops = "10000"  # Per container read IOPS cap (crucial for small-file compilation)
        self.io_write_iops = "10000" # Per container write IOPS cap
        self.io_device = "/dev/sda"  # Fallback block device if auto-detection fails (Linux only)
        self.tmpfs_size = "16g"      # RAM disk size for /tmp per container; set "" to disable
        # I/O watchdog thresholds (Linux only, reads /proc/stat)
        self.enable_iowait_monitor = True  # Enable/disable iowait monitoring and logging
        self.iowait_warn_pct = 70.0   # Print warning when host iowait exceeds this %
        self.iowait_pause_pct = 85.0  # Pause new container starts when iowait exceeds this %
        self.iowait_resume_pct = 50.0 # Resume spawning when iowait drops below this %
        # Docker control
        self.build_image = True  # Whether to build image before running
        self.stagger_secs = 10  # Seconds between container starts
        self.mock = False  # Use CPU image + skip --gpus; for M1/no-GPU testing
        self.gpu_memory_fraction = 0.0  # Fraction of GPU memory to reserve per container
        # AIDE search hyperparameters (see aideml/aide/utils/config.yaml)
        self.max_debug_depth = 5    # Max chain of debug iterations before new draft
        self.debug_prob = 0.5       # Probability of debugging a buggy node (vs drafting fresh)
        self.num_drafts = 3         # Number of initial draft solutions in the search tree
        # Checkpoint evaluation
        self.checkpoint_distance = 0  # Evaluate best kernel every N nodes; 0 = disabled


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


def build_docker_image(mock=False):
    """Build the Docker image from the repo root."""
    dockerfile = DOCKERFILE_CPU_PATH if mock else DOCKERFILE_PATH
    image = IMAGE_NAME_CPU if mock else IMAGE_NAME
    print(f"Building Docker image '{image}'...")
    result = subprocess.run(
        ["docker", "build", "-f", dockerfile, "-t", image, "."],
        timeout=3600,  # 1 hour max for build
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed with exit code {result.returncode}")
    print(f"Image '{image}' built successfully.")


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


def detect_io_device(fallback: str) -> str:
    """Auto-detect the block device that backs Docker's storage from /proc/mounts.

    Finds the device whose mountpoint is the longest prefix of /var/lib/docker,
    strips the partition suffix (e.g. nvme0n1p3 -> nvme0n1, sda1 -> sda),
    and returns /dev/<device>.  Falls back to `fallback` on any failure.

    Loop devices (/dev/loop*) are explicitly skipped — they are snap/overlay
    mounts and never back Docker storage.
    """
    if platform.system() != "Linux":
        return fallback

    target = "/var/lib/docker"
    best_mountpoint = ""
    best_device = ""

    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                device, mountpoint = parts[0], parts[1]
                if not device.startswith("/dev/"):
                    continue
                # Skip loop devices (snap mounts, overlayfs — never back Docker storage)
                if re.match(r"^/dev/loop\d", device):
                    continue
                # Pick the mount whose path is the longest prefix of target
                if target.startswith(mountpoint) and len(mountpoint) > len(best_mountpoint):
                    best_mountpoint = mountpoint
                    best_device = device
    except OSError as e:
        print(f"WARNING: Could not read /proc/mounts: {e}. Using fallback device '{fallback}'.")
        return fallback

    if not best_device:
        print(f"WARNING: Could not find mount for {target} in /proc/mounts. "
              f"Using fallback device '{fallback}'.")
        return fallback

    # Strip partition suffix: nvme0n1p3 -> nvme0n1, sda1 -> sda
    # Pattern: optional 'p' followed by trailing digits
    base_device = re.sub(r"p?\d+$", "", best_device)

    if not os.path.exists(base_device):
        print(f"WARNING: Auto-detected device '{base_device}' does not exist. "
              f"Using fallback '{fallback}'.")
        return fallback

    print(f"Auto-detected I/O device: {base_device}  (from mount {best_mountpoint} -> {best_device})")
    return base_device


def _parse_size_to_bytes(size_str: str) -> int:
    """Parse Docker-style size strings like '200mb', '16g', '20000' to bytes/count."""
    s = size_str.strip().lower()
    multipliers = {"k": 1024, "kb": 1024, "m": 1024**2, "mb": 1024**2,
                   "g": 1024**3, "gb": 1024**3, "t": 1024**4, "tb": 1024**4}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def print_aggregate_limit_warning(config, num_workers: int):
    """Print a budget table showing aggregate resource use across all workers.

    Warns if total tmpfs RAM allocation would exceed 75% of available system RAM.
    """
    print(f"\n{'='*60}")
    print(f"Aggregate I/O Budget  (num_workers={num_workers})")
    print(f"{'='*60}")

    def total_str(per_val, n):
        try:
            total_bytes = _parse_size_to_bytes(per_val) * n
            if total_bytes >= 1024**3:
                return f"{total_bytes / 1024**3:.1f}g"
            elif total_bytes >= 1024**2:
                return f"{total_bytes / 1024**2:.0f}mb"
            else:
                return f"{total_bytes / 1024:.0f}kb"
        except (ValueError, TypeError):
            return f"{per_val} x {n}"

    print(f"  Per-container write BPS   : {config.io_write_bps:<8}  -> Total: {total_str(config.io_write_bps, num_workers)}/s")
    print(f"  Per-container read  BPS   : {config.io_read_bps:<8}  -> Total: {total_str(config.io_read_bps, num_workers)}/s")
    print(f"  Per-container write IOPS  : {config.io_write_iops:<8}  -> Total: {int(config.io_write_iops) * num_workers}")
    print(f"  Per-container read  IOPS  : {config.io_read_iops:<8}  -> Total: {int(config.io_read_iops) * num_workers}")

    if config.tmpfs_size:
        total_tmpfs = total_str(config.tmpfs_size, num_workers)
        print(f"  Per-container tmpfs (/tmp): {config.tmpfs_size:<8}  -> Total RAM for tmpfs: {total_tmpfs}")

        # Check against available system RAM
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = int(line.split()[1])
                            mem_bytes = mem_kb * 1024
                            tmpfs_total = _parse_size_to_bytes(config.tmpfs_size) * num_workers
                            mem_limit_total = _parse_size_to_bytes(config.memory_limit) * num_workers
                            pct = tmpfs_total / mem_bytes * 100
                            print(f"  Per-container memory limit: {config.memory_limit:<8}  -> Total reserved: {total_str(config.memory_limit, num_workers)}")
                            print(f"  Host total RAM            : {mem_bytes / 1024**3:.0f}g")
                            print(f"  tmpfs as % of host RAM    : {pct:.1f}%")
                            if tmpfs_total + mem_limit_total > mem_bytes * 0.90:
                                print(f"  WARNING: Total tmpfs + memory limits ({total_str(config.tmpfs_size, num_workers)} + "
                                      f"{total_str(config.memory_limit, num_workers)}) exceeds 90% of "
                                      f"host RAM ({mem_bytes / 1024**3:.0f}g). "
                                      f"Consider reducing num_workers or tmpfs_size.")
                            elif pct > 75:
                                print(f"  WARNING: tmpfs alone consumes {pct:.0f}% of host RAM. "
                                      f"Consider reducing tmpfs_size or num_workers.")
                            break
            except OSError:
                pass
    else:
        print(f"  tmpfs                     : disabled  (SSD used for /tmp compilation artifacts)")
        print(f"  NOTE: Enabling tmpfs_size (e.g. tmpfs_size=16g) eliminates SSD I/O for builds.")

    print(f"{'='*60}\n")


class IOWaitMonitor:
    """Background thread that watches host iowait% and pauses container spawning when saturated.

    Reads /proc/stat every `interval` seconds to compute cumulative iowait %.
    Writes a CSV log to {run_dir}/iowait.log for post-mortem analysis.
    Sets an internal threading.Event when iowait > pause_pct; workers check this
    before starting a new container.
    """

    def __init__(self, run_dir: str, warn_pct: float, pause_pct: float,
                 resume_pct: float, interval: float = 2.0):
        self._run_dir = run_dir
        self._warn_pct = warn_pct
        self._pause_pct = pause_pct
        self._resume_pct = resume_pct
        self._interval = interval
        self._pause_event = threading.Event()   # set = paused, workers should wait
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="IOWaitMonitor")
        self._prev_stat: dict | None = None
        self._last_warn_time = 0.0
        self._log_file: str = os.path.join(run_dir, "iowait.log")

    def start(self):
        if platform.system() != "Linux":
            return  # /proc/stat not available on macOS/Windows
        self._thread.start()
        print(f"IOWaitMonitor started (warn={self._warn_pct}%, pause={self._pause_pct}%, "
              f"resume={self._resume_pct}%). Monitoring active, no log file written.")

    def stop(self):
        self._stop_event.set()

    def is_paused(self) -> bool:
        """Returns True when host iowait is above the pause threshold."""
        return self._pause_event.is_set()

    def _read_stat(self) -> dict | None:
        """Read /proc/stat and return the cpu line fields as a dict."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("cpu "):
                        fields = line.split()
                        # cpu user nice system idle iowait irq softirq steal guest guest_nice
                        return {
                            "user":    int(fields[1]),
                            "nice":    int(fields[2]),
                            "system":  int(fields[3]),
                            "idle":    int(fields[4]),
                            "iowait":  int(fields[5]),
                            "irq":     int(fields[6]),
                            "softirq": int(fields[7]),
                            "steal":   int(fields[8]) if len(fields) > 8 else 0,
                        }
        except (OSError, IndexError, ValueError):
            pass
        return None

    def _compute_iowait_pct(self) -> float:
        """Return iowait % since last call, or 0.0 on error."""
        curr = self._read_stat()
        if curr is None:
            return 0.0

        if self._prev_stat is None:
            self._prev_stat = curr
            return 0.0

        prev = self._prev_stat
        self._prev_stat = curr

        total_delta = sum(curr[k] - prev[k] for k in curr)
        if total_delta <= 0:
            return 0.0

        iowait_delta = curr["iowait"] - prev["iowait"]
        return max(0.0, iowait_delta / total_delta * 100.0)

    def _loop(self):
        while not self._stop_event.is_set():
            iowait = self._compute_iowait_pct()

            # Update pause event (no file logging, only in-memory monitoring)
            if iowait >= self._pause_pct:
                if not self._pause_event.is_set():
                    print(f"\n[IOWaitMonitor] PAUSING new containers: iowait={iowait:.1f}% "
                          f">= pause threshold {self._pause_pct}%")
                self._pause_event.set()
            elif iowait < self._resume_pct and self._pause_event.is_set():
                print(f"[IOWaitMonitor] RESUMING container spawning: iowait={iowait:.1f}% "
                      f"< resume threshold {self._resume_pct}%")
                self._pause_event.clear()

            # Throttled warning (at most once per 30s)
            if self._warn_pct <= iowait < self._pause_pct:
                now = time.time()
                if now - self._last_warn_time >= 30.0:
                    print(f"[IOWaitMonitor] WARNING: iowait={iowait:.1f}% "
                          f"(warn threshold={self._warn_pct}%)")
                    self._last_warn_time = now

            self._stop_event.wait(timeout=self._interval)


def get_completed_problems(run_dir):
    """Check which problems have entries in eval_results.json."""
    completed = set()
    eval_file = Path(run_dir) / "eval_results.json"
    if not eval_file.exists():
        return completed
    try:
        with open(eval_file) as f:
            data = json.load(f)
        for pid_str in data.keys():
            try:
                completed.add(int(pid_str))
            except ValueError:
                pass
    except Exception as e:
        print(f"Warning: Failed to read {eval_file}: {e}")
    return completed


def run_container(problem_id, level, config, gpu_id, run_dir, pbar=None):
    """Launch a Docker container for a single problem and wait for completion."""
    if is_shutting_down:
        if pbar:
            pbar.update(1)
        return

    # Each container mounts the shared run_dir; logs go to per-problem subdir
    log_subdir = os.path.join(run_dir, "container_logs")
    os.makedirs(log_subdir, exist_ok=True)
    log_file = os.path.join(log_subdir, f"L{level}_P{problem_id}.log")

    # Time limit with grace period for cleanup (provided by batch runner)
    time_limit_secs = int(getattr(config, "time_limit_secs", int(config.hours * 3600 * 2) + FINAL_EVAL_GRACE_SECS))

    container_name = f"kb-L{level}-P{problem_id}-{config.run_name}"

    # Build docker run command
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
    ]

    # GPU assignment: skip entirely in mock mode (M1/CPU testing)
    if not config.mock:
        cmd.extend(["--gpus", "all"])  # Use all GPUs at Docker level; restrict with CUDA_VISIBLE_DEVICES in env_vars

    cmd.extend([
        # Resource limits
        "--memory", config.memory_limit,
        "--memory-swap", config.memory_limit,  # Same as memory = no swap
        "--pids-limit", str(config.pids_limit),
        # Run as host user to preserve permission/ownership on mounted volumes
        "-u", f"{os.getuid()}:{os.getgid()}",
    ])

    # I/O limits (only work on Linux with cgroups v1; silently ignored elsewhere)
    is_linux = platform.system() == "Linux"
    if is_linux and config.io_device and config.io_read_bps:
        cmd.extend([
            "--device-read-bps",  f"{config.io_device}:{config.io_read_bps}",
            "--device-write-bps", f"{config.io_device}:{config.io_write_bps}",
        ])
    if is_linux and config.io_device and config.io_read_iops:
        cmd.extend([
            "--device-read-iops",  f"{config.io_device}:{config.io_read_iops}",
            "--device-write-iops", f"{config.io_device}:{config.io_write_iops}",
        ])

    # Mount base run directory (shared across all containers of this batch)
    # Each container receives /app/run pointing to run_integration/<run_name>/
    cmd.extend(["-v", f"{os.path.abspath(run_dir)}:/app/run:rw"])

    # tmpfs for /tmp: routes all compilation artifacts (nvcc .o/.cubin/.ptx) to RAM
    # This eliminates SSD I/O for the most write-heavy phase of each container run.
    if config.tmpfs_size:
        cmd.extend(["--tmpfs", f"/tmp:rw,exec,size={config.tmpfs_size}"])

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
        "RESULTS_DIR": "/app/run",  # Base results directory (mounted volume)
        "MOCK_EVAL": "1" if config.mock else "0",
        "GPU_MEMORY_FRACTION": str(config.gpu_memory_fraction),
        "MAX_DEBUG_DEPTH":     str(config.max_debug_depth),
        "DEBUG_PROB":          str(config.debug_prob),
        "NUM_DRAFTS":          str(config.num_drafts),
        "CHECKPOINT_DISTANCE": str(config.checkpoint_distance),
    }
    # Pass through API keys from host environment
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "GEMINI_API_KEY", "SGLANG_API_KEY", "OPENAI_BASE_URL",
    ]:
        if key in os.environ:
            env_vars[key] = os.environ[key]

    # Add CUDA_VISIBLE_DEVICES to restrict which GPUs the container can see
    # This prevents race conditions with multiple containers accessing GPU devices
    if not config.mock:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])

    # Image name (CPU image skips CUDA, used for M1/no-GPU testing)
    cmd.append(IMAGE_NAME_CPU if config.mock else IMAGE_NAME)

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
    """Read flat eval_results.json for results summary."""
    results_base = Path(run_dir)

    # Read the flat eval_results.json
    aggregated = defaultdict(list)
    flat_eval = results_base / "eval_results.json"
    if flat_eval.exists():
        try:
            with open(flat_eval) as f:
                data = json.load(f)
            for pid_key, results in data.items():
                aggregated[pid_key].extend(results)
        except Exception as e:
            print(f"Warning: Failed to read {flat_eval}: {e}")

    # Overwrite the original flat eval_results.json with aggregated/sorted results
    sorted_results = dict(sorted(aggregated.items(), key=lambda x: int(x[0])))
    with open(flat_eval, "w") as f:
        json.dump(sorted_results, f, indent=4)
    output_file = flat_eval

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {output_file}")
    print(f"  Total problems evaluated: {len(sorted_results)}")
    print(f"{'='*60}")

    correct_count = 0
    for pid_str, results in sorted_results.items():
        for r in results:
            status = "PASS" if r.get("correctness") else "FAIL"
            compiled = "compiled" if r.get("compiled") else "compile_fail"
            runtime = r.get("runtime", -1)
            runtime_str = f"{runtime:.1f}ms" if runtime > 0 else "N/A"
            print(f"  P{pid_str}: {status} ({compiled}, {runtime_str})")
            if r.get("correctness"):
                correct_count += 1

    if sorted_results:
        print(f"\n  Correctness rate: {correct_count}/{len(sorted_results)}")


def check_disk_space(run_dir, min_gb=50):
    """Check available disk space on run_dir partition.

    Prevents expensive batch run from starting if insufficient space,
    which would result in all containers failing and wasting API tokens.
    """
    import shutil
    try:
        stat = shutil.disk_usage(run_dir)
        free_gb = stat.free / (1024**3)
        if free_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.1f} GB available, "
                f"require {min_gb} GB minimum. Exiting to prevent cascade failure."
            )
        print(f"✓ Disk space OK: {free_gb:.1f} GB available on {run_dir}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        raise


@pydra.main(base=DockerBatchConfig)
def main(config: DockerBatchConfig):
    print(f"Docker Batch Run Config: {config}")

    # Platform check for I/O limits
    is_linux = platform.system() == "Linux"
    if not is_linux:
        print("WARNING: I/O rate/IOPS limits and iowait monitoring only work on Linux "
              "with cgroups v1. Other resource limits (memory, pids) still apply.")

    # Auto-detect block device (Linux only; falls back to config.io_device on failure)
    if is_linux:
        config.io_device = detect_io_device(config.io_device)

    # Compute expected total shell timeout and expose to containers via TIME_LIMIT_SECS
    # Formula matches the single-run logic: (hours * 3600 * 2) + FINAL_EVAL_GRACE_SECS
    expected_total_secs = int((config.hours * 3600 * 2) + FINAL_EVAL_GRACE_SECS)
    config.time_limit_secs = expected_total_secs

    # Build image if requested
    if config.build_image:
        build_docker_image(mock=config.mock)

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

    run_dir = os.path.join("run_integration", config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Check disk space before starting expensive batch
    check_disk_space(run_dir, min_gb=50)

    # Print aggregate budget table and RAM warnings before starting anything
    print_aggregate_limit_warning(config, config.num_workers)

    # Start I/O wait monitor (daemon thread, Linux only; can be disabled with enable_iowait_monitor=False)
    if config.enable_iowait_monitor:
        io_monitor = IOWaitMonitor(
            run_dir=run_dir,
            warn_pct=config.iowait_warn_pct,
            pause_pct=config.iowait_pause_pct,
            resume_pct=config.iowait_resume_pct,
        )
        io_monitor.start()
        atexit.register(io_monitor.stop)
    else:
        # Dummy monitor that does nothing (provides same interface)
        class DummyMonitor:
            def is_paused(self):
                return False
            def stop(self):
                pass
        io_monitor = DummyMonitor()
        print("[IOWaitMonitor] Disabled (enable_iowait_monitor=False)")

    completed = get_completed_problems(run_dir)
    print(f"Already completed: {len(completed)} problems: {sorted(completed)}")

    pending = [p for p in problems_to_run if p not in completed]
    print(f"Pending: {len(pending)} problems, including: {sorted(pending)}")

    if not pending:
        print("All problems completed!")
        aggregate_results(run_dir, config.level)
        return

    gpus = [g.strip().strip("()[]") for g in str(config.gpus).split(",")]
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

            # Back-pressure: wait if host iowait is above the pause threshold
            while io_monitor.is_paused() and not is_shutting_down:
                print(f"[Worker {worker_id}] I/O saturated — waiting for iowait to drop "
                      f"below {config.iowait_resume_pct}%...")
                time.sleep(5.0)

            # Rate limiter: stagger container starts to avoid simultaneous compilation spikes
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
    io_monitor.stop()

    # Aggregate results from all problem directories
    aggregate_results(run_dir, config.level)


if __name__ == "__main__":
    main()
