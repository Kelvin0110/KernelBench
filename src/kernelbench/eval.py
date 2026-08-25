"""
Helpers for Evaluations
"""

import hashlib
import importlib
import json
import linecache
import os, subprocess
import random
import sys
import tempfile
import time
import traceback
import types
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Union, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from pydantic import BaseModel

from . import timing, dataset

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_error_name(e: Exception) -> str:
    """
    Get the error name, for logging purposes
    """
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


# Markers of genuine build-lock contention. Kept explicit rather than testing for
# the bare substring "lock", which also matches "blockIdx" and "deadlock" in nvcc
# diagnostics and so swallowed ordinary compile errors.
_LOCK_ERROR_MARKERS = (
    ".lock",
    "lock file",
    "lockfile",
    "could not be acquired",
    "waiting for lock",
    "resource temporarily unavailable",
)


def is_transient_lock_error(e: Exception) -> bool:
    """
    True only for concurrent-build lock contention, which is worth retrying.

    A failed ninja build leaves no shared object behind, so the load that follows
    raises

        ImportError: <name>.so: cannot open shared object file: No such file or directory

    Treating "No such file or directory" as a lock error therefore reclassified
    every real compile failure as retryable and dropped the compiler diagnostics,
    leaving callers with a compile failure and no reason attached.
    """
    text = str(e)
    if isinstance(e, ImportError) and "cannot open shared object file" in text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _LOCK_ERROR_MARKERS)


def fetch_ref_arch_from_problem_id(problem_id: int, dataset: "BaseDataset", with_name=False) -> Union[str, tuple[str, str]]:
    """
    Fetches the reference architecture for a given problem_id from the dataset.
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem = dataset.get_problem_by_id(problem_id)
    ref_arch = problem.code
    
    if not with_name:
        return ref_arch
    else:
        # Use problem.name as fallback when path is None (e.g., for HuggingFace datasets)
        name = problem.path if problem.path is not None else problem.name
        return (name, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    kb_dataset = dataset.construct_kernelbench_dataset(level)
    return fetch_ref_arch_from_problem_id(problem_id, kb_dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

def get_torch_dtype_from_string(precision: str) -> torch.dtype:
    """
    Get the torch dtype for specific precision
    """
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else: # future, FP8, FP4, etc. support?
        raise ValueError(f"Invalid precision not supported: {precision}")

def get_tolerance_for_precision(precision: str | torch.dtype) -> float:
    """
    Get the tolerance from a string representing the percision.
    These tolerances are inspired by torchbench (PyTorch Benchmarking Suite): 
    Reference:
    https://github.com/pytorch/benchmark/blob/cfd835c35d04513ced9a59bd074eeb21dc8187d7/torchbenchmark/util/env_check.py#L519
    """
    if isinstance(precision, str):
        precision = get_torch_dtype_from_string(precision)

    PRECISION_TOLERANCES = {
        # By default for fp32, 1e-4 is used according to torchbench.
        torch.float32: 1e-4,
        # torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if it's too strict. 
        # @todo: Let user configure own tolerance as an option
        torch.float16: 1e-2, 
        torch.bfloat16: 1e-2,
    }
    assert precision in PRECISION_TOLERANCES, f"Invalid precision not supported: {precision}"
    return PRECISION_TOLERANCES[precision]
    

class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """
    # Execution
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {} # NOTE: to include warning if any

    # Timing
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

    # new: added ref time either through fetching prev runs or through execution
    # could do eager for level 1 and compile for level 2 and 3
    ref_runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    ref_runtime_stats: dict = {} # only recorded if we decide to measure performance


def _bind_function_to_context(fn: callable, context: dict) -> callable:
    """Re-bind *fn* so global lookups resolve in *context*, not a polluted namespace."""
    if fn is None:
        return None
    return types.FunctionType(
        fn.__code__,
        context,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = _bind_function_to_context(context.get("get_init_inputs"), context)
    get_inputs_fn = _bind_function_to_context(context.get("get_inputs"), context)
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
    extra_contexts: list[dict] | None = None,
):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    if extra_contexts:
        for ctx in extra_contexts:
            del ctx
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def _process_input_tensor(input, device, backend="cuda", precision=torch.float32):
    """
    Helper function to move tensors to the correct device and apply backend-specific dtype casting.
    
    Args:
        input: Input tensor or non-tensor value
        device: Target CUDA device
        backend: Backend type (e.g., 'cuda', 'triton', 'cute')
        precision: torch.dtype 
    Returns:
        Processed tensor on correct device with correct dtype, or original value if not a tensor
    """

    # sometimes things like init inputs are floats (like in the case of labels / targets, classification losses, etc.) 
    if not isinstance(input, torch.Tensor):
        return input
    
    # cast to the desired percision dtype for activations
    input_tensor = input.to(dtype=precision)
    
    # Default for all other backends and float types
    return input_tensor.to(device=device)


_GPU_LOCK_FN = None
_GPU_LOCK_TRIED = False


def _gpu_timing_lock(label: str = ""):
    """Serialise only the GPU phase of an eval against other concurrent runs.

    Correctness trials and the two timing windows are the parts whose *numbers*
    GPU contention corrupts. Model construction and nvcc/ninja loading are far
    longer and do not affect the measurement, so holding a lock across them
    only serialises unrelated work and caps concurrency.

    No-ops when the evolving-agent lock is unavailable, so stock KernelBench
    usage is unaffected.
    """
    global _GPU_LOCK_FN, _GPU_LOCK_TRIED
    if not _GPU_LOCK_TRIED:
        _GPU_LOCK_TRIED = True
        try:
            from evolving_common.governor.gpu_lock import gpu_eval_lock as _f

            _GPU_LOCK_FN = _f
        except Exception:
            _GPU_LOCK_FN = None
    if _GPU_LOCK_FN is None:
        return nullcontext()
    try:
        return _GPU_LOCK_FN(label=label)
    except Exception:
        return nullcontext()


@contextmanager
def _device_memory_reservation(need_bytes: int, device, timeout_sec: float):
    """Admission control on ESTIMATED device bytes, on top of the slot semaphore.

    The slot semaphore bounds how many evals are device-resident; it does not bound
    how much memory they need. On 2026-08-23 that gap cost 18 OOMs: 3 slots x ~52 GB
    on L1P34 (7.5 GB inputs) overruns a 143 GB card, and 16 of the 18 were that one
    problem. Each OOM is recorded as ``compiled=True correct=False``, so the governor
    then debugs a kernel that was never broken.

    A plain "is there free memory right now" check is racy in exactly the case that
    matters -- three evals can all observe free memory and then peak together. So this
    reserves *estimated* bytes instead, under a flock, which makes the check-and-set
    atomic. Effective concurrency becomes min(slots, budget / need).

    Dead reservers are pruned by liveness check, so a killed eval cannot leak budget.
    On timeout it proceeds anyway rather than blocking forever -- degrading to the
    current behaviour, the same contract as the GPU lock's own timeout.
    """
    if need_bytes <= 0:
        yield None
        return
    try:
        import fcntl
        from evolving_common.governor.gpu_lock import gpu_lock_key, lock_path
        budget_frac = float(os.environ.get("KB_EVAL_MEM_GATE_FRAC", "0.85"))
        total = torch.cuda.mem_get_info(device)[1]
        budget = int(total * budget_frac)
        path = str(lock_path()) + ".memresv"
        key = str(os.getpid())
    except Exception:  # never fail an eval over telemetry-grade machinery
        yield None
        return

    def _rw(mutate):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                raw = os.read(fd, 1 << 20).decode() or "{}"
                data = json.loads(raw)
            except Exception:
                data = {}
            data = {k: v for k, v in data.items()
                    if k == key or os.path.exists(f"/proc/{k}")}  # prune dead reservers
            out = mutate(data)
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(data).encode())
            return out
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _try_reserve(data):
        used = sum(v for k, v in data.items() if k != key)
        if used + need_bytes <= budget or not used:  # `not used` => never wedge alone
            data[key] = need_bytes
            return True
        return False

    # Publish the running wait so the SPAWNING PARENT discounts it from the eval
    # deadline, exactly as the GPU lock's own wait is discounted
    # (execution.evaluate_in_subprocess reads the shared counter every poll).
    # Without this the gate queues on the *work* budget: measured on the terra
    # wave, 12.6% of evals hit the valve and 5.2% were SIGTERM'd at 600s and
    # recorded as compile failures for kernels that were never broken.
    _report = _commit = None
    try:
        from evolving_common.governor.gpu_lock import (
            commit_external_wait as _commit,
            report_external_wait as _report,
        )
    except Exception:  # older submodule / direct callers: degrade to silence
        _report = _commit = None

    t0 = time.time()
    while True:
        try:
            if _rw(_try_reserve):
                break
        except Exception:
            break
        waited = time.time() - t0
        if _report is not None:
            try:
                _report(waited)
            except Exception:
                pass
        if waited >= timeout_sec:
            # Deliberately NOT printed. eval_runner folds BOTH stdout and stderr
            # into terminal_output, which governor.py splices into the coder's
            # system prompt -- a notice here mutates LLM input, i.e. it is an
            # experiment change rather than an observation. Measured on the terra
            # wave before this fix: 164 evals injected the old print, reaching 36
            # chat histories. The wait is already recorded as mem_gate_waited_sec
            # in KB_EVAL_PHASE_LOG, which is where telemetry belongs.
            break
        time.sleep(0.25)
    if _commit is not None:
        try:
            _commit(time.time() - t0)
        except Exception:
            pass
    try:
        yield round(time.time() - t0, 3)
    finally:
        try:
            _rw(lambda d: d.pop(key, None))
        except Exception:
            pass


def _gpu_lock_slots() -> int:
    """Slot count actually in force, recorded into the phase log for provenance.

    Read straight from the environment rather than importing gpu_lock, so this
    stays correct (and cheap) even when the evolving-agent lock is unavailable
    and _gpu_timing_lock degraded to a nullcontext.
    """
    try:
        return max(1, int(os.environ.get("KB_GPU_EVAL_LOCK_SLOTS", "1")))
    except (TypeError, ValueError):
        return 1


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env switch, failing closed.

    Only an explicit truthy value turns a switch on, so a typo (``fasle``,
    ``disabled``) leaves it off rather than silently enabling it. Unset or empty
    falls through to ``default``, because an exported-but-empty variable is a
    common shell accident and should not mean "off".
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _emit_phase_record(record: dict) -> None:
    """Append one eval's phase breakdown to $KB_EVAL_PHASE_LOG, if set.

    Deliberately NOT printed. The evolving-agent eval worker captures eval stdout
    and the governor splices it into the agent's prompt, so anything written to
    stdout here would change LLM input mid-run. Unset -> emit nothing at all,
    which keeps this file inert for a run already in flight.
    """
    path = os.environ.get("KB_EVAL_PHASE_LOG")
    if not path or not path.strip():
        return
    try:
        with open(path.strip(), "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # telemetry must never fail an eval
        pass


def _pregenerate_correctness_inputs(
    get_inputs_fn: callable, num_correct_trials: int, seed: int
) -> list:
    """Build the per-trial correctness inputs on the CPU, outside the GPU lock.

    Mirrors ``run_and_check_correctness``' seed derivation exactly, so the tensors
    are the ones it would have built itself. That function re-seeds before each
    model ``.to()`` regardless, so skipping its in-loop generation leaves the RNG
    state seen by the rest of it unchanged.
    """
    torch.manual_seed(seed)
    trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]
    inputs_by_trial = []
    with torch.no_grad():
        for trial_seed in trial_seeds:
            set_seed(trial_seed)
            inputs_by_trial.append(get_inputs_fn())
    return inputs_by_trial


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = False,
    timing_method: str = "cuda_event", # see timing.py
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = None,  # resolved below; avoids CUDA init at import time
    backend: str = "cuda",  # can be 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype = torch.float32,

    # Guard against potential reward hacking [optional but ongoing enhancement]
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 30, # flag if the kernel is more than <excessive_speedup_threshold>x faster than the reference
    baseline_runtime: float | None = None,  # fixed baseline for excessive-speedup flag (display speedup uses governor baseline)
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    NOTE: we are thinking about refactor this to be more modularized 
    and we can add more checks as our other ongiong PRs are working on

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    backend: str, one of 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype for computation (note: tilelang only supports fp16)
    timing_method: str, method to time kernel, see timing.py for more details 

    ONGOING EFFORT to refactor and modularize this, and adding more tests for eval.
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"

    if device is None:
        device = torch.cuda.current_device()
    
    # Backend-GPU vendor validation
    from .utils import get_gpu_vendor
    vendor = get_gpu_vendor(device)
    backend_lower = backend.lower()
    # HIP is AMD-only
    if backend_lower == "hip" and vendor != "amd":
        raise ValueError(f"HIP backend requires AMD GPU, got {vendor}")
    # cuda/cute/thunderkittens are NVIDIA-only (triton/tilelang work on both)
    if backend_lower in ["cuda", "cute", "thunderkittens"] and vendor == "amd":
        raise ValueError(f"{backend} backend requires NVIDIA GPU, got AMD")
    
    if backend_lower == "tilelang":
        assert precision == torch.float16 or precision == torch.bfloat16, "TileLang only supports fp16 or bfloat16"
    
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)
    
    # Backends that use tempfile approach and need CUDA_VISIBLE_DEVICES
    # TileLang, Triton, and CuTe all use tempfile for proper module loading
    uses_tempfile = backend.lower() in ["triton", "tilelang", "cute"]
    
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    if uses_tempfile:
        # need to set env var for triton/cute code to guarantee no wrong device shenanigans
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        # NVIDIA uses CUDA_VISIBLE_DEVICES, AMD uses HIP_VISIBLE_DEVICES
        if vendor == "amd":
            os.environ["HIP_VISIBLE_DEVICES"] = str(device_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    original_context: dict = {}
    custom_context: dict = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, original_context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    
    # Convert inputs to appropriate dtypes for GPU computation
    init_inputs = [_process_input_tensor(x, device, backend, precision) for x in init_inputs]
    
    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        tempfile = None
        # add hash for later to distinguish between multi-turn kernels
        
        backend_lower = backend.lower()
        if backend_lower in ["triton", "tilelang", "cute"]:
            # Use tempfile approach for triton, tilelang, and cute
            # These DSLs require proper module import for JIT decorators to work
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            # Default CUDA backend — isolated namespace so custom globals cannot pollute get_inputs
            ModelNew = load_custom_model(custom_model_src, custom_context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # Record the diagnostics on every path, so a caller that only reads
        # metadata still learns why the build failed.
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e

        if is_transient_lock_error(e):
            # concurrent compilation contended on the build lock; this does not
            # necessarily mean the compilation failed, so signal a retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(
                original_context, device, tempfile, extra_contexts=[custom_context]
            )
            return None
        else:
            graceful_eval_cleanup(
                original_context, device, tempfile, extra_contexts=[custom_context]
            )
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # Check if ModelNew was successfully loaded (load_custom_model returns None on syntax errors)
    if ModelNew is None:
        print(
            "Failed to load custom model: Syntax error or ModelNew not found in generated code. Record as compilation failure."
        )
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = "Syntax error in custom generated code or ModelNew not found"
        graceful_eval_cleanup(
            original_context, device, tempfile, extra_contexts=[custom_context]
        )
        return KernelExecResult(
            compiled=False, metadata=metadata
        )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            original_model = original_model.to(device=device, dtype=precision)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(
            original_context, device, tempfile, extra_contexts=[custom_context]
        )
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # --- Pre-lock input generation (opt-in) ---------------------------------
    # get_inputs() is pure-CPU torch.rand in every KernelBench problem -- it
    # touches no device -- and the timing functions receive already-materialised
    # tensors (timing.time_execution_with_cuda_event brackets only kernel_fn(*args)
    # between CUDA events), so building them here changes no measured number. Only
    # the host-to-device transfer inside _process_input_tensor is GPU work, and it
    # stays under the lock.
    #
    # Both timing windows re-seed with seed_num and build byte-identical tensors,
    # so one CPU master serves both; each window still takes its own fresh device
    # copy, preserving the existing guarantee that the reference window sees
    # pristine inputs even if the candidate mutated its own in place.
    #
    # Correctness pregeneration is limited to the single-trial case: at
    # num_correct_trials=5 (several scripts/ callers) this would hold five whole
    # input sets in host RAM at once, which for the largest dataset problems is
    # tens of GB for no benefit -- the correctness loop is not the part being
    # pulled out of the lock.
    _phase = {
        "input_gen_prelock": 0.0,
        "correct_input_gen": 0.0,
        "correct_h2d": 0.0,
        "timing_input_gen": 0.0,
        "timing_h2d": 0.0,
        "correctness": 0.0,
        "candidate": 0.0,
        "reference": 0.0,
    }
    _hoisted = False
    _cpu_inputs_timing = None
    _cpu_inputs_correct = None
    if _env_flag("KB_EVAL_HOIST_INPUT_GEN", default=False):
        # Falls back to the in-lock path on any failure, so a bad allocation is
        # still reported through the existing recoverable channel instead of
        # escaping as a worker error the governor would log as a compile failure.
        _t = time.perf_counter()
        try:
            if measure_performance:
                with torch.no_grad():
                    set_seed(seed_num)
                    _cpu_inputs_timing = get_inputs()
            if num_correct_trials == 1:
                _cpu_inputs_correct = _pregenerate_correctness_inputs(
                    get_inputs, num_correct_trials, seed_num
                )
            _hoisted = True
        except Exception:
            _cpu_inputs_timing = None
            _cpu_inputs_correct = None
        _phase["input_gen_prelock"] = time.perf_counter() - _t

    # --- GPU phase begins: correctness trials + both timing windows ---------
    # Everything above (source exec, reference-model construction, ninja load)
    # runs unlocked: it is the bulk of the wall time and does not affect the
    # numbers. There are no early returns between here and the release below;
    # if an exception escapes, the eval subprocess exits and the kernel drops
    # the flock, so the lock cannot leak beyond this eval.
    # KB_EVAL_UNLOCK_CORRECTNESS moves the correctness trials OUT of the lock, so
    # the held region is only the timing window(s). Correctness runs two real
    # forwards, so it is genuine GPU work -- but it is not a *measurement*, and
    # the 2026-08-23 concurrency probe on this host measured what overlapping it
    # actually costs: with the lock fully disabled, 6 kernels x 5 repeats at
    # degree 2/3 inflated the measured runtime by 1.2%/0.7% median and 2.3%/2.8%
    # worst -- an order of magnitude under the ~20-30% replicate noise. That probe
    # is strictly more aggressive than this flag, since it let timing windows
    # overlap each other too, which the lock still prevents.
    #
    # Pairs naturally with KB_GPU_EVAL_LOCK_SLOTS>1: slots bound how many timing
    # windows coexist, this bounds what else has to queue behind them.
    _unlock_correctness = _env_flag("KB_EVAL_UNLOCK_CORRECTNESS", default=False)

    # Estimated device need for THIS eval, from the tensors we already built on the
    # host. k covers the correctness copy + the timing copy + activations; measured
    # peak was ~7x input size on a level-1 problem, but the two copies are what the
    # slot semaphore multiplies, so k defaults to 2.5. Factor 0 (the default) leaves
    # the gate entirely off, so a run already in flight is untouched.
    try:
        _mem_k = float(os.environ.get("KB_EVAL_MEM_GATE_FACTOR", "0"))
    except (TypeError, ValueError):
        _mem_k = 0.0
    _need = 0
    if _mem_k > 0 and _cpu_inputs_timing is not None:
        try:
            _need = int(_mem_k * sum(
                t.numel() * t.element_size()
                for t in _cpu_inputs_timing if torch.is_tensor(t)))
        except Exception:
            _need = 0
    # A SHORT valve was only ever needed because this wait was billed to the eval
    # deadline; the gate proceeds UNGATED on timeout, so cutting it short trades
    # the memory guarantee for nothing once the parent discounts the wait. Floor
    # it when a parent is listening. This matters most on the biggest problems,
    # which are exactly where the gate binds: at factor 7 on a 143GiB card L1P34
    # (7.0GiB inputs) admits 2, so 9 arms/GPU in lockstep queue several rounds
    # deep, and an ungated third resident there is ~147GiB -- an OOM.
    try:
        _mem_gate_timeout = float(os.environ.get("KB_EVAL_MEM_GATE_TIMEOUT_SEC", "1800"))
    except (TypeError, ValueError):
        _mem_gate_timeout = 1800.0
    try:
        from evolving_common.governor.gpu_lock import wait_reporting_active

        if wait_reporting_active():
            _mem_gate_timeout = max(_mem_gate_timeout, 1800.0)
    except Exception:
        pass

    # Reserve BEFORE queueing for a slot, not after: holding a slot while waiting on
    # memory would block evals that could have run. Released after the GPU phase.
    _mem_resv = _device_memory_reservation(_need, device, _mem_gate_timeout)
    _mem_gate_waited = _mem_resv.__enter__()

    _gpu_phase = _gpu_timing_lock(label="eval_gpu_phase")
    _gpu_entered = False
    _gpu_waited = 0.0
    _gpu_held_t0 = None
    _gpu_wait_t0 = time.perf_counter()
    if not _unlock_correctness:
        _gpu_phase.__enter__()
        _gpu_entered = True
        _gpu_held_t0 = time.perf_counter()
        _gpu_waited = _gpu_held_t0 - _gpu_wait_t0

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    _t = time.perf_counter()
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            pregenerated_inputs=_cpu_inputs_correct,
            phase_timers=_phase,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
            precision=precision,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )
    # Net of the input work, which is accounted separately, so the phases
    # partition the hold instead of overlapping.
    _phase["correctness"] = (
        (time.perf_counter() - _t) - _phase["correct_input_gen"] - _phase["correct_h2d"]
    )
    _cpu_inputs_correct = None  # release the host copies as soon as they are dead

    if _unlock_correctness and not _gpu_entered:
        # Correctness ran unlocked; take the lock now so it covers only the timing
        # window(s). Acquire unconditionally rather than skipping for incorrect
        # kernels: the excessive-speedup block below reads kernel_exec_result and
        # must sit inside the same held region for the phase accounting (and the
        # single __exit__ below) to stay well defined.
        _gpu_wait_t0 = time.perf_counter()
        _gpu_phase.__enter__()
        _gpu_entered = True
        _gpu_held_t0 = time.perf_counter()
        _gpu_waited = _gpu_held_t0 - _gpu_wait_t0

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                _t = time.perf_counter()
                set_seed(seed_num)
                inputs = _cpu_inputs_timing if _cpu_inputs_timing is not None else get_inputs()
                _phase["timing_input_gen"] += time.perf_counter() - _t
                # Convert inputs for performance measurement
                _t = time.perf_counter()
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
                _phase["timing_h2d"] += time.perf_counter() - _t

                model_new = custom_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

                # support multiple timing backend
                timing_fn = timing.get_timing_function(timing_method)
                _t = time.perf_counter()
                elapsed_times = timing_fn(
                    model_new,
                    inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                _phase["candidate"] += time.perf_counter() - _t
                runtime_stats = timing.get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = timing.runtime_from_stats(runtime_stats, default=-1.0)
                kernel_exec_result.runtime_stats = runtime_stats

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    # To get base PyTorch time (eager, various compile modes)
    # please use timing.measure_ref_program_time()   


    ###############################################################
    # [Experimental] to be modularized
    # Condition: custom kernel ModelNew is correct and we are able to time it correctly with kernel_exec_result
    # We are working on preventing excessive speedup issues
    ##############################################################

    if measure_performance and check_for_excessive_speedup:  # experimental: hence able to shut off codepath if needed

        # The reference *measurement* is dead work under the GPU lock whenever a
        # fixed baseline is supplied: the flag below prefers the baseline, and the
        # evolving-agent governor overwrites the measured ref_runtime with that
        # same baseline before it reaches any metric or prompt. Skipping is opt-in
        # so it cannot perturb a run already in flight.
        #
        # Only the measurement is skipped. The excessive-speedup / reward-hack flag
        # depends on baseline_runtime and the candidate runtime, never on this
        # window, and must keep running -- it gates is_hack, which gates the
        # is_new_best veto and the primary aggregate metric.
        _skip_ref_measurement = False
        if _env_flag("KB_EVAL_SKIP_DEAD_REF_TIMING", default=False):
            try:
                _skip_ref_measurement = (
                    baseline_runtime is not None and float(baseline_runtime) > 0
                )
            except (TypeError, ValueError):
                _skip_ref_measurement = False

        if verbose:
            print("[Eval] Additional checks to flag excessive speedup")

        if not _skip_ref_measurement:
            # Drop custom model before reference timing to reduce peak GPU memory.
            del custom_model
            custom_model = None
            torch.cuda.synchronize(device=device)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

            torch.cuda.synchronize(device=device)
            _t = time.perf_counter()
            set_seed(seed_num)
            inputs = _cpu_inputs_timing if _cpu_inputs_timing is not None else get_inputs()
            _phase["timing_input_gen"] += time.perf_counter() - _t
            # Convert inputs for performance measurement -- a fresh device copy, so
            # the reference sees pristine values even if the candidate mutated its own.
            _t = time.perf_counter()
            inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
            _phase["timing_h2d"] += time.perf_counter() - _t

            torch.cuda.synchronize(device=device)

            # time PyTorch reference function
            # same timing_fn as specified from before
            timing_fn = timing.get_timing_function(timing_method)
            _t = time.perf_counter()
            reference_elapsed_times = timing_fn(
                original_model,
                inputs, # ideally cloned for extra safety but handled already in correctness check
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
            )
            _phase["reference"] += time.perf_counter() - _t
            reference_runtime_stats = timing.get_timing_stats(reference_elapsed_times, device=device)
            kernel_exec_result.ref_runtime = timing.runtime_from_stats(reference_runtime_stats, default=-1.0)
            kernel_exec_result.ref_runtime_stats = reference_runtime_stats
        elif verbose:
            print("[Eval] Reference timing window skipped (fixed baseline supplied)")

        # Prefer fixed baseline when provided (aligns flag with displayed governor speedup).
        ref_for_speedup = (
            float(baseline_runtime)
            if baseline_runtime is not None and float(baseline_runtime) > 0
            else float(kernel_exec_result.ref_runtime)
        )
        if kernel_exec_result.runtime > 0 and ref_for_speedup > 0:
            effective_speedup = ref_for_speedup / kernel_exec_result.runtime
        else:
            effective_speedup = 0.0

        # TODO: integrate SoL estimation for each unique program on designated hardware
        # 30x (was 10x): algebraic collapse after a huge op (e.g. conv-transpose + GAP)
        # can legitimately land in the 10-20x band, so 10x false-positived real kernels.

        if verbose:
            print(f"[Eval] Effective Speedup is {effective_speedup:.2f}x using timing method {timing_method}")

        if effective_speedup > excessive_speedup_threshold:
            kernel_exec_result.metadata["excessive_speedup"] = True
            
            print(f"[WARNING] Excessive speedup {effective_speedup:.2f}x over {excessive_speedup_threshold}x threshold detected")
            print(f"[WARNING] Double check your kernel carefully to ensure it is not reward hacking.")


    _cpu_inputs_timing = None
    if _gpu_entered:
        _gpu_phase.__exit__(None, None, None)  # --- GPU phase ends ---
        _gpu_held = time.perf_counter() - _gpu_held_t0
    else:  # measure_performance=False and correctness unlocked -> never held
        _gpu_held = 0.0
    _mem_resv.__exit__(None, None, None)  # release the byte reservation

    # Phases that ran OUTSIDE the held region, and so must not be subtracted from
    # held_sec when forming the residual. input_gen_prelock is always outside;
    # the correctness trio joins it under KB_EVAL_UNLOCK_CORRECTNESS. Getting this
    # wrong drives other_sec negative and silently corrupts the only telemetry we
    # have on where the hold goes.
    _outside = {"input_gen_prelock"}
    if _unlock_correctness:
        _outside |= {"correct_input_gen", "correct_h2d", "correctness"}

    _emit_phase_record(
        {
            "held_sec": round(_gpu_held, 4),
            "waited_sec": round(_gpu_waited, 4),
            "hoisted": _hoisted,
            "unlocked_correctness": _unlock_correctness,
            "lock_slots": _gpu_lock_slots(),
            "mem_need_gb": round(_need / 2**30, 3) if _need else 0,
            "mem_gate_waited_sec": _mem_gate_waited or 0,
            "ref_window": "ran" if kernel_exec_result.ref_runtime_stats else "skipped",
            # Non-overlapping; "other" is the residual -- empty_cache, syncs,
            # model .to(), get_timing_stats -- i.e. the part of the hold that no
            # named phase accounts for.
            "phases": {k: round(v, 4) for k, v in _phase.items()},
            "other_sec": round(
                _gpu_held - sum(v for k, v in _phase.items() if k not in _outside),
                4,
            ),
        }
    )

    graceful_eval_cleanup(
        original_context, device, tempfile, extra_contexts=[custom_context]
    )
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool =False,
    seed: int =42,
    device: Optional[torch.device] =None,
    backend: str ="cuda",
    precision: torch.dtype =torch.float32,
    pregenerated_inputs: Optional[list] = None,
    phase_timers: Optional[dict] = None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    backend: backend type for handling dtype conversions
    precision: torch.dtype
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            _t = time.perf_counter()
            set_seed(trial_seed)
            if pregenerated_inputs is not None:
                # Built on the CPU before the lock was taken; same seed sequence,
                # same tensors.
                inputs = pregenerated_inputs[trial]
            else:
                inputs = get_inputs_fn()
            if phase_timers is not None:
                phase_timers["correct_input_gen"] += time.perf_counter() - _t
            # Convert inputs to appropriate dtypes for GPU computation
            _t = time.perf_counter()
            inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
            if phase_timers is not None:
                phase_timers["correct_h2d"] += time.perf_counter() - _t

            set_seed(trial_seed)
    
            model = original_model_instance.to(device=device, dtype=precision)

            set_seed(trial_seed)
     
            model_new = new_model_instance.to(device=device, dtype=precision)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # in torchbench, they use both precisions for atol and rtol
                # kernelbench v0 and v0.1 uses fp32, atol = rtol = 1e-02
                # now we will return the tolerance from get_tolerance_for_precision
                tolerance = get_tolerance_for_precision(precision)
                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=tolerance, rtol=tolerance
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print("\n")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                # Also store the full traceback in metadata for debugging
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# Note: fetch_baseline_time is available in kernelbench.timing module