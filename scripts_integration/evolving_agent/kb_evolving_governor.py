"""KernelBench-specific governor loop for the Self-Evolving-Agent prototype."""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELF_EVOLVING_ROOT = _REPO_ROOT / "Self-Evolving-Agent"
if str(_SELF_EVOLVING_ROOT) not in sys.path:
    sys.path.append(str(_SELF_EVOLVING_ROOT))

import torch
from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from execution import (
    DEFAULT_RUN_TIMEOUT_SEC,
    extract_python_code,
    run_solution,
    write_solution_py,
)
from llm_client import call_coder, call_summarizer  # type: ignore  # noqa: E402
from memory_manager import (
    DEFAULT_L0_ENTRY_PROMOTE_THRESHOLD,
    DEFAULT_L0_TOKEN_BUDGET,
    L1_KNOWLEDGE_PATH,
    WORKSPACE_DIR,
    append_l0,
    format_l0_for_prompt,
    new_l0,
    promote_l0_to_l1,
    read_l1,
    should_promote_l0,
)

CODER_SYSTEM_PROMPT = """You are an expert GPU kernel engineer solving KernelBench optimization tasks.

Output rules:
1. Return exactly one fenced Python code block.
2. The code block must define ModelNew that implements the same behavior as the reference architecture.
3. Use the requested backend style in the task context.
4. Prioritize correctness first, then maximize speedup.
5. Do not include any explanation outside the code block.
"""

SUMMARIZER_SYSTEM_PROMPT = """You summarize optimization experiments across KernelBench problems.

Output concise bullet points only:
- What failed and why.
- What improved speedup while staying correct.
- Generalizable kernel optimization lessons for future problems.
"""

_FATAL_CUDA_ERROR_MARKERS = (
    "illegal memory access",
    "device-side assert",
    "cuda error",
)


@dataclass
class KBGovernorConfig:
    run_name: str
    level: int
    problem_id: int
    backend: str = "cuda"
    precision: str = "fp32"
    max_iterations: int = 20
    shared_l1_path: Path | None = None
    results_root: Path = Path("results/evolving_logs")
    promote_entry_threshold: int = DEFAULT_L0_ENTRY_PROMOTE_THRESHOLD
    promote_token_budget: int = DEFAULT_L0_TOKEN_BUDGET
    coder_max_tokens: int = 8192
    summarizer_max_tokens: int = 4096
    coder_timeout_sec: float = 90.0
    summarizer_timeout_sec: float = 60.0
    eval_timeout_sec: float = 300.0
    eval_start_method: str = "spawn"
    verbose: bool = True


@dataclass
class IterationRecord:
    iteration: int
    speedup: float
    correctness: bool
    compiled: bool
    error: str | None
    runtime: float = -1.0
    runtime_stats: dict | None = None


@dataclass
class KBGovernorResult:
    level: int
    problem_id: int
    backend: str
    precision: str
    best_speedup: float
    best_correct: bool
    best_compiled: bool
    best_code_path: str | None
    best_code: str | None
    iterations_run: int
    records: list[IterationRecord]
    runtime: float = -1.0
    runtime_stats: dict | None = None
    metadata: dict | None = None
    error: str | None = None


def _is_fatal_cuda_error(err: object | None) -> bool:
    if not err:
        return False
    lower = str(err).lower()
    return any(marker in lower for marker in _FATAL_CUDA_ERROR_MARKERS)


def _build_coder_user_prompt(
    *,
    task_prompt: str,
    l1_text: str,
    l0_text: str,
    level: int,
    problem_id: int,
    backend: str,
    precision: str,
) -> str:
    return (
        f"## Problem\n"
        f"KernelBench Level {level}, Problem {problem_id}\n"
        f"Backend: {backend}\n"
        f"Precision: {precision}\n\n"
        "## Official KernelBench task prompt\n"
        f"{task_prompt.strip()}\n\n"
        "## Shared L1 knowledge\n"
        f"{(l1_text or '(none yet)').strip()}\n\n"
        "## Recent L0 logs\n"
        f"{l0_text}\n\n"
        "Return only one fenced Python block containing complete candidate code."
    )


def _build_summarizer_user_prompt(l0_text: str) -> str:
    return (
        "Raw L0 logs from recent KernelBench attempts:\n\n"
        f"{l0_text}\n\n"
        "Summarize short reusable optimization lessons for shared L1."
    )


def _evaluate_candidate(
    *,
    reference_code: str,
    candidate_code: str,
    backend: str,
    precision: str,
) -> tuple[float, bool, bool, str | None, float, dict, dict]:
    try:
        if kb_eval is None:
            return 0.0, False, False, "kernelbench.eval unavailable", -1.0, {}, {}

        dtype = kb_eval.get_torch_dtype_from_string(precision)
        result = kb_eval.eval_kernel_against_ref(
            reference_code,
            candidate_code,
            backend=backend,
            precision=dtype,
            measure_performance=True,
        )
        correctness = bool(result.correctness)
        compiled = bool(result.compiled)
        speedup = 0.0
        runtime = float(result.runtime) if result.runtime is not None else -1.0
        runtime_stats = dict(result.runtime_stats or {})
        metadata = dict(result.metadata or {})
        if correctness and result.runtime and result.runtime > 0 and result.ref_runtime and result.ref_runtime > 0:
            speedup = float(result.ref_runtime / result.runtime)
        err = None
        if not correctness:
            err = (
                metadata.get("compilation_error")
                or metadata.get("runtime_error")
                or metadata.get("correctness_issue")
            )
        if err is not None:
            err = str(err)
        return speedup, correctness, compiled, err, runtime, runtime_stats, metadata
    except Exception as exc:
        return 0.0, False, False, str(exc), -1.0, {}, {}


def _evaluate_candidate_worker(
    reference_code: str,
    candidate_code: str,
    backend: str,
    precision: str,
    queue: mp.Queue,
) -> None:
    payload = _evaluate_candidate(
        reference_code=reference_code,
        candidate_code=candidate_code,
        backend=backend,
        precision=precision,
    )
    queue.put(payload)


def _evaluate_candidate_isolated(
    *,
    reference_code: str,
    candidate_code: str,
    backend: str,
    precision: str,
    timeout_sec: float,
    start_method: str,
) -> tuple[float, bool, bool, str | None, float, dict, dict]:
    ctx = mp.get_context(start_method)
    queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_candidate_worker,
        args=(reference_code, candidate_code, backend, precision, queue),
    )

    try:
        proc.start()
        proc.join(timeout=None if timeout_sec <= 0 else timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            return (
                0.0,
                False,
                False,
                f"evaluation_timeout: exceeded {timeout_sec:.1f}s for isolated eval",
                -1.0,
                {},
                {},
            )

        if not queue.empty():
            payload = queue.get_nowait()
            if isinstance(payload, tuple) and len(payload) == 7:
                speedup, correctness, compiled, err, runtime, runtime_stats, metadata = payload
                return (
                    float(speedup),
                    bool(correctness),
                    bool(compiled),
                    (str(err) if err is not None else None),
                    float(runtime),
                    (runtime_stats if isinstance(runtime_stats, dict) else {}),
                    (metadata if isinstance(metadata, dict) else {}),
                )

        return (
            0.0,
            False,
            False,
            f"evaluation_worker_error: worker exited with code {proc.exitcode}",
            -1.0,
            {},
            {},
        )
    except Exception as exc:
        return (
            0.0,
            False,
            False,
            f"evaluation_orchestration_error: {type(exc).__name__}: {exc}",
            -1.0,
            {},
            {},
        )
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        queue.close()


def run_kb_governor(cfg: KBGovernorConfig) -> KBGovernorResult:
    if not torch.cuda.is_available():
        return KBGovernorResult(
            level=cfg.level,
            problem_id=cfg.problem_id,
            backend=cfg.backend,
            precision=cfg.precision,
            best_speedup=0.0,
            best_correct=False,
            best_compiled=False,
            best_code_path=None,
            best_code=None,
            iterations_run=0,
            records=[],
            error="CUDA is not available.",
        )

    dataset = construct_kernelbench_dataset(level=cfg.level, source="local")
    problem = dataset.get_problem_by_id(cfg.problem_id)

    task_prompt = get_prompt_for_backend(
        ref_arch_src=problem.code,
        backend=cfg.backend,
        option="one_shot",
        precision=cfg.precision,
    )

    run_dir = cfg.results_root / cfg.run_name
    workspace_dir = run_dir / "workspaces" / f"level_{cfg.level}_problem_{cfg.problem_id}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    l1_path = cfg.shared_l1_path or run_dir / "shared_l1.txt"
    l1_path.parent.mkdir(parents=True, exist_ok=True)

    l0 = new_l0()
    records: list[IterationRecord] = []

    best_speedup = 0.0
    best_correct = False
    best_compiled = False
    best_code: str | None = None
    best_code_path: Path | None = None
    best_runtime = -1.0
    best_runtime_stats: dict = {}
    best_metadata: dict = {}
    last_eval_runtime = -1.0
    last_eval_runtime_stats: dict = {}
    last_eval_metadata: dict = {}
    fatal_cuda_error_count = 0
    run_error: str | None = None

    for iteration in range(1, cfg.max_iterations + 1):
        if cfg.verbose:
            print(
                f"[kb-governor] L{cfg.level}P{cfg.problem_id} "
                f"iteration {iteration}/{cfg.max_iterations}"
            )

        l1_text = read_l1(l1_path)
        l0_text = format_l0_for_prompt(l0)
        user_prompt = _build_coder_user_prompt(
            task_prompt=task_prompt,
            l1_text=l1_text,
            l0_text=l0_text,
            level=cfg.level,
            problem_id=cfg.problem_id,
            backend=cfg.backend,
            precision=cfg.precision,
        )

        raw, _ = call_coder(
            [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=cfg.coder_max_tokens,
            timeout_sec=cfg.coder_timeout_sec,
        )
        # `extract_python_code` has inconsistent return shapes across modules:
        # - kernelbench.utils.extract_python_code(text) -> str
        # - Self-Evolving-Agent.execution.extract_python_code(raw) -> (code, err)
        # Normalize both cases to (code, extract_err).
        _ex_res = extract_python_code(raw)
        if isinstance(_ex_res, tuple):
            # expected (code, err)
            code, extract_err = _ex_res if len(_ex_res) == 2 else (_ex_res[0], None)
        else:
            # kernelbench variant returns a string (possibly empty)
            code = _ex_res if _ex_res else None
            extract_err = None if code else "no code extracted"

        if extract_err or not code:
            msg = f"extract_error={extract_err}; raw={str(raw)[:1200]}"
            append_l0(l0, "terminal", msg)
            records.append(
                IterationRecord(
                    iteration=iteration,
                    speedup=0.0,
                    correctness=False,
                    compiled=False,
                    error=extract_err or "no code extracted",
                )
            )
        else:
            append_l0(l0, "code", code)
            speedup, correctness, compiled, err, runtime, runtime_stats, metadata = _evaluate_candidate_isolated(
                reference_code=problem.code,
                candidate_code=code,
                backend=cfg.backend,
                precision=cfg.precision,
                timeout_sec=cfg.eval_timeout_sec,
                start_method=cfg.eval_start_method,
            )

            last_eval_runtime = runtime
            last_eval_runtime_stats = runtime_stats
            last_eval_metadata = metadata

            terminal_log = (
                f"KERNEL_BENCH_CORRECT: {correctness}\n"
                f"KERNEL_BENCH_SPEEDUP: {speedup:.6f}\n"
            )
            if err:
                terminal_log += f"KERNEL_BENCH_ERROR: {err}\n"
            append_l0(l0, "terminal", terminal_log)

            records.append(
                IterationRecord(
                    iteration=iteration,
                    speedup=speedup,
                    correctness=correctness,
                    compiled=compiled,
                    error=err,
                    runtime=runtime,
                    runtime_stats=runtime_stats,
                )
            )

            if correctness and speedup >= best_speedup:
                best_speedup = speedup
                best_correct = correctness
                best_compiled = compiled
                best_code = code
                best_runtime = runtime
                best_runtime_stats = runtime_stats
                best_metadata = metadata
                best_code_path = workspace_dir / f"best_iter_{iteration}.py"
                best_code_path.write_text(code, encoding="utf-8")

            if cfg.verbose:
                print(
                    f"[kb-governor] iter={iteration} "
                    f"correct={correctness} compiled={compiled} speedup={speedup:.4f}"
                )

            if _is_fatal_cuda_error(err):
                fatal_cuda_error_count += 1
                if cfg.verbose:
                    print("[kb-governor] detected fatal CUDA runtime error in candidate; continuing next iteration")

        if should_promote_l0(
            l0,
            entry_threshold=cfg.promote_entry_threshold,
            token_budget=cfg.promote_token_budget,
        ):
            l0_snapshot = format_l0_for_prompt(l0)
            try:
                summary, _ = call_summarizer(
                    [
                        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_summarizer_user_prompt(l0_snapshot)},
                    ],
                    max_tokens=cfg.summarizer_max_tokens,
                    timeout_sec=cfg.summarizer_timeout_sec,
                )
            except Exception as exc:
                summary = (
                    "- Summarizer request failed; promoting fallback note instead.\n"
                    f"- Error: {type(exc).__name__}: {exc}"
                )
            summary_text = (summary or "").strip() or "- No summary returned."
            promote_l0_to_l1(l0, summary_text, l1_path=l1_path)
            if cfg.verbose:
                print("[kb-governor] promoted L0 -> shared L1")

    if fatal_cuda_error_count > 0:
        run_error = (
            f"observed_fatal_cuda_error_in_{fatal_cuda_error_count}_iterations; "
            "continued via isolated per-iteration eval workers"
        )

    if not best_metadata:
        best_metadata = dict(last_eval_metadata)

    if best_runtime < 0 and last_eval_runtime >= 0:
        best_runtime = last_eval_runtime

    if not best_runtime_stats and last_eval_runtime_stats:
        best_runtime_stats = dict(last_eval_runtime_stats)

    return KBGovernorResult(
        level=cfg.level,
        problem_id=cfg.problem_id,
        backend=cfg.backend,
        precision=cfg.precision,
        best_speedup=best_speedup,
        best_correct=best_correct,
        best_compiled=best_compiled,
        best_code_path=str(best_code_path) if best_code_path else None,
        best_code=best_code,
        iterations_run=len(records),
        records=records,
        runtime=float(best_runtime),
        runtime_stats=best_runtime_stats,
        metadata=best_metadata,
        error=run_error,
    )


def governor_result_to_dict(result: KBGovernorResult) -> dict:
    return {
        "level": result.level,
        "problem_id": result.problem_id,
        "backend": result.backend,
        "precision": result.precision,
        "best_speedup": result.best_speedup,
        "best_correct": result.best_correct,
        "best_compiled": result.best_compiled,
        "best_code_path": result.best_code_path,
        "iterations_run": result.iterations_run,
        "records": [
            {
                "iteration": r.iteration,
                "speedup": r.speedup,
                "correctness": r.correctness,
                "compiled": r.compiled,
                "error": r.error,
                "runtime": r.runtime,
                "runtime_stats": r.runtime_stats or {},
            }
            for r in result.records
        ],
        "runtime": result.runtime,
        "runtime_stats": result.runtime_stats or {},
        "metadata": result.metadata or {},
        "error": result.error,
    }


def safe_run_kb_governor(cfg: KBGovernorConfig) -> KBGovernorResult:
    try:
        return run_kb_governor(cfg)
    except Exception:
        return KBGovernorResult(
            level=cfg.level,
            problem_id=cfg.problem_id,
            backend=cfg.backend,
            precision=cfg.precision,
            best_speedup=0.0,
            best_correct=False,
            best_compiled=False,
            best_code_path=None,
            best_code=None,
            iterations_run=0,
            records=[],
            runtime=-1.0,
            runtime_stats={},
            metadata={},
            error=traceback.format_exc(),
        )
