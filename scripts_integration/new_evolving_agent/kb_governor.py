"""Concrete evolving governor implementation for KernelBench-compatible runs."""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import traceback
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELF_EVOLVING_ROOT = _REPO_ROOT / "Self-Evolving-Agent"
if str(_SELF_EVOLVING_ROOT) not in sys.path:
    # Keep root package precedence for KernelBench dataset/eval imports.
    sys.path.append(str(_SELF_EVOLVING_ROOT))

from evolving_common.benchmark_memory import fresh_l0_for_problem, l0_entries_to_json_serializable
from evolving_common.governor import maybe_promote_l0_to_l1, normalize_extracted_python
from evolving_common.governor.base import BaseEvolvingGovernor
from evolving_common.llm_client import call_coder_with_meta
from evolving_common.memory_manager import append_l0, format_l0_for_prompt, read_l1
from evolving_common.metrics_holder import BestMetricsHolder
from evolving_common.prompt_context import (
    SUMMARIZER_SYSTEM_PROMPT,
    build_summarizer_user_message,
    build_user_prompt_with_memory,
)
from evolving_common.run_recorder import BenchmarkRunRecorder, RunRecorderConfig


def _load_module_from_file(file_path: Path, module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module for {file_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


try:
    from kernelbench.config import KBGovernorConfig
    from kernelbench.schemas import KBEvalResult, KBGovernorResult, KBIterationRecord
except Exception:
    _config_path = _SELF_EVOLVING_ROOT / "kernelbench" / "config.py"
    _schema_path = _SELF_EVOLVING_ROOT / "kernelbench" / "schemas.py"
    _cfg_module = _load_module_from_file(_config_path, "sea_kb_config")
    _schema_module = _load_module_from_file(_schema_path, "sea_kb_schema")
    KBGovernorConfig = getattr(_cfg_module, "KBGovernorConfig")
    KBEvalResult = getattr(_schema_module, "KBEvalResult")
    KBIterationRecord = getattr(_schema_module, "KBIterationRecord")
    KBGovernorResult = getattr(_schema_module, "KBGovernorResult")


CODER_SYSTEM_PROMPT = """You are an expert GPU kernel engineer solving KernelBench optimization tasks.

Output rules:
1. Return exactly one fenced Python code block.
2. The code block must define ModelNew that implements the same behavior as the reference architecture.
3. Use the requested backend style in the task context.
4. Prioritize correctness first, then maximize speedup.
5. Do not include any explanation outside the code block.
"""


def _run_kernelbench_eval(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        kb_eval = import_module("kernelbench.eval")
    except ModuleNotFoundError:
        return {
            "ok": False,
            "worker_error": "kernelbench.eval unavailable in this environment",
        }

    try:
        dtype = kb_eval.get_torch_dtype_from_string(payload["precision"])
        result = kb_eval.eval_kernel_against_ref(
            payload["reference_code"],
            payload["candidate_code"],
            backend=payload["backend"],
            precision=dtype,
            measure_performance=True,
            build_dir=payload.get("build_dir"),
        )
    except Exception as exc:  # pragma: no cover - runtime external branch
        return {
            "ok": False,
            "worker_error": f"kernelbench evaluation error: {type(exc).__name__}: {exc}",
        }

    runtime_stats = getattr(result, "runtime_stats", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    return {
        "ok": True,
        "compiled": bool(getattr(result, "compiled", False)),
        "correct": bool(getattr(result, "correctness", False)),
        "runtime": getattr(result, "runtime", None),
        "ref_runtime": getattr(result, "ref_runtime", None),
        "runtime_stats": dict(runtime_stats),
        "metadata": dict(metadata),
    }


def _kernelbench_eval_worker(payload: dict[str, Any], out_queue: Any) -> None:
    out_queue.put(_run_kernelbench_eval(payload))


class KBGovernor(BaseEvolvingGovernor[KBEvalResult]):
    """KernelBench governor that plugs into evolving_common memory/prompt/logging utilities."""

    def __init__(self, config: KBGovernorConfig) -> None:
        super().__init__(max_iterations=config.max_iterations)
        self.config = config

    def _get_coder_prompt(self, task_prompt: str, l1_text: str, l0_text: str) -> str:
        return build_user_prompt_with_memory(
            task_section=task_prompt,
            l1_text=l1_text,
            l0_formatted=l0_text,
            task_heading="## Official KernelBench task prompt",
            closing_instruction=(
                f"KernelBench Level {self.config.level}, Problem {self.config.problem_id}. "
                "Return exactly one fenced Python implementation."
            ),
        )

    def _evaluate_in_subprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        start_method = str(self.config.evaluation_start_method).strip().lower()
        if start_method not in {"spawn", "fork", "forkserver"}:
            start_method = "spawn"

        ctx = mp.get_context(start_method)
        out_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_kernelbench_eval_worker, args=(payload, out_queue), daemon=True)
        proc.start()
        proc.join(timeout=float(self.config.evaluation_timeout_s))

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            return {
                "ok": False,
                "worker_error": f"kernelbench evaluation timeout after {self.config.evaluation_timeout_s}s",
            }

        try:
            return out_queue.get_nowait()
        except queue.Empty:
            return {
                "ok": False,
                "worker_error": f"kernelbench evaluation worker exited with code {proc.exitcode}",
            }

    def _build_eval_result(self, payload: dict[str, Any]) -> KBEvalResult:
        if not payload.get("ok", False):
            return KBEvalResult(
                compiled=False,
                correct=False,
                error_message=str(payload.get("worker_error") or "kernelbench evaluation failed"),
            )

        compiled = bool(payload.get("compiled", False))
        correct = bool(payload.get("correct", False))
        runtime = payload.get("runtime")
        ref_runtime = payload.get("ref_runtime")
        runtime_stats = payload.get("runtime_stats") if isinstance(payload.get("runtime_stats"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        speedup: float | None = None
        if correct and runtime and ref_runtime and runtime > 0 and ref_runtime > 0:
            speedup = float(ref_runtime / runtime)

        error_message = None
        if not correct:
            error_message = (
                metadata.get("compilation_error")
                or metadata.get("runtime_error")
                or metadata.get("correctness_issue")
            )
            if error_message is not None:
                error_message = str(error_message)

        runtime_value: float | None
        ref_runtime_value: float | None
        try:
            runtime_value = float(runtime) if runtime is not None and float(runtime) >= 0 else None
        except Exception:
            runtime_value = None

        try:
            ref_runtime_value = (
                float(ref_runtime)
                if ref_runtime is not None and float(ref_runtime) >= 0
                else None
            )
        except Exception:
            ref_runtime_value = None

        return KBEvalResult(
            compiled=compiled,
            correct=correct,
            speedup=speedup,
            error_message=error_message,
            runtime=runtime_value,
            ref_runtime=ref_runtime_value,
            runtime_stats=dict(runtime_stats),
            metadata=dict(metadata),
        )

    def _evaluate_candidate(self, code: str, *, attempt: int) -> KBEvalResult:
        normalized = code.strip()
        if not normalized:
            return KBEvalResult(
                compiled=False,
                correct=False,
                error_message="empty candidate code",
            )

        if not self.config.reference_code:
            return KBEvalResult(
                compiled=False,
                correct=False,
                error_message="missing reference_code for KernelBench evaluation",
            )

        build_dir = (
            Path(self.config.results_root)
            / self.config.run_name
            / "builds"
            / f"l{self.config.level}_p{self.config.problem_id}_iter{attempt}_{uuid4().hex[:8]}"
        )
        build_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "reference_code": self.config.reference_code,
            "candidate_code": normalized,
            "backend": self.config.backend,
            "precision": self.config.precision,
            "build_dir": str(build_dir),
        }

        if self.config.isolate_evaluation_process:
            eval_payload = self._evaluate_in_subprocess(payload)
        else:
            eval_payload = _run_kernelbench_eval(payload)

        return self._build_eval_result(eval_payload)

    def _get_summarizer_prompt(self, result: KBEvalResult) -> str:
        terminal = (
            f"compiled={result.compiled}, correct={result.correct}, "
            f"speedup={result.speedup if result.speedup is not None else 'n/a'}, "
            f"error={result.error_message or 'none'}"
        )
        return build_summarizer_user_message(terminal)

    def _handle_fatal_error(self, error: Exception) -> bool:
        return isinstance(error, (MemoryError, SystemError))

    def run(self, *, task_prompt: str) -> KBGovernorResult:
        run_dir = Path(self.config.results_root) / self.config.run_name
        workspace_dir = (
            run_dir
            / "workspaces"
            / f"level_{self.config.level}_problem_{self.config.problem_id}"
        )
        workspace_dir.mkdir(parents=True, exist_ok=True)

        l1_path = self.config.shared_l1_path or run_dir / "shared_l1.txt"
        l1_path.parent.mkdir(parents=True, exist_ok=True)
        if not l1_path.exists():
            l1_path.write_text("# Shared L1 journal for evolving KernelBench batch\n", encoding="utf-8")

        holder = BestMetricsHolder()
        recorder = BenchmarkRunRecorder(
            RunRecorderConfig(
                output_dir=workspace_dir,
                time_sample_interval_sec=self.config.run_recorder_time_sample_interval_sec,
            ),
            holder,
            run_metadata={
                "benchmark": "kernelbench",
                "level": self.config.level,
                "problem_id": self.config.problem_id,
                "backend": self.config.backend,
                "precision": self.config.precision,
            },
        )

        l0 = fresh_l0_for_problem()
        records: list[KBIterationRecord] = []
        best_speedup = 0.0
        best_correct = False
        best_compiled = False
        best_code: str | None = None
        best_code_path: Path | None = None
        best_runtime = -1.0
        best_runtime_stats: dict[str, Any] = {}
        best_metadata: dict[str, Any] = {}
        run_error: str | None = None
        fatal_error_count = 0

        recorder.start()
        try:
            for attempt in range(1, self.config.max_iterations + 1):
                holder.set_iteration(attempt)
                l1_text = read_l1(l1_path)
                l0_text = format_l0_for_prompt(l0)
                coder_prompt = self._get_coder_prompt(task_prompt, l1_text, l0_text)
                coder_messages = [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": coder_prompt},
                ]

                try:
                    raw, _tokens, coder_meta = call_coder_with_meta(
                        coder_messages,
                        max_tokens=self.config.coder_max_tokens,
                        timeout_sec=self.config.coder_timeout_sec,
                    )
                except Exception as exc:
                    fatal_error_count += 1
                    err = f"coder_call_error: {type(exc).__name__}: {exc}"
                    append_l0(l0, "terminal", err)
                    metrics_iteration = {
                        "attempt": attempt,
                        "compiled": False,
                        "correct": False,
                        "speedup": 0.0,
                        "error": err,
                    }
                    holder.update_iteration_metrics(metrics_iteration)
                    recorder.record_iteration_snapshot(
                        iteration=attempt,
                        l0_entries=l0_entries_to_json_serializable(l0),
                        l1_text=l1_text,
                        l1_path=str(l1_path),
                        metrics_iteration=metrics_iteration,
                        metrics_best=holder.get_snapshot().get("metrics_best", {}),
                    )
                    if self._handle_fatal_error(exc) or fatal_error_count >= self.config.max_fatal_errors:
                        run_error = err
                        break
                    continue

                recorder.record_llm_turn(
                    iteration=attempt,
                    phase="coder",
                    messages=coder_messages,
                    assistant_text=raw,
                    extra=coder_meta,
                )

                code, extract_err = normalize_extracted_python(raw)
                if extract_err or not code:
                    extraction_error = extract_err or "no code extracted"
                    append_l0(l0, "terminal", f"extract_error={extraction_error}; raw={(raw or '')[:1000]}")
                    eval_result = KBEvalResult(
                        compiled=False,
                        correct=False,
                        error_message=extraction_error,
                    )
                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code="# extraction_failed",
                        evaluation=eval_result,
                    )
                    records.append(record)
                else:
                    append_l0(l0, "code", code)
                    eval_result = self._evaluate_candidate(code, attempt=attempt)
                    terminal_log = (
                        f"KERNEL_BENCH_CORRECT: {eval_result.correct}\n"
                        f"KERNEL_BENCH_SPEEDUP: {float(eval_result.speedup or 0.0):.6f}\n"
                    )
                    if eval_result.error_message:
                        terminal_log += f"KERNEL_BENCH_ERROR: {eval_result.error_message}\n"
                    append_l0(l0, "terminal", terminal_log)

                    record = KBIterationRecord(
                        attempt=attempt,
                        candidate_code=code,
                        evaluation=eval_result,
                    )
                    records.append(record)

                    speedup = float(eval_result.speedup or 0.0)
                    if eval_result.correct and speedup >= best_speedup:
                        best_speedup = speedup
                        best_correct = bool(eval_result.correct)
                        best_compiled = bool(eval_result.compiled)
                        best_code = code
                        best_runtime = float(eval_result.runtime) if eval_result.runtime is not None else -1.0
                        best_runtime_stats = dict(eval_result.runtime_stats)
                        best_metadata = dict(eval_result.metadata)
                        best_code_path = workspace_dir / f"best_iter_{attempt}.py"
                        best_code_path.write_text(code, encoding="utf-8")

                metrics_iteration = {
                    "attempt": attempt,
                    "compiled": bool(eval_result.compiled),
                    "correct": bool(eval_result.correct),
                    "speedup": float(eval_result.speedup or 0.0),
                    "error": eval_result.error_message,
                }
                metrics_best = {
                    "compiled": best_compiled,
                    "correct": best_correct,
                    "speedup": best_speedup,
                }
                holder.update_iteration_metrics(metrics_iteration)
                holder.update_best(metrics_best)

                maybe_promote_l0_to_l1(
                    l0,
                    l1_path=l1_path,
                    entry_threshold=self.config.promote_entry_threshold,
                    token_budget=self.config.promote_token_budget,
                    summarizer_system_prompt=SUMMARIZER_SYSTEM_PROMPT,
                    build_summarizer_user_message=build_summarizer_user_message,
                    summarizer_max_tokens=self.config.summarizer_max_tokens,
                    summarizer_timeout_sec=self.config.summarizer_timeout_sec,
                    enable_promotion=self.config.enable_promotion,
                    catch_summarizer_errors=True,
                    verbose=self.config.verbose,
                    log_prefix="[kb-governor]",
                    on_summarizer_round=recorder.summarizer_callback(attempt),
                    on_l0_cleared_without_l1=recorder.flush_without_l1_callback(attempt),
                )

                recorder.record_iteration_snapshot(
                    iteration=attempt,
                    l0_entries=l0_entries_to_json_serializable(l0),
                    l1_text=read_l1(l1_path),
                    l1_path=str(l1_path),
                    metrics_iteration=metrics_iteration,
                    metrics_best=metrics_best,
                )

                if self.config.verbose:
                    print(
                        f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f}"
                    )

        except Exception:
            run_error = traceback.format_exc()
        finally:
            recorder.stop(
                final_metadata={
                    "best_speedup": best_speedup,
                    "best_correct": best_correct,
                    "best_compiled": best_compiled,
                    "error": run_error,
                }
            )

        return KBGovernorResult(
            level=self.config.level,
            problem_id=self.config.problem_id,
            backend=self.config.backend,
            precision=self.config.precision,
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


def governor_result_to_dict(result: KBGovernorResult) -> dict[str, Any]:
    return result.model_dump()


def safe_run_kb_governor(cfg: KBGovernorConfig, *, task_prompt: str) -> KBGovernorResult:
    try:
        return KBGovernor(cfg).run(task_prompt=task_prompt)
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
