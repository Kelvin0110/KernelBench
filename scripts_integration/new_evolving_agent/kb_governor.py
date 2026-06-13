"""Concrete evolving governor implementation for KernelBench-compatible runs."""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import shutil
import queue
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
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
from evolving_common.eval_feedback import (
    BEST_EVAL_FEEDBACK_EMPTY,
    format_best_eval_feedback,
    format_kernelbench_best_metrics,
)
from evolving_common.execution import evaluate_in_subprocess
from evolving_common.governor import maybe_promote_l0_to_l1, normalize_extracted_python
from evolving_common.governor.l0_round_summary import maybe_summarize_l0_round
from evolving_common.governor.base import BaseEvolvingGovernor
from evolving_common.governor.code_extract import parse_selected_entry_ids, parse_unfold_round_ids
from evolving_common.governor.gpu_reserver import GPUMemoryReserver
from evolving_common.governor.util import (
    extract_optional_tag,
    get_fallback_action,
    parse_action_selector_response,
)
from evolving_common.l0_context import (
    build_l0_archived_catalog,
    build_l0_global_summary,
    build_l0_recent_full,
    build_l0_unfolded_full,
)
from evolving_common.llm_client import (
    call_action_selector_with_meta,
    call_coder_with_meta,
    call_extractor_with_meta,
    get_action_selector_model_id,
    resolve_nvidia_model_id,
)
from evolving_common.memory_manager import (
    L0Round,
    PromotionTrigger,
    finalize_l0_round,
    l0_rounds_promotion_window,
    read_l1,
    read_l1_jsonl,
    read_recent_l1_jsonl,
)
from evolving_common.metrics_holder import BestMetricsHolder
from evolving_common.prompt_context import (
    ACTION_SELECTOR_SYSTEM_PROMPT,
    ALLOWED_CODER_ACTIONS,
    BASE_EVOLVING_CODER_SYSTEM_PROMPT,
    CODER_PREFLIGHT_SYSTEM_PROMPT,
    DEFAULT_EXTRACTOR_SYSTEM_PROMPT,
    GEN3_CODER_FINAL_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    build_action_coder_user_prompt,
    build_action_selector_user_message,
    build_coder_final_user_appendix,
    build_coder_preflight_user_appendix,
    build_extractor_user_message,
    build_l1_skill_catalog,
    build_selected_l1_skills_section,
    build_summarizer_user_message,
    build_user_prompt_with_memory,
    format_l0_for_coder_prompt,
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


CODER_SYSTEM_PROMPT = f"""You are an expert GPU kernel engineer solving kernel code optimization tasks.

{GEN3_CODER_FINAL_SYSTEM_PROMPT}

KernelBench rules:
1. The code block must define ModelNew that implements the same behavior as the reference architecture.
2. Use the requested backend style in the task context.
3. Prioritize correctness first, then maximize speedup.
"""

LEGACY_CODER_SYSTEM_PROMPT = f"""You are an expert GPU kernel engineer solving kernel code optimization tasks.

{BASE_EVOLVING_CODER_SYSTEM_PROMPT}

Output rules:
1. Return tags followed by exactly one fenced Python code block and no extra prose.
2. The code block must define ModelNew that implements the same behavior as the reference architecture.
3. Use the requested backend style in the task context.
4. Prioritize correctness first, then maximize speedup.
"""

EXTRACTOR_SYSTEM_PROMPT = DEFAULT_EXTRACTOR_SYSTEM_PROMPT


def _problem_build_name_prefix(level: int, problem_id: str | int) -> str:
    return f"l{int(level)}_p{int(problem_id)}_"


def _remove_build_dir(build_dir: Path) -> None:
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)


def cleanup_problem_build_artifacts(
    results_root: Path | str,
    run_name: str,
    *,
    level: int,
    problem_id: str | int,
) -> None:
    """Remove KernelBench CUDA extension build dirs for one problem under a run."""
    builds_dir = Path(results_root) / run_name / "builds"
    if not builds_dir.is_dir():
        return
    prefix = _problem_build_name_prefix(level, problem_id)
    for path in list(builds_dir.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            shutil.rmtree(path, ignore_errors=True)


def _run_kernelbench_eval(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        kb_eval = import_module("kernelbench.eval")
    except ModuleNotFoundError:
        return {
            "ok": False,
            "worker_error": "kernelbench.eval unavailable in this environment",
        }

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        dtype = kb_eval.get_torch_dtype_from_string(payload["precision"])
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = kb_eval.eval_kernel_against_ref(
                payload["reference_code"],
                payload["candidate_code"],
                backend=payload["backend"],
                precision=dtype,
                measure_performance=True,
                build_dir=payload.get("build_dir"),
            )
    except Exception as exc:  # pragma: no cover - runtime external branch
        terminal_stdout = stdout_capture.getvalue()
        terminal_stderr = stderr_capture.getvalue()
        terminal_output = "\n".join(
            chunk.strip()
            for chunk in (
                f"[stdout]\n{terminal_stdout}" if terminal_stdout.strip() else "",
                f"[stderr]\n{terminal_stderr}" if terminal_stderr.strip() else "",
            )
            if chunk
        )
        return {
            "ok": False,
            "worker_error": f"kernelbench evaluation error: {type(exc).__name__}: {exc}",
            "terminal_stdout": terminal_stdout,
            "terminal_stderr": terminal_stderr,
            "terminal_output": terminal_output or None,
        }

    runtime_stats = getattr(result, "runtime_stats", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    terminal_stdout = stdout_capture.getvalue()
    terminal_stderr = stderr_capture.getvalue()
    terminal_output = "\n".join(
        chunk.strip()
        for chunk in (
            f"[stdout]\n{terminal_stdout}" if terminal_stdout.strip() else "",
            f"[stderr]\n{terminal_stderr}" if terminal_stderr.strip() else "",
        )
        if chunk
    )
    return {
        "ok": True,
        "compiled": bool(getattr(result, "compiled", False)),
        "correct": bool(getattr(result, "correctness", False)),
        "runtime": getattr(result, "runtime", None),
        "ref_runtime": getattr(result, "ref_runtime", None),
        "runtime_stats": dict(runtime_stats),
        "metadata": dict(metadata),
        "terminal_stdout": terminal_stdout,
        "terminal_stderr": terminal_stderr,
        "terminal_output": terminal_output or None,
    }


def _kernelbench_eval_worker(payload: dict[str, Any], out_queue: Any) -> None:
    out_queue.put(_run_kernelbench_eval(payload))


class KBGovernor(BaseEvolvingGovernor[KBEvalResult]):
    """KernelBench governor that plugs into evolving_common memory/prompt/logging utilities."""

    def __init__(self, config: KBGovernorConfig) -> None:
        super().__init__(max_iterations=config.max_iterations)
        self.config = config
        self._last_promoted_round_count = 0
        self.reserver = GPUMemoryReserver(reserve_gb=46.0)  #  49140MiB for NVIDIA RTX A6000
        self.reserver.acquire()

    def _get_coder_prompt(
        self,
        task_prompt: str,
        l1_text: str,
        l0_text: str,
        *,
        selected_l1_entries: list[dict[str, str]] | None = None,
        allowed_actions: list[str] | None = None,
        iteration_context: str | None = None,
        best_eval_feedback: str | None = None,
    ) -> str:
        return build_user_prompt_with_memory(
            task_section=task_prompt,
            l1_text=l1_text,
            l0_formatted=l0_text,
            selected_l1_entries=selected_l1_entries,
            allowed_actions=allowed_actions,
            iteration_context=iteration_context,
            best_eval_feedback=best_eval_feedback,
            task_heading="## Official KernelBench task prompt",
            closing_instruction=(
                f"KernelBench Level {self.config.level}, Problem {self.config.problem_id}. "
                "Return exactly one fenced Python implementation."
            ),
        )

    def _promote_gen3_window(
        self,
        l0: list[L0Round],
        *,
        l1_path: Path,
        trigger: PromotionTrigger,
        attempt: int,
        recorder: BenchmarkRunRecorder,
    ) -> None:
        rounds_to_promote, end_exclusive = l0_rounds_promotion_window(
            l0,
            last_promoted_round_count=self._last_promoted_round_count,
            trigger=trigger,
        )
        self._last_promoted_round_count = maybe_promote_l0_to_l1(
            l0,
            l1_path=l1_path,
            rounds_to_promote=rounds_to_promote,
            promotion_end_exclusive=end_exclusive,
            summarizer_system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            build_summarizer_user_message=build_summarizer_user_message,
            summarizer_max_tokens=self.config.summarizer_max_tokens,
            summarizer_timeout_sec=self.config.summarizer_timeout_sec,
            enable_promotion=self.config.enable_promotion,
            should_write_l1=None,
            catch_summarizer_errors=True,
            verbose=self.config.verbose,
            log_prefix="[kb-governor]",
            source=f"Level {self.config.level} problem {self.config.problem_id}",
            on_summarizer_round=recorder.summarizer_callback(attempt),
            on_l0_cleared_without_l1=recorder.flush_without_l1_callback(attempt),
            clear_l0_after_promotion=False,
            last_promoted_round_count=self._last_promoted_round_count,
        )

    def _build_iteration_context(
        self,
        *,
        attempt: int,
        best_speedup: float,
        best_correct: bool,
        best_compiled: bool,
    ) -> str:
        return (
            f"iteration={attempt}/{self.config.max_iterations}; "
            f"best_speedup_so_far={best_speedup:.6f}; "
            f"best_correct_so_far={best_correct}; "
            f"best_compiled_so_far={best_compiled}"
        )

    def _format_l0_for_coder_prompt(self, l0: list[dict[str, str]]) -> str:
        return format_l0_for_coder_prompt(
            l0,
            max_entries=self.config.action_coder_l0_full_recent,
        )

    def _closing_instruction(self) -> str:
        return (
            f"KernelBench Level {self.config.level}, Problem {self.config.problem_id}. "
            "Return exactly one fenced Python implementation defining ModelNew."
        )

    def _previous_attempt_artifacts(
        self,
        records: list[KBIterationRecord],
    ) -> tuple[str | None, str | None]:
        if not records:
            return None, None
        latest = records[-1]
        code = (latest.candidate_code or "").strip()
        if not code or code == "# extraction_failed":
            return None, None
        terminal = latest.evaluation.terminal_output or latest.evaluation.error_message
        return code, terminal

    def _finalize_iteration_l0(
        self,
        l0: list[L0Round],
        *,
        attempt: int,
        action_norm: str,
        selector_rationale: str | None,
        coder_meta: dict[str, Any] | None,
        code: str | None,
        terminal_lines: list[str],
        metrics_iteration: dict[str, Any],
        recorder: BenchmarkRunRecorder,
    ) -> L0Round:
        reasoning_full: str | None = None
        if coder_meta:
            reasoning = coder_meta.get("assistant_reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                reasoning_full = reasoning.strip()
        rnd = finalize_l0_round(
            l0,
            attempt=attempt,
            action=action_norm,
            action_selector_rationale=selector_rationale,
            reasoning_full=reasoning_full,
            code=code,
            terminal=terminal_lines,
            metrics=metrics_iteration,
        )
        maybe_summarize_l0_round(
            rnd,
            enable=self.config.enable_l0_round_summary,
            max_tokens=self.config.l0_round_summary_max_tokens,
            timeout_sec=self.config.l0_round_summary_timeout_sec,
            code_excerpt_chars=self.config.l0_round_code_excerpt_chars,
            on_round=lambda msgs, text, meta: recorder.record_llm_turn(
                iteration=attempt,
                phase="l0_round_summarizer",
                messages=msgs,
                assistant_text=text,
                extra=meta,
            ),
        )
        return rnd

    def _build_extractor_messages(
        self,
        *,
        task_prompt: str,
        l1_entries: list[dict[str, str]],
        iteration_context: str,
        best_eval_feedback: str,
        selected_action: str | None = None,
        l1_catalog: str | None = None,
    ) -> list[dict[str, str]]:
        user_prompt = build_extractor_user_message(
            task_prompt=task_prompt,
            l1_entries=l1_entries,
            iteration_context=iteration_context,
            best_eval_feedback=best_eval_feedback,
            max_memories=self.config.extractor_max_memories,
            selected_action=selected_action,
            l1_catalog=l1_catalog,
        )
        return [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_selected_entry_ids(
        self,
        raw_text: str | None,
        *,
        valid_ids: set[str],
    ) -> list[str]:
        return parse_selected_entry_ids(
            raw_text,
            valid_ids=valid_ids,
            max_memories=self.config.extractor_max_memories,
        )

    @staticmethod
    def _extract_optional_tag(raw_text: str | None, tag_name: str) -> str | None:
        return extract_optional_tag(raw_text, tag_name)

    def _fallback_action(
        self,
        *,
        attempt: int,
        records: list[KBIterationRecord],
        is_last_correct: bool | None = None,
    ) -> str:
        if is_last_correct is None:
            is_last_correct = bool(records and records[-1].evaluation.correct)
        return get_fallback_action(attempt, records, is_last_correct=is_last_correct)

    def _fallback_reasoning(self, raw_text: str | None, action: str) -> str:
        if raw_text:
            stripped = re.sub(r"<action>.*?</action>", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
            stripped = re.sub(r"<reasoning>.*?</reasoning>", "", stripped, flags=re.IGNORECASE | re.DOTALL)
            stripped = re.sub(r"```[\s\S]*?```", "", stripped, flags=re.DOTALL)
            for line in stripped.splitlines():
                candidate = line.strip()
                if candidate:
                    return candidate[:240]
        return f"Fallback reasoning: continue with action={action} based on latest evaluation feedback."

    def _evaluate_in_subprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        return evaluate_in_subprocess(
            target_worker=_kernelbench_eval_worker,
            payload=payload,
            timeout_s=float(self.config.evaluation_timeout_s),
            start_method=str(self.config.evaluation_start_method or "spawn"),
        )

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
        terminal_output = payload.get("terminal_output")
        if not isinstance(terminal_output, str) or not terminal_output.strip():
            terminal_output = None

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

        normalized_metadata = dict(metadata)
        if terminal_output:
            normalized_metadata.setdefault("evaluation_terminal_output", terminal_output)

        return KBEvalResult(
            compiled=compiled,
            correct=correct,
            speedup=speedup,
            error_message=error_message,
            runtime=runtime_value,
            ref_runtime=ref_runtime_value,
            terminal_output=terminal_output,
            runtime_stats=dict(runtime_stats),
            metadata=normalized_metadata,
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

        self.reserver.release()
        try:
            if self.config.isolate_evaluation_process:
                eval_payload = self._evaluate_in_subprocess(payload)
            else:
                eval_payload = _run_kernelbench_eval(payload)
        finally:
            self.reserver.acquire()
            _remove_build_dir(build_dir)

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

    def _run_staged_llm_pipeline(
        self,
        *,
        attempt: int,
        task_prompt: str,
        l0: list,
        l1_path: Path,
        records: list[KBIterationRecord],
        iteration_context: str,
        best_eval_feedback: str,
        recorder: BenchmarkRunRecorder,
    ) -> tuple[
        str | None,
        dict[str, Any] | None,
        list[dict[str, str]],
        str,
        str | None,
        str | None,
    ]:
        """
        Gen3 flow: action_selector -> l1_skill_picker -> action prompt -> preflight unfold -> coder.
        Returns (raw, meta, coder_messages, action_norm, selector_rationale).
        """
        l0_summary = build_l0_global_summary(l0)
        l0_recent_selector = build_l0_recent_full(
            l0,
            self.config.action_selector_recent_l0_full,
        )

        action_norm: str | None = None
        selector_rationale: str | None = None
        if self.config.enable_action_selector:
            selector_messages = [
                {"role": "system", "content": ACTION_SELECTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_action_selector_user_message(
                        task_section=task_prompt,
                        l0_summary=l0_summary,
                        l0_recent_full_last_n=l0_recent_selector,
                    ),
                },
            ]
            selector_model = (
                resolve_nvidia_model_id(self.config.action_selector_model)
                if self.config.action_selector_model
                else get_action_selector_model_id()
            )
            selector_raw, _stok, selector_meta = call_action_selector_with_meta(
                selector_messages,
                max_tokens=self.config.action_selector_max_tokens,
                timeout_sec=self.config.action_selector_timeout_sec,
                model_id=selector_model,
            )
            recorder.record_llm_turn(
                iteration=attempt,
                phase="action_selector",
                messages=selector_messages,
                assistant_text=selector_raw,
                extra=selector_meta,
            )
            action_norm, selector_rationale = parse_action_selector_response(selector_raw)

        is_last_correct = bool(records and records[-1].evaluation.correct)
        if not action_norm or action_norm not in ALLOWED_CODER_ACTIONS:
            action_norm = self._fallback_action(
                attempt=attempt,
                records=records,
                is_last_correct=is_last_correct,
            )
        l1_picker_error: str | None = None
        l1_entries = read_recent_l1_jsonl(
            l1_path,
            max_entries=self.config.l1_catalog_max_skills,
        )
        l1_catalog = build_l1_skill_catalog(l1_entries)
        selected_l1_entries: list[dict[str, str]] = []
        if self.config.enable_l1_extractor and l1_entries:
            max_entries = max(1, int(self.config.extractor_max_memories))
            fallback_selected = l1_entries[-max_entries:]
            extractor_messages = self._build_extractor_messages(
                task_prompt=task_prompt,
                l1_entries=l1_entries,
                iteration_context=iteration_context,
                best_eval_feedback=best_eval_feedback,
                selected_action=action_norm,
                l1_catalog=l1_catalog,
            )
            try:
                extractor_model_id = (
                    resolve_nvidia_model_id(self.config.extractor_model)
                    if self.config.extractor_model
                    else None
                )
                extractor_raw, _etok, extractor_meta = call_extractor_with_meta(
                    extractor_messages,
                    max_tokens=self.config.extractor_max_tokens,
                    timeout_sec=self.config.extractor_timeout_sec,
                    model_id=extractor_model_id,
                )
                recorder.record_llm_turn(
                    iteration=attempt,
                    phase="l1_skill_picker",
                    messages=extractor_messages,
                    assistant_text=extractor_raw,
                    extra=extractor_meta,
                )
                selected_ids = self._parse_selected_entry_ids(
                    extractor_raw,
                    valid_ids={entry.get("entry_id", "") for entry in l1_entries},
                )
                if selected_ids:
                    by_id = {
                        str(entry.get("entry_id", "")): entry
                        for entry in l1_entries
                        if entry.get("entry_id")
                    }
                    selected_l1_entries = [
                        by_id[entry_id]
                        for entry_id in selected_ids
                        if entry_id in by_id
                    ][:max_entries]
            except Exception as exc:
                l1_picker_error = f"l1_skill_picker_error: {type(exc).__name__}: {exc}"
            if not selected_l1_entries:
                selected_l1_entries = fallback_selected

        l0_recent_coder = build_l0_recent_full(
            l0,
            self.config.action_coder_l0_full_recent,
        )
        archived_catalog, archived_pairs = build_l0_archived_catalog(
            l0,
            exclude_recent=self.config.action_coder_l0_full_recent,
        )
        archived_summary = archived_catalog

        prev_code, prev_terminal = self._previous_attempt_artifacts(records)
        user_prompt = build_action_coder_user_prompt(
            action=action_norm,
            task_section=task_prompt,
            iteration_context=iteration_context,
            best_eval_feedback=best_eval_feedback,
            l0_recent_full=l0_recent_coder,
            l0_archived_summary=archived_summary,
            l1_catalog=l1_catalog,
            selected_l1_skills=build_selected_l1_skills_section(selected_l1_entries),
            closing_instruction=self._closing_instruction(),
            previous_code=prev_code,
            previous_terminal=prev_terminal,
        )

        valid_round_ids = {rid for rid, _ in archived_pairs}
        unfolded_ids: list[str] = []
        if (
            self.config.enable_l0_unfold
            and valid_round_ids
            and self.config.l0_unfold_max_attempts > 0
        ):
            for unfold_attempt in range(1, self.config.l0_unfold_max_attempts + 1):
                remaining = self.config.l0_unfold_max_attempts - unfold_attempt + 1
                preflight_user = user_prompt + build_coder_preflight_user_appendix(
                    l0_archived_catalog=archived_catalog,
                    already_unfolded=unfolded_ids,
                    remaining_attempts=remaining,
                )
                preflight_messages = [
                    {"role": "system", "content": CODER_PREFLIGHT_SYSTEM_PROMPT},
                    {"role": "user", "content": preflight_user},
                ]
                preflight_raw, _ptok, preflight_meta = call_coder_with_meta(
                    preflight_messages,
                    max_tokens=min(1024, self.config.coder_max_tokens),
                    timeout_sec=self.config.coder_timeout_sec,
                )
                recorder.record_llm_turn(
                    iteration=attempt,
                    phase="coder_preflight",
                    messages=preflight_messages,
                    assistant_text=preflight_raw,
                    extra=preflight_meta,
                )
                new_ids = parse_unfold_round_ids(
                    preflight_raw,
                    valid_round_ids=valid_round_ids - set(unfolded_ids),
                    max_rounds=self.config.l0_unfold_max_rounds_per_attempt,
                )
                if not new_ids:
                    break
                for rid in new_ids:
                    if rid not in unfolded_ids:
                        unfolded_ids.append(rid)
                unfolded_block = build_l0_unfolded_full(l0, unfolded_ids)
                user_prompt = build_action_coder_user_prompt(
                    action=action_norm,
                    task_section=task_prompt,
                    iteration_context=iteration_context,
                    best_eval_feedback=best_eval_feedback,
                    l0_recent_full=l0_recent_coder,
                    l0_archived_summary=archived_summary,
                    l1_catalog=l1_catalog,
                    selected_l1_skills=build_selected_l1_skills_section(selected_l1_entries),
                    closing_instruction=self._closing_instruction(),
                    previous_code=prev_code,
                    previous_terminal=prev_terminal,
                    unfolded_l0=unfolded_block,
                )

        user_prompt = user_prompt + build_coder_final_user_appendix()
        coder_messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw, _tokens, coder_meta = call_coder_with_meta(
            coder_messages,
            max_tokens=self.config.coder_max_tokens,
            timeout_sec=self.config.coder_timeout_sec,
        )
        return raw, coder_meta, coder_messages, action_norm, selector_rationale, l1_picker_error

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
        best_eval_feedback = BEST_EVAL_FEEDBACK_EMPTY
        run_error: str | None = None
        fatal_error_count = 0

        recorder.start()
        try:
            for attempt in range(1, self.config.max_iterations + 1):
                holder.set_iteration(attempt)
                l1_text = read_l1(l1_path)
                iteration_context = self._build_iteration_context(
                    attempt=attempt,
                    best_speedup=best_speedup,
                    best_correct=best_correct,
                    best_compiled=best_compiled,
                )
                action_norm = self._fallback_action(
                    attempt=attempt,
                    records=records,
                    is_last_correct=bool(records and records[-1].evaluation.correct),
                )
                selector_rationale: str | None = None
                l1_picker_error: str | None = None
                coder_meta: dict[str, Any] | None = None
                coder_messages: list[dict[str, str]] = []

                try:
                    if self.config.enable_action_selector:
                        (
                            raw,
                            coder_meta,
                            coder_messages,
                            action_norm,
                            selector_rationale,
                            l1_picker_error,
                        ) = self._run_staged_llm_pipeline(
                            attempt=attempt,
                            task_prompt=task_prompt,
                            l0=l0,
                            l1_path=l1_path,
                            records=records,
                            iteration_context=iteration_context,
                            best_eval_feedback=best_eval_feedback,
                            recorder=recorder,
                        )
                    else:
                        l0_text = self._format_l0_for_coder_prompt(l0)
                        selected_l1_entries: list[dict[str, str]] | None = None
                        l1_entries = read_recent_l1_jsonl(
                            l1_path,
                            max_entries=self.config.l1_catalog_max_skills,
                        )
                        if self.config.enable_l1_extractor and l1_entries:
                            max_entries = max(1, int(self.config.extractor_max_memories))
                            fallback_selected = l1_entries[-max_entries:]
                            extractor_messages = self._build_extractor_messages(
                                task_prompt=task_prompt,
                                l1_entries=l1_entries,
                                iteration_context=iteration_context,
                                best_eval_feedback=best_eval_feedback,
                            )
                            try:
                                extractor_model_id = (
                                    resolve_nvidia_model_id(self.config.extractor_model)
                                    if self.config.extractor_model
                                    else None
                                )
                                extractor_raw, _extractor_tokens, extractor_meta = (
                                    call_extractor_with_meta(
                                        extractor_messages,
                                        max_tokens=self.config.extractor_max_tokens,
                                        timeout_sec=self.config.extractor_timeout_sec,
                                        model_id=extractor_model_id,
                                    )
                                )
                                recorder.record_llm_turn(
                                    iteration=attempt,
                                    phase="l1_skill_picker",
                                    messages=extractor_messages,
                                    assistant_text=extractor_raw,
                                    extra=extractor_meta,
                                )
                                selected_ids = self._parse_selected_entry_ids(
                                    extractor_raw,
                                    valid_ids={
                                        entry.get("entry_id", "")
                                        for entry in l1_entries
                                    },
                                )
                                if selected_ids:
                                    by_id = {
                                        str(entry.get("entry_id", "")): entry
                                        for entry in l1_entries
                                        if entry.get("entry_id")
                                    }
                                    selected_l1_entries = [
                                        by_id[entry_id]
                                        for entry_id in selected_ids
                                        if entry_id in by_id
                                    ][:max_entries]
                            except Exception as exc:
                                l1_picker_error = (
                                    f"l1_skill_picker_error: {type(exc).__name__}: {exc}"
                                )
                            if not selected_l1_entries:
                                selected_l1_entries = fallback_selected

                        l1_for_prompt = l1_text
                        if selected_l1_entries:
                            l1_for_prompt = (
                                "(Selected L1 skills are in the section below.)"
                            )
                        coder_prompt = self._get_coder_prompt(
                            task_prompt,
                            l1_for_prompt,
                            l0_text,
                            selected_l1_entries=selected_l1_entries,
                            allowed_actions=ALLOWED_CODER_ACTIONS,
                            iteration_context=iteration_context,
                            best_eval_feedback=best_eval_feedback,
                        )
                        coder_messages = [
                            {"role": "system", "content": LEGACY_CODER_SYSTEM_PROMPT},
                            {"role": "user", "content": coder_prompt},
                        ]
                        raw, _tokens, coder_meta = call_coder_with_meta(
                            coder_messages,
                            max_tokens=self.config.coder_max_tokens,
                            timeout_sec=self.config.coder_timeout_sec,
                        )
                        action_text = self._extract_optional_tag(raw, "action")
                        if action_text:
                            action_norm = action_text.strip().lower()
                        if action_norm not in ALLOWED_CODER_ACTIONS:
                            action_norm = self._fallback_action(
                                attempt=attempt,
                                records=records,
                                is_last_correct=bool(
                                    records and records[-1].evaluation.correct
                                ),
                            )
                except Exception as exc:
                    fatal_error_count += 1
                    err = f"coder_call_error: {type(exc).__name__}: {exc}"
                    metrics_iteration = {
                        "attempt": attempt,
                        "compiled": False,
                        "correct": False,
                        "speedup": 0.0,
                        "error": err,
                    }
                    terminal_lines = [err]
                    if l1_picker_error:
                        terminal_lines.insert(0, l1_picker_error)
                    self._finalize_iteration_l0(
                        l0,
                        attempt=attempt,
                        action_norm=action_norm,
                        selector_rationale=selector_rationale,
                        coder_meta=coder_meta,
                        code=None,
                        terminal_lines=terminal_lines,
                        metrics_iteration=metrics_iteration,
                        recorder=recorder,
                    )
                    holder.update_iteration_metrics(metrics_iteration)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="coder_call",
                        terminal_output=err,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
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
                terminal_lines: list[str] = []
                if l1_picker_error:
                    terminal_lines.append(l1_picker_error)
                if extract_err or not code:
                    extraction_error = extract_err or "no code extracted"
                    extract_terminal = f"extract_error={extraction_error}; raw={(raw or '')[:1000]}"
                    terminal_lines.append(extract_terminal)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="extract",
                        terminal_output=extract_terminal,
                        extra={
                            "compiled": False,
                            "correct": False,
                            "speedup": 0.0,
                        },
                    )
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
                    eval_result = self._evaluate_candidate(code, attempt=attempt)
                    runtime_value = (
                        float(eval_result.runtime)
                        if eval_result.runtime is not None
                        else None
                    )
                    ref_runtime_value = (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    )
                    terminal_log = (
                        f"KERNEL_BENCH_CORRECT: {eval_result.correct}\n"
                        f"KERNEL_BENCH_SPEEDUP: {float(eval_result.speedup or 0.0):.6f}\n"
                        f"KERNEL_BENCH_RUNTIME: {runtime_value if runtime_value is not None else 'n/a'}\n"
                        f"KERNEL_BENCH_REF_RUNTIME: {ref_runtime_value if ref_runtime_value is not None else 'n/a'}\n"
                    )
                    if eval_result.error_message:
                        terminal_log += f"KERNEL_BENCH_ERROR: {eval_result.error_message}\n"
                    if eval_result.terminal_output:
                        terminal_log += (
                            "KERNEL_BENCH_EVAL_TERMINAL_OUTPUT:\n"
                            f"{eval_result.terminal_output}\n"
                        )
                    terminal_lines.append(terminal_log)
                    recorder.record_evaluation_terminal_output(
                        iteration=attempt,
                        phase="evaluation",
                        terminal_output=eval_result.terminal_output or terminal_log,
                        extra={
                            "compiled": bool(eval_result.compiled),
                            "correct": bool(eval_result.correct),
                            "speedup": float(eval_result.speedup or 0.0),
                            "runtime": runtime_value,
                            "ref_runtime": ref_runtime_value,
                            "error": eval_result.error_message,
                        },
                    )

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
                        best_eval_feedback = format_best_eval_feedback(
                            attempt=attempt,
                            metrics_line=format_kernelbench_best_metrics(
                                compiled=bool(eval_result.compiled),
                                correct=bool(eval_result.correct),
                                speedup=speedup,
                                error_message=eval_result.error_message,
                            ),
                            evaluation_output=terminal_log,
                            candidate_code=code,
                        )

                metrics_iteration = {
                    "attempt": attempt,
                    "compiled": bool(eval_result.compiled),
                    "correct": bool(eval_result.correct),
                    "speedup": float(eval_result.speedup or 0.0),
                    "runtime": float(eval_result.runtime) if eval_result.runtime is not None else None,
                    "ref_runtime": (
                        float(eval_result.ref_runtime)
                        if eval_result.ref_runtime is not None
                        else None
                    ),
                    "error": eval_result.error_message,
                }
                metrics_best = {
                    "compiled": best_compiled,
                    "correct": best_correct,
                    "speedup": best_speedup,
                    "runtime": best_runtime if best_runtime >= 0 else None,
                }
                holder.update_iteration_metrics(metrics_iteration)
                holder.update_best(metrics_best)

                self._finalize_iteration_l0(
                    l0,
                    attempt=attempt,
                    action_norm=action_norm,
                    selector_rationale=selector_rationale,
                    coder_meta=coder_meta,
                    code=code if (code and not extract_err) else None,
                    terminal_lines=terminal_lines,
                    metrics_iteration=metrics_iteration,
                    recorder=recorder,
                )

                if action_norm == "propose_new":
                    self._promote_gen3_window(
                        l0,
                        l1_path=l1_path,
                        trigger="propose_new",
                        attempt=attempt,
                        recorder=recorder,
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
                        f"speedup={float(eval_result.speedup or 0.0):.4f} "
                        f"runtime={float(eval_result.runtime):.6f}"
                        if eval_result.runtime is not None
                        else f"[kb-governor] iter={attempt} "
                        f"correct={eval_result.correct} compiled={eval_result.compiled} "
                        f"speedup={float(eval_result.speedup or 0.0):.4f} runtime=n/a"
                    )

            if self.config.enable_promotion:
                self._promote_gen3_window(
                    l0,
                    l1_path=l1_path,
                    trigger="iteration_end",
                    attempt=self.config.max_iterations,
                    recorder=recorder,
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
            cleanup_problem_build_artifacts(
                self.config.results_root,
                self.config.run_name,
                level=self.config.level,
                problem_id=self.config.problem_id,
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
