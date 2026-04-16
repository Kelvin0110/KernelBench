"""Concrete evolving governor implementation for KernelBench-compatible runs."""

from __future__ import annotations

import io
import json
import multiprocessing as mp
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
from evolving_common.governor import maybe_promote_l0_to_l1, normalize_extracted_python
from evolving_common.governor.base import BaseEvolvingGovernor
from evolving_common.llm_client import (
    call_coder_with_meta,
    call_extractor_with_meta,
    resolve_nvidia_model_id,
)
from evolving_common.memory_manager import append_l0, format_l0_for_prompt, read_l1, read_l1_jsonl
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


CODER_SYSTEM_PROMPT = """You are an expert GPU kernel engineer solving kernel code optimization tasks.

Memory + planning rules:
1. L0 is recent raw attempt/evaluation history for this problem. L1 is shared cross-problem lessons.
2. You have a fixed iteration budget for this run. Choose one action each turn: propose_new, debug_current, or refine_current.
3. You must include both tags before code: <action>...</action> and <reasoning>...</reasoning>.

Output rules:
1. Return tags followed by exactly one fenced Python code block and no extra prose.
2. The code block must define ModelNew that implements the same behavior as the reference architecture.
3. Use the requested backend style in the task context.
4. Prioritize correctness first, then maximize speedup.
"""

EXTRACTOR_SYSTEM_PROMPT = """You select relevant L1 memory IDs for the next coding attempt by comparing the current condition with the summary descriptions of each L1 entry.
Return ONLY JSON: {"selected_entry_ids": ["id1", "id2", ...]}.
Do not include text outside JSON.
"""

ALLOWED_CODER_ACTIONS = ["propose_new", "debug_current", "refine_current"]


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
        self._last_promoted_count = 0

    def _get_coder_prompt(
        self,
        task_prompt: str,
        l1_text: str,
        l0_text: str,
        *,
        selected_l1_entries: list[dict[str, str]] | None = None,
        allowed_actions: list[str] | None = None,
        iteration_context: str | None = None,
        latest_eval_feedback: str | None = None,
    ) -> str:
        return build_user_prompt_with_memory(
            task_section=task_prompt,
            l1_text=l1_text,
            l0_formatted=l0_text,
            selected_l1_entries=selected_l1_entries,
            allowed_actions=allowed_actions,
            iteration_context=iteration_context,
            latest_eval_feedback=latest_eval_feedback,
            task_heading="## Official KernelBench task prompt",
            closing_instruction=(
                f"KernelBench Level {self.config.level}, Problem {self.config.problem_id}. "
                "Return exactly one fenced Python implementation."
            ),
        )

    def _latest_eval_feedback(self, records: list[KBIterationRecord]) -> str:
        if not records:
            return "No prior evaluation feedback in this run."
        latest = records[-1]
        evaluation = latest.evaluation
        return (
            f"attempt={latest.attempt}, compiled={evaluation.compiled}, "
            f"correct={evaluation.correct}, speedup={float(evaluation.speedup or 0.0):.6f}, "
            f"error={evaluation.error_message or 'none'}"
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
        if not l0:
            return "(no L0 entries yet)"

        groups: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def _ensure_group() -> dict[str, Any]:
            nonlocal current
            if current is None:
                current = {
                    "action": "(unknown)",
                    "reasoning": "(none)",
                    "code": None,
                    "terminal": [],
                }
            return current

        for entry in l0:
            role = entry.get("role", "")
            content = (entry.get("content") or "").strip()
            if not content:
                continue

            if role == "system" and content.startswith("coder_action="):
                if current is not None and current.get("code"):
                    groups.append(current)
                    current = None
                grp = _ensure_group()
                grp["action"] = content.split("=", 1)[1].strip() or "(unknown)"
                continue

            if role == "system" and content.startswith("coder_reasoning="):
                grp = _ensure_group()
                grp["reasoning"] = content.split("=", 1)[1].strip() or "(none)"
                continue

            if role == "code":
                grp = _ensure_group()
                grp["code"] = content
                continue

            if role == "terminal":
                grp = _ensure_group()
                grp["terminal"].append(content)
                continue

            grp = _ensure_group()
            grp["terminal"].append(f"{role}: {content}")

        if current is not None:
            groups.append(current)

        if not groups:
            return format_l0_for_prompt(l0)

        rendered: list[str] = []
        for idx, grp in enumerate(groups[-8:], start=1):
            code_text = (grp.get("code") or "(no code captured)").strip()
            if len(code_text) > 800:
                code_text = f"{code_text[:800]}\n... [truncated]"

            terminal_items = grp.get("terminal") or []
            terminal_text = "\n\n".join(terminal_items[-2:]).strip() or "(no terminal output captured)"
            if len(terminal_text) > 1200:
                terminal_text = f"{terminal_text[:1200]}\n... [truncated]"

            rendered.append(
                "\n".join(
                    [
                        f"### Attempt history #{idx}",
                        f"Action: {grp.get('action')}",
                        f"Reasoning: {grp.get('reasoning')}",
                        "Code:",
                        code_text,
                        "",
                        "Terminal:",
                        terminal_text,
                    ]
                )
            )

        return "\n\n".join(rendered)

    def _build_extractor_messages(
        self,
        *,
        task_prompt: str,
        l1_entries: list[dict[str, str]],
        iteration_context: str,
        latest_eval_feedback: str,
    ) -> list[dict[str, str]]:
        candidate_lines: list[str] = []
        for entry in l1_entries:
            entry_id = (entry.get("entry_id") or "").strip()
            description = (entry.get("description") or "").strip()
            content = (entry.get("content") or "").strip()
            content_preview = content[:280]
            candidate_lines.append(
                f"id={entry_id}\ndescription={description}\ncontent={content_preview}"
            )
        candidates = "\n\n".join(candidate_lines)
        user_prompt = (
            f"Select up to {self.config.extractor_max_memories} entry IDs relevant for the next iteration.\n\n"
            f"Task:\n{task_prompt.strip()}\n\n"
            f"{iteration_context}\n"
            f"Latest evaluation feedback:\n{latest_eval_feedback}\n\n"
            f"Candidates:\n{candidates}\n"
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
        if not raw_text or not valid_ids:
            return []

        selected: list[str] = []
        text = raw_text.strip()
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

        candidate_ids: list[str] = []
        if isinstance(parsed, dict):
            for key in ("selected_entry_ids", "entry_ids", "selected_ids", "ids"):
                maybe_list = parsed.get(key)
                if isinstance(maybe_list, list):
                    candidate_ids = [str(item).strip() for item in maybe_list]
                    break
        elif isinstance(parsed, list):
            candidate_ids = [str(item).strip() for item in parsed]

        if not candidate_ids:
            for entry_id in valid_ids:
                if entry_id and entry_id in text:
                    candidate_ids.append(entry_id)

        seen: set[str] = set()
        for entry_id in candidate_ids:
            if entry_id in valid_ids and entry_id not in seen:
                selected.append(entry_id)
                seen.add(entry_id)
            if len(selected) >= self.config.extractor_max_memories:
                break
        return selected

    @staticmethod
    def _extract_optional_tag(raw_text: str | None, tag_name: str) -> str | None:
        if not raw_text:
            return None
        match = re.search(
            rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return None
        value = match.group(1).strip()
        return value or None

    def _fallback_action(self, *, attempt: int, records: list[KBIterationRecord]) -> str:
        if attempt <= 1:
            return "propose_new"
        if records and records[-1].evaluation.correct:
            return "refine_current"
        return "debug_current"

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
                l0_text = self._format_l0_for_coder_prompt(l0)
                iteration_context = self._build_iteration_context(
                    attempt=attempt,
                    best_speedup=best_speedup,
                    best_correct=best_correct,
                    best_compiled=best_compiled,
                )
                latest_eval_feedback = self._latest_eval_feedback(records)
                selected_l1_entries: list[dict[str, str]] | None = None

                l1_entries = read_l1_jsonl(l1_path)
                if self.config.enable_l1_extractor and l1_entries:
                    max_entries = max(1, int(self.config.extractor_max_memories))
                    fallback_selected = l1_entries[-max_entries:]
                    extractor_messages = self._build_extractor_messages(
                        task_prompt=task_prompt,
                        l1_entries=l1_entries,
                        iteration_context=iteration_context,
                        latest_eval_feedback=latest_eval_feedback,
                    )
                    try:
                        extractor_model_id = (
                            resolve_nvidia_model_id(self.config.extractor_model)
                            if self.config.extractor_model
                            else None
                        )
                        extractor_raw, _extractor_tokens, extractor_meta = call_extractor_with_meta(
                            extractor_messages,
                            max_tokens=self.config.extractor_max_tokens,
                            timeout_sec=self.config.extractor_timeout_sec,
                            model_id=extractor_model_id,
                        )
                        recorder.record_llm_turn(
                            iteration=attempt,
                            phase="extractor",
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
                        append_l0(
                            l0,
                            "terminal",
                            f"extractor_selection_error: {type(exc).__name__}: {exc}",
                        )

                    if not selected_l1_entries:
                        selected_l1_entries = fallback_selected

                l1_for_prompt = l1_text
                if selected_l1_entries:
                    l1_for_prompt = "(Extractor-selected L1 entries are provided in the section below.)"

                coder_prompt = self._get_coder_prompt(
                    task_prompt,
                    l1_for_prompt,
                    l0_text,
                    selected_l1_entries=selected_l1_entries,
                    allowed_actions=ALLOWED_CODER_ACTIONS,
                    iteration_context=iteration_context,
                    latest_eval_feedback=latest_eval_feedback,
                )
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

                action_text = self._extract_optional_tag(raw, "action")
                if not action_text:
                    action_text = self._fallback_action(attempt=attempt, records=records)

                action_norm = action_text.strip().lower()
                if action_norm not in ALLOWED_CODER_ACTIONS:
                    action_norm = self._fallback_action(attempt=attempt, records=records)
                append_l0(l0, "system", f"coder_action={action_norm}")

                reasoning_text = self._extract_optional_tag(raw, "reasoning")
                if not reasoning_text:
                    reasoning_text = self._fallback_reasoning(raw, action_norm)
                append_l0(l0, "system", f"coder_reasoning={reasoning_text}")

                code, extract_err = normalize_extracted_python(raw)
                if extract_err or not code:
                    extraction_error = extract_err or "no code extracted"
                    extract_terminal = f"extract_error={extraction_error}; raw={(raw or '')[:1000]}"
                    append_l0(l0, "terminal", extract_terminal)
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
                    append_l0(l0, "code", code)
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
                    append_l0(l0, "terminal", terminal_log)
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

                self._last_promoted_count = maybe_promote_l0_to_l1(
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
                    clear_l0_after_promotion=False,
                    last_promoted_count=self._last_promoted_count,
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
