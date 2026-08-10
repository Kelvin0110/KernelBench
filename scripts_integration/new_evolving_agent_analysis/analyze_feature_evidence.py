"""Extract compact, read-only feature evidence from completed evolving-agent runs.

The extractor reads only the explicitly selected run directories and the recorded
artifacts named below.  It never imports the agent, regenerates performance data,
or writes into a run directory.  Its only writes are deterministic
``feature_evidence.json`` and ``feature_evidence.csv`` files under ``--output-dir``.

Metric semantics
----------------
* A chat turn is one valid JSON object in ``chat_history.jsonl``.  Token totals
  sum endpoint-reported values; per-call ``total_tokens`` falls back to
  ``prompt_tokens + completion_tokens`` only when both are present.  Missing
  usage remains visible through reported/missing call counts.
* Phase call counts count every valid chat row.  Action mix counts parsable
  ``action_selector`` calls (not unique iterations) and reports parse failures.
* Metrics counts use only each row's ``metrics_iteration`` object.  A field rate
  is ``true_count / observed_count`` for that field, so missing booleans are not
  silently treated as false.  ``metrics_best`` and ``run_finished.json`` are
  used only for compact per-problem final/best result fields.
* A valid comparison speedup is finite and positive, with ``best_correct=true``
  and ``best_is_hack=false``.  Speedup deltas require valid values on both sides.
* L0 summaries use recorded ``l0_entry_count`` values (or legacy
  ``len(l0_entries)``).  L1/refinement counts describe rows in ``shared_l1.jsonl``;
  governance event counts describe their corresponding sidecars.
* Error taxonomy is heuristic and descriptive.  Error signatures are sanitized,
  first-line excerpts rather than full terminal output or code.
* Case-study candidates are deterministic descriptive selections.  They are not
  causal evidence: ties are broken by workspace name, and selection rules are
  embedded in the JSON output.

Malformed JSON/JSONL rows are counted and warned about without aborting other
work.  The CLI rejects missing or incomplete selected runs before extraction so
partial runs cannot silently enter a comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = "feature_evidence.json"
OUTPUT_CSV = "feature_evidence.csv"

WORKSPACE_ARTIFACTS = (
    "chat_history.jsonl",
    "metrics_by_iteration.jsonl",
    "iteration_snapshots.jsonl",
    "run_finished.json",
)
GOVERNANCE_SIDECARS = (
    "l1_skill_usage.json",
    "l1_skill_deletions.jsonl",
    "l1_skill_merges.jsonl",
    "l1_skill_merge_clustering.jsonl",
    "l1_skill_merge_state.json",
    "l1_skill_catalog_stats.json",
    "l1_skill_unit_test_runs.jsonl",
    "skill_merges.txt",
    "skill_revisions.txt",
)
TRACKED_ACTIONS = ("propose_new", "refine_current", "debug_current")
ACTION_ALIASES = {
    "propose": "propose_new",
    "propose_new": "propose_new",
    "new": "propose_new",
    "refine": "refine_current",
    "refine_current": "refine_current",
    "debug": "debug_current",
    "debug_current": "debug_current",
    "fix": "debug_current",
}
ACTION_KEYS = ("action", "selected_action", "parsed_action", "choice", "decision")

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_FIELD_RE = re.compile(
    r"""(?:["']?(?:action|selected_action|selected action|choice|decision)["']?)"""
    r"""\s*(?::|=|is)\s*["'`*]*"""
    r"""(propose[_ -]?new|refine[_ -]?current|debug[_ -]?current|[A-Za-z][A-Za-z0-9_-]{0,63})""",
    re.IGNORECASE,
)
_ACTION_WORD_RE = re.compile(
    r"\b(propose[_ -]?new|refine[_ -]?current|debug[_ -]?current)\b",
    re.IGNORECASE,
)
_WORKSPACE_RE = re.compile(r"^level_(\d+)_problem_(.+)$")
_ABS_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s/:]+/)+[^\s:]+")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LONG_WS_RE = re.compile(r"\s+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _round(value: float | None, digits: int = 9) -> float | None:
    return None if value is None else round(float(value), digits)


def _rate(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator, 6) if denominator else None


def _clean_label(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:80] or fallback


def _bounded_excerpt(value: Any, limit: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    text = _LONG_WS_RE.sub(" ", value).strip()
    text = _ABS_PATH_RE.sub("<path>", text)
    text = _HEX_RE.sub("<hex>", text)
    if not text:
        return None
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _error_signature(value: Any) -> str | None:
    excerpt = _bounded_excerpt(value, 180)
    if excerpt is None:
        return None
    first = excerpt.split("\\n", 1)[0]
    return first[:180]


def classify_error(value: Any) -> str:
    text = str(value or "").lower()
    if not text.strip():
        return "unknown"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "out of memory" in text or re.search(r"\boom\b", text):
        return "out_of_memory"
    if "output mismatch" in text or "incorrect result" in text:
        return "output_mismatch"
    if "undefined symbol" in text or "cannot open shared object" in text or "importerror" in text:
        return "load_or_link_error"
    if any(
        token in text
        for token in (
            "failed to compile",
            "compilation failure",
            "compile error",
            "nvcc",
            "ninja:",
            "c++ compilation",
        )
    ):
        return "compilation_error"
    if any(
        token in text
        for token in (
            "cuda error",
            "cudaerror",
            "illegal memory",
            "illegal address",
            "device-side assert",
            "launch failed",
        )
    ):
        return "cuda_runtime_error"
    if any(
        token in text
        for token in (
            "shape",
            "dimension",
            "size mismatch",
            "dtype",
            "expected scalar type",
        )
    ):
        return "shape_or_dtype_error"
    if "runtimeerror" in text or "fake tensor" in text or "exception" in text:
        return "runtime_error"
    return "other"


def _summary_stats(values: Iterable[int | float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "sum": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    total = sum(clean)
    return {
        "count": len(clean),
        "sum": _round(total, 6),
        "min": _round(min(clean), 6),
        "max": _round(max(clean), 6),
        "mean": _round(total / len(clean), 6),
        "median": _round(float(statistics.median(clean)), 6),
    }


@dataclass
class WarningCollector:
    base: Path
    _counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)

    def add(self, path: Path, code: str, message: str, count: int = 1) -> None:
        try:
            display = path.relative_to(self.base).as_posix()
        except ValueError:
            display = str(path)
        self._counts[(display, code, message)] += max(1, int(count))

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"path": path, "code": code, "message": message, "count": count}
            for (path, code, message), count in sorted(self._counts.items())
        ]


@dataclass
class ArtifactInventory:
    base: Path
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, path: Path) -> None:
        try:
            display = path.relative_to(self.base).as_posix()
        except ValueError:
            display = str(path)
        if display in self._entries:
            return
        entry: dict[str, Any] = {"path": display, "exists": path.is_file()}
        if entry["exists"]:
            try:
                stat = path.stat()
                entry["size_bytes"] = stat.st_size
                entry["mtime_ns"] = stat.st_mtime_ns
            except OSError:
                entry["size_bytes"] = None
                entry["mtime_ns"] = None
        else:
            entry["size_bytes"] = None
            entry["mtime_ns"] = None
        self._entries[display] = entry

    def rows(self) -> list[dict[str, Any]]:
        return [self._entries[key] for key in sorted(self._entries)]


def _read_json(
    path: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
    optional: bool = True,
) -> Any:
    inventory.add(path)
    key = path.relative_to(inventory.base).as_posix()
    diag = {"kind": "json", "valid": 0, "malformed": 0, "missing": 0}
    diagnostics[key] = diag
    if not path.is_file():
        diag["missing"] = 1
        warnings.add(path, "missing_optional_artifact" if optional else "missing_artifact", "file is missing")
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        diag["malformed"] = 1
        warnings.add(path, "malformed_json", f"{type(exc).__name__}: {_bounded_excerpt(str(exc), 160)}")
        return None
    diag["valid"] = 1
    return payload


def _iter_jsonl(
    path: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
    optional: bool = True,
) -> Iterator[dict[str, Any]]:
    inventory.add(path)
    key = path.relative_to(inventory.base).as_posix()
    diag = {
        "kind": "jsonl",
        "nonempty_rows": 0,
        "valid_object_rows": 0,
        "malformed_rows": 0,
        "non_object_rows": 0,
        "missing": 0,
    }
    diagnostics[key] = diag
    if not path.is_file():
        diag["missing"] = 1
        warnings.add(path, "missing_optional_artifact" if optional else "missing_artifact", "file is missing")
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                diag["nonempty_rows"] += 1
                try:
                    payload = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeError):
                    diag["malformed_rows"] += 1
                    continue
                if not isinstance(payload, dict):
                    diag["non_object_rows"] += 1
                    continue
                diag["valid_object_rows"] += 1
                yield payload
    except OSError as exc:
        warnings.add(path, "read_error", f"{type(exc).__name__}: {_bounded_excerpt(str(exc), 160)}")
    if diag["malformed_rows"]:
        warnings.add(
            path,
            "malformed_jsonl_rows",
            "malformed JSONL rows were skipped",
            diag["malformed_rows"],
        )
    if diag["non_object_rows"]:
        warnings.add(
            path,
            "non_object_jsonl_rows",
            "non-object JSONL rows were skipped",
            diag["non_object_rows"],
        )


def _read_text_shape(
    path: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inventory.add(path)
    key = path.relative_to(inventory.base).as_posix()
    result = {"present": path.is_file(), "nonempty_lines": 0, "characters": 0}
    diagnostics[key] = {"kind": "text", "missing": 0 if path.is_file() else 1}
    if not path.is_file():
        warnings.add(path, "missing_optional_artifact", "file is missing")
        return result
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                result["characters"] += len(line)
                if line.strip():
                    result["nonempty_lines"] += 1
    except OSError as exc:
        warnings.add(path, "read_error", f"{type(exc).__name__}: {_bounded_excerpt(str(exc), 160)}")
    return result


def _normalize_action(value: Any) -> str | None:
    if value is None:
        return None
    label = _clean_label(value, fallback="")
    if not label:
        return None
    label = ACTION_ALIASES.get(label, label)
    return label if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", label) else None


def _action_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ACTION_KEYS:
        action = _normalize_action(payload.get(key))
        if action:
            return action
    for key in ("result", "response", "output", "extra"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            action = _action_from_payload(nested)
            if action:
                return action
    return None


def _selector_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)

    add(row.get("assistant_text"))
    add(row.get("assistant_content"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
                add(message.get("content"))
                break
    return texts


def parse_action_selector(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return ``(normalized action, parse method/error)`` for common recorder forms."""
    direct = _action_from_payload(row)
    if direct:
        return direct, "structured_field"
    for text in _selector_texts(row):
        stripped = _JSON_FENCE_RE.sub("", text).strip()
        for candidate in (text.strip(), stripped):
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            action = _action_from_payload(payload)
            if action:
                return action, "json_text"
        match = _ACTION_FIELD_RE.search(text)
        if match:
            action = _normalize_action(match.group(1).splitlines()[0])
            if action:
                return action, "keyed_text"
        words = {_normalize_action(match.group(1)) for match in _ACTION_WORD_RE.finditer(text)}
        words.discard(None)
        if len(words) == 1:
            return next(iter(words)), "unique_keyword"
        direct_text = _normalize_action(stripped.strip("`*_ .'\""))
        if direct_text in TRACKED_ACTIONS:
            return direct_text, "bare_text"
    return None, "action_not_found"


def _empty_token_bucket() -> dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_reported_calls": 0,
        "prompt_missing_calls": 0,
        "completion_reported_calls": 0,
        "completion_missing_calls": 0,
        "total_reported_calls": 0,
        "total_derived_calls": 0,
        "total_missing_calls": 0,
    }


def _add_token_row(bucket: dict[str, int], row: dict[str, Any]) -> None:
    bucket["calls"] += 1
    prompt = _as_int(row.get("prompt_tokens"))
    completion = _as_int(row.get("completion_tokens"))
    total = _as_int(row.get("total_tokens"))
    if prompt is not None:
        bucket["prompt_tokens"] += prompt
        bucket["prompt_reported_calls"] += 1
    else:
        bucket["prompt_missing_calls"] += 1
    if completion is not None:
        bucket["completion_tokens"] += completion
        bucket["completion_reported_calls"] += 1
    else:
        bucket["completion_missing_calls"] += 1
    if total is not None:
        bucket["total_tokens"] += total
        bucket["total_reported_calls"] += 1
    elif prompt is not None and completion is not None:
        bucket["total_tokens"] += prompt + completion
        bucket["total_derived_calls"] += 1
    else:
        bucket["total_missing_calls"] += 1


def _merge_token_bucket(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += int(source.get(key, 0))


def _content_characters(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_characters(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return _content_characters(value.get("content"))
        return sum(_content_characters(item) for item in value.values())
    return 0


def analyze_chat(
    path: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    overall = _empty_token_bucket()
    phases: dict[str, dict[str, int]] = {}
    action_counts: Counter[str] = Counter()
    action_methods: Counter[str] = Counter()
    action_errors: Counter[str] = Counter()
    report_input_chars = 0
    report_output_chars = 0

    for row in _iter_jsonl(
        path,
        warnings=warnings,
        inventory=inventory,
        diagnostics=diagnostics,
        optional=True,
    ):
        phase = _clean_label(row.get("phase"), fallback="unknown")
        phase_bucket = phases.setdefault(phase, _empty_token_bucket())
        _add_token_row(overall, row)
        _add_token_row(phase_bucket, row)
        if phase == "action_selector":
            action, method = parse_action_selector(row)
            if action is None:
                action_errors[method] += 1
            else:
                action_counts[action] += 1
                action_methods[method] += 1
        if phase == "evolving_report":
            report_input_chars += _content_characters(row.get("messages"))
            visible = row.get("assistant_text")
            if not isinstance(visible, str):
                visible = row.get("assistant_content")
            report_output_chars += len(visible) if isinstance(visible, str) else 0

    ordered_actions = {action: action_counts.get(action, 0) for action in TRACKED_ACTIONS}
    for action in sorted(set(action_counts) - set(TRACKED_ACTIONS)):
        ordered_actions[action] = action_counts[action]
    return {
        "turn_count": overall["calls"],
        "tokens": overall,
        "by_phase": {phase: phases[phase] for phase in sorted(phases)},
        "phase_call_counts": {phase: phases[phase]["calls"] for phase in sorted(phases)},
        "action_selector": {
            "call_count": phases.get("action_selector", {}).get("calls", 0),
            "parsed_count": sum(action_counts.values()),
            "parse_error_count": sum(action_errors.values()),
            "actions": ordered_actions,
            "parse_methods": dict(sorted(action_methods.items())),
            "parse_errors": dict(sorted(action_errors.items())),
        },
        "evolving_report_characters": {
            "input": report_input_chars,
            "output": report_output_chars,
            "total": report_input_chars + report_output_chars,
        },
    }


def _merge_chat(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["turn_count"] += source["turn_count"]
    _merge_token_bucket(target["tokens"], source["tokens"])
    for phase, bucket in source["by_phase"].items():
        _merge_token_bucket(target["by_phase"].setdefault(phase, _empty_token_bucket()), bucket)
    selector = target["action_selector"]
    other = source["action_selector"]
    selector["call_count"] += other["call_count"]
    selector["parsed_count"] += other["parsed_count"]
    selector["parse_error_count"] += other["parse_error_count"]
    for group in ("actions", "parse_methods", "parse_errors"):
        for key, value in other[group].items():
            selector[group][key] = selector[group].get(key, 0) + value
    for key in ("input", "output", "total"):
        target["evolving_report_characters"][key] += source["evolving_report_characters"][key]


def _empty_chat_summary() -> dict[str, Any]:
    return {
        "turn_count": 0,
        "tokens": _empty_token_bucket(),
        "by_phase": {},
        "action_selector": {
            "call_count": 0,
            "parsed_count": 0,
            "parse_error_count": 0,
            "actions": {action: 0 for action in TRACKED_ACTIONS},
            "parse_methods": {},
            "parse_errors": {},
        },
        "evolving_report_characters": {"input": 0, "output": 0, "total": 0},
    }


def _metric_bool(metrics: dict[str, Any], key: str) -> bool | None:
    return _as_bool(metrics.get(key))


def _result_locator(workspace: str, iteration: int | None) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "iteration": iteration,
        "phase": "metrics_iteration",
        "artifact": f"workspaces/{workspace}/metrics_by_iteration.jsonl",
    }


def analyze_metrics(
    path: Path,
    *,
    workspace: str,
    marker: dict[str, Any] | None,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_count = 0
    valid_metrics_rows = 0
    iterations: set[int] = set()
    true_counts = {key: 0 for key in ("compiled", "correct", "is_hack")}
    observed_counts = {key: 0 for key in ("compiled", "correct", "is_hack")}
    latest_key: tuple[int, int] | None = None
    latest_iteration: int | None = None
    latest_metrics: dict[str, Any] = {}
    latest_best: dict[str, Any] = {}
    best_candidate: dict[str, Any] | None = None
    report_chars: list[int] = []
    error_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(
        _iter_jsonl(
            path,
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        ),
        start=1,
    ):
        row_count += 1
        metrics = row.get("metrics_iteration")
        if not isinstance(metrics, dict):
            warnings.add(path, "missing_metrics_iteration", "row has no metrics_iteration object")
            continue
        valid_metrics_rows += 1
        iteration = _as_int(row.get("iteration"))
        if iteration is not None:
            iterations.add(iteration)
        for key in true_counts:
            parsed = _metric_bool(metrics, key)
            if parsed is not None:
                observed_counts[key] += 1
                if parsed:
                    true_counts[key] += 1
        chars = _as_int(metrics.get("evolving_report_chars"))
        if chars is not None and chars >= 0:
            report_chars.append(chars)
        error = metrics.get("error")
        signature = _error_signature(error)
        if signature:
            error_rows.append(
                {
                    "category": classify_error(error),
                    "signature": signature,
                    "workspace": workspace,
                    "iteration": iteration,
                }
            )
        order_key = (iteration if iteration is not None else -1, row_index)
        if latest_key is None or order_key > latest_key:
            latest_key = order_key
            latest_iteration = iteration
            latest_metrics = metrics
            latest_best = row.get("metrics_best") if isinstance(row.get("metrics_best"), dict) else {}
        speedup = _as_float(metrics.get("speedup"))
        correct = _metric_bool(metrics, "correct")
        is_hack = _metric_bool(metrics, "is_hack")
        if correct is True and is_hack is not True and speedup is not None and speedup > 0:
            candidate = {
                "iteration": iteration,
                "speedup": speedup,
                "runtime": _as_float(metrics.get("runtime")),
                "ref_runtime": _as_float(metrics.get("ref_runtime")),
            }
            if best_candidate is None:
                best_candidate = candidate
            else:
                candidate_key = (candidate["speedup"], -(candidate["iteration"] or 10**12))
                best_key = (best_candidate["speedup"], -(best_candidate["iteration"] or 10**12))
                if candidate_key > best_key:
                    best_candidate = candidate

    marker_metadata = marker.get("metadata") if isinstance(marker, dict) else None
    if not isinstance(marker_metadata, dict):
        marker_metadata = {}

    def marker_or_best(marker_key: str, best_key: str, parser: Any) -> Any:
        marker_value = parser(marker_metadata.get(marker_key))
        return marker_value if marker_value is not None else parser(latest_best.get(best_key))

    best_correct = marker_or_best("best_correct", "correct", _as_bool)
    best_compiled = marker_or_best("best_compiled", "compiled", _as_bool)
    best_is_hack = marker_or_best("best_is_hack", "is_hack", _as_bool)
    best_speedup = marker_or_best("best_speedup", "speedup", _as_float)
    best_runtime = marker_or_best("best_runtime", "runtime", _as_float)
    if best_speedup is None and best_candidate:
        best_speedup = best_candidate["speedup"]
    if best_runtime is None and best_candidate:
        best_runtime = best_candidate["runtime"]

    final_error = latest_metrics.get("error")
    final_result = {
        "iteration": latest_iteration,
        "compiled": _metric_bool(latest_metrics, "compiled"),
        "correct": _metric_bool(latest_metrics, "correct"),
        "is_hack": _metric_bool(latest_metrics, "is_hack"),
        "speedup": _round(_as_float(latest_metrics.get("speedup"))),
        "runtime": _round(_as_float(latest_metrics.get("runtime"))),
        "ref_runtime": _round(_as_float(latest_metrics.get("ref_runtime"))),
        "error_category": classify_error(final_error) if _error_signature(final_error) else None,
        "error_excerpt": _bounded_excerpt(final_error),
        "locator": _result_locator(workspace, latest_iteration),
    }
    best_iteration = best_candidate.get("iteration") if best_candidate else None
    best_result = {
        "iteration": best_iteration,
        "compiled": best_compiled,
        "correct": best_correct,
        "is_hack": best_is_hack,
        "speedup": _round(best_speedup),
        "runtime": _round(best_runtime),
        "ref_runtime": _round(best_candidate.get("ref_runtime")) if best_candidate else None,
        "locator": _result_locator(workspace, best_iteration),
    }
    summary = {
        "row_count": row_count,
        "metrics_iteration_row_count": valid_metrics_rows,
        "unique_iteration_count": len(iterations),
        "compiled_count": true_counts["compiled"],
        "compiled_observed": observed_counts["compiled"],
        "compiled_rate": _rate(true_counts["compiled"], observed_counts["compiled"]),
        "correct_count": true_counts["correct"],
        "correct_observed": observed_counts["correct"],
        "correct_rate": _rate(true_counts["correct"], observed_counts["correct"]),
        "is_hack_count": true_counts["is_hack"],
        "is_hack_observed": observed_counts["is_hack"],
        "is_hack_rate": _rate(true_counts["is_hack"], observed_counts["is_hack"]),
        "evolving_report_chars": report_chars,
        "final": final_result,
        "best": best_result,
    }
    return summary, error_rows


def analyze_snapshots(
    path: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = 0
    observations: list[int] = []
    latest_key: tuple[int, int] | None = None
    first_count: int | None = None
    final_count: int | None = None
    for row_index, row in enumerate(
        _iter_jsonl(
            path,
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        ),
        start=1,
    ):
        rows += 1
        count = _as_int(row.get("l0_entry_count"))
        if count is None and isinstance(row.get("l0_entries"), list):
            count = len(row["l0_entries"])
        if count is None or count < 0:
            continue
        observations.append(count)
        if first_count is None:
            first_count = count
        iteration = _as_int(row.get("iteration"))
        key = (iteration if iteration is not None else -1, row_index)
        if latest_key is None or key > latest_key:
            latest_key = key
            final_count = count
    return {
        "snapshot_row_count": rows,
        "entry_count_observations": observations,
        "first_entry_count": first_count,
        "final_entry_count": final_count,
        "growth": final_count - first_count
        if final_count is not None and first_count is not None
        else None,
    }


def _parse_workspace(workspace: str) -> tuple[int | None, str | None]:
    match = _WORKSPACE_RE.match(workspace)
    if match is None:
        return None, None
    return _as_int(match.group(1)), match.group(2)


def _merge_metric_totals(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in (
        "row_count",
        "metrics_iteration_row_count",
        "unique_iteration_count",
        "compiled_count",
        "compiled_observed",
        "correct_count",
        "correct_observed",
        "is_hack_count",
        "is_hack_observed",
    ):
        target[key] += int(source.get(key, 0))


def _finalize_metric_totals(totals: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(totals)
    for key in ("compiled", "correct", "is_hack"):
        result[f"{key}_rate"] = _rate(result[f"{key}_count"], result[f"{key}_observed"])
    return result


def _top_error_summary(error_rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(row["category"] for row in error_rows)
    signatures = Counter((row["category"], row["signature"]) for row in error_rows)
    first_locator: dict[tuple[str, str], dict[str, Any]] = {}
    for row in error_rows:
        key = (row["category"], row["signature"])
        first_locator.setdefault(
            key,
            {
                "workspace": row["workspace"],
                "iteration": row["iteration"],
                "phase": "metrics_iteration",
            },
        )
    top = []
    for (category, signature), count in sorted(
        signatures.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:10]:
        top.append(
            {
                "category": category,
                "signature": signature,
                "count": count,
                "example_locator": first_locator[(category, signature)],
            }
        )
    return {
        "error_row_count": len(error_rows),
        "categories": dict(sorted(categories.items())),
        "top_errors": top,
    }


def _entry_is_refinement(entry: dict[str, Any]) -> bool:
    return bool(
        entry.get("parent_id")
        or entry.get("refinement_round") is not None
        or str(entry.get("source") or "") == "skill_refinement"
        or entry.get("refinement_meta")
        or entry.get("revision_trace")
    )


def _compact_json_shape(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        scalars = {
            str(key): (_bounded_excerpt(value) if isinstance(value, str) else value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
            if value is None or isinstance(value, (bool, int, float, str))
        }
        list_lengths = {
            str(key): len(value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
            if isinstance(value, list)
        }
        return {
            "type": "object",
            "key_count": len(payload),
            "keys": sorted(str(key) for key in payload),
            "scalars": scalars,
            "list_lengths": list_lengths,
        }
    if isinstance(payload, list):
        return {"type": "array", "length": len(payload)}
    return {"type": type(payload).__name__, "value": payload}


def _event_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(_clean_label(row.get("status")) for row in rows)
    reasons = Counter(_clean_label(row.get("reason")) for row in rows)
    accepted = statuses.get("accepted", 0)
    return {
        "event_count": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "reasons": dict(sorted(reasons.items())),
        "accepted_count": accepted,
        "acceptance_rate": _rate(accepted, len(rows)),
    }


def analyze_governance(
    run_dir: Path,
    *,
    warnings: WarningCollector,
    inventory: ArtifactInventory,
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    l1_path = run_dir / "shared_l1.jsonl"
    statuses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    refinement_rounds: Counter[str] = Counter()
    refinement_count = 0
    refinement_active_count = 0
    parent_ids: set[str] = set()
    l1_rows = 0
    for entry in _iter_jsonl(
        l1_path,
        warnings=warnings,
        inventory=inventory,
        diagnostics=diagnostics,
        optional=True,
    ):
        l1_rows += 1
        status = _clean_label(entry.get("status") or "active")
        statuses[status] += 1
        sources[_clean_label(entry.get("source"))] += 1
        if _entry_is_refinement(entry):
            refinement_count += 1
            if status == "active":
                refinement_active_count += 1
            parent = entry.get("parent_id")
            if parent is not None and str(parent).strip():
                parent_ids.add(str(parent))
            round_value = entry.get("refinement_round")
            refinement_rounds[str(round_value) if round_value is not None else "unspecified"] += 1

    deletion_rows = list(
        _iter_jsonl(
            run_dir / "l1_skill_deletions.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        )
    )
    deletion_reasons = Counter(_clean_label(row.get("reason")) for row in deletion_rows)

    merge_rows = list(
        _iter_jsonl(
            run_dir / "l1_skill_merges.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        )
    )
    merge_summary = _event_summary(merge_rows)
    absorbed: set[str] = set()
    for row in merge_rows:
        if _clean_label(row.get("status")) != "accepted":
            continue
        source_ids = row.get("source_entry_ids")
        if isinstance(source_ids, list):
            absorbed.update(str(value) for value in source_ids if str(value).strip())
    merge_summary["unique_source_entries_absorbed"] = len(absorbed)

    clustering_rows = list(
        _iter_jsonl(
            run_dir / "l1_skill_merge_clustering.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        )
    )
    unit_test_rows = list(
        _iter_jsonl(
            run_dir / "l1_skill_unit_test_runs.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        )
    )

    json_sidecars: dict[str, Any] = {}
    for name in ("l1_skill_usage.json", "l1_skill_merge_state.json", "l1_skill_catalog_stats.json"):
        payload = _read_json(
            run_dir / name,
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=True,
        )
        json_sidecars[name] = _compact_json_shape(payload) if payload is not None else None

    text_sidecars = {
        name: _read_text_shape(
            run_dir / name,
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
        )
        for name in ("skill_merges.txt", "skill_revisions.txt")
    }
    present = [name for name in GOVERNANCE_SIDECARS if (run_dir / name).is_file()]
    missing = [name for name in GOVERNANCE_SIDECARS if name not in present]
    return {
        "shared_l1": {
            "entry_count": l1_rows,
            "statuses": dict(sorted(statuses.items())),
            "sources": dict(sorted(sources.items())),
            "active_count": statuses.get("active", 0),
            "superseded_count": statuses.get("superseded", 0),
            "deleted_count": statuses.get("deleted", 0),
        },
        "refinement": {
            "entry_count": refinement_count,
            "active_entry_count": refinement_active_count,
            "unique_parent_count": len(parent_ids),
            "rounds": dict(sorted(refinement_rounds.items())),
            "revision_log": text_sidecars["skill_revisions.txt"],
        },
        "deletion": {
            "event_count": len(deletion_rows),
            "reasons": dict(sorted(deletion_reasons.items())),
        },
        "merge": merge_summary,
        "merge_clustering": _event_summary(clustering_rows),
        "unit_tests": _event_summary(unit_test_rows),
        "sidecars_present": present,
        "sidecars_missing": missing,
        "json_sidecar_shapes": json_sidecars,
        "text_sidecars": text_sidecars,
    }


def _run_summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "batch_started_at_utc",
        "batch_finished_at_utc",
        "total_attempted",
        "total_completed",
        "total_correct",
        "context_management",
        "model",
        "coder_model",
        "summarizer_model",
        "extractor_model",
        "action_selector_model",
        "skill_deletion",
        "skill_merging",
        "enable_skill_refinement",
        "enable_l1_skill_unit_test_gc",
        "max_iterations",
        "evolving_report_max_tokens",
    )
    return {key: summary.get(key) for key in keys}


def analyze_run(run_name: str, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / run_name
    warnings = WarningCollector(run_dir)
    inventory = ArtifactInventory(run_dir)
    diagnostics: dict[str, dict[str, Any]] = {}
    summary_payload = _read_json(
        run_dir / "run_summary.json",
        warnings=warnings,
        inventory=inventory,
        diagnostics=diagnostics,
        optional=False,
    )
    summary = summary_payload if isinstance(summary_payload, dict) else {}

    workspaces_dir = run_dir / "workspaces"
    workspace_dirs = sorted(
        (path for path in workspaces_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    run_chat = _empty_chat_summary()
    metric_totals = {
        "row_count": 0,
        "metrics_iteration_row_count": 0,
        "unique_iteration_count": 0,
        "compiled_count": 0,
        "compiled_observed": 0,
        "correct_count": 0,
        "correct_observed": 0,
        "is_hack_count": 0,
        "is_hack_observed": 0,
    }
    all_errors: list[dict[str, Any]] = []
    report_char_observations: list[int] = []
    report_final_chars: list[int] = []
    l0_observations: list[int] = []
    l0_finals: list[int] = []
    l0_growth: list[int] = []
    problems: list[dict[str, Any]] = []
    finished_count = 0

    for workspace_dir in workspace_dirs:
        workspace = workspace_dir.name
        level, problem_id = _parse_workspace(workspace)
        marker_payload = _read_json(
            workspace_dir / "run_finished.json",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
            optional=False,
        )
        marker = marker_payload if isinstance(marker_payload, dict) else None
        finished = marker is not None
        if finished:
            finished_count += 1

        chat = analyze_chat(
            workspace_dir / "chat_history.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
        )
        _merge_chat(run_chat, chat)
        metrics, error_rows = analyze_metrics(
            workspace_dir / "metrics_by_iteration.jsonl",
            workspace=workspace,
            marker=marker,
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
        )
        _merge_metric_totals(metric_totals, metrics)
        all_errors.extend(error_rows)
        workspace_report_values = metrics.pop("evolving_report_chars")
        report_char_observations.extend(workspace_report_values)
        snapshots = analyze_snapshots(
            workspace_dir / "iteration_snapshots.jsonl",
            warnings=warnings,
            inventory=inventory,
            diagnostics=diagnostics,
        )
        l0_observations.extend(snapshots.pop("entry_count_observations"))
        if snapshots["final_entry_count"] is not None:
            l0_finals.append(snapshots["final_entry_count"])
        if snapshots["growth"] is not None:
            l0_growth.append(snapshots["growth"])
        final_report_chars = None
        if workspace_report_values:
            final_report_chars = workspace_report_values[-1]
            report_final_chars.append(final_report_chars)

        problems.append(
            {
                "workspace": workspace,
                "level": level,
                "problem_id": problem_id,
                "finished": finished,
                "finished_metadata": {
                    "ended_utc": marker.get("ended_utc") if marker else None,
                    "elapsed_monotonic_sec": _round(
                        _as_float(marker.get("elapsed_monotonic_sec")) if marker else None,
                        3,
                    ),
                    "error_excerpt": _bounded_excerpt(
                        marker.get("metadata", {}).get("error")
                        if marker and isinstance(marker.get("metadata"), dict)
                        else None
                    ),
                },
                "chat_turn_count": chat["turn_count"],
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"final", "best"}
                },
                "final": metrics["final"],
                "best": metrics["best"],
                "l0": snapshots,
                "evolving_report_final_chars": final_report_chars,
            }
        )

    run_chat["by_phase"] = {
        phase: run_chat["by_phase"][phase] for phase in sorted(run_chat["by_phase"])
    }
    run_chat["phase_call_counts"] = {
        phase: bucket["calls"] for phase, bucket in run_chat["by_phase"].items()
    }
    for group in ("actions", "parse_methods", "parse_errors"):
        run_chat["action_selector"][group] = dict(
            sorted(run_chat["action_selector"][group].items())
        )

    governance = analyze_governance(
        run_dir,
        warnings=warnings,
        inventory=inventory,
        diagnostics=diagnostics,
    )
    evolving_phase = run_chat["by_phase"].get("evolving_report", _empty_token_bucket())
    evolving_report = {
        "call_count": evolving_phase["calls"],
        "prompt_tokens": evolving_phase["prompt_tokens"],
        "completion_tokens": evolving_phase["completion_tokens"],
        "total_tokens": evolving_phase["total_tokens"],
        "input_characters": run_chat["evolving_report_characters"]["input"],
        "output_characters": run_chat["evolving_report_characters"]["output"],
        "total_characters": run_chat["evolving_report_characters"]["total"],
        "recorded_report_size_chars": {
            "all_iterations": _summary_stats(report_char_observations),
            "final_per_workspace": _summary_stats(report_final_chars),
        },
    }
    return {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "status": "complete",
        "run_summary": _run_summary_view(summary),
        "workspace_count": len(workspace_dirs),
        "finished_workspace_count": finished_count,
        "chat": {
            "turn_count": run_chat["turn_count"],
            "tokens": run_chat["tokens"],
            "phase_call_counts": run_chat["phase_call_counts"],
            "tokens_by_phase": run_chat["by_phase"],
        },
        "action_selector": run_chat["action_selector"],
        "metrics": _finalize_metric_totals(metric_totals),
        "errors": _top_error_summary(all_errors),
        "evolving_report_overhead": evolving_report,
        "l0": {
            "snapshot_entry_counts": _summary_stats(l0_observations),
            "final_entry_counts_per_workspace": _summary_stats(l0_finals),
            "growth_per_workspace": _summary_stats(l0_growth),
        },
        "l1_governance": governance,
        "problems": sorted(problems, key=lambda row: row["workspace"]),
        "parse_diagnostics": {
            key: diagnostics[key] for key in sorted(diagnostics)
        },
        "input_artifacts": inventory.rows(),
        "warnings": warnings.rows(),
    }


def _valid_speedup(result: dict[str, Any]) -> float | None:
    if result.get("correct") is not True or result.get("is_hack") is True:
        return None
    speedup = _as_float(result.get("speedup"))
    return speedup if speedup is not None and speedup > 0 else None


def _comparison_result(problem: dict[str, Any]) -> dict[str, Any]:
    best = problem.get("best") if isinstance(problem.get("best"), dict) else {}
    final = problem.get("final") if isinstance(problem.get("final"), dict) else {}
    return {
        "best_iteration": best.get("iteration"),
        "best_compiled": best.get("compiled"),
        "best_correct": best.get("correct"),
        "best_is_hack": best.get("is_hack"),
        "best_speedup": best.get("speedup"),
        "best_runtime": best.get("runtime"),
        "final_iteration": final.get("iteration"),
        "final_correct": final.get("correct"),
        "final_error_category": final.get("error_category"),
        "final_error_excerpt": final.get("error_excerpt"),
        "best_locator": best.get("locator"),
        "final_locator": final.get("locator"),
    }


def _correctness_change(baseline: Any, candidate: Any) -> str:
    baseline_bool = _as_bool(baseline)
    candidate_bool = _as_bool(candidate)
    if baseline_bool is None or candidate_bool is None:
        return "unknown"
    if not baseline_bool and candidate_bool:
        return "gain"
    if baseline_bool and not candidate_bool:
        return "loss"
    return "same_correct" if baseline_bool else "same_incorrect"


def _case_candidate(
    row: dict[str, Any],
    selection: str,
    *,
    excerpt_from: str | None = None,
) -> dict[str, Any]:
    baseline = row["baseline"]
    candidate = row["candidate"]
    excerpt = (
        row[excerpt_from].get("final_error_excerpt")
        if excerpt_from in {"baseline", "candidate"}
        else None
    )
    locators = []
    for result in (baseline, candidate):
        best_locator = result.get("best_locator")
        locator = (
            best_locator
            if isinstance(best_locator, dict) and best_locator.get("iteration") is not None
            else result.get("final_locator")
        )
        if isinstance(locator, dict) and locator not in locators:
            locators.append(locator)
    return {
        "workspace": row["workspace"],
        "selection": selection,
        "correctness_change": row["correctness_change"],
        "baseline_valid_speedup": row["baseline_valid_speedup"],
        "candidate_valid_speedup": row["candidate_valid_speedup"],
        "speedup_delta": row["speedup_delta"],
        "evidence_locators": locators,
        "excerpt": excerpt,
    }


def _select_case_studies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [row for row in rows if row["speedup_delta"] is not None and row["speedup_delta"] > 0]
    regressions = [row for row in rows if row["speedup_delta"] is not None and row["speedup_delta"] < 0]
    gains = [row for row in rows if row["correctness_change"] == "gain"]
    losses = [row for row in rows if row["correctness_change"] == "loss"]
    same = [
        row
        for row in rows
        if row["correctness_change"] in {"same_correct", "same_incorrect"}
    ]

    improvement = min(improvements, key=lambda row: (-row["speedup_delta"], row["workspace"])) if improvements else None
    regression = min(regressions, key=lambda row: (row["speedup_delta"], row["workspace"])) if regressions else None
    gain = (
        min(
            gains,
            key=lambda row: (
                -(row["candidate_valid_speedup"] or -1.0),
                row["workspace"],
            ),
        )
        if gains
        else None
    )
    loss = (
        min(
            losses,
            key=lambda row: (
                -(row["baseline_valid_speedup"] or -1.0),
                row["workspace"],
            ),
        )
        if losses
        else None
    )

    def same_key(row: dict[str, Any]) -> tuple[Any, ...]:
        if row["correctness_change"] == "same_incorrect":
            return (0, 0.0, row["workspace"])
        delta = row["speedup_delta"]
        return (1, abs(delta) if delta is not None else math.inf, row["workspace"])

    representative = min(same, key=same_key) if same else None
    representative_label = (
        "lexicographically first matched workspace where both runs remain incorrect"
        if representative and representative["correctness_change"] == "same_incorrect"
        else "smallest absolute valid speedup delta among unchanged-correctness matches"
    )
    return {
        "largest_valid_speedup_improvement": _case_candidate(
            improvement, "largest positive candidate-minus-baseline valid speedup delta"
        )
        if improvement
        else None,
        "largest_valid_regression": _case_candidate(
            regression, "most negative candidate-minus-baseline valid speedup delta"
        )
        if regression
        else None,
        "correctness_gain": _case_candidate(
            gain,
            "baseline best incorrect and candidate best correct; highest candidate valid speedup, then workspace",
            excerpt_from="baseline",
        )
        if gain
        else None,
        "correctness_loss": _case_candidate(
            loss,
            "baseline best correct and candidate best incorrect; highest baseline valid speedup, then workspace",
            excerpt_from="candidate",
        )
        if loss
        else None,
        "representative_no_change_or_stall": _case_candidate(
            representative,
            representative_label,
            excerpt_from=(
                "candidate"
                if representative and representative["correctness_change"] == "same_incorrect"
                else None
            ),
        )
        if representative
        else None,
    }


COMPARISON_SELECTION_RULES = {
    "valid_speedup": (
        "finite positive best speedup with best_correct=true and best_is_hack=false; "
        "a delta exists only when both runs have valid speedups"
    ),
    "speedup_improvement": (
        "largest positive candidate-minus-baseline valid speedup delta; workspace name breaks ties"
    ),
    "speedup_regression": (
        "most negative candidate-minus-baseline valid speedup delta; workspace name breaks ties"
    ),
    "correctness_gain": (
        "baseline best incorrect and candidate best correct; prefer highest candidate valid speedup, "
        "then workspace name"
    ),
    "correctness_loss": (
        "baseline best correct and candidate best incorrect; prefer highest baseline valid speedup, "
        "then workspace name"
    ),
    "representative_no_change_or_stall": (
        "prefer the lexicographically first matched problem where both remain incorrect; otherwise "
        "choose unchanged correctness with the smallest absolute valid speedup delta, then workspace"
    ),
    "interpretation": "descriptive candidate selection only; no causal attribution is made",
}


def build_comparisons(runs: list[dict[str, Any]], baseline_run: str | None) -> dict[str, Any] | None:
    if baseline_run is None:
        return None
    by_run = {run["run_name"]: run for run in runs}
    baseline = by_run[baseline_run]
    baseline_problems = {row["workspace"]: row for row in baseline["problems"]}
    comparisons = []
    for run in runs:
        if run["run_name"] == baseline_run:
            continue
        candidate_problems = {row["workspace"]: row for row in run["problems"]}
        matched_names = sorted(set(baseline_problems) & set(candidate_problems))
        matched = []
        for workspace in matched_names:
            baseline_result = _comparison_result(baseline_problems[workspace])
            candidate_result = _comparison_result(candidate_problems[workspace])
            baseline_valid = _valid_speedup(baseline_problems[workspace]["best"])
            candidate_valid = _valid_speedup(candidate_problems[workspace]["best"])
            delta = (
                candidate_valid - baseline_valid
                if baseline_valid is not None and candidate_valid is not None
                else None
            )
            matched.append(
                {
                    "workspace": workspace,
                    "baseline": baseline_result,
                    "candidate": candidate_result,
                    "correctness_change": _correctness_change(
                        baseline_result["best_correct"], candidate_result["best_correct"]
                    ),
                    "baseline_valid_speedup": _round(baseline_valid),
                    "candidate_valid_speedup": _round(candidate_valid),
                    "speedup_delta": _round(delta),
                    "speedup_relative_change": _round(delta / baseline_valid, 6)
                    if delta is not None and baseline_valid
                    else None,
                }
            )
        change_counts = Counter(row["correctness_change"] for row in matched)
        comparisons.append(
            {
                "run_name": run["run_name"],
                "matched_problem_count": len(matched),
                "baseline_only_workspaces": sorted(set(baseline_problems) - set(candidate_problems)),
                "candidate_only_workspaces": sorted(set(candidate_problems) - set(baseline_problems)),
                "correctness_change_counts": dict(sorted(change_counts.items())),
                "matched_problems": matched,
                "case_study_candidates": _select_case_studies(matched),
            }
        )
    return {
        "baseline_run": baseline_run,
        "selection_rules": COMPARISON_SELECTION_RULES,
        "runs": comparisons,
    }


def _completion_check(run_name: str, runs_root: Path) -> tuple[bool, str]:
    run_dir = runs_root / run_name
    if not run_dir.is_dir():
        return False, f"run directory not found: {run_dir}"
    workspaces = run_dir / "workspaces"
    if not workspaces.is_dir():
        return False, f"workspaces directory not found: {workspaces}"
    summary_path = run_dir / "run_summary.json"
    try:
        with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return False, f"run_summary.json unavailable or malformed: {type(exc).__name__}: {exc}"
    if not isinstance(summary, dict):
        return False, "run_summary.json is not a JSON object"
    workspace_dirs = sorted(path for path in workspaces.iterdir() if path.is_dir())
    finished = sum((path / "run_finished.json").is_file() for path in workspace_dirs)
    attempted = _as_int(summary.get("total_attempted")) or 0
    completed = _as_int(summary.get("total_completed")) or 0
    if attempted <= 0 or completed < attempted:
        return False, f"run summary is incomplete: completed={completed}, attempted={attempted}"
    if not workspace_dirs or finished != len(workspace_dirs):
        return False, f"workspace markers are incomplete: finished={finished}, workspaces={len(workspace_dirs)}"
    if len(workspace_dirs) < attempted:
        return False, f"workspace count {len(workspace_dirs)} is below attempted count {attempted}"
    return True, "complete"


CSV_FIELDS = (
    "run_name",
    "is_baseline",
    "workspace",
    "level",
    "problem_id",
    "finished",
    "workspace_count",
    "finished_workspace_count",
    "chat_turn_count_run",
    "prompt_tokens_run",
    "completion_tokens_run",
    "total_tokens_run",
    "phase_call_counts_json",
    "action_mix_json",
    "action_selector_parse_errors",
    "metrics_rows_run",
    "compiled_count_run",
    "compiled_rate_run",
    "correct_count_run",
    "correct_rate_run",
    "is_hack_count_run",
    "is_hack_rate_run",
    "error_categories_json",
    "evolving_report_calls",
    "evolving_report_tokens",
    "evolving_report_output_characters",
    "l0_final_mean",
    "l1_entry_count",
    "l1_active_count",
    "refinement_entry_count",
    "deletion_event_count",
    "merge_event_count",
    "merge_accepted_count",
    "merge_acceptance_rate",
    "problem_chat_turn_count",
    "problem_metrics_rows",
    "problem_l0_final_count",
    "final_iteration",
    "final_compiled",
    "final_correct",
    "final_is_hack",
    "final_speedup",
    "final_runtime",
    "final_ref_runtime",
    "final_error_category",
    "final_error_excerpt",
    "best_iteration",
    "best_compiled",
    "best_correct",
    "best_is_hack",
    "best_speedup",
    "best_runtime",
    "matched_to_baseline",
    "baseline_best_correct",
    "baseline_best_is_hack",
    "baseline_valid_speedup",
    "correctness_change",
    "speedup_delta",
    "speedup_relative_change",
    "run_warning_count",
)


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_csv_rows(
    runs: list[dict[str, Any]],
    comparisons: dict[str, Any] | None,
    baseline_run: str | None,
) -> list[dict[str, Any]]:
    comparison_maps: dict[str, dict[str, dict[str, Any]]] = {}
    if comparisons:
        for comparison in comparisons["runs"]:
            comparison_maps[comparison["run_name"]] = {
                row["workspace"]: row for row in comparison["matched_problems"]
            }
    rows: list[dict[str, Any]] = []
    for run in runs:
        tokens = run["chat"]["tokens"]
        metrics = run["metrics"]
        governance = run["l1_governance"]
        shared_l1 = governance["shared_l1"]
        refinement = governance["refinement"]
        deletion = governance["deletion"]
        merge = governance["merge"]
        l0_mean = run["l0"]["final_entry_counts_per_workspace"]["mean"]
        matched_map = comparison_maps.get(run["run_name"], {})
        for problem in run["problems"]:
            final = problem["final"]
            best = problem["best"]
            matched = matched_map.get(problem["workspace"])
            baseline_result = matched.get("baseline", {}) if matched else {}
            row = {
                "run_name": run["run_name"],
                "is_baseline": run["run_name"] == baseline_run,
                "workspace": problem["workspace"],
                "level": problem["level"],
                "problem_id": problem["problem_id"],
                "finished": problem["finished"],
                "workspace_count": run["workspace_count"],
                "finished_workspace_count": run["finished_workspace_count"],
                "chat_turn_count_run": run["chat"]["turn_count"],
                "prompt_tokens_run": tokens["prompt_tokens"],
                "completion_tokens_run": tokens["completion_tokens"],
                "total_tokens_run": tokens["total_tokens"],
                "phase_call_counts_json": _json_cell(run["chat"]["phase_call_counts"]),
                "action_mix_json": _json_cell(run["action_selector"]["actions"]),
                "action_selector_parse_errors": run["action_selector"]["parse_error_count"],
                "metrics_rows_run": metrics["metrics_iteration_row_count"],
                "compiled_count_run": metrics["compiled_count"],
                "compiled_rate_run": metrics["compiled_rate"],
                "correct_count_run": metrics["correct_count"],
                "correct_rate_run": metrics["correct_rate"],
                "is_hack_count_run": metrics["is_hack_count"],
                "is_hack_rate_run": metrics["is_hack_rate"],
                "error_categories_json": _json_cell(run["errors"]["categories"]),
                "evolving_report_calls": run["evolving_report_overhead"]["call_count"],
                "evolving_report_tokens": run["evolving_report_overhead"]["total_tokens"],
                "evolving_report_output_characters": run["evolving_report_overhead"][
                    "output_characters"
                ],
                "l0_final_mean": l0_mean,
                "l1_entry_count": shared_l1["entry_count"],
                "l1_active_count": shared_l1["active_count"],
                "refinement_entry_count": refinement["entry_count"],
                "deletion_event_count": deletion["event_count"],
                "merge_event_count": merge["event_count"],
                "merge_accepted_count": merge["accepted_count"],
                "merge_acceptance_rate": merge["acceptance_rate"],
                "problem_chat_turn_count": problem["chat_turn_count"],
                "problem_metrics_rows": problem["metrics"]["metrics_iteration_row_count"],
                "problem_l0_final_count": problem["l0"]["final_entry_count"],
                "final_iteration": final["iteration"],
                "final_compiled": final["compiled"],
                "final_correct": final["correct"],
                "final_is_hack": final["is_hack"],
                "final_speedup": final["speedup"],
                "final_runtime": final["runtime"],
                "final_ref_runtime": final["ref_runtime"],
                "final_error_category": final["error_category"],
                "final_error_excerpt": final["error_excerpt"],
                "best_iteration": best["iteration"],
                "best_compiled": best["compiled"],
                "best_correct": best["correct"],
                "best_is_hack": best["is_hack"],
                "best_speedup": best["speedup"],
                "best_runtime": best["runtime"],
                "matched_to_baseline": matched is not None,
                "baseline_best_correct": baseline_result.get("best_correct"),
                "baseline_best_is_hack": baseline_result.get("best_is_hack"),
                "baseline_valid_speedup": matched.get("baseline_valid_speedup") if matched else None,
                "correctness_change": matched.get("correctness_change") if matched else None,
                "speedup_delta": matched.get("speedup_delta") if matched else None,
                "speedup_relative_change": matched.get("speedup_relative_change")
                if matched
                else None,
                "run_warning_count": sum(warning["count"] for warning in run["warnings"]),
            }
            rows.append(row)
    return rows


def write_outputs(
    doc: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / OUTPUT_JSON
    csv_path = output_dir / OUTPUT_CSV
    json_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = build_csv_rows(
        doc["runs"],
        doc.get("comparisons"),
        doc["inputs"].get("baseline_run"),
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in CSV_FIELDS})
    return json_path, csv_path


def extract_feature_evidence(
    *,
    runs_root: Path,
    run_names: list[str],
    output_dir: Path,
    baseline_run: str | None = None,
    generated_at_utc: str | None = None,
    write: bool = True,
    requested_runs: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze already-validated completed runs and optionally write both outputs."""
    records = [analyze_run(run_name, runs_root) for run_name in run_names]
    comparisons = build_comparisons(records, baseline_run)
    doc = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "inputs": {
            "repo_root": str(REPO_ROOT),
            "runs_root": str(runs_root.resolve()),
            "requested_runs": list(requested_runs if requested_runs is not None else run_names),
            "selected_runs": list(run_names),
            "baseline_run": baseline_run,
            "output_dir": str(output_dir.resolve()),
            "workspace_artifacts": list(WORKSPACE_ARTIFACTS),
            "run_artifacts": ["run_summary.json", "shared_l1.jsonl"],
            "governance_sidecars": list(GOVERNANCE_SIDECARS),
        },
        "semantics": {
            "metrics_rate_denominator": "rows where the metrics_iteration boolean field is present",
            "token_total_policy": "reported total_tokens, else prompt+completion when both are reported",
            "valid_speedup": COMPARISON_SELECTION_RULES["valid_speedup"],
            "error_taxonomy": "heuristic classification of metrics_iteration.error",
            "content_policy": "no full chat text or code; only bounded error excerpts in final/case evidence",
            "completion_policy": (
                "run_summary completed>=attempted>0 and every selected workspace has run_finished.json"
            ),
        },
        "runs": records,
        "comparisons": comparisons,
        "warning_count": sum(
            warning["count"] for run in records for warning in run.get("warnings", [])
        ),
    }
    if write:
        write_outputs(doc, output_dir=output_dir)
    return doc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic, compact feature evidence from explicitly selected "
            "completed evolving-agent runs."
        ),
        epilog=(
            "Paths are resolved relative to the repository root. Example:\n"
            "  python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \\\n"
            "    --runs-root runs_evolving/inference_oss_120b \\\n"
            "    --runs RUN_A --runs RUN_B \\\n"
            "    --baseline-run RUN_A \\\n"
            "    --output-dir scripts_integration/new_evolving_agent_analysis/output/example\n\n"
            "Rates use observed metrics_iteration fields; valid comparison speedups must be "
            "positive, correct, and non-hack. Partial runs are rejected."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        metavar="PATH",
        help="Root containing selected run directories (relative paths use repo root)",
    )
    parser.add_argument(
        "--runs",
        action="append",
        required=True,
        metavar="RUN_NAME",
        help="Completed run directory name to analyze; repeat for each run",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        help=f"Destination for {OUTPUT_JSON} and {OUTPUT_CSV} (relative to repo root)",
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        metavar="RUN_NAME",
        help="Selected run used for matched per-problem descriptive comparisons",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = _resolve_repo_path(args.runs_root)
    output_dir = _resolve_repo_path(args.output_dir)
    requested_runs = list(args.runs)
    run_names = list(dict.fromkeys(requested_runs))

    if not runs_root.is_dir():
        print(f"[feature-evidence] runs root not found: {runs_root}", file=sys.stderr)
        return 2
    if args.baseline_run is not None and args.baseline_run not in run_names:
        print(
            "[feature-evidence] --baseline-run must also be supplied with --runs: "
            f"{args.baseline_run}",
            file=sys.stderr,
        )
        return 2
    incomplete = []
    for run_name in run_names:
        complete, detail = _completion_check(run_name, runs_root)
        if not complete:
            incomplete.append((run_name, detail))
    if incomplete:
        for run_name, detail in incomplete:
            print(f"[feature-evidence] rejected {run_name}: {detail}", file=sys.stderr)
        print("[feature-evidence] no outputs written because selected runs must be complete", file=sys.stderr)
        return 2

    doc = extract_feature_evidence(
        runs_root=runs_root,
        run_names=run_names,
        requested_runs=requested_runs,
        output_dir=output_dir,
        baseline_run=args.baseline_run,
        write=True,
    )
    print(
        f"[feature-evidence] runs={len(doc['runs'])} "
        f"workspaces={sum(run['workspace_count'] for run in doc['runs'])} "
        f"warnings={doc['warning_count']}"
    )
    print(f"[feature-evidence] json={output_dir / OUTPUT_JSON}")
    print(f"[feature-evidence] csv={output_dir / OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
