"""Aggregate cross-run metrics for evolving-agent KernelBench experiments.

Walks every run folder under ``runs_evolving/`` (skipping ``archived/`` and any
directory without a ``workspaces/`` subdir) and flattens each run into a single
record combining:

- run configuration from ``<run>/run_summary.json`` (context management mode,
  models, skill-governance flags, ...)
- batch outcomes/timing from ``run_summary.json`` with a fallback to
  ``batch_timing.jsonl`` for in-flight runs that have no summary yet
- L1 skill-governance counters from ``shared_l1.jsonl`` plus the optional
  governance sidecars (``l1_skill_merges.jsonl``, ``l1_skill_deletions.jsonl``);
  every counter degrades to 0 when the artifact is absent
- final-iteration speedup aggregates and fast-p values read from
  ``<run>/visualizations/performance_stats.json``, generated on demand by
  importing ``build_performance_stats`` (never shelled out)

Speedup aggregates and fast-p are computed by ``generate_run_performance_stats``
over **correct, non-hack samples only**; failed problems are excluded from the
mean/median/geomean rather than scored as 0, while fast-p still penalizes them
through the full-problem denominator.

Outputs (default ``scripts_integration/new_evolving_agent_analysis/output/``):
    aggregate_runs.json   nested doc, one entry per run plus per-iteration series
    aggregate_runs.csv    one flat row per run

Example:
    uv run python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
        --hardware NVIDIA_GH200x2 \
        --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
        --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
SERVER_DIR = SEA_ROOT / "visualizations" / "kernelbench" / "server"
for _path in (str(SEA_ROOT), str(SERVER_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kernelbench.performance_stats import (  # noqa: E402
    DEFAULT_FAST_P_THRESHOLDS,
    parse_fastp_values,
    read_json,
    read_jsonl,
    resolve_threshold_key,
    safe_float,
    write_json,
)
from generate_run_performance_stats import build_performance_stats, discover_run_names  # noqa: E402

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs_evolving"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts_integration" / "new_evolving_agent_analysis" / "output"
DEFAULT_HARDWARE = "NVIDIA_GH200x2"
DEFAULT_BASELINE_STEM = "baseline_time_torch"

SKIP_DIR_NAMES = {"archived"}

#: Governance sidecars written flat into the run dir when skill governance is on.
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

SPEEDUP_SERIES_KEYS = (
    "current_mean",
    "current_median",
    "current_geometric_mean",
    "best_mean",
    "best_median",
    "best_geometric_mean",
)

_RUN_NAME_TIMESTAMP_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$")
_RUN_NAME_ITERATIONS_RE = re.compile(r"itr(\d+)")

#: Cached performance_stats.json is treated as stale when a run artifact is newer
#: than this many seconds past the cache mtime (guards filesystem timestamp jitter).
STALE_TOLERANCE_SEC = 1.0


# --------------------------------------------------------------------------- #
# small defensive helpers
# --------------------------------------------------------------------------- #
def _read_json_safe(path: Path) -> tuple[Any, str | None]:
    """``read_json`` that never raises on a truncated / half-written file."""
    try:
        return read_json(path, default=None), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"



def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    total = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total += 1
    except OSError:
        return 0
    return total


def _run_name_timestamp(run_name: str) -> str | None:
    match = _RUN_NAME_TIMESTAMP_RE.search(run_name)
    if match is None:
        return None
    year, month, day, hour, minute = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}"


# --------------------------------------------------------------------------- #
# run discovery / status
# --------------------------------------------------------------------------- #
def list_run_names(*, runs_root: Path, only: list[str] | None = None) -> list[str]:
    """Discovered run names under ``runs_root``, optionally filtered by ``only``."""
    if not runs_root.is_dir():
        return []
    try:
        discovered = [name for name in discover_run_names(runs_root=runs_root) if name not in SKIP_DIR_NAMES]
    except Exception:
        discovered = sorted(
            child.name
            for child in runs_root.iterdir()
            if child.is_dir() and child.name not in SKIP_DIR_NAMES and (child / "workspaces").is_dir()
        )
    if not only:
        return discovered
    wanted = list(dict.fromkeys(only))
    return [name for name in wanted if name in discovered]


def workspace_progress(run_dir: Path) -> dict[str, int]:
    """Count workspaces, finished markers, and correct problems from those markers.

    ``run_finished.json`` carries ``metadata.best_correct``, the same predicate the
    batch runner uses for ``run_summary.total_correct`` (``[e for e in runs if
    e.get("best_correct")]``), so a run with no ``run_summary.json`` yet can still
    report a truthful correctness count instead of a fabricated 0.
    """
    workspaces_dir = run_dir / "workspaces"
    if not workspaces_dir.is_dir():
        return {"workspace_count": 0, "workspaces_finished": 0, "workspaces_correct": 0}
    total = 0
    finished = 0
    correct = 0
    for child in sorted(workspaces_dir.iterdir(), key=lambda x: x.name):
        if not child.is_dir():
            continue
        total += 1
        marker = child / "run_finished.json"
        if not marker.is_file():
            continue
        finished += 1
        doc, _ = _read_json_safe(marker)
        metadata = doc.get("metadata") if isinstance(doc, dict) else None
        if isinstance(metadata, dict) and bool(metadata.get("best_correct")):
            correct += 1
    return {
        "workspace_count": total,
        "workspaces_finished": finished,
        "workspaces_correct": correct,
    }


def classify_status(*, summary: dict[str, Any], progress: dict[str, int], has_summary: bool) -> str:
    """``complete`` only when the batch summary exists and everything finished."""
    if not has_summary:
        return "partial"
    attempted = _as_int(summary.get("total_attempted"))
    completed = _as_int(summary.get("total_completed"))
    if attempted <= 0 or completed < attempted:
        return "partial"
    workspace_count = progress.get("workspace_count", 0)
    workspaces_finished = progress.get("workspaces_finished", 0)
    if workspace_count <= 0 or workspaces_finished < workspace_count:
        return "partial"
    return "complete"


# --------------------------------------------------------------------------- #
# iteration counts / hardware
# --------------------------------------------------------------------------- #
def observed_max_iteration(run_dir: Path) -> int:
    """Largest ``iteration`` seen across ``workspaces/*/metrics_by_iteration.jsonl``."""
    workspaces_dir = run_dir / "workspaces"
    if not workspaces_dir.is_dir():
        return 0
    best = 0
    for child in sorted(workspaces_dir.iterdir(), key=lambda x: x.name):
        if not child.is_dir():
            continue
        for row in read_jsonl(child / "metrics_by_iteration.jsonl"):
            best = max(best, _as_int(row.get("iteration")))
    return best


def infer_max_iterations(
    *,
    run_name: str,
    run_dir: Path,
    summary: dict[str, Any],
    stats_doc: dict[str, Any] | None,
) -> tuple[int | None, str]:
    """Best-effort per-problem iteration budget plus the source it came from."""
    for key in ("max_iterations", "iterations", "num_iterations", "max_iters"):
        value = summary.get(key)
        if value is not None:
            parsed = _as_int(value, default=0)
            if parsed > 0:
                return parsed, f"run_summary.{key}"

    match = _RUN_NAME_ITERATIONS_RE.search(run_name)
    if match is not None:
        parsed = _as_int(match.group(1), default=0)
        if parsed > 0:
            return parsed, "run_name"

    if isinstance(stats_doc, dict):
        parsed = _as_int(stats_doc.get("iteration_count"), default=0)
        if parsed > 0:
            return parsed, "performance_stats.iteration_count"

    parsed = observed_max_iteration(run_dir)
    if parsed > 0:
        return parsed, "metrics_by_iteration"
    return None, "unknown"


def detect_hardware(run_dir: Path) -> str | None:
    """First non-null ``metadata.hardware`` in ``eval_results.json`` (small file)."""
    eval_results, _ = _read_json_safe(run_dir / "eval_results.json")
    if not isinstance(eval_results, dict):
        return None
    for entries_by_problem in eval_results.values():
        if not isinstance(entries_by_problem, dict):
            continue
        for entries in entries_by_problem.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                metadata = entry.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                hardware = _as_str(metadata.get("hardware"))
                if hardware:
                    return hardware
    return None


#: Tokens too generic to identify an accelerator family.
_GENERIC_HARDWARE_TOKENS = {"NVIDIA", "GPU", "CUDA", "HBM", "GB", "G"}


def hardware_matches_baseline(hardware: str, baseline_file: Path) -> bool:
    """Loose check that the run's accelerator appears in the baseline folder name."""
    haystack = re.sub(r"[^0-9A-Za-z]", "", baseline_file.parent.name).upper()
    tokens = [
        token.upper()
        for token in re.split(r"[^0-9A-Za-z]+", hardware)
        if len(token) >= 3 and token.upper() not in _GENERIC_HARDWARE_TOKENS
    ]
    if not tokens or not haystack:
        return True
    return any(token in haystack for token in tokens)


# --------------------------------------------------------------------------- #
# governance counters
# --------------------------------------------------------------------------- #
def _catalog_stats_fallback(entries: list[dict[str, Any]]) -> dict[str, int]:
    active = [e for e in entries if (e.get("status") or "active") == "active"]
    superseded = [e for e in entries if e.get("status") == "superseded"]
    deleted = [e for e in entries if e.get("status") == "deleted"]

    def _is_merged(entry: dict[str, Any]) -> bool:
        if str(entry.get("source") or "") == "skill_merge":
            return True
        merge_meta = entry.get("merge_meta")
        return isinstance(merge_meta, dict) and bool(merge_meta.get("source_entry_ids"))

    def _is_refined(entry: dict[str, Any]) -> bool:
        if entry.get("parent_id") or entry.get("refinement_round") is not None:
            return True
        if str(entry.get("source") or "") == "skill_refinement":
            return True
        return bool(entry.get("refinement_meta"))

    return {
        "total_entries": len(entries),
        "active": len(active),
        "superseded": len(superseded),
        "deleted": len(deleted),
        "active_merged_skills": sum(1 for e in active if _is_merged(e)),
        "active_refined_skills": sum(1 for e in active if _is_refined(e)),
        "active_standard_skills": sum(1 for e in active if not _is_merged(e) and not _is_refined(e)),
        "superseded_by_merge": sum(
            1
            for e in superseded
            if isinstance(e.get("merge_meta"), dict) and e["merge_meta"].get("merged_into")
        ),
    }


def collect_governance(run_dir: Path) -> dict[str, Any]:
    """Skill-governance counters; every field is 0/empty when artifacts are absent."""
    l1_jsonl = run_dir / "shared_l1.jsonl"
    entries = read_jsonl(l1_jsonl)

    catalog: dict[str, Any] = {}
    try:
        from evolving_common.memory_manager import summarize_l1_catalog_stats

        candidate = summarize_l1_catalog_stats(l1_jsonl)
        if isinstance(candidate, dict):
            catalog = candidate
    except Exception:
        catalog = {}
    if not catalog:
        catalog = _catalog_stats_fallback(entries)

    merge_rows = read_jsonl(run_dir / "l1_skill_merges.jsonl")
    accepted = [row for row in merge_rows if str(row.get("status")) == "accepted"]
    absorbed: set[str] = set()
    for row in accepted:
        source_ids = row.get("source_entry_ids")
        if isinstance(source_ids, list):
            absorbed.update(str(sid) for sid in source_ids if str(sid).strip())

    deletion_rows = read_jsonl(run_dir / "l1_skill_deletions.jsonl")
    deletion_reasons: dict[str, int] = {}
    for row in deletion_rows:
        reason = _as_str(row.get("reason")) or "unknown"
        deletion_reasons[reason] = deletion_reasons.get(reason, 0) + 1

    refined_count = _as_int(catalog.get("active_refined_skills"))
    lineage_refined = sum(
        1
        for entry in entries
        if entry.get("parent_id")
        or entry.get("refinement_round") is not None
        or str(entry.get("source") or "") == "skill_refinement"
        or bool(entry.get("refinement_meta"))
    )
    refined_count = max(refined_count, lineage_refined)

    l1_entry_count = _as_int(catalog.get("total_entries"), default=len(entries))
    active_count = _as_int(catalog.get("active"), default=l1_entry_count)

    return {
        "l1_entry_count": l1_entry_count,
        "l1_active_count": active_count,
        "l1_superseded_count": _as_int(catalog.get("superseded")),
        "deleted_count": _as_int(catalog.get("deleted")),
        "refined_count": refined_count,
        "merge_count": len(accepted),
        "merge_events_total": len(merge_rows),
        "merge_events_rejected": sum(1 for row in merge_rows if str(row.get("status")) == "rejected"),
        "merge_events_skipped": sum(1 for row in merge_rows if str(row.get("status")) == "skipped"),
        "skills_absorbed_by_merge": len(absorbed),
        "deletion_event_count": len(deletion_rows),
        "deletion_reasons": deletion_reasons,
        "merge_passes_executed": _count_jsonl_rows(run_dir / "l1_skill_merge_clustering.jsonl"),
        "unit_test_runs_total": _count_jsonl_rows(run_dir / "l1_skill_unit_test_runs.jsonl"),
        "catalog_compression_ratio": _round(active_count / l1_entry_count) if l1_entry_count else None,
        "governance_sidecars_present": [
            name for name in GOVERNANCE_SIDECARS if (run_dir / name).exists()
        ],
    }


# --------------------------------------------------------------------------- #
# performance_stats.json
# --------------------------------------------------------------------------- #
def stats_path_for(run_dir: Path) -> Path:
    return run_dir / "visualizations" / "performance_stats.json"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def stats_inputs_newest_mtime(run_dir: Path) -> float:
    """Newest mtime across everything ``build_performance_stats`` reads.

    Used to invalidate a cached ``performance_stats.json``: an in-flight run gains
    workspaces and iterations after the cache was written, so a stale cache would
    otherwise report (for example) a 1-problem snapshot for a 9-problem run.
    """
    newest = _mtime(run_dir / "run_summary.json")
    workspaces_dir = run_dir / "workspaces"
    if not workspaces_dir.is_dir():
        return newest
    newest = max(newest, _mtime(workspaces_dir))
    for child in workspaces_dir.iterdir():
        if not child.is_dir():
            continue
        metrics = child / "metrics_by_iteration.jsonl"
        newest = max(newest, _mtime(metrics) if metrics.is_file() else _mtime(child))
    return newest


def load_or_build_stats(
    *,
    run_name: str,
    runs_root: Path,
    baseline_file: Path,
    fast_p_thresholds: list[float],
    regenerate: bool,
) -> tuple[dict[str, Any] | None, str | None, str, str | None]:
    """``(stats_doc, error, source, stale_reason)``; source is cached|generated|regenerated_stale|missing."""
    run_dir = runs_root / run_name
    path = stats_path_for(run_dir)
    stale_reason: str | None = None
    if not regenerate and path.is_file():
        doc, read_error = _read_json_safe(path)
        if read_error is not None:
            stale_reason = f"cached performance_stats.json unreadable ({read_error})"
        elif not isinstance(doc, dict) or not doc.get("iterations"):
            stale_reason = "cached performance_stats.json has no iterations"
        else:
            newest_input = stats_inputs_newest_mtime(run_dir)
            if newest_input > _mtime(path) + STALE_TOLERANCE_SEC:
                stale_reason = "run artifacts are newer than cached performance_stats.json"
            else:
                return doc, None, "cached", None

    try:
        result = build_performance_stats(
            run_name=run_name,
            runs_root=runs_root,
            baseline_file=baseline_file,
            fast_p_thresholds=fast_p_thresholds,
        )
    except Exception as exc:  # missing workspaces / baseline / no parsable problems
        return None, f"{type(exc).__name__}: {exc}", "missing", stale_reason

    doc = result.get("doc") if isinstance(result, dict) else None
    if not isinstance(doc, dict):
        return None, "build_performance_stats returned no doc", "missing", stale_reason
    return doc, None, ("regenerated_stale" if stale_reason else "generated"), stale_reason


def _fast_p_map(source: Any, thresholds: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    values = source if isinstance(source, dict) else {}
    for threshold in thresholds:
        key = resolve_threshold_key(values, threshold)
        out[str(threshold)] = _round(safe_float(values.get(key))) if key is not None else None
    return out


def summarize_stats(stats_doc: dict[str, Any] | None, thresholds: list[float]) -> dict[str, Any]:
    """Final-iteration aggregates + fast-p, plus the per-iteration series."""
    empty = {
        "iteration_count": 0,
        "problem_count": 0,
        "final_iteration": None,
        "final_aligned_count": 0,
        "hack_iteration_count": 0,
        "problems_with_hack": 0,
        "speedup_current": {"mean": None, "median": None, "geometric_mean": None, "n": 0},
        "speedup_best": {"mean": None, "median": None, "geometric_mean": None, "n": 0},
        "fast_p_best": {str(p): None for p in thresholds},
        "fast_p_current": {str(p): None for p in thresholds},
    }
    if not isinstance(stats_doc, dict):
        return empty

    iterations = stats_doc.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        return empty
    final = iterations[-1] if isinstance(iterations[-1], dict) else {}

    aggregates = final.get("aggregates") if isinstance(final.get("aggregates"), dict) else {}
    current = aggregates.get("current") if isinstance(aggregates.get("current"), dict) else {}
    best = aggregates.get("best") if isinstance(aggregates.get("best"), dict) else {}

    # Sample size actually entering each aggregate. generate_run_performance_stats
    # gates the best curves on `best_correct` alone (hack iterations never form a
    # best; a later hack does not revoke an earlier non-hack best). Do NOT AND with
    # `best_is_hack` — that recorder field is run_had_hack, a run-level "any hack
    # seen" latch, not "this best kernel is a hack". ANDing it understates n by
    # roughly problems_with_hack while the geomean it annotates is unfiltered.
    points = final.get("points") if isinstance(final.get("points"), list) else []
    best_n = 0
    current_n = 0
    for point in points:
        if not isinstance(point, dict):
            continue
        if point.get("best_correct"):
            best_n += 1
        current_correct = point.get("current_correct")
        if current_correct is None:
            current_correct = bool(point.get("correct")) and not bool(point.get("is_hack"))
        if current_correct:
            current_n += 1

    return {
        "iteration_count": _as_int(stats_doc.get("iteration_count"), default=len(iterations)),
        "problem_count": _as_int(stats_doc.get("problem_count")),
        "final_iteration": _as_int(final.get("iteration")) or None,
        "final_aligned_count": _as_int(final.get("aligned_count")),
        "hack_iteration_count": _as_int(stats_doc.get("hack_iteration_count")),
        "problems_with_hack": _as_int(stats_doc.get("problems_with_hack")),
        "speedup_current": {
            "mean": _round(safe_float(current.get("mean"))),
            "median": _round(safe_float(current.get("median"))),
            "geometric_mean": _round(safe_float(current.get("geometric_mean"))),
            "n": current_n,
        },
        "speedup_best": {
            "mean": _round(safe_float(best.get("mean"))),
            "median": _round(safe_float(best.get("median"))),
            "geometric_mean": _round(safe_float(best.get("geometric_mean"))),
            "n": best_n,
        },
        "fast_p_best": _fast_p_map(final.get("fast_p_best"), thresholds),
        "fast_p_current": _fast_p_map(final.get("fast_p_current"), thresholds),
    }


def _best_n_by_iteration(stats_doc: dict[str, Any] | None) -> dict[int, int]:
    """Count correct, non-hack bests at each iteration from performance_stats points."""
    out: dict[int, int] = {}
    if not isinstance(stats_doc, dict):
        return out
    iterations = stats_doc.get("iterations")
    if not isinstance(iterations, list):
        return out
    for item in iterations:
        if not isinstance(item, dict):
            continue
        iteration = _as_int(item.get("iteration"))
        points = item.get("points") if isinstance(item.get("points"), list) else []
        sample_n = 0
        for point in points:
            if not isinstance(point, dict):
                continue
            # Same gate as the generator's best curves; see aggregate n note above.
            if point.get("best_correct"):
                sample_n += 1
        out[iteration] = sample_n
    return out


def extract_series(stats_doc: dict[str, Any] | None, thresholds: list[float]) -> dict[str, Any]:
    """Per-iteration series kept in the aggregate doc so compare_runs is self-contained."""
    out: dict[str, Any] = {"speedup": {}, "fast_p_best": {}, "fast_p_current": {}}
    if not isinstance(stats_doc, dict):
        return out
    series = stats_doc.get("series")
    if not isinstance(series, dict):
        return out

    speedup_series = series.get("speedup") if isinstance(series.get("speedup"), dict) else {}
    for key in SPEEDUP_SERIES_KEYS:
        points = speedup_series.get(key)
        if isinstance(points, list):
            out["speedup"][key] = [
                {
                    "iteration": _as_int(p.get("iteration")),
                    "value": _round(safe_float(p.get("value"))),
                    **(
                        {"n": _as_int(p.get("n"))}
                        if p.get("n") is not None
                        else {}
                    ),
                }
                for p in points
                if isinstance(p, dict)
            ]

    n_by_iteration = _best_n_by_iteration(stats_doc)
    if n_by_iteration:
        for point in out["speedup"].get("best_geometric_mean") or []:
            if "n" not in point:
                sample_n = n_by_iteration.get(point.get("iteration"))
                if sample_n is not None:
                    point["n"] = sample_n

    for field in ("fast_p_best", "fast_p_current"):
        raw = series.get(field) if isinstance(series.get(field), dict) else {}
        for threshold in thresholds:
            key = resolve_threshold_key(raw, threshold)
            if key is None:
                continue
            points = raw.get(key)
            if not isinstance(points, list):
                continue
            out[field][str(threshold)] = [
                {"iteration": _as_int(p.get("iteration")), "value": _round(safe_float(p.get("value")))}
                for p in points
                if isinstance(p, dict)
            ]
    return out


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def collect_timing(*, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Wall-clock totals; falls back to batch_timing.jsonl for in-flight runs."""
    rows = read_jsonl(run_dir / "batch_timing.jsonl")
    row_total = 0.0
    row_count = 0
    # A resume appends new rows for replayed problems, so the file can hold more
    # rows than problems. Keep the last timing per problem for the distinct count.
    per_problem: dict[str, float] = {}
    for row in rows:
        value = safe_float(row.get("wall_time_sec"))
        if value is None:
            continue
        row_total += value
        row_count += 1
        # subset_index is the canonical position; problem_id alone collides
        # across levels (L1P100 vs L3P100).
        key = _as_str(row.get("subset_index"))
        if not key:
            key = f"{_as_str(row.get('level'))}:{_as_str(row.get('problem_id'))}"
        per_problem[key] = value

    total = safe_float(summary.get("total_wall_time_sec"))
    if total is None:
        total = row_total if row_count else None
    problems_timed = _as_int(summary.get("problems_timed_this_session"), default=row_count)

    # Do NOT trust summary["avg_wall_time_sec"] on a resumed run: it is
    # total_wall_time_sec / problems_timed_this_session, and a resume that
    # replayed 2 problems reports the whole batch's wall time divided by 2
    # (observed: 2143 min/problem instead of the true 85.7).
    distinct = len(per_problem)
    if distinct:
        avg = sum(per_problem.values()) / distinct
    else:
        avg = safe_float(summary.get("avg_wall_time_sec"))
        if avg is None and total is not None and problems_timed > 0:
            avg = total / problems_timed

    return {
        "batch_started_at_utc": _as_str(summary.get("batch_started_at_utc"))
        or _as_str(rows[0].get("started_at_utc") if rows else None),
        "batch_finished_at_utc": _as_str(summary.get("batch_finished_at_utc"))
        or _as_str(rows[-1].get("finished_at_utc") if rows else None),
        "batch_session_wall_time_sec": _round(safe_float(summary.get("batch_session_wall_time_sec")), 3),
        "total_wall_time_sec": _round(total, 3),
        "total_wall_time_hours": _round(total / 3600.0, 4) if total is not None else None,
        "avg_wall_time_sec": _round(avg, 3),
        "avg_wall_time_min": _round(avg / 60.0, 4) if avg is not None else None,
        "problems_timed_this_session": problems_timed,
        "batch_timing_rows": row_count,
        "batch_timing_status_counts": _status_counts(rows),
    }


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = _as_str(row.get("status")) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def build_run_record(
    *,
    run_name: str,
    runs_root: Path,
    baseline_file: Path,
    fast_p_thresholds: list[float],
    regenerate_stats: bool,
) -> dict[str, Any]:
    run_dir = runs_root / run_name
    warnings: list[str] = []

    summary_raw, summary_error = _read_json_safe(run_dir / "run_summary.json")
    has_summary = isinstance(summary_raw, dict)
    summary: dict[str, Any] = summary_raw if has_summary else {}
    if summary_error is not None:
        warnings.append(f"run_summary.json unreadable ({summary_error})")
    elif not has_summary:
        warnings.append("run_summary.json missing (run still in progress or aborted)")

    progress = workspace_progress(run_dir)
    status = classify_status(summary=summary, progress=progress, has_summary=has_summary)

    stats_doc, stats_error, stats_source, stale_reason = load_or_build_stats(
        run_name=run_name,
        runs_root=runs_root,
        baseline_file=baseline_file,
        fast_p_thresholds=fast_p_thresholds,
        regenerate=regenerate_stats,
    )
    if stats_error:
        warnings.append(f"performance_stats unavailable: {stats_error}")
    elif stale_reason:
        warnings.append(f"performance_stats rebuilt: {stale_reason}")

    max_iterations, max_iterations_source = infer_max_iterations(
        run_name=run_name, run_dir=run_dir, summary=summary, stats_doc=stats_doc
    )

    hardware = detect_hardware(run_dir)
    if hardware and not hardware_matches_baseline(hardware, baseline_file):
        # Purely informational: e.g. scoring an A6000 run against a GH200 baseline.
        warnings.append(
            f"run hardware {hardware!r} may not match baseline dir {baseline_file.parent.name!r}"
        )

    attempted = _as_int(summary.get("total_attempted"), default=progress["workspace_count"])
    completed = _as_int(summary.get("total_completed"), default=progress["workspaces_finished"])

    # Correctness: prefer run_summary; otherwise derive from the per-workspace
    # run_finished.json markers instead of reporting a fabricated 0 for in-flight runs.
    correct: int | None
    if has_summary and summary.get("total_correct") is not None:
        correct = _as_int(summary.get("total_correct"))
        outcomes_source = "run_summary"
        correct_rate_basis = "total_attempted"
        denominator = attempted
    elif progress["workspaces_finished"] > 0:
        correct = progress["workspaces_correct"]
        outcomes_source = "run_finished_markers"
        correct_rate_basis = "workspaces_finished"
        denominator = progress["workspaces_finished"]
        warnings.append(
            "total_correct derived from workspaces/*/run_finished.json "
            f"(correct_rate is over {denominator} finished problems, not the full batch)"
        )
    else:
        correct = None
        outcomes_source = "unavailable"
        correct_rate_basis = None
        denominator = 0

    per_level_summary = summary.get("per_level_summary")
    if not isinstance(per_level_summary, dict):
        per_level_summary = {}

    record: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": status,
        "timestamp": _as_str(summary.get("batch_started_at_utc")) or _run_name_timestamp(run_name),
        "run_name_timestamp": _run_name_timestamp(run_name),
        "hardware": hardware,
        "max_iterations": max_iterations,
        "max_iterations_source": max_iterations_source,
        "config": {
            "context_management": _as_str(summary.get("context_management")),
            "model": _as_str(summary.get("model")),
            "coder_model": _as_str(summary.get("coder_model")),
            "summarizer_model": _as_str(summary.get("summarizer_model")),
            "extractor_model": _as_str(summary.get("extractor_model")),
            "action_selector_model": _as_str(summary.get("action_selector_model")),
            "nvidia_endpoint": _as_str(summary.get("nvidia_endpoint")),
            "subset_csv": _as_str(summary.get("subset_csv")),
            "skill_deletion": _as_bool(summary.get("skill_deletion")),
            "skill_merging": _as_bool(summary.get("skill_merging")),
            "enable_skill_refinement": _as_bool(summary.get("enable_skill_refinement")),
            "enable_l1_skill_unit_test_gc": _as_bool(summary.get("enable_l1_skill_unit_test_gc")),
            # L2 promotion tier. Without these the analysis renders an L2 arm and
            # a plain truncation control as the SAME design (CLAUDE.md open item 7),
            # so every delta table silently compares an arm against itself.
            "enable_l2": _as_bool(summary.get("enable_l2")),
            "l2_render": _as_str(summary.get("l2_render")),
            "l2_min_tasks": _as_int(summary.get("l2_min_tasks"))
            if summary.get("l2_min_tasks") is not None
            else None,
            "l2_min_selections": _as_int(summary.get("l2_min_selections"))
            if summary.get("l2_min_selections") is not None
            else None,
            "l2_min_rate": safe_float(summary.get("l2_min_rate")),
            "l2_min_new_bests": _as_int(summary.get("l2_min_new_bests"))
            if summary.get("l2_min_new_bests") is not None
            else None,
            "l2_max_entries": _as_int(summary.get("l2_max_entries"))
            if summary.get("l2_max_entries") is not None
            else None,
            "l2_use_hit_rate": _as_bool(summary.get("l2_use_hit_rate")),
            "l2_min_hit_rate": safe_float(summary.get("l2_min_hit_rate")),
            "l2_standing_cap": _as_int(summary.get("l2_standing_cap"))
            if summary.get("l2_standing_cap") is not None
            else None,
            "l2_dedup_similarity": safe_float(summary.get("l2_dedup_similarity")),
            "l2_judge": _as_bool(summary.get("l2_judge")),
            "l2_freeze": _as_bool(summary.get("l2_freeze")),
            "redesign_l2": _as_bool(summary.get("redesign_l2")),
            "l2_standing_count": _as_int(summary.get("l2_standing_count"))
            if summary.get("l2_standing_count") is not None
            else None,
            "skill_merge_similarity": safe_float(summary.get("skill_merge_similarity")),
            "skill_merge_interval": _as_int(summary.get("skill_merge_interval"))
            if summary.get("skill_merge_interval") is not None
            else None,
            "skill_refinement_max_rounds": _as_int(summary.get("skill_refinement_max_rounds"))
            if summary.get("skill_refinement_max_rounds") is not None
            else None,
            "evolving_report_max_tokens": _as_int(summary.get("evolving_report_max_tokens"))
            if summary.get("evolving_report_max_tokens") is not None
            else None,
            "enable_static_check": _as_bool(summary.get("enable_static_check")),
            "dry_run": _as_bool(summary.get("dry_run")),
            "resume": _as_bool(summary.get("resume")),
        },
        "outcomes": {
            "total_attempted": attempted,
            "total_completed": completed,
            "total_correct": correct,
            "correct_rate": (
                _round(correct / denominator, 6) if correct is not None and denominator > 0 else None
            ),
            "correct_rate_basis": correct_rate_basis,
            "outcomes_source": outcomes_source,
            "best_speedup_overall": _round(safe_float(summary.get("best_speedup_overall"))),
            "best_runtime_overall": _round(safe_float(summary.get("best_runtime_overall"))),
            "suspicious_speedup_count": _as_int(summary.get("suspicious_speedup_count")),
            "workspace_count": progress["workspace_count"],
            "workspaces_finished": progress["workspaces_finished"],
            "per_level_summary": per_level_summary,
        },
        "timing": collect_timing(run_dir=run_dir, summary=summary),
        "governance": collect_governance(run_dir),
        "performance": summarize_stats(stats_doc, fast_p_thresholds),
        "series": extract_series(stats_doc, fast_p_thresholds),
        "performance_stats_path": str(stats_path_for(run_dir)) if stats_doc is not None else None,
        "performance_stats_source": stats_source,
        "performance_stats_error": stats_error,
        "warnings": warnings,
    }
    return record


def flatten_record(record: dict[str, Any], thresholds: list[float]) -> dict[str, Any]:
    """One flat CSV row per run."""
    config = record.get("config", {})
    outcomes = record.get("outcomes", {})
    timing = record.get("timing", {})
    governance = record.get("governance", {})
    performance = record.get("performance", {})
    speedup_current = performance.get("speedup_current", {})
    speedup_best = performance.get("speedup_best", {})

    row: dict[str, Any] = {
        "run_name": record.get("run_name"),
        "status": record.get("status"),
        "timestamp": record.get("timestamp"),
        "hardware": record.get("hardware"),
        "context_management": config.get("context_management"),
        "model": config.get("model"),
        "nvidia_endpoint": config.get("nvidia_endpoint"),
        "skill_deletion": config.get("skill_deletion"),
        "skill_merging": config.get("skill_merging"),
        "enable_skill_refinement": config.get("enable_skill_refinement"),
        "enable_l1_skill_unit_test_gc": config.get("enable_l1_skill_unit_test_gc"),
        "enable_l2": config.get("enable_l2"),
        "l2_render": config.get("l2_render"),
        "l2_use_hit_rate": config.get("l2_use_hit_rate"),
        "l2_standing_cap": config.get("l2_standing_cap"),
        "l2_dedup_similarity": config.get("l2_dedup_similarity"),
        "l2_judge": config.get("l2_judge"),
        "l2_freeze": config.get("l2_freeze"),
        "redesign_l2": config.get("redesign_l2"),
        "l2_standing_count": config.get("l2_standing_count"),
        "max_iterations": record.get("max_iterations"),
        "total_attempted": outcomes.get("total_attempted"),
        "total_completed": outcomes.get("total_completed"),
        "total_correct": outcomes.get("total_correct"),
        "correct_rate": outcomes.get("correct_rate"),
        "correct_rate_basis": outcomes.get("correct_rate_basis"),
        "outcomes_source": outcomes.get("outcomes_source"),
        "best_speedup_overall": outcomes.get("best_speedup_overall"),
        "best_runtime_overall": outcomes.get("best_runtime_overall"),
        "suspicious_speedup_count": outcomes.get("suspicious_speedup_count"),
        "workspace_count": outcomes.get("workspace_count"),
        "workspaces_finished": outcomes.get("workspaces_finished"),
        "total_wall_time_sec": timing.get("total_wall_time_sec"),
        "total_wall_time_hours": timing.get("total_wall_time_hours"),
        "avg_wall_time_sec": timing.get("avg_wall_time_sec"),
        "avg_wall_time_min": timing.get("avg_wall_time_min"),
        "problems_timed_this_session": timing.get("problems_timed_this_session"),
        "l1_entry_count": governance.get("l1_entry_count"),
        "l1_active_count": governance.get("l1_active_count"),
        "merge_count": governance.get("merge_count"),
        "deleted_count": governance.get("deleted_count"),
        "refined_count": governance.get("refined_count"),
        "deletion_event_count": governance.get("deletion_event_count"),
        "merge_events_total": governance.get("merge_events_total"),
        "catalog_compression_ratio": governance.get("catalog_compression_ratio"),
        "problem_count": performance.get("problem_count"),
        "iteration_count": performance.get("iteration_count"),
        "final_iteration": performance.get("final_iteration"),
        "hack_iteration_count": performance.get("hack_iteration_count"),
        "problems_with_hack": performance.get("problems_with_hack"),
        "speedup_current_mean": speedup_current.get("mean"),
        "speedup_current_median": speedup_current.get("median"),
        "speedup_current_geomean": speedup_current.get("geometric_mean"),
        "speedup_current_n": speedup_current.get("n"),
        "speedup_best_mean": speedup_best.get("mean"),
        "speedup_best_median": speedup_best.get("median"),
        "speedup_best_geomean": speedup_best.get("geometric_mean"),
        "speedup_best_n": speedup_best.get("n"),
    }
    fast_p_best = performance.get("fast_p_best", {})
    fast_p_current = performance.get("fast_p_current", {})
    for threshold in thresholds:
        key = str(threshold)
        row[f"fast_p_best@{key}"] = fast_p_best.get(key)
        row[f"fast_p_current@{key}"] = fast_p_current.get(key)

    row["performance_stats_source"] = record.get("performance_stats_source")
    row["warnings"] = " | ".join(record.get("warnings") or [])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def aggregate_runs(
    *,
    runs_root: Path,
    output_dir: Path,
    baseline_file: Path,
    fast_p_thresholds: list[float] | None = None,
    only_runs: list[str] | None = None,
    regenerate_stats: bool = False,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Build the cross-run aggregate doc (and optionally write JSON + CSV)."""
    thresholds = list(fast_p_thresholds or DEFAULT_FAST_P_THRESHOLDS)
    run_names = list_run_names(runs_root=runs_root, only=only_runs)

    missing = [name for name in (only_runs or []) if name not in run_names]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for run_name in run_names:
        try:
            records.append(
                build_run_record(
                    run_name=run_name,
                    runs_root=runs_root,
                    baseline_file=baseline_file,
                    fast_p_thresholds=thresholds,
                    regenerate_stats=regenerate_stats,
                )
            )
        except Exception as exc:  # never let one bad run kill the aggregation
            failures.append({"run_name": run_name, "error": f"{type(exc).__name__}: {exc}"})

    doc: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "baseline_file": str(baseline_file),
        "fast_p_thresholds": thresholds,
        "speedup_aggregate_policy": "correct_only_exclude_hack",
        "discovered": len(run_names),
        "aggregated": len(records),
        "complete_runs": sum(1 for r in records if r.get("status") == "complete"),
        "partial_runs": sum(1 for r in records if r.get("status") == "partial"),
        "requested_runs_not_found": missing,
        "failures": failures,
        "runs": records,
    }

    json_path = output_dir / "aggregate_runs.json"
    csv_path = output_dir / "aggregate_runs.csv"
    if write_outputs:
        write_json(json_path, doc)
        write_csv(csv_path, [flatten_record(record, thresholds) for record in records])

    return {"doc": doc, "json_path": str(json_path), "csv_path": str(csv_path)}


def resolve_baseline_path(args: argparse.Namespace) -> Path:
    if args.baseline_file:
        candidate = Path(args.baseline_file)
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    return REPO_ROOT / "results" / "timing" / args.hardware / f"{args.baseline}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help=f"Root containing evolving-agent run directories (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for aggregate_runs.json / aggregate_runs.csv (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=DEFAULT_HARDWARE,
        help=f"Hardware folder under results/timing when --baseline-file is not given (default: {DEFAULT_HARDWARE})",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE_STEM,
        help=f"Baseline filename stem under results/timing/<hardware>/ (default: {DEFAULT_BASELINE_STEM})",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=None,
        help="Explicit baseline timing JSON path; overrides --hardware/--baseline",
    )
    parser.add_argument(
        "--runs",
        action="append",
        default=None,
        metavar="RUN_NAME",
        help="Restrict aggregation to this run name (repeatable)",
    )
    parser.add_argument(
        "--fast-p-values",
        type=str,
        default=None,
        help="Comma-separated fast-p thresholds (default: 0.0,0.5,0.8,1.0,1.5,2.0)",
    )
    parser.add_argument(
        "--regenerate-stats",
        action="store_true",
        help="Rebuild <run>/visualizations/performance_stats.json even when cached",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    baseline_file = resolve_baseline_path(args)

    if not baseline_file.is_file():
        print(f"[aggregate] baseline file not found: {baseline_file}", file=sys.stderr)
        return 2

    result = aggregate_runs(
        runs_root=runs_root,
        output_dir=output_dir,
        baseline_file=baseline_file,
        fast_p_thresholds=parse_fastp_values(args.fast_p_values),
        only_runs=args.runs,
        regenerate_stats=bool(args.regenerate_stats),
    )
    doc = result["doc"]

    print(f"[aggregate] runs_root={runs_root}")
    print(f"[aggregate] baseline_file={baseline_file}")
    print(
        f"[aggregate] discovered={doc['discovered']} aggregated={doc['aggregated']} "
        f"complete={doc['complete_runs']} partial={doc['partial_runs']}"
    )
    for record in doc["runs"]:
        performance = record["performance"]
        best = performance["speedup_best"]
        fast_p_1 = performance["fast_p_best"].get("1.0")
        outcomes = record["outcomes"]
        correct = outcomes["total_correct"]
        # Denominator must match correct_rate's basis so the two never disagree.
        denominator = (
            outcomes["total_completed"]
            if outcomes.get("correct_rate_basis") == "workspaces_finished"
            else outcomes["total_attempted"]
        )
        print(
            f"  - {record['run_name']} [{record['status']}] "
            f"ctx={record['config']['context_management']} "
            f"itr={record['max_iterations']} "
            f"correct={'?' if correct is None else correct}/{denominator} "
            f"best_geomean={best['geometric_mean']}(n={best['n']}) fast_p@1.0={fast_p_1} "
            f"l1={record['governance']['l1_entry_count']} stats={record['performance_stats_source']}"
        )
        for warning in record["warnings"]:
            print(f"      ! {warning}")
    for failure in doc["failures"]:
        print(f"  ! failed {failure['run_name']}: {failure['error']}", file=sys.stderr)
    for name in doc["requested_runs_not_found"]:
        print(f"  ! requested run not found under runs_root: {name}", file=sys.stderr)

    print(f"[aggregate] json={result['json_path']}")
    print(f"[aggregate] csv={result['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
