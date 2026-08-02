"""Re-apply narrowed is_hack policy to a completed evolving run (offline, no GPU).

Re-runs static checks on stored candidate code, replays governor best-tracking,
and rebuilds batch aggregates (evolving_runs, eval_results, run_summary, viz stats).

Example::

    uv run python scripts_integration/new_evolving_agent/repair/repair_is_hack_policy.py \\
        --run-name base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr10_2026_07_09_15_16 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
for _path in (REPO_ROOT, SEA_ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from kernelbench.performance_stats import DEFAULT_FAST_P_THRESHOLDS, min_non_outlier_runtime
from kernelbench_integration import static_check

_BATCH_DIR = Path(__file__).resolve().parents[1]
if str(_BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BATCH_DIR))
import evolve_kb_batch as batch_module

_to_kernelbench_eval_entry = batch_module._to_kernelbench_eval_entry
_summarize_per_level_runs = batch_module._summarize_per_level_runs
_collect_suspicious_speedup_problems = batch_module._collect_suspicious_speedup_problems
_level_eval_path = batch_module._level_eval_path
_extract_best_kernel_code = batch_module._extract_best_kernel_code
_write_kernel_export = batch_module._write_kernel_export
_read_json = batch_module._read_json
_write_json = batch_module._write_json

REPAIR_POLICY = "strict_or_excessive_speedup"


@dataclass
class ProblemChangeStats:
    iterations_is_hack_cleared: int = 0
    iterations_is_hack_set: int = 0
    best_changed: bool = False
    best_attempt: int | None = None
    replay_rows: list[dict[str, Any]] = field(default_factory=list)


def _resolve_run_dir(*, runs_root: Path, run_name: str | None, run_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = run_dir.resolve()
    elif run_name:
        resolved = (runs_root / run_name).resolve()
    else:
        raise ValueError("Provide --run-name or --run-dir")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
    return resolved


def _backup_file(path: Path, *, backup: bool, timestamp: str) -> None:
    if backup and path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak.{timestamp}"))


def _load_generate_run_performance_stats():
    script_path = (
        REPO_ROOT
        / "Self-Evolving-Agent"
        / "visualizations"
        / "kernelbench"
        / "server"
        / "generate_run_performance_stats.py"
    )
    spec = importlib.util.spec_from_file_location("generate_run_performance_stats", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_baseline_file() -> Path:
    return REPO_ROOT / "results" / "timing" / "SONG_CPU6_A6000x4" / "baseline_time_torch.json"


def _recompute_record_evaluation(
    record: dict[str, Any],
    *,
    backend: str,
    precision: str,
    enable_static_check: bool,
) -> tuple[dict[str, Any], bool]:
    evaluation = deepcopy(record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {})
    old_is_hack = bool(evaluation.get("is_hack", False))
    metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
    metadata = dict(metadata)

    code = record.get("candidate_code")
    static_errors: list[str] = []
    static_warnings: list[str] = []

    if enable_static_check and isinstance(code, str) and code.strip():
        valid, static_errors, static_warnings = static_check.run_static_check(
            code.strip(),
            backend=backend,
            precision=precision,
        )
        if not valid:
            static_check_warnings = static_check.collect_static_check_warnings(static_errors, static_warnings)
            is_hack = True
            metadata["static_check_errors"] = list(static_errors)
            metadata["static_check_warnings"] = list(static_check_warnings)
            evaluation["is_hack"] = is_hack
            evaluation["static_check_warnings"] = static_check_warnings
            evaluation["metadata"] = metadata
            return evaluation, old_is_hack != is_hack

    static_check_warnings = static_check.collect_static_check_warnings([], static_warnings)
    is_hack = static_check.resolve_is_hack(static_warnings=static_warnings, metadata=metadata)
    if static_check_warnings:
        metadata["static_check_warnings"] = list(static_check_warnings)
    elif "static_check_warnings" in metadata:
        metadata.pop("static_check_warnings", None)
    if "static_check_errors" in metadata and enable_static_check:
        metadata.pop("static_check_errors", None)

    evaluation["is_hack"] = is_hack
    evaluation["static_check_warnings"] = static_check_warnings
    evaluation["metadata"] = metadata
    return evaluation, old_is_hack != is_hack


def recompute_run_entry(
    run_entry: dict[str, Any],
    *,
    enable_static_check: bool = True,
) -> tuple[dict[str, Any], ProblemChangeStats]:
    updated = deepcopy(run_entry)
    stats = ProblemChangeStats()

    old_best_runtime = updated.get("runtime", -1.0)
    try:
        old_best_runtime_f = float(old_best_runtime)
    except Exception:
        old_best_runtime_f = -1.0
    old_best_correct = bool(updated.get("best_correct", False))
    old_best_speedup = float(updated.get("best_speedup", 0.0) or 0.0)

    backend = str(updated.get("backend") or "cuda")
    precision = str(updated.get("precision") or "fp32")
    records = updated.get("records")
    if not isinstance(records, list):
        records = []

    sorted_records = sorted(
        [r for r in records if isinstance(r, dict)],
        key=lambda r: int(r.get("attempt") or 0),
    )

    best_speedup = 0.0
    best_correct = False
    best_compiled = False
    best_runtime = -1.0
    best_code: str | None = None
    best_code_path: str | None = None
    best_metadata: dict[str, Any] = {}
    best_runtime_stats: dict[str, Any] = {}
    best_attempt: int | None = None
    run_had_hack = False

    new_records: list[dict[str, Any]] = []
    for record in sorted_records:
        record_copy = deepcopy(record)
        evaluation, hack_changed = _recompute_record_evaluation(
            record_copy,
            backend=backend,
            precision=precision,
            enable_static_check=enable_static_check,
        )
        record_copy["evaluation"] = evaluation
        new_records.append(record_copy)

        is_hack = bool(evaluation.get("is_hack", False))
        if is_hack:
            run_had_hack = True
        if hack_changed:
            if bool(evaluation.get("is_hack", False)):
                stats.iterations_is_hack_set += 1
            else:
                stats.iterations_is_hack_cleared += 1

        attempt = int(record_copy.get("attempt") or 0)
        compiled = bool(evaluation.get("compiled", False))
        correct = bool(evaluation.get("correct", False))
        metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
        runtime_raw = evaluation.get("runtime")
        runtime_value: float | None
        try:
            runtime_value = float(runtime_raw) if runtime_raw is not None and float(runtime_raw) >= 0 else None
        except Exception:
            runtime_value = None

        speedup_raw = evaluation.get("speedup")
        try:
            speedup_value = float(speedup_raw) if speedup_raw is not None else 0.0
        except Exception:
            speedup_value = 0.0

        metrics_iteration = {
            "attempt": attempt,
            "compiled": compiled,
            "correct": correct,
            "speedup": speedup_value,
            "is_hack": is_hack,
            "static_check_warnings": list(evaluation.get("static_check_warnings") or []),
            "runtime": runtime_value,
            "ref_runtime": evaluation.get("ref_runtime"),
            "error": evaluation.get("error_message"),
        }

        is_new_best = (
            correct
            and runtime_value is not None
            and runtime_value >= 0
            and not is_hack
            and not bool(metadata.get("excessive_speedup"))
            and (best_runtime < 0 or runtime_value < best_runtime)
        )
        if is_new_best:
            best_speedup = speedup_value
            best_correct = True
            best_compiled = compiled
            best_runtime = float(runtime_value)
            best_runtime_stats = (
                dict(evaluation.get("runtime_stats"))
                if isinstance(evaluation.get("runtime_stats"), dict)
                else {}
            )
            best_metadata = dict(metadata)
            candidate = record_copy.get("candidate_code")
            best_code = candidate if isinstance(candidate, str) else None
            best_attempt = attempt

        metrics_best = {
            "compiled": best_compiled,
            "correct": best_correct,
            "speedup": best_speedup,
            "runtime": best_runtime if best_runtime >= 0 else None,
            "is_hack": run_had_hack,
        }
        stats.replay_rows.append(
            {
                "iteration": attempt,
                "metrics_iteration": metrics_iteration,
                "metrics_best": metrics_best,
            }
        )

    updated["records"] = new_records
    updated["best_speedup"] = best_speedup
    updated["best_correct"] = best_correct
    updated["best_compiled"] = best_compiled
    updated["best_is_hack"] = run_had_hack
    updated["runtime"] = best_runtime
    updated["runtime_stats"] = best_runtime_stats
    updated["metadata"] = best_metadata
    updated["best_code"] = best_code
    stats.best_attempt = best_attempt
    if best_attempt is not None:
        updated["best_code_path"] = None  # set by repair_run with absolute path
    else:
        updated["best_code_path"] = None

    new_best_runtime_f = float(best_runtime)
    stats.best_changed = (
        old_best_correct != best_correct
        or abs(old_best_speedup - best_speedup) > 1e-9
        or (
            (old_best_runtime_f >= 0 or new_best_runtime_f >= 0)
            and abs(old_best_runtime_f - new_best_runtime_f) > 1e-9
        )
    )

    return updated, stats


def sync_workspace_metrics(
    workspace_dir: Path,
    replay_rows: list[dict[str, Any]],
    *,
    best_code: str | None,
    best_attempt: int | None,
    dry_run: bool,
) -> bool:
    metrics_path = workspace_dir / "metrics_by_iteration.jsonl"
    if not metrics_path.is_file():
        return False

    replay_by_iter = {int(row["iteration"]): row for row in replay_rows}
    out_lines: list[str] = []
    changed = False
    for raw in metrics_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        iteration = int(row.get("iteration") or 0)
        replay = replay_by_iter.get(iteration)
        if replay is not None:
            if row.get("metrics_iteration") != replay["metrics_iteration"]:
                changed = True
            if row.get("metrics_best") != replay["metrics_best"]:
                changed = True
            row["metrics_iteration"] = replay["metrics_iteration"]
            row["metrics_best"] = replay["metrics_best"]
        out_lines.append(json.dumps(row, ensure_ascii=False))

    if changed and not dry_run:
        metrics_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if best_code and best_attempt is not None and not dry_run:
        for old_best in workspace_dir.glob("best_iter_*.py"):
            old_best.unlink(missing_ok=True)
        (workspace_dir / f"best_iter_{best_attempt}.py").write_text(best_code, encoding="utf-8")
        changed = True
    elif best_attempt is None and not dry_run:
        for old_best in workspace_dir.glob("best_iter_*.py"):
            old_best.unlink(missing_ok=True)
            changed = True

    return changed


def rebuild_batch_artifacts(
    run_dir: Path,
    runs: list[dict[str, Any]],
    *,
    run_summary: dict[str, Any],
    baseline_results: dict[str, Any] | None,
    repair_stats: dict[str, Any],
) -> dict[str, Any]:
    eval_doc: dict[str, dict[str, list]] = {}
    level_eval_docs: dict[int, dict[str, list]] = {}

    for entry in runs:
        if not isinstance(entry, dict):
            continue
        level = int(entry.get("level"))
        problem_id = str(entry.get("problem_id"))
        eval_entry = _to_kernelbench_eval_entry(entry, level=level, problem_id=problem_id)
        eval_doc.setdefault(str(level), {})[problem_id] = [eval_entry]
        level_eval_docs.setdefault(level, {})[problem_id] = [eval_entry]

    _write_json(run_dir / "eval_results.json", eval_doc)
    for level, level_doc in level_eval_docs.items():
        _write_json(_level_eval_path(run_dir, level), level_doc)

    successful = [e for e in runs if isinstance(e, dict) and e.get("best_correct")]
    suspicious_speedup_problems, likely_hack_keys = _collect_suspicious_speedup_problems(
        runs,
        baseline_results=baseline_results if isinstance(baseline_results, dict) else None,
    )

    best_overall = 0.0
    best_runtime_overall: float | None = None
    runtime_candidates: list[float] = []
    for entry in successful:
        entry_key = f"L{entry.get('level')}P{entry.get('problem_id')}"
        if entry_key in likely_hack_keys:
            continue
        try:
            runtime_f = float(entry.get("runtime"))
        except Exception:
            continue
        if runtime_f >= 0:
            runtime_candidates.append(runtime_f)

    if runtime_candidates:
        best_runtime_overall = min_non_outlier_runtime(runtime_candidates)
        if best_runtime_overall is not None:
            for entry in successful:
                entry_key = f"L{entry.get('level')}P{entry.get('problem_id')}"
                if entry_key in likely_hack_keys:
                    continue
                try:
                    runtime_f = float(entry.get("runtime"))
                except Exception:
                    continue
                if runtime_f == best_runtime_overall:
                    best_overall = float(entry.get("best_speedup", 0.0) or 0.0)
                    break

    per_level_results = {
        str(level): str(_level_eval_path(run_dir, level))
        for level in sorted(level_eval_docs.keys())
    }

    updated_summary = dict(run_summary)
    updated_summary["total_correct"] = len(successful)
    updated_summary["best_speedup_overall"] = best_overall
    updated_summary["best_runtime_overall"] = best_runtime_overall
    updated_summary["per_level_summary"] = _summarize_per_level_runs(
        runs,
        likely_hack_keys=likely_hack_keys,
    )
    updated_summary["suspicious_speedup_problems"] = suspicious_speedup_problems
    updated_summary["suspicious_speedup_count"] = len(suspicious_speedup_problems)
    updated_summary["results_path"] = str(run_dir / "eval_results.json")
    updated_summary["evolving_runs_path"] = str(run_dir / "evolving_runs.json")
    updated_summary["per_level_results"] = per_level_results
    updated_summary["is_hack_policy_repair"] = {
        "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": REPAIR_POLICY,
        **repair_stats,
    }
    _write_json(run_dir / "run_summary.json", updated_summary)
    return updated_summary


def repair_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
    skip_viz: bool = False,
    baseline_file: Path | None = None,
    enable_static_check: bool | None = None,
) -> dict[str, Any]:
    evolving_runs_path = run_dir / "evolving_runs.json"
    summary_path = run_dir / "run_summary.json"
    if not evolving_runs_path.is_file():
        raise FileNotFoundError(f"Missing evolving_runs.json: {evolving_runs_path}")

    evolving_doc = _read_json(evolving_runs_path, default={"runs": []})
    runs = evolving_doc.get("runs") if isinstance(evolving_doc, dict) else []
    if not isinstance(runs, list):
        raise RuntimeError(f"Invalid evolving_runs.json shape: {evolving_runs_path}")

    run_summary = _read_json(summary_path, default={})
    if not isinstance(run_summary, dict):
        run_summary = {}

    if enable_static_check is None:
        enable_static_check = bool(run_summary.get("enable_static_check", True))

    baseline_path = baseline_file or _default_baseline_file()
    baseline_results = _read_json(baseline_path, default={})
    if not isinstance(baseline_results, dict):
        baseline_results = {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_hack_cleared = 0
    total_hack_set = 0
    problems_best_changed = 0
    updated_runs: list[dict[str, Any]] = []

    for entry in runs:
        if not isinstance(entry, dict):
            continue
        level = int(entry.get("level"))
        problem_id = str(entry.get("problem_id"))
        entry["run_name"] = run_dir.name
        updated_entry, stats = recompute_run_entry(entry, enable_static_check=enable_static_check)
        if stats.best_attempt is not None and isinstance(updated_entry.get("best_code"), str):
            updated_entry["best_code_path"] = str(
                run_dir
                / "workspaces"
                / f"level_{level}_problem_{problem_id}"
                / f"best_iter_{stats.best_attempt}.py"
            )
        updated_runs.append(updated_entry)
        total_hack_cleared += stats.iterations_is_hack_cleared
        total_hack_set += stats.iterations_is_hack_set
        if stats.best_changed:
            problems_best_changed += 1

        workspace_dir = run_dir / "workspaces" / f"level_{level}_problem_{problem_id}"
        if workspace_dir.is_dir():
            sync_workspace_metrics(
                workspace_dir,
                stats.replay_rows,
                best_code=updated_entry.get("best_code") if isinstance(updated_entry.get("best_code"), str) else None,
                best_attempt=stats.best_attempt,
                dry_run=dry_run,
            )

        export_code = _extract_best_kernel_code(updated_entry)
        if export_code and not dry_run:
            _write_kernel_export(run_dir, level=level, problem_id=problem_id, code=export_code)

    repair_stats = {
        "problems_best_changed": problems_best_changed,
        "iterations_is_hack_cleared": total_hack_cleared,
        "iterations_is_hack_set": total_hack_set,
        "problems_processed": len(updated_runs),
    }

    if dry_run:
        return {
            "dry_run": True,
            "run_dir": str(run_dir),
            **repair_stats,
        }

    _backup_file(evolving_runs_path, backup=backup, timestamp=timestamp)
    _backup_file(summary_path, backup=backup, timestamp=timestamp)
    _backup_file(run_dir / "eval_results.json", backup=backup, timestamp=timestamp)
    for level_path in run_dir.glob("eval_results_level_*.json"):
        _backup_file(level_path, backup=backup, timestamp=timestamp)

    evolving_doc["runs"] = updated_runs
    _write_json(evolving_runs_path, evolving_doc)

    updated_summary = rebuild_batch_artifacts(
        run_dir,
        updated_runs,
        run_summary=run_summary,
        baseline_results=baseline_results,
        repair_stats=repair_stats,
    )

    viz_path: str | None = None
    if not skip_viz and baseline_path.is_file():
        gen_module = _load_generate_run_performance_stats()
        result = gen_module.build_performance_stats(
            run_name=run_dir.name,
            runs_root=run_dir.parent,
            baseline_file=baseline_path,
            fast_p_thresholds=list(DEFAULT_FAST_P_THRESHOLDS),
        )
        viz_path = result.get("output_path")

    return {
        "dry_run": False,
        "run_dir": str(run_dir),
        "viz_output_path": viz_path,
        "total_correct": updated_summary.get("total_correct"),
        "best_speedup_overall": updated_summary.get("best_speedup_overall"),
        **repair_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-apply narrowed is_hack policy to a completed evolving run (offline).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs_evolving",
        help="Root directory containing evolving run folders",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name under --runs-root")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit path to the run directory")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak.* copies before overwriting")
    parser.add_argument("--skip-viz", action="store_true", help="Skip performance_stats.json regeneration")
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Baseline timing JSON for run_summary / viz (default: results/timing/.../baseline_time_torch.json)",
    )
    parser.add_argument(
        "--no-static-check",
        action="store_true",
        help="Only use stored excessive_speedup metadata; do not re-run validate_kernel_static",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(
        runs_root=args.runs_root.resolve(),
        run_name=args.run_name,
        run_dir=args.run_dir,
    )
    stats = repair_run(
        run_dir,
        dry_run=bool(args.dry_run),
        backup=not args.no_backup,
        skip_viz=bool(args.skip_viz),
        baseline_file=args.baseline_file.resolve() if args.baseline_file else None,
        enable_static_check=not args.no_static_check,
    )

    mode = "DRY RUN" if stats.get("dry_run") else "UPDATED"
    print(f"[{mode}] run_dir={stats['run_dir']}")
    print(
        f"  problems_processed={stats['problems_processed']} "
        f"best_changed={stats['problems_best_changed']} "
        f"hack_cleared={stats['iterations_is_hack_cleared']} "
        f"hack_set={stats['iterations_is_hack_set']}"
    )
    if stats.get("total_correct") is not None:
        print(f"  total_correct={stats['total_correct']} best_speedup_overall={stats.get('best_speedup_overall')}")
    if stats.get("viz_output_path"):
        print(f"  viz_output_path={stats['viz_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
