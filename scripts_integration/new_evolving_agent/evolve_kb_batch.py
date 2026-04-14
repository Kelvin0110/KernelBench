"""Batch orchestrator for evolving-agent KernelBench integration (new governor path)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Support direct execution: python scripts_integration/new_evolving_agent/evolve_kb_batch.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from scripts_integration.new_evolving_agent.kb_governor import (
    KBGovernorConfig,
    governor_result_to_dict,
    safe_run_kb_governor,
)


def _load_subset_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            level = int(row["level"])
            problem_id = int(row["problem_id"])
            rows.append({"level": level, "problem_id": problem_id, **row})
    return rows


def _read_json(path: Path, default: dict | list):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict | list) -> None:
    def _json_default(obj: Any) -> str:
        if isinstance(obj, BaseException):
            return f"{type(obj).__name__}: {obj}"
        if isinstance(obj, Path):
            return str(obj)
        return repr(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _to_kernelbench_eval_entry(run_entry: dict[str, Any], *, level: int, problem_id: str) -> dict[str, Any]:
    runtime_stats = run_entry.get("runtime_stats")
    if not isinstance(runtime_stats, dict):
        runtime_stats = {}

    raw_metadata = run_entry.get("metadata")
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}

    runtime = run_entry.get("runtime", -1.0)
    try:
        runtime = float(runtime)
    except Exception:
        runtime = -1.0

    merged_metadata = {
        "hardware": raw_metadata.get("hardware") or runtime_stats.get("hardware"),
        "device": raw_metadata.get("device") or runtime_stats.get("device"),
        "correctness_trials": raw_metadata.get("correctness_trials"),
        "source": "new_evolving_agent",
        "level": int(level),
        "problem_id": int(problem_id),
        "best_speedup": float(run_entry.get("best_speedup", 0.0) or 0.0),
        "backend": run_entry.get("backend"),
        "precision": run_entry.get("precision"),
        "iterations_run": int(run_entry.get("iterations_run", 0) or 0),
        "error": run_entry.get("error"),
    }

    return {
        "sample_id": 0,
        "compiled": bool(run_entry.get("best_compiled", False)),
        "correctness": bool(run_entry.get("best_correct", False)),
        "metadata": merged_metadata,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
    }


def _normalize_level_first_eval_doc(payload: dict | list | None) -> dict[str, dict[str, list]]:
    if not isinstance(payload, dict):
        return {}

    if payload and all(isinstance(v, dict) for v in payload.values()):
        return payload

    normalized: dict[str, dict[str, list]] = {}
    for problem_id, entries in payload.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            level_value = metadata.get("level")
            if level_value is None:
                continue
            level_key = str(level_value)
            pid_key = str(problem_id)
            normalized.setdefault(level_key, {})
            normalized[level_key].setdefault(pid_key, [])
            normalized[level_key][pid_key].append(entry)
    return normalized


def _level_eval_path(run_dir: Path, level: int) -> Path:
    return run_dir / f"eval_results_level_{level}.json"


def _extract_best_kernel_code(run_entry: dict[str, Any]) -> str | None:
    best_code = run_entry.get("best_code")
    if isinstance(best_code, str) and best_code.strip():
        return best_code

    records = run_entry.get("records")
    if isinstance(records, list):
        for record in reversed(records):
            if not isinstance(record, dict):
                continue
            candidate = record.get("candidate_code")
            if isinstance(candidate, str) and candidate.strip():
                return candidate

    return None


def _write_kernel_export(run_dir: Path, *, level: int, problem_id: str, code: str) -> Path:
    kernels_dir = run_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    export_path = kernels_dir / f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
    export_path.write_text(code, encoding="utf-8")
    return export_path


def _summarize_per_level_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_level: dict[str, dict[str, Any]] = {}
    for entry in runs:
        level = str(entry.get("level"))
        bucket = per_level.setdefault(
            level,
            {
                "completed": 0,
                "correct": 0,
                "best_speedup": 0.0,
                "best_runtime": None,
            },
        )
        bucket["completed"] += 1
        if entry.get("best_correct"):
            bucket["correct"] += 1
        bucket["best_speedup"] = max(
            float(bucket["best_speedup"]),
            float(entry.get("best_speedup", 0.0) or 0.0),
        )
        runtime = entry.get("runtime")
        try:
            runtime_f = float(runtime)
        except Exception:
            runtime_f = -1.0
        if runtime_f >= 0:
            current = bucket.get("best_runtime")
            bucket["best_runtime"] = runtime_f if current is None else min(float(current), runtime_f)
    return per_level


def _check_integration_dependencies(*, dry_run: bool) -> None:
    missing: list[str] = []
    if "kernelbench" not in sys.modules:
        try:
            __import__("kernelbench.dataset")
            __import__("kernelbench.prompt_constructor_toml")
        except Exception:
            missing.append("kernelbench package")

    try:
        __import__("evolving_common")
    except Exception:
        missing.append("Self-Evolving-Agent/evolving_common")

    if missing:
        raise RuntimeError(
            "missing integration dependencies: "
            + ", ".join(missing)
            + ". Ensure repository root is installed with `uv pip install -e .` and "
            "Self-Evolving-Agent deps are installed."
        )

    if not dry_run and not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY is required for non-dry runs because llm_client uses NVIDIA OpenAI-compatible API."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run new evolving-agent KernelBench batch.")
    parser.add_argument(
        "--subset-csv",
        type=str,
        default="subset_selection/selected_problems_50.csv",
        help="CSV from subset_selection/select_problems.py",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV parsing/output generation without LLM or GPU eval.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="runs_evolving/",
        help="Base directory for run artifacts",
    )
    parser.add_argument(
        "--time-sample-interval-sec",
        type=float,
        default=300.0,
        help="Recorder sampling interval for metrics_by_time.jsonl.",
    )
    args = parser.parse_args()

    # Append UTC timestamp to run-name to avoid collisions and make runs unique.
    # Format: YYYY_MM_DD_HH_MM (year_month_day_hour_minute)
    now_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M")
    args.run_name = f"{args.run_name}_{now_str}"

    _check_integration_dependencies(dry_run=args.dry_run)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.dry_run:
        raise RuntimeError(
            "CUDA GPU is required for this integration run. Use --dry-run for local validation."
        )

    subset_csv = Path(args.subset_csv)
    if not subset_csv.is_file():
        raise FileNotFoundError(f"Subset CSV not found: {subset_csv}")

    rows = _load_subset_rows(subset_csv)
    if args.max_problems > 0:
        rows = rows[: args.max_problems]

    run_dir = Path(args.results_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    shared_l1_path = run_dir / "shared_l1.txt"
    if not shared_l1_path.exists():
        shared_l1_path.write_text(
            "# Shared L1 journal for evolving KernelBench batch\n",
            encoding="utf-8",
        )

    eval_path = run_dir / "eval_results.json"
    evolving_runs_path = run_dir / "evolving_runs.json"
    summary_path = run_dir / "run_summary.json"

    evolving_doc_raw = _read_json(evolving_runs_path, default={"runs": []})
    if isinstance(evolving_doc_raw, list):
        evolving_doc = {"runs": evolving_doc_raw}
    elif isinstance(evolving_doc_raw, dict):
        evolving_doc = evolving_doc_raw
    else:
        evolving_doc = {"runs": []}
    runs_raw = evolving_doc.get("runs", [])
    runs = runs_raw if isinstance(runs_raw, list) else []
    completed_keys = {
        f"L{entry['level']}P{entry['problem_id']}"
        for entry in runs
        if isinstance(entry, dict) and "level" in entry and "problem_id" in entry
    }
    eval_doc = _normalize_level_first_eval_doc(_read_json(eval_path, default={}))
    level_eval_docs: dict[int, dict[str, list]] = {}

    for idx, row in enumerate(rows, start=1):
        level = int(row["level"])
        problem_id = str(int(row["problem_id"]))
        key = f"L{level}P{problem_id}"

        if key in completed_keys:
            print(f"[batch] skip completed {key}")
            continue

        row_backend = (row.get("backend") or "").strip().lower()
        backend = row_backend if row_backend else args.backend

        print(
            f"[batch] ({idx}/{len(rows)}) running L{level}P{problem_id} "
            f"backend={backend} precision={args.precision}"
        )

        if args.dry_run:
            entry = {
                "level": level,
                "problem_id": problem_id,
                "backend": backend,
                "precision": args.precision,
                "best_speedup": 0.0,
                "best_correct": False,
                "best_compiled": False,
                "best_code_path": None,
                "best_code": None,
                "iterations_run": 0,
                "records": [],
                "runtime": -1.0,
                "runtime_stats": {},
                "metadata": {},
                "error": "dry_run_no_gpu_execution",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        else:
            dataset = construct_kernelbench_dataset(level=level, source="local")
            problem = dataset.get_problem_by_id(int(problem_id))
            task_prompt = get_prompt_for_backend(
                ref_arch_src=problem.code,
                backend=backend,
                option="one_shot",
                precision=args.precision,
            )
            cfg = KBGovernorConfig(
                run_name=args.run_name,
                level=level,
                problem_id=problem_id,
                backend=backend,
                precision=args.precision,
                max_iterations=args.max_iterations,
                shared_l1_path=shared_l1_path,
                results_root=Path(args.results_root),
                reference_code=problem.code,
                run_recorder_time_sample_interval_sec=args.time_sample_interval_sec,
                verbose=True,
            )
            result = safe_run_kb_governor(cfg, task_prompt=task_prompt)
            entry = governor_result_to_dict(result)
            entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        export_code = _extract_best_kernel_code(entry)
        if export_code:
            export_path = _write_kernel_export(run_dir, level=level, problem_id=problem_id, code=export_code)
            entry["exported_kernel_path"] = str(export_path)
            if not entry.get("best_code_path"):
                entry["best_code_path"] = str(export_path)

        runs.append(entry)
        completed_keys.add(key)
        level_key = str(level)
        pid_key = str(problem_id)
        eval_doc.setdefault(level_key, {})
        eval_doc[level_key].setdefault(pid_key, [])
        eval_doc[level_key][pid_key].append(
            _to_kernelbench_eval_entry(entry, level=level, problem_id=problem_id)
        )

        if level not in level_eval_docs:
            level_eval_docs[level] = _read_json(_level_eval_path(run_dir, level), default={})
        level_eval_docs[level].setdefault(pid_key, [])
        level_eval_docs[level][pid_key].append(
            _to_kernelbench_eval_entry(entry, level=level, problem_id=problem_id)
        )

        evolving_doc["runs"] = runs
        _write_json(evolving_runs_path, evolving_doc)
        _write_json(eval_path, eval_doc)
        _write_json(_level_eval_path(run_dir, level), level_eval_docs[level])

    successful = [e for e in runs if e.get("best_correct")]
    best_overall = 0.0
    if successful:
        best_overall = max(float(e.get("best_speedup", 0.0)) for e in successful)

    best_runtime_overall: float | None = None
    runtime_candidates: list[float] = []
    for entry in runs:
        runtime = entry.get("runtime")
        try:
            runtime_f = float(runtime)
        except Exception:
            continue
        if runtime_f >= 0:
            runtime_candidates.append(runtime_f)
    if runtime_candidates:
        best_runtime_overall = min(runtime_candidates)

    summary = {
        "run_name": args.run_name,
        "subset_csv": str(subset_csv),
        "dry_run": bool(args.dry_run),
        "cuda_available": bool(has_cuda),
        "total_attempted": len(rows),
        "total_completed": len(runs),
        "total_correct": len(successful),
        "best_speedup_overall": best_overall,
        "best_runtime_overall": best_runtime_overall,
        "per_level_summary": _summarize_per_level_runs(runs),
        "results_path": str(eval_path),
        "evolving_runs_path": str(evolving_runs_path),
        "per_level_results": {
            str(level): str(_level_eval_path(run_dir, level))
            for level in sorted(level_eval_docs.keys())
        },
        "shared_l1_path": str(shared_l1_path),
    }
    _write_json(summary_path, summary)

    print("[batch] complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
