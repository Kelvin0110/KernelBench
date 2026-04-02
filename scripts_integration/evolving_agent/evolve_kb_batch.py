"""Batch orchestrator for Self-Evolving-Agent x KernelBench prototype."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
# Support direct execution: python scripts_integration/evolving_agent/evolve_kb_batch.py
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from scripts_integration.evolving_agent.kb_evolving_governor import (
    KBGovernorConfig,
    governor_result_to_dict,
    safe_run_kb_governor,
)


def _load_subset_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _to_kernelbench_eval_entry(run_entry: dict, *, level: int, problem_id: int) -> dict:
    """Convert evolving run output into KernelBench-style eval entry shape."""
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
        "source": "evolving_agent_prototype",
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


def _normalize_level_first_eval_doc(payload: dict | list | None) -> dict:
    """Normalize eval json payload into level-first shape: {level: {problem_id: [entries]}}."""
    if not isinstance(payload, dict):
        return {}

    # Already level-first.
    if payload and all(isinstance(v, dict) for v in payload.values()):
        return payload

    # Legacy shape: {problem_id: [entries]}
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evolving-agent KernelBench batch.")
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
        "--eval-timeout-sec",
        type=float,
        default=300.0,
        help="Per-iteration isolated eval timeout in seconds; <=0 disables timeout.",
    )
    parser.add_argument(
        "--eval-start-method",
        type=str,
        default="spawn",
        help="Multiprocessing start method for isolated eval worker (spawn or fork).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV parsing and run wiring without executing GPU evaluation.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/evolving_logs",
        help="Base directory for run artifacts",
    )
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.dry_run:
        raise RuntimeError(
            "CUDA GPU is required for this prototype run. "
            "Use --dry-run for local non-GPU validation."
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
    level_eval_docs: dict[int, dict] = {}

    for idx, row in enumerate(rows, start=1):
        level = int(row["level"])
        problem_id = int(row["problem_id"])
        key = f"L{level}P{problem_id}"

        if key in completed_keys:
            print(f"[batch] skip completed {key}")
            continue

        row_backend = (row.get("backend") or "").strip().lower()
        backend = row_backend if row_backend else args.backend

        cfg = KBGovernorConfig(
            run_name=args.run_name,
            level=level,
            problem_id=problem_id,
            backend=backend,
            precision=args.precision,
            max_iterations=args.max_iterations,
            shared_l1_path=shared_l1_path,
            results_root=Path(args.results_root),
            eval_timeout_sec=args.eval_timeout_sec,
            eval_start_method=args.eval_start_method,
            verbose=True,
        )

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
                "iterations_run": 0,
                "records": [],
                "error": "dry_run_no_gpu_execution",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
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
            continue

        result = safe_run_kb_governor(cfg)
        entry = governor_result_to_dict(result)
        entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
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

    summary = {
        "run_name": args.run_name,
        "subset_csv": str(subset_csv),
        "dry_run": bool(args.dry_run),
        "cuda_available": bool(has_cuda),
        "total_attempted": len(rows),
        "total_completed": len(runs),
        "total_correct": len(successful),
        "best_speedup_overall": best_overall,
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
