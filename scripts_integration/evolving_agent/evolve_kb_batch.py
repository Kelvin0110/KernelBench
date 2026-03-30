"""Batch orchestrator for Self-Evolving-Agent x KernelBench prototype."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from scripts_integration.evolving_agent.kb_evolving_governor import (
        KBGovernorConfig,
        governor_result_to_dict,
        safe_run_kb_governor,
    )
except ModuleNotFoundError:
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
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-problems", type=int, default=10)
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
    summary_path = run_dir / "run_summary.json"

    eval_doc = _read_json(eval_path, default={"runs": []})
    runs = eval_doc.get("runs", [])
    completed_keys = {
        f"L{entry['level']}P{entry['problem_id']}"
        for entry in runs
        if isinstance(entry, dict) and "level" in entry and "problem_id" in entry
    }

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
            eval_doc["runs"] = runs
            _write_json(eval_path, eval_doc)
            continue

        result = safe_run_kb_governor(cfg)
        entry = governor_result_to_dict(result)
        entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        runs.append(entry)
        completed_keys.add(key)
        eval_doc["runs"] = runs
        _write_json(eval_path, eval_doc)

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
        "shared_l1_path": str(shared_l1_path),
    }
    _write_json(summary_path, summary)

    print("[batch] complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
