"""Analyze runs_evolving eval artifacts for the new evolving-agent integration.

This script supports both:
- level-first eval docs: eval_results.json => {"1": {"100": [entry], ...}, ...}
- per-level eval docs: eval_results_level_1.json => {"100": [entry], ...}

Outputs are written under:
- runs_evolving/<run_name>/analysis/

Example:
    uv run python scripts_integration/new_evolving_agent/analysis/analyze_evolving_run.py \
        --run-name memory_evolving_agent_2026_04_14_15_44 \
        --hardware SONG_CPU6_A6000x4 \
        --baseline baseline_time_torch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.score import fastp, geometric_mean_speed_ratio_correct_only

FAST_P_THRESHOLDS = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]


@dataclass
class EvalRecord:
    level: int
    problem_id: int
    compiled: bool
    correctness: bool
    runtime: float


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pick_sample_zero(entries: Any) -> dict[str, Any] | None:
    if isinstance(entries, dict):
        return entries
    if not isinstance(entries, list):
        return None

    sample_zero = [e for e in entries if isinstance(e, dict) and e.get("sample_id") == 0]
    if len(sample_zero) == 1:
        return sample_zero[0]
    if len(sample_zero) > 1:
        return sample_zero[-1]

    # Fallback if sample_id key is absent.
    for entry in entries:
        if isinstance(entry, dict):
            return entry
    return None


def _normalize_level_first_eval_payload(payload: Any) -> dict[int, dict[int, dict[str, Any]]]:
    """Normalize eval payload into {level: {problem_id: entry}}."""
    normalized: dict[int, dict[int, dict[str, Any]]] = {}

    if not isinstance(payload, dict):
        return normalized

    # Format A: level-first map, e.g. {"1": {"100": [entry]}}
    if payload and all(isinstance(v, dict) for v in payload.values()):
        for level_key, level_map in payload.items():
            try:
                level = int(level_key)
            except Exception:
                continue
            if not isinstance(level_map, dict):
                continue
            for pid_key, entries in level_map.items():
                try:
                    pid = int(pid_key)
                except Exception:
                    continue
                entry = _pick_sample_zero(entries)
                if entry is None:
                    continue
                normalized.setdefault(level, {})[pid] = entry
        return normalized

    # Format B: pid-first map, e.g. {"100": [entry]} where level in metadata.
    for pid_key, entries in payload.items():
        try:
            pid = int(pid_key)
        except Exception:
            continue
        entry = _pick_sample_zero(entries)
        if entry is None:
            continue

        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        level_raw = metadata.get("level")
        if level_raw is None:
            continue

        try:
            level = int(level_raw)
        except Exception:
            continue

        normalized.setdefault(level, {})[pid] = entry

    return normalized


def _load_eval_records(run_dir: Path) -> dict[int, dict[int, dict[str, Any]]]:
    """Load and merge eval records from eval_results.json and eval_results_level_*.json."""
    merged: dict[int, dict[int, dict[str, Any]]] = {}

    eval_results_path = run_dir / "eval_results.json"
    level_first_payload = _read_json(eval_results_path, default={})
    level_first_map = _normalize_level_first_eval_payload(level_first_payload)
    for level, level_map in level_first_map.items():
        merged.setdefault(level, {}).update(level_map)

    for path in sorted(run_dir.glob("eval_results_level_*.json")):
        match = re.match(r"eval_results_level_(\d+)\.json$", path.name)
        if not match:
            continue
        level = int(match.group(1))

        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            continue

        for pid_key, entries in payload.items():
            try:
                pid = int(pid_key)
            except Exception:
                continue
            entry = _pick_sample_zero(entries)
            if entry is None:
                continue
            # per-level file is authoritative for this level/pid
            merged.setdefault(level, {})[pid] = entry

    return merged


def _build_baseline_lookup(baseline_results: dict[str, Any], level: int) -> dict[int, float]:
    level_key = f"level{level}"
    level_baseline = baseline_results.get(level_key)
    if not isinstance(level_baseline, dict):
        return {}

    dataset = construct_kernelbench_dataset(level=level, source="local")
    lookup: dict[int, float] = {}
    for pid in dataset.get_problem_ids():
        problem = dataset.get_problem_by_id(pid)
        baseline_entry = level_baseline.get(problem.name)
        if not isinstance(baseline_entry, dict):
            continue
        # median, falling back to mean for pre-median baselines
        value = baseline_entry.get("median", baseline_entry.get("mean"))
        try:
            value_float = float(value)
        except Exception:
            continue
        lookup[int(pid)] = value_float
    return lookup


def _to_eval_record(level: int, problem_id: int, entry: dict[str, Any]) -> EvalRecord:
    runtime_raw = entry.get("runtime", -1.0)
    try:
        runtime = float(runtime_raw)
    except Exception:
        runtime = -1.0

    return EvalRecord(
        level=level,
        problem_id=problem_id,
        compiled=bool(entry.get("compiled", False)),
        correctness=bool(entry.get("correctness", False)),
        runtime=runtime,
    )


def _compute_metrics(level: int | None, records: list[EvalRecord], baseline_lookup: dict[int, float]) -> dict[str, Any]:
    total_count = len(records)
    compiled_count = sum(1 for r in records if r.compiled)
    correct_count = sum(1 for r in records if r.correctness)

    aligned_records = [r for r in records if r.problem_id in baseline_lookup]
    missing_baseline_problem_ids = sorted(
        r.problem_id for r in records if r.problem_id not in baseline_lookup
    )

    is_correct = np.array([bool(r.correctness) for r in aligned_records])
    baseline_speed = np.array([float(baseline_lookup[r.problem_id]) for r in aligned_records])
    actual_speed = np.array([float(r.runtime) for r in aligned_records])
    n = len(aligned_records)

    if n > 0:
        geo_mean_speedup = float(
            geometric_mean_speed_ratio_correct_only(
                is_correct,
                baseline_speed,
                actual_speed,
                n,
            )
        )
        fast_p = {
            str(p): float(fastp(is_correct, baseline_speed, actual_speed, n, p))
            for p in FAST_P_THRESHOLDS
        }
    else:
        geo_mean_speedup = 0.0
        fast_p = {str(p): 0.0 for p in FAST_P_THRESHOLDS}

    result = {
        "total_count": total_count,
        "compiled_count": compiled_count,
        "correct_count": correct_count,
        "compilation_rate": (compiled_count / total_count) if total_count > 0 else 0.0,
        "correctness_rate": (correct_count / total_count) if total_count > 0 else 0.0,
        "geo_mean_speedup": geo_mean_speedup,
        "fast_p": fast_p,
        "aligned_count": n,
        "missing_baseline_problem_ids": missing_baseline_problem_ids,
    }

    if level is not None:
        result["level"] = int(level)

    return result


def _resolve_baseline_path(args: argparse.Namespace) -> Path:
    if args.baseline_file:
        return Path(args.baseline_file)
    return Path("results") / "timing" / args.hardware / f"{args.baseline}.json"


def analyze_run(
    *,
    run_name: str,
    runs_root: Path,
    baseline_file: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = runs_root / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if not baseline_file.is_file():
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    eval_by_level = _load_eval_records(run_dir)
    if not eval_by_level:
        raise RuntimeError(
            f"No eval entries found in {run_dir}. Expected eval_results.json and/or eval_results_level_*.json"
        )

    baseline_results = _read_json(baseline_file, default={})
    if not isinstance(baseline_results, dict):
        raise RuntimeError(f"Invalid baseline JSON shape in: {baseline_file}")

    per_level_metrics: dict[str, dict[str, Any]] = {}
    all_records: list[EvalRecord] = []
    all_baseline_lookup: dict[int, float] = {}

    for level in sorted(eval_by_level.keys()):
        level_entries = eval_by_level[level]
        level_baseline_lookup = _build_baseline_lookup(baseline_results, level)

        level_records = [
            _to_eval_record(level, pid, entry)
            for pid, entry in sorted(level_entries.items(), key=lambda kv: kv[0])
        ]
        all_records.extend(level_records)

        # Build a level-encoded baseline key for full-subset aggregation.
        for pid, baseline_mean in level_baseline_lookup.items():
            all_baseline_lookup[(level * 1000) + pid] = baseline_mean

        level_metrics = _compute_metrics(level, level_records, level_baseline_lookup)
        per_level_metrics[str(level)] = level_metrics

    full_subset_records = [
        EvalRecord(
            level=r.level,
            problem_id=(r.level * 1000) + r.problem_id,
            compiled=r.compiled,
            correctness=r.correctness,
            runtime=r.runtime,
        )
        for r in all_records
    ]

    full_subset_metrics = _compute_metrics(None, full_subset_records, all_baseline_lookup)

    analysis_doc = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "baseline_file": str(baseline_file),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "levels": per_level_metrics,
        "full_subset": full_subset_metrics,
    }

    if output_dir is None:
        output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / "analysis_results.json"
    _write_json(aggregate_path, analysis_doc)

    for level_key, metrics in per_level_metrics.items():
        _write_json(output_dir / f"level_{level_key}_metrics.json", metrics)

    return {
        "analysis_doc": analysis_doc,
        "output_dir": str(output_dir),
        "aggregate_path": str(aggregate_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze runs_evolving eval results for new evolving agent.")
    parser.add_argument("--run-name", type=str, required=True, help="Run folder name under runs_root")
    parser.add_argument("--runs-root", type=str, default="runs_evolving", help="Root directory containing run folders")
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=None,
        help="Direct path to baseline timing JSON (overrides --hardware/--baseline)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="SONG_CPU6_A6000x4",
        help="Hardware folder under results/timing when --baseline-file is not provided",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline_time_torch",
        help="Baseline filename stem under results/timing/<hardware>/ when --baseline-file is not provided",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write output JSON files (default: runs_root/run_name/analysis)",
    )

    args = parser.parse_args()

    baseline_path = _resolve_baseline_path(args)
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = analyze_run(
        run_name=args.run_name,
        runs_root=Path(args.runs_root),
        baseline_file=baseline_path,
        output_dir=output_dir,
    )

    print(json.dumps(result["analysis_doc"], indent=2))
    print(f"\n[analysis] output_dir={result['output_dir']}")
    print(f"[analysis] aggregate={result['aggregate_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
