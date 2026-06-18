"""Aggregate fast-p checkpoint metrics for selected evolving and AIDE runs.

Reads cached visualization stats:
  - Evolving: runs_evolving/<run>/visualizations/performance_stats.json
  - AIDE: run_integration/analysis/aide_subset_run_<run_name>.json

Samples fast-p at thresholds 0.0, 1.0, 2.0 for iterations 10, 20, 30, ... through
the final iteration (always includes the last iteration when it is not a multiple of 10).

Output:
  scripts_integration/new_evolving_agent/analysis/aggregated_fast_p_checkpoints.json

Example:
    uv run python scripts_integration/new_evolving_agent/analysis/aggregate_fast_p_checkpoints.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernelbench.performance_stats import read_json, resolve_threshold_key, sanitize_model_name, write_json

# Edit these lists to choose which runs to aggregate.
EVOLVING_RUN_NAMES: list[str] = [
    "memory_evolving_agent_gen3_itr20_2026_06_04_11_34",
    "memory_evolving_agent_base_itr50_new_prompt_2026_06_13_13_00",
]

AIDE_RUN_NAMES: list[str] = [
    "aide_subset_gpt_oss_120b_step40_new_problem_set",
]

FAST_P_THRESHOLDS = [0.0, 1.0, 2.0]
ITERATION_STRIDE = 10
FAST_P_FIELD = "fast_p_best"

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "scripts_integration"
    / "new_evolving_agent"
    / "analysis"
    / "aggregated_fast_p_checkpoints.json"
)


def _checkpoint_iterations(max_iteration: int, *, stride: int = ITERATION_STRIDE) -> list[int]:
    if max_iteration < 1:
        return []
    checkpoints = list(range(stride, max_iteration + 1, stride))
    if not checkpoints or checkpoints[-1] != max_iteration:
        checkpoints.append(max_iteration)
    return sorted(set(checkpoints))


def _fast_p_at_iteration(
    iteration_doc: dict[str, Any],
    *,
    thresholds: list[float],
    fast_p_field: str,
) -> dict[str, float]:
    fast_p_map = iteration_doc.get(fast_p_field)
    if not isinstance(fast_p_map, dict):
        fast_p_map = iteration_doc.get("fast_p", {})
    if not isinstance(fast_p_map, dict):
        return {str(p): 0.0 for p in thresholds}

    out: dict[str, float] = {}
    for threshold in thresholds:
        key = resolve_threshold_key(fast_p_map, threshold)
        if key is None:
            out[str(threshold)] = 0.0
            continue
        try:
            out[str(threshold)] = float(fast_p_map[key])
        except (TypeError, ValueError):
            out[str(threshold)] = 0.0
    return out


def _extract_checkpoints(
    stats_doc: dict[str, Any],
    *,
    thresholds: list[float],
    fast_p_field: str,
    stride: int,
) -> list[dict[str, Any]]:
    iterations = stats_doc.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        return []

    by_iteration: dict[int, dict[str, Any]] = {}
    for row in iterations:
        if not isinstance(row, dict):
            continue
        try:
            iteration = int(row.get("iteration"))
        except (TypeError, ValueError):
            continue
        by_iteration[iteration] = row

    if not by_iteration:
        return []

    max_iteration = max(by_iteration)
    checkpoints: list[dict[str, Any]] = []
    for iteration in _checkpoint_iterations(max_iteration, stride=stride):
        row = by_iteration.get(iteration)
        if row is None:
            continue
        checkpoints.append(
            {
                "iteration": iteration,
                "fast_p": _fast_p_at_iteration(
                    row,
                    thresholds=thresholds,
                    fast_p_field=fast_p_field,
                ),
            }
        )
    return checkpoints


def _evolving_stats_path(runs_root: Path, run_name: str) -> Path:
    return runs_root / run_name / "visualizations" / "performance_stats.json"


def _aide_stats_path(run_integration_analysis: Path, run_name: str) -> Path:
    safe_run = sanitize_model_name(run_name)
    return run_integration_analysis / f"aide_subset_run_{safe_run}.json"


def aggregate_evolving_run(
    *,
    run_name: str,
    runs_root: Path,
    thresholds: list[float],
    fast_p_field: str,
    stride: int,
) -> dict[str, Any]:
    stats_path = _evolving_stats_path(runs_root, run_name)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing performance stats: {stats_path}")

    stats_doc = read_json(stats_path, default={})
    if not isinstance(stats_doc, dict) or not stats_doc:
        raise RuntimeError(f"Invalid performance stats JSON: {stats_path}")

    checkpoints = _extract_checkpoints(
        stats_doc,
        thresholds=thresholds,
        fast_p_field=fast_p_field,
        stride=stride,
    )
    return {
        "source": "evolving",
        "run_name": run_name,
        "stats_path": str(stats_path),
        "agent_generation": stats_doc.get("agent_generation"),
        "iteration_count": int(stats_doc.get("iteration_count") or 0),
        "fast_p_field": fast_p_field,
        "checkpoints": checkpoints,
    }


def aggregate_aide_run(
    *,
    run_name: str,
    run_integration_analysis: Path,
    thresholds: list[float],
    fast_p_field: str,
    stride: int,
) -> dict[str, Any]:
    stats_path = _aide_stats_path(run_integration_analysis, run_name)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing AIDE analysis stats: {stats_path}")

    stats_doc = read_json(stats_path, default={})
    if not isinstance(stats_doc, dict) or not stats_doc:
        raise RuntimeError(f"Invalid AIDE stats JSON: {stats_path}")

    checkpoints = _extract_checkpoints(
        stats_doc,
        thresholds=thresholds,
        fast_p_field=fast_p_field,
        stride=stride,
    )
    return {
        "source": "aide",
        "run_name": run_name,
        "stats_path": str(stats_path),
        "layout": stats_doc.get("layout"),
        "iteration_count": int(stats_doc.get("iteration_count") or 0),
        "fast_p_field": fast_p_field,
        "checkpoints": checkpoints,
    }


def build_aggregate_doc(
    *,
    evolving_run_names: list[str],
    aide_run_names: list[str],
    runs_root: Path,
    run_integration_analysis: Path,
    thresholds: list[float],
    fast_p_field: str,
    stride: int,
) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for run_name in evolving_run_names:
        try:
            runs[run_name] = aggregate_evolving_run(
                run_name=run_name,
                runs_root=runs_root,
                thresholds=thresholds,
                fast_p_field=fast_p_field,
                stride=stride,
            )
        except Exception as exc:
            errors[run_name] = str(exc)

    for run_name in aide_run_names:
        try:
            runs[run_name] = aggregate_aide_run(
                run_name=run_name,
                run_integration_analysis=run_integration_analysis,
                thresholds=thresholds,
                fast_p_field=fast_p_field,
                stride=stride,
            )
        except Exception as exc:
            errors[run_name] = str(exc)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fast_p_thresholds": thresholds,
        "iteration_stride": stride,
        "fast_p_field": fast_p_field,
        "evolving_run_names": list(evolving_run_names),
        "aide_run_names": list(aide_run_names),
        "runs": runs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate fast-p checkpoints for selected evolving and AIDE runs."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(REPO_ROOT / "runs_evolving"),
        help="Root directory containing evolving run folders",
    )
    parser.add_argument(
        "--run-integration-analysis",
        type=str,
        default=str(REPO_ROOT / "run_integration" / "analysis"),
        help="Directory containing aide_subset_run_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    parser.add_argument(
        "--fast-p-field",
        type=str,
        default=FAST_P_FIELD,
        choices=["fast_p_best", "fast_p_current", "fast_p"],
        help="Which per-iteration fast-p map to read from cached stats",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=ITERATION_STRIDE,
        help="Iteration interval for checkpoints (default: 10)",
    )
    args = parser.parse_args()

    doc = build_aggregate_doc(
        evolving_run_names=EVOLVING_RUN_NAMES,
        aide_run_names=AIDE_RUN_NAMES,
        runs_root=Path(args.runs_root),
        run_integration_analysis=Path(args.run_integration_analysis),
        thresholds=FAST_P_THRESHOLDS,
        fast_p_field=args.fast_p_field,
        stride=max(1, int(args.stride)),
    )

    output_path = Path(args.output)
    write_json(output_path, doc)

    summary = {
        "output_path": str(output_path),
        "run_count": len(doc["runs"]),
        "error_count": len(doc["errors"]),
        "run_names": list(doc["runs"].keys()),
        "errors": doc["errors"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not doc["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
