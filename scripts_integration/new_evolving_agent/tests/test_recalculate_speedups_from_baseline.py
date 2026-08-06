from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def _load_recalculate_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = (
        repo_root
        / "Self-Evolving-Agent"
        / "visualizations"
        / "kernelbench"
        / "server"
        / "fix_data"
        / "recalculate_speedups_from_baseline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "recalculate_speedups_from_baseline", script_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewrite_metrics_jsonl_syncs_ref_runtime_and_speedup(tmp_path: Path) -> None:
    module = _load_recalculate_module()
    baseline = 22.2
    runtime = 49.7
    stale_ref = 10.0
    stale_speedup = 0.99
    expected_speedup = baseline / runtime

    metrics_path = tmp_path / "metrics_by_iteration.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "record_type": "metrics_by_iteration",
                "iteration": 1,
                "metrics_iteration": {
                    "correct": True,
                    "runtime": runtime,
                    "ref_runtime": stale_ref,
                    "speedup": stale_speedup,
                },
                "metrics_best": {
                    "correct": True,
                    "runtime": runtime,
                    "speedup": stale_speedup,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.rewrite_metrics_jsonl(
        metrics_path,
        level=1,
        problem_id=22,
        baseline=baseline,
        dry_run=False,
    )
    assert result["changed"] is True
    assert result["rows_changed"] == 1

    row = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    mi = row["metrics_iteration"]
    assert math.isclose(mi["ref_runtime"], baseline)
    assert math.isclose(mi["speedup"], expected_speedup)
    mb = row["metrics_best"]
    assert math.isclose(mb["ref_runtime"], baseline)
    assert math.isclose(mb["speedup"], expected_speedup)


def test_rewrite_metrics_jsonl_updates_metrics_last_iteration(tmp_path: Path) -> None:
    module = _load_recalculate_module()
    baseline = 40.0
    runtime = 20.0
    expected_speedup = 2.0

    metrics_path = tmp_path / "metrics_by_time.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "record_type": "metrics_by_time",
                "iteration": 1,
                "metrics_last_iteration": {
                    "correct": True,
                    "runtime": runtime,
                    "ref_runtime": 15.0,
                    "speedup": 0.5,
                },
                "metrics_best": {
                    "correct": True,
                    "runtime": runtime,
                    "speedup": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.rewrite_metrics_jsonl(
        metrics_path,
        level=1,
        problem_id=1,
        baseline=baseline,
        dry_run=False,
    )
    assert result["changed"] is True

    row = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    last = row["metrics_last_iteration"]
    assert math.isclose(last["ref_runtime"], baseline)
    assert math.isclose(last["speedup"], expected_speedup)


def test_rewrite_evolving_runs_best_speedup(tmp_path: Path) -> None:
    module = _load_recalculate_module()
    path = tmp_path / "evolving_runs.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "level": 1,
                        "problem_id": "22",
                        "best_correct": True,
                        "runtime": 49.2,
                        "best_speedup": 0.99,
                    },
                    {
                        "level": 2,
                        "problem_id": 3,
                        "best_correct": False,
                        "runtime": 10.0,
                        "best_speedup": 1.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    lookups = {1: {22: 22.2}, 2: {3: 30.0}}
    result = module.rewrite_evolving_runs(path, lookups=lookups, dry_run=False)
    assert result["changed"] is True
    assert result["entries_changed"] == 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert math.isclose(payload["runs"][0]["best_speedup"], 22.2 / 49.2)
    assert payload["runs"][1]["best_speedup"] == 0.0
