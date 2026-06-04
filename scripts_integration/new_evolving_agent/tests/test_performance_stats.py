from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from kernelbench.performance_stats import (
    align_series_for_comparison,
    compute_fastp_from_records,
    format_result_key,
    parse_fastp_values,
    parse_result_key,
)


def _load_generate_run_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "Self-Evolving-Agent" / "visualizations" / "kernelbench" / "server" / "generate_run_performance_stats.py"
    spec = importlib.util.spec_from_file_location("generate_run_performance_stats", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_and_parse_result_key() -> None:
    assert format_result_key(2, 10) == "L2P10"
    assert parse_result_key("L2P10") == (2, 10)
    assert parse_result_key("10") == (None, 10)


def test_parse_fastp_values_dedup_and_sorted() -> None:
    values = parse_fastp_values("1.0,0.8,1.0,0.5")
    assert values == [0.5, 0.8, 1.0]


def test_compute_fastp_from_records_supports_best_runtime_mode() -> None:
    records = [
        {
            "correct": True,
            "baseline_runtime": 10.0,
            "current_runtime": 20.0,
            "best_runtime": 5.0,
        }
    ]

    scores_current, aligned_current = compute_fastp_from_records(
        records,
        thresholds=[1.0],
        runtime_field="current_runtime",
        correct_field="correct",
    )
    scores_best, aligned_best = compute_fastp_from_records(
        records,
        thresholds=[1.0],
        runtime_field="best_runtime",
        correct_field="correct",
    )

    assert aligned_current == 1
    assert aligned_best == 1
    assert scores_current["1.0"] == 0.0
    assert scores_best["1.0"] == 1.0


def test_align_series_for_comparison_intersects_iterations_only() -> None:
    evolving = [
        {"iteration": 1, "value": 0.2},
        {"iteration": 2, "value": 0.3},
        {"iteration": 4, "value": 0.5},
    ]
    aide = [
        {"iteration": 1, "value": 0.25},
        {"iteration": 3, "value": 0.35},
        {"iteration": 4, "value": 0.55},
        {"iteration": 10, "value": 0.9},
    ]

    aligned = align_series_for_comparison(evolving=evolving, aide=aide)
    assert aligned == [
        {"iteration": 1, "evolving": 0.2, "aide": 0.25},
        {"iteration": 4, "evolving": 0.5, "aide": 0.55},
    ]


def test_generate_run_uses_best_runtime_for_fastp(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_run_module()

    run_name = "run_a"
    run_dir = tmp_path / run_name
    ws_dir = run_dir / "workspaces" / "level_1_problem_1"
    ws_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({}), encoding="utf-8")

    metrics_path = ws_dir / "metrics_by_iteration.jsonl"
    metrics_row = {
        "iteration": 1,
        "metrics_iteration": {
            "compiled": True,
            "correct": True,
            "runtime": 20.0,
        },
        "metrics_best": {
            "compiled": True,
            "correct": True,
            "runtime": 5.0,
        },
    }
    metrics_path.write_text(json.dumps(metrics_row) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "build_baseline_lookup", lambda _base, _level: {1: 10.0})

    result = module.build_performance_stats(
        run_name=run_name,
        runs_root=tmp_path,
        baseline_file=baseline_file,
        fast_p_thresholds=[1.0],
    )

    fastp_score = result["doc"]["iterations"][0]["fast_p_current"]["1.0"]
    assert fastp_score == 1.0
