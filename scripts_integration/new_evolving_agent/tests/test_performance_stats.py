from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from kernelbench.performance_stats import (
    align_series_for_comparison,
    aggregate_speedups,
    compute_fastp_from_records,
    parse_fastp_values,
)


def _load_generate_run_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "Self-Evolving-Agent" / "visualizations" / "kernelbench" / "server" / "generate_run_performance_stats.py"
    spec = importlib.util.spec_from_file_location("generate_run_performance_stats", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generate_aide_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "Self-Evolving-Agent" / "visualizations" / "kernelbench" / "server" / "generate_aide_integration_stats.py"
    spec = importlib.util.spec_from_file_location("generate_aide_integration_stats", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_for_current_fastp_gate_unchanged_code() -> None:
    module = _load_generate_aide_module()
    records = [
        {
            "level": 1,
            "problem_id": 1,
            "baseline_runtime": 10.0,
            "runtime": 25.1,
            "correct": True,
            "code_changed_since_last_checkpoint": False,
        },
        {
            "level": 1,
            "problem_id": 2,
            "baseline_runtime": 10.0,
            "runtime": 5.0,
            "correct": True,
            "code_changed_since_last_checkpoint": True,
        },
    ]
    gated = module._records_for_current_fastp(records)
    assert gated[0]["correct"] is False
    assert gated[1]["correct"] is True


def test_parse_fastp_values_dedup_and_sorted() -> None:
    values = parse_fastp_values("1.0,0.8,1.0,0.5")
    assert values == [0.5, 0.8, 1.0]


def test_aggregate_speedups_uses_correct_samples_only() -> None:
    speedups = [2.0, 0.0, 4.0, 0.0]
    correct_flags = [True, False, True, True]

    agg = aggregate_speedups(speedups, correct_flags)

    assert agg["mean"] == 2.0
    assert agg["median"] == 2.0
    assert agg["geometric_mean"] == (2.0 * 4.0) ** 0.5


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


def test_discover_run_names_requires_workspaces_dir(tmp_path: Path) -> None:
    module = _load_generate_run_module()

    runs_root = tmp_path / "runs_evolving"
    runs_root.mkdir()
    (runs_root / "run_with_ws" / "workspaces").mkdir(parents=True)
    (runs_root / "run_without_ws").mkdir()
    (runs_root / "not_a_run.txt").write_text("x", encoding="utf-8")

    assert module.discover_run_names(runs_root=runs_root) == ["run_with_ws"]


def test_generate_all_run_performance_stats(tmp_path: Path, monkeypatch) -> None:
    module = _load_generate_run_module()

    runs_root = tmp_path / "runs_evolving"
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({}), encoding="utf-8")

    for run_name in ("run_a", "run_b"):
        ws_dir = runs_root / run_name / "workspaces" / "level_1_problem_1"
        ws_dir.mkdir(parents=True, exist_ok=True)
        metrics_row = {
            "iteration": 1,
            "metrics_iteration": {"compiled": True, "correct": True, "runtime": 20.0},
            "metrics_best": {"compiled": True, "correct": True, "runtime": 5.0},
        }
        (ws_dir / "metrics_by_iteration.jsonl").write_text(
            json.dumps(metrics_row) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "build_baseline_lookup", lambda _base, _level: {1: 10.0})

    summary = module.generate_all_run_performance_stats(
        runs_root=runs_root,
        baseline_file=baseline_file,
        fast_p_thresholds=[1.0],
    )

    assert summary["discovered"] == 2
    assert summary["generated"] == 2
    assert summary["skipped"] == 0
    assert set(summary["run_names"]) == {"run_a", "run_b"}
    for run_name in ("run_a", "run_b"):
        out_path = runs_root / run_name / "visualizations" / "performance_stats.json"
        assert out_path.is_file()
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        assert doc["run_name"] == run_name


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

    fastp_score = result["doc"]["iterations"][0]["fast_p_best"]["1.0"]
    assert fastp_score == 1.0


def test_fast_p_best_forward_fills_when_problem_stops_early(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_run_module()

    run_name = "run_a"
    run_dir = tmp_path / run_name
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({}), encoding="utf-8")

    for ws_name, rows in (
        (
            "level_1_problem_1",
            [
                {
                    "iteration": 1,
                    "metrics_iteration": {"compiled": True, "correct": True, "runtime": 5.0},
                    "metrics_best": {"compiled": True, "correct": True, "runtime": 5.0},
                },
            ],
        ),
        (
            "level_1_problem_2",
            [
                {
                    "iteration": 1,
                    "metrics_iteration": {"compiled": True, "correct": True, "runtime": 8.0},
                    "metrics_best": {"compiled": True, "correct": True, "runtime": 8.0},
                },
                {
                    "iteration": 2,
                    "metrics_iteration": {"compiled": True, "correct": True, "runtime": 4.0},
                    "metrics_best": {"compiled": True, "correct": True, "runtime": 4.0},
                },
                {
                    "iteration": 3,
                    "metrics_iteration": {"compiled": True, "correct": True, "runtime": 3.0},
                    "metrics_best": {"compiled": True, "correct": True, "runtime": 3.0},
                },
            ],
        ),
    ):
        ws_dir = run_dir / "workspaces" / ws_name
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "metrics_by_iteration.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        module,
        "build_baseline_lookup",
        lambda _base, _level: {1: 10.0, 2: 10.0},
    )

    result = module.build_performance_stats(
        run_name=run_name,
        runs_root=tmp_path,
        baseline_file=baseline_file,
        fast_p_thresholds=[1.0],
    )

    series = result["doc"]["series"]["fast_p_best"]["1.0"]
    values = [point["value"] for point in series]
    assert values == [1.0, 1.0, 1.0]
    assert values == sorted(values)


def test_fast_p_best_does_not_drop_after_problem_finishes(monkeypatch, tmp_path: Path) -> None:
    module = _load_generate_run_module()

    run_name = "run_a"
    run_dir = tmp_path / run_name
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({}), encoding="utf-8")

    early_rows = [
        {
            "iteration": i,
            "metrics_iteration": {
                "compiled": i < 10,
                "correct": i < 10,
                "runtime": 65.1 if i < 10 else None,
            },
            "metrics_best": {
                "compiled": True,
                "correct": True,
                "runtime": 65.1,
            },
        }
        for i in range(1, 11)
    ]
    late_rows = [
        {
            "iteration": i,
            "metrics_iteration": {"compiled": True, "correct": True, "runtime": 3.0},
            "metrics_best": {"compiled": True, "correct": True, "runtime": 3.0},
        }
        for i in range(1, 21)
    ]

    for ws_name, rows in (("level_1_problem_1", early_rows), ("level_1_problem_2", late_rows)):
        ws_dir = run_dir / "workspaces" / ws_name
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "metrics_by_iteration.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "build_baseline_lookup", lambda _base, _level: {1: 84.5, 2: 10.0})

    result = module.build_performance_stats(
        run_name=run_name,
        runs_root=tmp_path,
        baseline_file=baseline_file,
        fast_p_thresholds=[1.0],
    )

    series = result["doc"]["series"]["fast_p_best"]["1.0"]
    values = [point["value"] for point in series]
    assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))

