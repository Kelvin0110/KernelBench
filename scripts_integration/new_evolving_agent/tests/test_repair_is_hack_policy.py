from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_repair_module():
    repo_root = Path(__file__).resolve().parents[3]
    batch_dir = repo_root / "scripts_integration" / "new_evolving_agent"
    for entry in (str(repo_root), str(repo_root / "Self-Evolving-Agent"), str(batch_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    script_path = batch_dir / "repair" / "repair_is_hack_policy.py"
    spec = importlib.util.spec_from_file_location("repair_is_hack_policy", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recompute_run_entry_promotes_best_after_hack_cleared(monkeypatch) -> None:
    module = _load_repair_module()
    calls: list[str] = []

    def _fake_static_check(code: str, *, backend: str, precision: str):
        calls.append(code)
        if "nn.Linear" in code or "torch.nn.Linear" in code:
            return True, [], ["pytorch_wrap: Uses torch.nn compute layer"]
        return True, [], []

    monkeypatch.setattr(module.static_check, "run_static_check", _fake_static_check)

    run_entry = {
        "level": 1,
        "problem_id": "1",
        "backend": "cuda",
        "precision": "fp32",
        "best_speedup": 0.0,
        "best_correct": False,
        "runtime": -1.0,
        "records": [
            {
                "attempt": 1,
                "candidate_code": "class ModelNew: pass",
                "evaluation": {
                    "compiled": True,
                    "correct": True,
                    "is_hack": True,
                    "speedup": 2.0,
                    "runtime": 20.0,
                    "metadata": {},
                    "runtime_stats": {"mean": 20.0},
                },
            },
            {
                "attempt": 2,
                "candidate_code": "import torch\nimport torch.nn as nn\nclass ModelNew(nn.Linear): pass",
                "evaluation": {
                    "compiled": True,
                    "correct": True,
                    "is_hack": True,
                    "speedup": 4.0,
                    "runtime": 10.0,
                    "metadata": {},
                    "runtime_stats": {"mean": 10.0},
                },
            },
        ],
    }

    updated, stats = module.recompute_run_entry(run_entry)
    assert stats.iterations_is_hack_cleared == 2
    assert stats.best_changed is True
    assert stats.best_attempt == 2
    assert updated["best_correct"] is True
    assert updated["best_speedup"] == 4.0
    assert updated["runtime"] == 10.0
    assert updated["records"][1]["evaluation"]["is_hack"] is False
    assert updated["records"][1]["evaluation"]["static_check_warnings"]


def test_recompute_run_entry_clears_workload_shrink_hack(monkeypatch) -> None:
    module = _load_repair_module()

    monkeypatch.setattr(
        module.static_check,
        "run_static_check",
        lambda *_a, **_k: (True, [], ["workload_shrink: Defines get_inputs/get_init_inputs"]),
    )

    run_entry = {
        "level": 1,
        "problem_id": "2",
        "backend": "cuda",
        "precision": "fp32",
        "best_speedup": 0.0,
        "best_correct": False,
        "runtime": -1.0,
        "records": [
            {
                "attempt": 1,
                "candidate_code": "def get_inputs(): return []",
                "evaluation": {
                    "compiled": True,
                    "correct": True,
                    "is_hack": True,
                    "speedup": 5.0,
                    "runtime": 4.0,
                    "metadata": {},
                    "runtime_stats": {"mean": 4.0},
                },
            },
        ],
    }

    updated, stats = module.recompute_run_entry(run_entry)
    assert updated["records"][0]["evaluation"]["is_hack"] is False
    assert updated["best_correct"] is True
    assert updated["runtime"] == 4.0
    assert stats.best_attempt == 1


def test_repair_run_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    module = _load_repair_module()
    run_dir = tmp_path / "demo_run"
    ws_dir = run_dir / "workspaces" / "level_1_problem_1"
    ws_dir.mkdir(parents=True)

    evolving = {
        "runs": [
            {
                "level": 1,
                "problem_id": "1",
                "backend": "cuda",
                "precision": "fp32",
                "best_speedup": 0.0,
                "best_correct": False,
                "runtime": -1.0,
                "records": [
                    {
                        "attempt": 1,
                        "candidate_code": "print('ok')",
                        "evaluation": {
                            "compiled": True,
                            "correct": True,
                            "is_hack": True,
                            "speedup": 2.0,
                            "runtime": 5.0,
                            "metadata": {},
                            "runtime_stats": {"mean": 5.0},
                        },
                    }
                ],
            }
        ]
    }
    (run_dir / "evolving_runs.json").write_text(json.dumps(evolving), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps({"enable_static_check": True, "total_correct": 0}),
        encoding="utf-8",
    )
    metrics_path = ws_dir / "metrics_by_iteration.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "iteration": 1,
                "metrics_iteration": {"is_hack": True, "correct": True, "speedup": 2.0},
                "metrics_best": {"is_hack": True, "correct": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module.static_check, "run_static_check", lambda *_a, **_k: (True, [], []))

    before = metrics_path.read_text(encoding="utf-8")
    stats = module.repair_run(run_dir, dry_run=True, skip_viz=True)
    after = metrics_path.read_text(encoding="utf-8")

    assert stats["dry_run"] is True
    assert stats["iterations_is_hack_cleared"] == 1
    assert before == after
    assert json.loads((run_dir / "evolving_runs.json").read_text(encoding="utf-8")) == evolving
