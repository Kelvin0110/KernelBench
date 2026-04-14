from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_integration.new_evolving_agent import evolve_kb_batch


class _DummyGovernorResult:
    def __init__(self) -> None:
        self.level = 1
        self.problem_id = "100"
        self.backend = "cuda"
        self.precision = "fp32"
        self.best_speedup = 1.5
        self.best_correct = True
        self.best_compiled = True
        self.best_code_path = None
        self.best_code = "print('x')"
        self.iterations_run = 1
        self.records = []
        self.runtime = 2.0
        self.runtime_stats = {"mean": 2.0}
        self.metadata = {"hardware": "fake-gpu", "device": "cuda:0"}
        self.error = None


def test_to_kernelbench_eval_entry_includes_runtime_and_metadata() -> None:
    run_entry = {
        "best_compiled": True,
        "best_correct": True,
        "best_speedup": 1.5,
        "backend": "cuda",
        "precision": "fp32",
        "iterations_run": 2,
        "error": None,
        "runtime": 25.1,
        "runtime_stats": {
            "mean": 25.1,
            "std": 26.1,
            "min": 22.2,
            "max": 285.0,
            "num_trials": 100,
            "hardware": "NVIDIA RTX A6000",
            "device": "cuda:0",
        },
        "metadata": {
            "hardware": "NVIDIA RTX A6000",
            "device": "cuda:0",
            "correctness_trials": "(5 / 5)",
        },
    }

    entry = evolve_kb_batch._to_kernelbench_eval_entry(run_entry, level=1, problem_id="100")

    assert entry["runtime"] == 25.1
    assert entry["runtime_stats"]["mean"] == 25.1
    assert entry["metadata"]["hardware"] == "NVIDIA RTX A6000"
    assert entry["metadata"]["device"] == "cuda:0"
    assert entry["metadata"]["correctness_trials"] == "(5 / 5)"


def test_main_dry_run_writes_level_first_eval_results(tmp_path: Path, monkeypatch) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text(
        "level,problem_id\n1,100\n2,5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    run_name = "dry_run_level_first"
    results_root = tmp_path / "results"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            run_name,
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "2",
        ],
    )

    rc = evolve_kb_batch.main()
    assert rc == 0

    eval_results_path = results_root / run_name / "eval_results.json"
    payload = json.loads(eval_results_path.read_text(encoding="utf-8"))

    assert "1" in payload
    assert "2" in payload
    assert "100" in payload["1"]
    assert "5" in payload["2"]
    assert payload["1"]["100"][0]["metadata"]["level"] == 1
    assert payload["2"]["5"][0]["metadata"]["level"] == 2
