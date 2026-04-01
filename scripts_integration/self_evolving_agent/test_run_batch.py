from __future__ import annotations

import json
from pathlib import Path

from scripts_integration.self_evolving_agent import run_batch


class _FakeReflectionEngine:
    def __init__(self, local_memory, global_memory):
        self.local_memory = local_memory
        self.global_memory = global_memory


class _FakeEnvironment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_main_dry_run_writes_level_first_output(tmp_path: Path, monkeypatch) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n2,5\n", encoding="utf-8")

    output_path = tmp_path / "eval_results.json"

    monkeypatch.setattr(run_batch, "SimpleKernelBenchReflectionEngine", _FakeReflectionEngine)
    monkeypatch.setattr(run_batch, "KernelBenchEnvironment", _FakeEnvironment)
    monkeypatch.setattr(run_batch, "KernelBenchEvolvingAgent", _FakeAgent)

    def _fake_run_subset(*, agent, subset_rows, output_path, max_steps, backend, precision):
        _ = agent
        _ = max_steps
        _ = backend
        _ = precision
        payload = {
            "1": {
                "100": [
                    {
                        "sample_id": 0,
                        "compiled": False,
                        "correctness": False,
                        "metadata": {"level": 1, "problem_id": 100},
                        "runtime": -1.0,
                        "runtime_stats": {},
                    }
                ]
            },
            "2": {
                "5": [
                    {
                        "sample_id": 0,
                        "compiled": False,
                        "correctness": False,
                        "metadata": {"level": 2, "problem_id": 5},
                        "runtime": -1.0,
                        "runtime_stats": {},
                    }
                ]
            },
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        assert subset_rows == [{"level": 1, "problem_id": 100}, {"level": 2, "problem_id": 5}]
        return payload

    monkeypatch.setattr(run_batch, "run_subset", _fake_run_subset)

    rc = run_batch.main(
        [
            "--subset-csv",
            str(subset_csv),
            "--output-path",
            str(output_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "1" in payload and "2" in payload
    assert "100" in payload["1"]
    assert "5" in payload["2"]
