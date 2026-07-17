from __future__ import annotations

import json
import sys
from pathlib import Path

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


def test_extract_best_kernel_code_prefers_best_code_then_records() -> None:
    with_best = {
        "best_code": "print('best')",
        "records": [{"candidate_code": "print('candidate')"}],
    }
    from_records = {
        "best_code": None,
        "records": [{"candidate_code": "print('older')"}, {"candidate_code": "print('newer')"}],
    }

    assert evolve_kb_batch._extract_best_kernel_code(with_best) == "print('best')"
    assert evolve_kb_batch._extract_best_kernel_code(from_records) == "print('newer')"


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

    matching_runs = sorted(p for p in results_root.glob(f"{run_name}*") if p.is_dir())
    assert matching_runs
    eval_results_path = matching_runs[-1] / "eval_results.json"
    payload = json.loads(eval_results_path.read_text(encoding="utf-8"))

    assert "1" in payload
    assert "2" in payload
    assert "100" in payload["1"]
    assert "5" in payload["2"]
    assert payload["1"]["100"][0]["metadata"]["level"] == 1
    assert payload["2"]["5"][0]["metadata"]["level"] == 2


def test_main_dry_run_accepts_skill_refinement_flag(tmp_path: Path, monkeypatch) -> None:
    """The skill-refinement CLI flags are accepted and default off (dry-run smoke)."""
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            "skill_refine_flag",
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "1",
            "--enable-skill-refinement",
            "--skill-refinement-max-rounds",
            "2",
        ],
    )

    assert evolve_kb_batch.main() == 0


def test_main_dry_run_accepts_skill_deletion_only_flags(tmp_path: Path, monkeypatch) -> None:
    """Deletion-only: --skill-deletion without --skill-merging."""
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            "skill_deletion_only_flag",
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "1",
            "--skill-deletion",
            "--no-skill-merging",
            "--no-enable-l1-skill-unit-test-gc",
        ],
    )

    assert evolve_kb_batch.main() == 0


def test_main_dry_run_accepts_skill_merge_only_flags(tmp_path: Path, monkeypatch) -> None:
    """Merge-only: --skill-merging without --skill-deletion."""
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            "skill_merge_only_flag",
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "1",
            "--no-skill-deletion",
            "--skill-merging",
            "--skill-merge-similarity",
            "0.85",
            "--skill-merge-interval",
            "25",
        ],
    )

    assert evolve_kb_batch.main() == 0

    matching_runs = sorted(p for p in results_root.glob("skill_merge_only_flag*") if p.is_dir())
    assert matching_runs
    summary = json.loads((matching_runs[-1] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["skill_deletion"] is False
    assert summary["skill_merging"] is True


def test_main_dry_run_accepts_markov_report_context_management(
    tmp_path: Path, monkeypatch
) -> None:
    """markov_report context mode is accepted; deletion can be left off."""
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            "markov_report_flag",
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "1",
            "--context-management",
            "markov_report",
            "--no-skill-deletion",
        ],
    )

    assert evolve_kb_batch.main() == 0

    matching_runs = sorted(p for p in results_root.glob("markov_report_flag*") if p.is_dir())
    assert matching_runs
    summary = json.loads((matching_runs[-1] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["context_management"] == "markov_report"
    assert summary["skill_deletion"] is False


def test_main_dry_run_accepts_skill_deletion_flags(tmp_path: Path, monkeypatch) -> None:
    """Skill-deletion CLI flags are accepted; deletion is on by default."""
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--subset-csv",
            str(subset_csv),
            "--run-name",
            "skill_deletion_flag",
            "--dry-run",
            "--results-root",
            str(results_root),
            "--max-problems",
            "1",
            "--skill-deletion",
            "--l1-skill-consecutive-unused-delete-after",
            "40",
            "--no-enable-l1-skill-unit-test-gc",
        ],
    )

    assert evolve_kb_batch.main() == 0


def test_main_dry_run_accepts_skill_merging_and_timing(tmp_path: Path, monkeypatch) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n1,200\n", encoding="utf-8")

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)

    results_root = tmp_path / "results"
    run_name = "skill_merge_timing"
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
            "--skill-deletion",
            "--skill-merging",
            "--skill-merge-similarity",
            "0.85",
            "--skill-merge-interval",
            "25",
        ],
    )

    assert evolve_kb_batch.main() == 0

    matching_runs = sorted(p for p in results_root.glob(f"{run_name}*") if p.is_dir())
    assert matching_runs
    run_dir = matching_runs[-1]
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["skill_merging"] is True
    assert summary["skill_merge_similarity"] == 0.85
    assert summary["skill_merge_interval"] == 25
    assert summary["batch_timing_jsonl"]
    assert summary["total_wall_time_sec"] >= 0.0
    assert summary["problems_timed_this_session"] == 2

    timing_lines = (run_dir / "batch_timing.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(timing_lines) == 2
    first = json.loads(timing_lines[0])
    assert "wall_time_sec" in first
    assert first["subset_index"] == 1

    runs_doc = json.loads((run_dir / "evolving_runs.json").read_text(encoding="utf-8"))
    for entry in runs_doc["runs"]:
        assert "wall_time_sec" in entry
        assert "started_at_utc" in entry
        assert "finished_at_utc" in entry


def _seed_resume_run_dir(run_dir: Path, *, runs: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shared_l1.txt").write_text("# shared l1\n", encoding="utf-8")
    evolve_kb_batch._write_json(run_dir / "evolving_runs.json", {"runs": runs})


def test_resume_does_not_append_timestamp(tmp_path: Path, monkeypatch) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("level,problem_id\n1,100\n1,200\n", encoding="utf-8")

    run_name = "my_run_2020_01_01_00_00"
    results_root = tmp_path / "results"
    run_dir = results_root / run_name
    _seed_resume_run_dir(
        run_dir,
        runs=[
            {
                "level": 1,
                "problem_id": "100",
                "error": "prior",
                "timestamp_utc": "old",
            }
        ],
    )

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--resume",
            "--run-name",
            run_name,
            "--subset-csv",
            str(subset_csv),
            "--dry-run",
            "--results-root",
            str(results_root),
            "--start-problem",
            "2",
            "--max-problems",
            "2",
        ],
    )

    assert evolve_kb_batch.main() == 0
    assert (results_root / run_name).is_dir()
    assert not list(results_root.glob(f"{run_name}_*"))


def test_resume_replaces_from_start_problem(tmp_path: Path, monkeypatch) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text(
        "level,problem_id\n1,1\n1,2\n1,3\n",
        encoding="utf-8",
    )

    run_name = "resume_replace_test"
    results_root = tmp_path / "results"
    run_dir = results_root / run_name
    _seed_resume_run_dir(
        run_dir,
        runs=[
            {
                "level": 1,
                "problem_id": "1",
                "error": "ok_p1",
                "timestamp_utc": "old1",
            },
            {
                "level": 1,
                "problem_id": "2",
                "error": "coder_call_error: RateLimitError",
                "timestamp_utc": "old2",
            },
            {
                "level": 1,
                "problem_id": "3",
                "error": "ok_p3",
                "timestamp_utc": "old3",
            },
        ],
    )

    monkeypatch.setattr(evolve_kb_batch.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_kb_batch.py",
            "--resume",
            "--run-name",
            run_name,
            "--subset-csv",
            str(subset_csv),
            "--dry-run",
            "--results-root",
            str(results_root),
            "--start-problem",
            "2",
            "--max-problems",
            "3",
        ],
    )

    assert evolve_kb_batch.main() == 0

    doc = json.loads((run_dir / "evolving_runs.json").read_text(encoding="utf-8"))
    by_pid = {str(e["problem_id"]): e for e in doc["runs"]}
    assert by_pid["1"]["error"] == "ok_p1"
    assert by_pid["1"]["timestamp_utc"] == "old1"
    assert "RateLimitError" not in (by_pid["2"].get("error") or "")
    assert by_pid["2"]["error"] == "dry_run_no_gpu_execution"
    assert by_pid["3"]["error"] == "dry_run_no_gpu_execution"
    assert by_pid["3"]["timestamp_utc"] != "old3"


def test_purge_problem_state_clears_workspace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = run_dir / "workspaces" / "level_1_problem_100"
    workspace.mkdir(parents=True)
    marker = workspace / "stale.txt"
    marker.write_text("stale", encoding="utf-8")

    runs = [{"level": 1, "problem_id": "100", "error": "old"}]
    eval_doc: dict = {"1": {"100": [{"sample_id": 0}]}}
    level_eval_docs: dict = {1: {"100": [{"sample_id": 0}]}}

    runs = evolve_kb_batch._purge_problem_state(
        run_dir=run_dir,
        runs=runs,
        eval_doc=eval_doc,
        level_eval_docs=level_eval_docs,
        level=1,
        problem_id="100",
    )

    assert runs == []
    assert "100" not in eval_doc["1"]
    assert "100" not in level_eval_docs[1]
    assert not workspace.exists()


def test_write_json_serializes_exception_objects(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    payload = {
        "runs": [
            {
                "metadata": {
                    "runtime_error": RuntimeError("cuda illegal memory access"),
                }
            }
        ]
    }

    evolve_kb_batch._write_json(output, payload)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "cuda illegal memory access" in data["runs"][0]["metadata"]["runtime_error"]
