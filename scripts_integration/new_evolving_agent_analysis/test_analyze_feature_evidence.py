from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_feature_evidence as analysis


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object], *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if malformed:
        text += "{not-json\n"
    path.write_text(text, encoding="utf-8")


def _make_completed_run(root: Path, name: str) -> None:
    run = root / name
    _write_json(
        run / "run_summary.json",
        {
            "total_attempted": 2,
            "total_completed": 2,
            "total_correct": 1,
            "context_management": "markov",
            "model": "fixture-model",
            "skill_deletion": True,
            "skill_merging": True,
            "enable_skill_refinement": True,
        },
    )
    _write_jsonl(
        run / "shared_l1.jsonl",
        [
            {"entry_id": "1", "status": "active", "source": "promotion"},
            {
                "entry_id": "2",
                "status": "active",
                "source": "skill_refinement",
                "parent_id": "1",
                "refinement_round": 1,
            },
        ],
    )
    _write_jsonl(
        run / "l1_skill_deletions.jsonl",
        [{"entry_id": "old", "reason": "consecutive_unused"}],
        malformed=True,
    )
    _write_jsonl(
        run / "l1_skill_merges.jsonl",
        [
            {
                "status": "accepted",
                "reason": "unit_test_pass",
                "source_entry_ids": ["1", "2"],
            },
            {"status": "rejected", "reason": "unit_test_fail"},
        ],
    )
    _write_json(run / "l1_skill_usage.json", {"global_iteration": 4})
    _write_json(run / "l1_skill_merge_state.json", {"last_merge_iteration": 4})
    _write_json(run / "l1_skill_catalog_stats.json", {"active": 2, "superseded": 0})
    _write_jsonl(run / "l1_skill_merge_clustering.jsonl", [])
    _write_jsonl(run / "l1_skill_unit_test_runs.jsonl", [])
    (run / "skill_revisions.txt").write_text("refinement 1\n", encoding="utf-8")

    for index, action in enumerate(("propose_new", "debug_current"), start=1):
        workspace = run / "workspaces" / f"level_1_problem_{index}"
        chat_rows = [
            {
                "iteration": 1,
                "phase": "action_selector",
                "assistant_text": json.dumps({"action": action}),
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            {
                "iteration": 1,
                "phase": "coder",
                "assistant_text": "short response",
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": None,
            },
            {
                "iteration": 1,
                "phase": "evolving_report",
                "messages": [{"role": "user", "content": "update report"}],
                "assistant_text": "compact report",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        ]
        _write_jsonl(workspace / "chat_history.jsonl", chat_rows, malformed=True)
        correct = index == 1
        metrics_rows = [
            {
                "iteration": 1,
                "metrics_iteration": {
                    "compiled": False,
                    "correct": False,
                    "is_hack": False,
                    "speedup": 0,
                    "error": "Failed to compile custom CUDA kernel",
                    "evolving_report_chars": 20,
                },
                "metrics_best": {
                    "compiled": False,
                    "correct": False,
                    "is_hack": False,
                    "speedup": 0,
                },
            },
            {
                "iteration": 2,
                "metrics_iteration": {
                    "compiled": True,
                    "correct": correct,
                    "is_hack": False,
                    "speedup": 1.5 if correct else 0,
                    "runtime": 1.0 if correct else None,
                    "ref_runtime": 1.5,
                    "error": None if correct else "Output mismatch",
                    "evolving_report_chars": 30,
                },
                "metrics_best": {
                    "compiled": True,
                    "correct": correct,
                    "is_hack": False,
                    "speedup": 1.5 if correct else 0,
                    "runtime": 1.0 if correct else None,
                },
            },
        ]
        _write_jsonl(workspace / "metrics_by_iteration.jsonl", metrics_rows)
        _write_jsonl(
            workspace / "iteration_snapshots.jsonl",
            [
                {"iteration": 1, "l0_entry_count": 1},
                {"iteration": 2, "l0_entry_count": 2},
            ],
            malformed=True,
        )
        _write_json(
            workspace / "run_finished.json",
            {
                "ended_utc": "2026-01-01T00:00:00+00:00",
                "metadata": {
                    "best_compiled": True,
                    "best_correct": correct,
                    "best_is_hack": False,
                    "best_speedup": 1.5 if correct else 0,
                    "best_runtime": 1.0 if correct else None,
                    "error": None if correct else "Output mismatch",
                },
            },
        )


def _problem(workspace: str, *, correct: bool, speedup: float | None) -> dict[str, object]:
    return {
        "workspace": workspace,
        "best": {
            "iteration": 2,
            "compiled": True,
            "correct": correct,
            "is_hack": False,
            "speedup": speedup,
            "runtime": None,
            "locator": {"workspace": workspace, "iteration": 2, "phase": "metrics_iteration"},
        },
        "final": {
            "iteration": 2,
            "correct": correct,
            "error_category": None if correct else "output_mismatch",
            "error_excerpt": None if correct else "Output mismatch",
            "locator": {"workspace": workspace, "iteration": 2, "phase": "metrics_iteration"},
        },
    }


class ActionSelectorParsingTests(unittest.TestCase):
    def test_common_structured_and_text_forms(self) -> None:
        cases = [
            ({"action": "Propose New"}, "propose_new"),
            ({"assistant_text": '```json\n{"action":"refine_current"}\n```'}, "refine_current"),
            ({"assistant_text": "Selected action: debug-current."}, "debug_current"),
            (
                {
                    "assistant_text": None,
                    "messages": [{"role": "assistant", "content": "ACTION = propose_new"}],
                },
                "propose_new",
            ),
        ]
        for row, expected in cases:
            with self.subTest(row=row):
                self.assertEqual(analysis.parse_action_selector(row)[0], expected)


class FeatureEvidenceTests(unittest.TestCase):
    def test_extracts_metrics_warnings_governance_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            output = Path(tmp) / "output"
            _make_completed_run(root, "run-a")

            complete, _ = analysis._completion_check("run-a", root)
            self.assertTrue(complete)
            doc = analysis.extract_feature_evidence(
                runs_root=root,
                run_names=["run-a"],
                output_dir=output,
                generated_at_utc="2026-01-02T00:00:00+00:00",
            )
            run = doc["runs"][0]
            self.assertEqual(run["workspace_count"], 2)
            self.assertEqual(run["finished_workspace_count"], 2)
            self.assertEqual(run["chat"]["turn_count"], 6)
            self.assertEqual(run["chat"]["tokens"]["total_tokens"], 88)
            self.assertEqual(run["action_selector"]["actions"]["propose_new"], 1)
            self.assertEqual(run["action_selector"]["actions"]["debug_current"], 1)
            self.assertEqual(run["metrics"]["metrics_iteration_row_count"], 4)
            self.assertEqual(run["metrics"]["compiled_count"], 2)
            self.assertEqual(run["metrics"]["correct_count"], 1)
            self.assertEqual(run["errors"]["categories"]["compilation_error"], 2)
            self.assertEqual(run["l0"]["final_entry_counts_per_workspace"]["mean"], 2.0)
            self.assertEqual(run["l1_governance"]["refinement"]["entry_count"], 1)
            self.assertEqual(run["l1_governance"]["deletion"]["event_count"], 1)
            self.assertEqual(run["l1_governance"]["merge"]["accepted_count"], 1)
            self.assertGreater(doc["warning_count"], 0)
            warning_codes = {warning["code"] for warning in run["warnings"]}
            self.assertIn("malformed_jsonl_rows", warning_codes)
            self.assertIn("missing_optional_artifact", warning_codes)
            self.assertEqual(doc["inputs"]["selected_runs"], ["run-a"])
            self.assertTrue((output / analysis.OUTPUT_JSON).is_file())
            self.assertTrue((output / analysis.OUTPUT_CSV).is_file())
            with (output / analysis.OUTPUT_CSV).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

            repeated = analysis.extract_feature_evidence(
                runs_root=root,
                run_names=["run-a"],
                output_dir=output,
                generated_at_utc="2026-01-02T00:00:00+00:00",
                write=False,
            )
            self.assertEqual(doc, repeated)

    def test_matched_comparisons_and_case_selection(self) -> None:
        baseline = {
            "run_name": "baseline",
            "problems": [
                _problem("p1", correct=True, speedup=1.0),
                _problem("p2", correct=True, speedup=2.0),
                _problem("p3", correct=False, speedup=0),
                _problem("p4", correct=True, speedup=3.0),
                _problem("p5", correct=False, speedup=0),
            ],
        }
        candidate = {
            "run_name": "candidate",
            "problems": [
                _problem("p1", correct=True, speedup=2.0),
                _problem("p2", correct=True, speedup=1.0),
                _problem("p3", correct=True, speedup=1.5),
                _problem("p4", correct=False, speedup=0),
                _problem("p5", correct=False, speedup=0),
            ],
        }
        comparisons = analysis.build_comparisons([baseline, candidate], "baseline")
        self.assertIsNotNone(comparisons)
        comparison = comparisons["runs"][0]
        cases = comparison["case_study_candidates"]
        self.assertEqual(cases["largest_valid_speedup_improvement"]["workspace"], "p1")
        self.assertEqual(cases["largest_valid_regression"]["workspace"], "p2")
        self.assertEqual(cases["correctness_gain"]["workspace"], "p3")
        self.assertEqual(cases["correctness_loss"]["workspace"], "p4")
        self.assertEqual(cases["representative_no_change_or_stall"]["workspace"], "p5")
        self.assertIn("no causal", comparisons["selection_rules"]["interpretation"])


if __name__ == "__main__":
    unittest.main()
