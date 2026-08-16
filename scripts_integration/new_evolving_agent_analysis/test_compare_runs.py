from __future__ import annotations

import unittest

import compare_runs as compare


def _series(values: dict[int, float], *, n: int | None = None) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for iteration, value in values.items():
        point: dict[str, object] = {"iteration": iteration, "value": value}
        if n is not None:
            point["n"] = n
        points.append(point)
    return points


def _record(*, name: str, mode: str, deletion: bool = False) -> dict[str, object]:
    return {
        "run_name": name,
        "status": "complete",
        "config": {
            "context_management": mode,
            "skill_deletion": deletion,
            "skill_merging": False,
            "enable_skill_refinement": False,
        },
        "outcomes": {"total_correct": 48, "total_attempted": 50},
        "performance": {"final_iteration": 30, "speedup_best": {"n": 40}},
        "series": {
            "speedup": {
                "best_geometric_mean": _series({10: 1.2, 30: 1.4}, n=40),
            },
            "fast_p_best": {
                "0.0": _series({10: 0.90, 30: 0.96}),
                "1.0": _series({10: 0.50, 30: 0.72}),
                "2.0": _series({10: 0.10, 30: 0.20}),
            },
        },
    }


class CompareCheckpointTests(unittest.TestCase):
    def test_design_variant_includes_governance(self) -> None:
        record = _record(name="r", mode="truncation", deletion=True)
        self.assertEqual(compare.design_variant_label(record), "truncation+deletion")

    def test_checkpoint_snapshot_reads_iter_10_and_30(self) -> None:
        snap = compare.checkpoint_snapshot(_record(name="t0", mode="truncation"))
        self.assertEqual(snap["design"], "truncation")
        self.assertEqual(snap["checkpoints"]["10"]["fast_p_best"]["1.0"], 0.50)
        self.assertEqual(snap["checkpoints"]["30"]["fast_p_best"]["2.0"], 0.20)
        self.assertEqual(snap["checkpoints"]["30"]["speedup_best_geomean"], 1.4)
        self.assertEqual(snap["checkpoints"]["10"]["speedup_best_n"], 40)

    def test_markdown_contains_required_checkpoint_section(self) -> None:
        records = [
            _record(name="base_agent_t0", mode="truncation"),
            _record(name="base_agent_markov", mode="markov_report"),
        ]
        markdown = compare.build_markdown(
            doc={
                "generated_at_utc": "2026-08-16T00:00:00+00:00",
                "runs_root": "/tmp/runs",
                "baseline_file": "/tmp/baseline.json",
                "speedup_aggregate_policy": "correct_only_exclude_hack",
                "fast_p_thresholds": [0.0, 1.0, 2.0],
            },
            records=records,
            baseline_run="base_agent_t0",
            fast_p=1.0,
            stride=5,
            thresholds=[0.0, 1.0, 2.0],
        )
        self.assertIn("## Required checkpoints: iterations 10 and 30", markdown)
        self.assertIn("I10 @0", markdown)
        self.assertIn("I30 geomean", markdown)
        self.assertIn("markov_report", markdown)


if __name__ == "__main__":
    unittest.main()
