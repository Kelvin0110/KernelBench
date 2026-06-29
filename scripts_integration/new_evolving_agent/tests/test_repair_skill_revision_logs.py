"""Tests for repair_skill_revision_logs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts_integration.new_evolving_agent.repair.repair_skill_revision_logs import (
    _repair_entry,
    rebuild_skill_revisions_txt,
    repair_run,
)


def test_repair_entry_parses_json_blob() -> None:
    blob = (
        '{"revised_skill": "Title: Fixed\\nDescription: d\\nApplicability Trigger: t\\n'
        'Details:\\n- ```cpp\\nconstexpr int N = 32;\\n```", '
        '"revision_trace": "Defined N."}'
    )
    entry = {
        "entry_id": "2",
        "parent_id": "1",
        "content": blob,
        "revision_trace": None,
        "title": blob[:40],
        "description": blob[:40],
        "trigger": "t",
        "version": 2,
        "status": "active",
        "refinement_round": 1,
        "refinement_meta": {},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    fixed, changed = _repair_entry(entry)
    assert changed
    assert fixed["revision_trace"] == "Defined N."
    assert "constexpr int N = 32" in fixed["content"]
    assert not fixed["content"].lstrip().startswith("{")


def test_repair_run_rewrites_jsonl_and_skill_revisions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    blob = (
        '{"revised_skill": "Title: R\\nDescription: d\\nApplicability Trigger: t\\n'
        'Details:\\n- x", "revision_trace": "trace text"}'
    )
    rows = [
        {
            "entry_id": "1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "title": "Base",
            "description": "d",
            "trigger": "t",
            "content": "Title: Base\nDescription: d\nApplicability Trigger: t\nDetails:\n- x",
            "parent_id": None,
            "version": 1,
            "status": "superseded",
            "refinement_round": None,
            "refinement_meta": {},
            "revision_trace": None,
        },
        {
            "entry_id": "2",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "title": "Base",
            "description": blob[:20],
            "trigger": "t",
            "content": blob,
            "parent_id": "1",
            "version": 2,
            "status": "active",
            "refinement_round": 1,
            "refinement_meta": {"refinement_round": 1},
            "revision_trace": None,
        },
    ]
    (run_dir / "shared_l1.txt").write_text("", encoding="utf-8")
    (run_dir / "shared_l1.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "skill_revisions.txt").write_text("stale\n", encoding="utf-8")

    stats = repair_run(run_dir, dry_run=False, backup=False)
    assert stats["repaired"] == 1

    fixed_rows = [
        json.loads(line)
        for line in (run_dir / "shared_l1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert fixed_rows[1]["revision_trace"] == "trace text"
    txt = (run_dir / "skill_revisions.txt").read_text(encoding="utf-8")
    assert "trace text" in txt
    assert "Title: R" in txt
    assert "stale" not in txt

    rebuilt = rebuild_skill_revisions_txt(fixed_rows)
    assert rebuilt == txt
