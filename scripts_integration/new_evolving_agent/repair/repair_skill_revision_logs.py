"""Re-parse refined L1 skills and rebuild skill_revisions.txt for one evolving run.

Use when an older run stored raw revision JSON in ``shared_l1.jsonl`` because
``parse_revision`` mishandled `` ``` `` fences inside ``revised_skill``.

Example::

    uv run python scripts_integration/new_evolving_agent/repair/repair_skill_revision_logs.py \\
        --run-name base_agent_with_skill_refinement_ver2_itr20_2026_06_27_08_35

    uv run python scripts_integration/new_evolving_agent/repair/repair_skill_revision_logs.py \\
        --run-dir runs_evolving/my_run --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
for _path in (REPO_ROOT, SEA_ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evolving_common.governor.skill_refinement import parse_revision
from evolving_common.memory_manager import (
    _extract_summary_description,
    _extract_summary_title,
    _extract_summary_trigger,
    resolve_l1_jsonl_path,
    resolve_skill_revisions_path,
)


def _resolve_run_dir(*, runs_root: Path, run_name: str | None, run_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = run_dir.resolve()
    elif run_name:
        resolved = (runs_root / run_name).resolve()
    else:
        raise ValueError("Provide --run-name or --run-dir")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"L1 JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _looks_like_unparsed_revision_blob(content: str) -> bool:
    text = content.strip()
    return text.startswith('{"revised_skill"') or text.startswith("{\"revised_skill\"")


def _entry_needs_repair(entry: dict[str, Any]) -> bool:
    if not entry.get("parent_id"):
        return False
    content = str(entry.get("content") or "").strip()
    if not content:
        return False
    if _looks_like_unparsed_revision_blob(content):
        return True
    if not entry.get("revision_trace"):
        revised, trace = parse_revision(content)
        return bool(trace and revised and not _looks_like_unparsed_revision_blob(revised))
    return False


def _repair_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    content = str(entry.get("content") or "").strip()
    revised, trace = parse_revision(content)
    if not revised or _looks_like_unparsed_revision_blob(revised):
        return entry, False
    if revised == content and trace == entry.get("revision_trace"):
        return entry, False

    updated = dict(entry)
    updated["content"] = revised
    if trace:
        updated["revision_trace"] = trace
    if _looks_like_unparsed_revision_blob(str(updated.get("title") or "")):
        updated["title"] = _extract_summary_title(revised)
    if _looks_like_unparsed_revision_blob(str(updated.get("description") or "")):
        updated["description"] = _extract_summary_description(revised)
    trigger = str(updated.get("trigger") or "").strip()
    if not trigger or _looks_like_unparsed_revision_blob(trigger):
        updated["trigger"] = _extract_summary_trigger(revised)
    return updated, True


def _format_revision_block(entry: dict[str, Any], parent_entry: dict[str, Any]) -> str:
    meta = entry.get("refinement_meta") or {}
    return (
        f"\n=== Skill revision | {entry.get('timestamp')} | "
        f"entry_id={entry.get('entry_id')} parent_id={entry.get('parent_id')} "
        f"version={entry.get('version')} round={entry.get('refinement_round')} "
        f"status={entry.get('status')} ===\n"
        f"parent_title: {parent_entry.get('title')}\n"
        f"refinement_meta: {json.dumps(meta, ensure_ascii=False)}\n"
        f"revision_trace:\n{(entry.get('revision_trace') or '(none)')}\n"
        f"refined_content:\n{entry.get('content') or '(empty)'}\n"
    )


def rebuild_skill_revisions_txt(entries: list[dict[str, Any]]) -> str:
    by_id = {str(e.get("entry_id", "")).strip(): e for e in entries if e.get("entry_id")}
    blocks: list[str] = []
    for entry in entries:
        parent_id = str(entry.get("parent_id") or "").strip()
        if not parent_id:
            continue
        parent = by_id.get(parent_id, {"title": "(unknown parent)"})
        blocks.append(_format_revision_block(entry, parent))
    return "".join(blocks)


def repair_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, int]:
    l1_anchor = run_dir / "shared_l1.txt"
    if not l1_anchor.is_file():
        l1_anchor = run_dir / "shared_l1.jsonl"
    jsonl_path = resolve_l1_jsonl_path(l1_anchor)
    revisions_path = resolve_skill_revisions_path(l1_anchor)

    entries = _read_jsonl(jsonl_path)
    repaired = 0
    skipped = 0
    out_rows: list[dict[str, Any]] = []
    for entry in entries:
        if _entry_needs_repair(entry):
            fixed, changed = _repair_entry(entry)
            if changed:
                repaired += 1
                out_rows.append(fixed)
            else:
                skipped += 1
                out_rows.append(entry)
        else:
            out_rows.append(entry)

    refined_count = sum(1 for e in out_rows if e.get("parent_id"))
    new_txt = rebuild_skill_revisions_txt(out_rows)

    if dry_run:
        return {
            "entries": len(entries),
            "refined_entries": refined_count,
            "repaired": repaired,
            "skipped": skipped,
            "jsonl_path": 0,
            "revisions_path": 0,
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if backup:
        if jsonl_path.is_file():
            shutil.copy2(jsonl_path, jsonl_path.with_suffix(jsonl_path.suffix + f".bak.{stamp}"))
        if revisions_path.is_file():
            shutil.copy2(
                revisions_path,
                revisions_path.with_name(revisions_path.name + f".bak.{stamp}"),
            )

    _write_jsonl(jsonl_path, out_rows)
    revisions_path.write_text(new_txt, encoding="utf-8")
    return {
        "entries": len(entries),
        "refined_entries": refined_count,
        "repaired": repaired,
        "skipped": skipped,
        "jsonl_path": 1,
        "revisions_path": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-parse refined skills and rebuild skill_revisions.txt for one run.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs_evolving",
        help="Root directory containing evolving run folders (default: runs_evolving)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name under --runs-root")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit path to the run directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many entries would be repaired without writing files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak.* copies before overwriting JSONL / skill_revisions.txt",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(
        runs_root=args.runs_root.resolve(),
        run_name=args.run_name,
        run_dir=args.run_dir,
    )
    stats = repair_run(run_dir, dry_run=bool(args.dry_run), backup=not args.no_backup)

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"[{mode}] run_dir={run_dir}")
    print(
        f"  entries={stats['entries']} refined={stats['refined_entries']} "
        f"repaired={stats['repaired']} skipped_unparsed={stats['skipped']}"
    )
    if not args.dry_run:
        print(f"  wrote {run_dir / 'shared_l1.jsonl'} (or sibling) and skill_revisions.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
