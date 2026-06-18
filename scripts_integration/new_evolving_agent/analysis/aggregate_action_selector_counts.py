"""Aggregate action-selector choices for selected evolving-agent runs.

For each workspace under runs_evolving/<run>/workspaces/, reads chat_history.jsonl and
counts how often the action_selector phase chose:
  - propose_new
  - refine_current
  - debug_current

Output:
  scripts_integration/new_evolving_agent/analysis/aggregated_action_selector_counts.json

Example:
    uv run python scripts_integration/new_evolving_agent/analysis/aggregate_action_selector_counts.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernelbench.performance_stats import parse_workspace_name, write_json

# Edit this list to choose which runs to aggregate.
EVOLVING_RUN_NAMES: list[str] = [
    "memory_evolving_agent_gen3_itr20_2026_06_04_11_34",
    "memory_evolving_agent_base_itr50_new_prompt_2026_06_13_13_00",
]

TRACKED_ACTIONS = ("propose_new", "refine_current", "debug_current")
ACTION_SELECTOR_PHASE = "action_selector"
_ACTION_FIELD_RE = re.compile(
    r"""["']action["']\s*:\s*["'](propose_new|refine_current|debug_current)["']""",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "scripts_integration"
    / "new_evolving_agent"
    / "analysis"
    / "aggregated_action_selector_counts.json"
)


def _empty_action_counts() -> dict[str, int]:
    return {action: 0 for action in TRACKED_ACTIONS}


def _normalize_action(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    text = re.sub(r"[\s-]+", "_", text)
    aliases = {
        "propose_new": "propose_new",
        "propose": "propose_new",
        "refine_current": "refine_current",
        "refine": "refine_current",
        "debug_current": "debug_current",
        "debug": "debug_current",
    }
    return aliases.get(text)


def _read_chat_history_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read chat_history.jsonl with explicit UTF-8 decoding."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _strip_json_fence(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


def _collect_selector_text_candidates(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add_text(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    add_text(row.get("assistant_text"))
    add_text(row.get("assistant_content"))

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "assistant":
                continue
            add_text(message.get("content"))
            break

    return candidates


def _extract_action_from_text(text: str) -> tuple[str | None, str]:
    """Return (normalized_action, parse_method_or_error)."""
    text = text.strip()
    if not text:
        return None, "empty_selector_response"

    for candidate in (text, _strip_json_fence(text)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            action = _normalize_action(payload.get("action"))
            if action is not None:
                return action, "json"

    match = _ACTION_FIELD_RE.search(text)
    if match is not None:
        action = _normalize_action(match.group(1))
        if action is not None:
            return action, "regex"

    return None, "action_not_found"


def _parse_action_from_selector_row(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return (normalized_action, parse_method_or_error)."""
    for text in _collect_selector_text_candidates(row):
        action, method = _extract_action_from_text(text)
        if action is not None:
            return action, method
    return None, "empty_selector_response"


def _count_actions_for_workspace(chat_path: Path) -> dict[str, Any]:
    rows = _read_chat_history_jsonl(chat_path)
    per_iteration: dict[int, str] = {}
    parse_errors = 0
    parse_recoveries = 0
    unknown_actions: Counter[str] = Counter()
    parse_methods: Counter[str] = Counter()

    for row in rows:
        if row.get("phase") != ACTION_SELECTOR_PHASE:
            continue
        try:
            iteration = int(row.get("iteration"))
        except (TypeError, ValueError):
            continue

        action, method = _parse_action_from_selector_row(row)
        if action is None:
            parse_errors += 1
            unknown_actions[method] += 1
            continue

        parse_methods[method] += 1
        if method == "regex":
            parse_recoveries += 1
        per_iteration[iteration] = action

    counts = _empty_action_counts()
    for action in per_iteration.values():
        counts[action] += 1

    return {
        "selector_iterations": len(per_iteration),
        "actions": counts,
        "parse_errors": parse_errors,
        "parse_recoveries": parse_recoveries,
        "parse_methods": dict(parse_methods),
        "unknown_action_labels": dict(unknown_actions),
    }


def aggregate_run_actions(*, run_name: str, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / run_name
    workspaces_dir = run_dir / "workspaces"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not workspaces_dir.is_dir():
        raise FileNotFoundError(f"Workspaces directory not found: {workspaces_dir}")

    problems: dict[str, dict[str, Any]] = {}
    totals = _empty_action_counts()
    total_parse_errors = 0
    total_parse_recoveries = 0
    total_selector_iterations = 0

    for workspace_dir in sorted(workspaces_dir.iterdir(), key=lambda p: p.name):
        if not workspace_dir.is_dir():
            continue
        chat_path = workspace_dir / "chat_history.jsonl"
        if not chat_path.is_file():
            continue

        parsed = parse_workspace_name(workspace_dir.name)
        level, problem_id = parsed if parsed is not None else (None, None)

        workspace_stats = _count_actions_for_workspace(chat_path)
        actions = workspace_stats["actions"]
        for action in TRACKED_ACTIONS:
            totals[action] += int(actions.get(action, 0))
        total_parse_errors += int(workspace_stats.get("parse_errors", 0))
        total_parse_recoveries += int(workspace_stats.get("parse_recoveries", 0))
        total_selector_iterations += int(workspace_stats.get("selector_iterations", 0))

        problems[workspace_dir.name] = {
            "workspace_id": workspace_dir.name,
            "level": level,
            "problem_id": problem_id,
            "chat_history_path": str(chat_path),
            "selector_iterations": workspace_stats["selector_iterations"],
            "actions": actions,
            "parse_errors": workspace_stats["parse_errors"],
            "parse_recoveries": workspace_stats.get("parse_recoveries", 0),
            "parse_methods": workspace_stats.get("parse_methods", {}),
            "unknown_action_labels": workspace_stats["unknown_action_labels"],
        }

    if not problems:
        raise RuntimeError(f"No workspaces with chat_history.jsonl found under {workspaces_dir}")

    return {
        "source": "evolving",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "workspaces_dir": str(workspaces_dir),
        "problem_count": len(problems),
        "selector_iterations": total_selector_iterations,
        "totals": totals,
        "parse_errors": total_parse_errors,
        "parse_recoveries": total_parse_recoveries,
        "problems": problems,
    }


def build_aggregate_doc(
    *,
    evolving_run_names: list[str],
    runs_root: Path,
) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for run_name in evolving_run_names:
        try:
            runs[run_name] = aggregate_run_actions(run_name=run_name, runs_root=runs_root)
        except Exception as exc:
            errors[run_name] = str(exc)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracked_actions": list(TRACKED_ACTIONS),
        "selector_phase": ACTION_SELECTOR_PHASE,
        "evolving_run_names": list(evolving_run_names),
        "runs": runs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate action_selector action counts for evolving-agent runs."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(REPO_ROOT / "runs_evolving"),
        help="Root directory containing evolving run folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    args = parser.parse_args()

    doc = build_aggregate_doc(
        evolving_run_names=EVOLVING_RUN_NAMES,
        runs_root=Path(args.runs_root),
    )

    output_path = Path(args.output)
    write_json(output_path, doc)

    summary = {
        "output_path": str(output_path),
        "run_count": len(doc["runs"]),
        "error_count": len(doc["errors"]),
        "run_names": list(doc["runs"].keys()),
        "errors": doc["errors"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not doc["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
