"""
Merge one problem's AIDE run artifacts from a source run folder into a target batch run.

Typical use: replace a failed/problematic problem entry in a subset run with results
from a dedicated single-problem re-run (e.g. drop L1P38 and insert L1P58 as-is).

By default the source problem id and keys are preserved in the target run
(`L1P58`, `level_1_problem_58`, ...). Use ``--remap-to-target-slot`` to rewrite
everything to the target slot id instead (``L1P38``, ...).

Folder layout follows scripts_integration/docker/docker_single_run.py:

    {run_dir}/
        eval_results.json
        kernels/level_{L}_problem_{P}_sample_0_kernel.py
        logs/level_{L}_problem_{P}/...
        workspaces/level_{L}_problem_{P}/...   (optional)
        container_logs/L{L}_P{P}.log
        checkpoints/node_{NNNN}/
            eval_results.json
            checkpoint_summary.json
            kernels/level_{L}_problem_{P}_kernel.py

Example (replace subset slot 38 with single-problem L1P58 run, keep L1P58 keys):

    python scripts_integration/docker/update_run_from_source.py \\
        --target-run aide_subset_gpt_oss_120b_step40_new_problem_set \\
        --source-run aide_subset_gpt_oss_120b_step40_L1P58 \\
        --source-problem-id 58 \\
        --target-problem-id 38
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_COMPOSITE_RESULT_KEY = re.compile(r"^L(\d+)P(\d+)$", re.IGNORECASE)


def format_result_key(level: int, problem_id: int) -> str:
    return f"L{level}P{problem_id}"


def parse_result_key(key: str) -> tuple[int, int] | None:
    text = str(key).strip()
    match = _COMPOSITE_RESULT_KEY.match(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    try:
        return None, int(text)
    except Exception:
        return None


def _sort_result_key(key: str) -> tuple[Any, ...]:
    text = str(key)
    match = re.fullmatch(r"L(\d+)P(\d+)", text)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), text)
    try:
        return (1, 0, int(text), text)
    except Exception:
        return (2, 0, 0, text)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any], *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def _sorted_eval_results(data: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted(data.items(), key=lambda item: _sort_result_key(item[0])))


def _result_keys(level: int, problem_id: int) -> tuple[str, str]:
    return format_result_key(level, problem_id), str(problem_id)


def _pick_entries(eval_results: dict[str, Any], level: int, problem_id: int) -> list[dict[str, Any]] | None:
    composite, legacy = _result_keys(level, problem_id)
    entries = eval_results.get(composite)
    if entries is None:
        entries = eval_results.get(legacy)
    if not entries:
        return None
    return entries


def _normalize_final_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Final eval_results.json uses the slimmer schema from add_to_eval_results_file()."""
    keep = ("sample_id", "compiled", "correctness", "metadata", "runtime", "runtime_stats")
    out = {k: copy.deepcopy(entry[k]) for k in keep if k in entry}
    if "sample_id" not in out:
        out["sample_id"] = 0
    return out


def _normalize_checkpoint_entry(
    entry: dict[str, Any],
    *,
    level: int,
    problem_id: int,
) -> dict[str, Any]:
    out = copy.deepcopy(entry)
    out["level"] = level
    out["problem_id"] = problem_id
    if "sample_id" not in out:
        out["sample_id"] = 0
    return out


def rebuild_checkpoint_summary(
    sorted_results: dict[str, Any],
    *,
    checkpoint_node: int,
    timestamp: str | None = None,
    elapsed_hours: float | None = None,
) -> dict[str, Any]:
    """Mirror checkpoint_summary.json generation in docker_single_run.py."""
    problems_list: list[dict[str, Any]] = []
    default_level = 1

    for key_str, results in sorted_results.items():
        if not results:
            continue
        row = results[-1]
        level = row.get("level")
        problem_id = row.get("problem_id")
        parsed = parse_result_key(str(key_str))
        if parsed is not None:
            parsed_level, parsed_pid = parsed
            if level is None:
                level = parsed_level
            if problem_id is None:
                problem_id = parsed_pid
        if problem_id is None:
            try:
                problem_id = int(key_str)
            except Exception:
                continue
        if level is None:
            level = default_level
        problems_list.append(
            {
                "level": int(level),
                "problem_id": int(problem_id),
                "eval_skipped": row.get("eval_skipped", False),
                "skip_reason": row.get("skip_reason"),
                "code_changed": row.get("code_changed_since_last_checkpoint", False),
                "compiled": row.get("compiled"),
                "correct": row.get("correctness"),
                "aide_metric": row.get("aide_metric"),
                "runtime_secs": row.get("runtime"),
            }
        )

    problems_list.sort(key=lambda p: (p["level"], p["problem_id"]))
    return {
        "checkpoint_node": checkpoint_node,
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_hours": elapsed_hours,
        "problems_evaluated": problems_list,
        "summary": {
            "total_problems": len(problems_list),
            "problems_evaluated": sum(1 for p in problems_list if not p.get("eval_skipped", False)),
            "problems_skipped": sum(1 for p in problems_list if p.get("eval_skipped", False)),
            "avg_aide_metric": (
                sum(p.get("aide_metric") or 0 for p in problems_list) / len(problems_list)
                if problems_list
                else None
            ),
        },
    }


def _checkpoint_node_number(node_dir: Path) -> int | None:
    match = re.fullmatch(r"node_(\d+)", node_dir.name)
    if not match:
        return None
    return int(match.group(1))


def _insert_identity(
    *,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool,
) -> tuple[int, int]:
    if keep_source_keys:
        return src_level, src_problem_id
    return dst_level, dst_problem_id


def _purge_target_problem_from_run(
    target_run: Path,
    *,
    dst_level: int,
    dst_problem_id: int,
    dry_run: bool,
) -> None:
    """Remove all target-problem entries and artifacts before copying source data."""
    dst_key = format_result_key(dst_level, dst_problem_id)
    dst_tag = f"level_{dst_level}_problem_{dst_problem_id}"

    final_path = target_run / "eval_results.json"
    final_data = _read_json(final_path)
    if dst_key in final_data:
        if dry_run:
            print(f"[dry-run] purge {dst_key} from {final_path}")
        else:
            final_data.pop(dst_key, None)
            _write_json(final_path, _sorted_eval_results(final_data), indent=4)

    artifact_paths = [
        target_run / "kernels" / f"{dst_tag}_sample_0_kernel.py",
        target_run / "container_logs" / f"L{dst_level}_P{dst_problem_id}.log",
        target_run / "logs" / dst_tag,
        target_run / "workspaces" / dst_tag,
    ]
    for path in artifact_paths:
        _remove_path(path, dry_run=dry_run)

    checkpoints_dir = target_run / "checkpoints"
    if checkpoints_dir.is_dir():
        for node_dir in sorted(checkpoints_dir.iterdir(), key=lambda p: p.name):
            if not node_dir.is_dir():
                continue
            eval_path = node_dir / "eval_results.json"
            eval_data = _read_json(eval_path)
            if dst_key not in eval_data:
                continue
            if dry_run:
                print(f"[dry-run] purge {dst_key} from {eval_path}")
            else:
                eval_data.pop(dst_key, None)
                _write_json(eval_path, _sorted_eval_results(eval_data), indent=2)
            ck_kernel = node_dir / "kernels" / f"{dst_tag}_kernel.py"
            _remove_path(ck_kernel, dry_run=dry_run)


def _load_source_checkpoint_entries(
    source_ckpts: Path,
    *,
    src_level: int,
    src_problem_id: int,
) -> dict[int, tuple[Path, dict[str, Any]]]:
    """Map checkpoint node number -> (node_dir, normalized entry)."""
    out: dict[int, tuple[Path, dict[str, Any]]] = {}
    if not source_ckpts.is_dir():
        return out
    for node_dir in source_ckpts.iterdir():
        if not node_dir.is_dir():
            continue
        node_num = _checkpoint_node_number(node_dir)
        if node_num is None:
            continue
        source_eval = _read_json(node_dir / "eval_results.json")
        src_entries = _pick_entries(source_eval, src_level, src_problem_id)
        if not src_entries:
            continue
        out[node_num] = (node_dir, src_entries[-1])
    return out


def _resolve_source_for_target_node(
    target_node_num: int,
    source_by_num: dict[int, tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any], str] | None:
    """Pick source checkpoint for a target node (exact match or forward-fill)."""
    if not source_by_num:
        return None
    if target_node_num in source_by_num:
        node_dir, entry = source_by_num[target_node_num]
        return node_dir, entry, "exact"
    prior = [n for n in source_by_num if n <= target_node_num]
    if prior:
        node_num = max(prior)
        node_dir, entry = source_by_num[node_num]
        return node_dir, entry, "forward_prior"
    first_num = min(source_by_num)
    node_dir, entry = source_by_num[first_num]
    return node_dir, entry, "forward_first"


def _remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run] remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _build_replacement_pairs(
    *,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
) -> list[tuple[str, str]]:
    """Text replacements applied inside copied artifacts."""
    src_tag = f"level_{src_level}_problem_{src_problem_id}"
    dst_tag = f"level_{dst_level}_problem_{dst_problem_id}"
    pairs = [
        (src_tag, dst_tag),
        (format_result_key(src_level, src_problem_id), format_result_key(dst_level, dst_problem_id)),
        (f"L{src_level}_P{src_problem_id}", f"L{dst_level}_P{dst_problem_id}"),
        (f"/P{src_problem_id}", f"/P{dst_problem_id}"),
        (f"- P{src_problem_id}\r\n", f"- P{dst_problem_id}\r\n"),
        (f"- P{src_problem_id}\n", f"- P{dst_problem_id}\n"),
        (f"problem_id: {src_problem_id}", f"problem_id: {dst_problem_id}"),
        (f"\"problem_id\": {src_problem_id}", f"\"problem_id\": {dst_problem_id}"),
    ]
    # De-duplicate while preserving order (longer strings first).
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for old, new in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        if old == new or old in seen:
            continue
        seen.add(old)
        ordered.append((old, new))
    return ordered


_TEXT_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".log",
    ".html",
    ".txt",
    ".md",
    ".toml",
    ".csv",
}


def _apply_text_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _rewrite_text_file(path: Path, replacements: list[tuple[str, str]], *, dry_run: bool) -> bool:
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    updated = _apply_text_replacements(original, replacements)
    if updated == original:
        return False
    if dry_run:
        print(f"[dry-run] rewrite contents {path}")
        return True
    path.write_text(updated, encoding="utf-8")
    return True


def _rename_if_needed(path: Path, replacements: list[tuple[str, str]], *, dry_run: bool) -> Path:
    new_name = path.name
    for old, new in replacements:
        new_name = new_name.replace(old, new)
    if new_name == path.name:
        return path
    target = path.with_name(new_name)
    if dry_run:
        print(f"[dry-run] rename {path} -> {target}")
        return target
    path.rename(target)
    return target


def _remap_problem_references(
    root: Path,
    *,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    dry_run: bool,
) -> None:
    """Rename leftover source paths and rewrite embedded source keys to target keys."""
    if not root.exists():
        return

    replacements = _build_replacement_pairs(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
    )

    if root.is_file():
        _rewrite_text_file(root, replacements, dry_run=dry_run)
        return

    src_tag = f"level_{src_level}_problem_{src_problem_id}"

    paths = sorted(
        [root, *root.rglob("*")],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    renamed: dict[Path, Path] = {root: root}
    for path in paths:
        parent = renamed.get(path.parent, path.parent)
        current = parent / path.name
        if src_tag in current.name:
            current = _rename_if_needed(current, replacements, dry_run=dry_run)
        renamed[path] = current

    final_root = renamed[root]
    for path in sorted(final_root.rglob("*"), key=lambda p: p.as_posix()):
        if path.is_file():
            _rewrite_text_file(path, replacements, dry_run=dry_run)


def _copy_file(
    src: Path,
    dst: Path,
    *,
    dry_run: bool,
    remap: dict[str, Any] | None = None,
) -> bool:
    if not src.is_file():
        return False
    if dry_run:
        print(f"[dry-run] copy {src} -> {dst}")
        if remap is not None:
            print(f"[dry-run] remap references under {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if remap is not None:
        _remap_problem_references(
            dst,
            src_level=remap["src_level"],
            src_problem_id=remap["src_problem_id"],
            dst_level=remap["dst_level"],
            dst_problem_id=remap["dst_problem_id"],
            dry_run=dry_run,
        )
    return True


def _copy_problem_tree(
    src_root: Path,
    dst_root: Path,
    *,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool,
    dry_run: bool,
) -> list[str]:
    copied: list[str] = []
    src_tag = f"level_{src_level}_problem_{src_problem_id}"
    insert_level, insert_problem_id = _insert_identity(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
    )
    insert_tag = f"level_{insert_level}_problem_{insert_problem_id}"
    remap = None if keep_source_keys else {
        "src_level": src_level,
        "src_problem_id": src_problem_id,
        "dst_level": dst_level,
        "dst_problem_id": dst_problem_id,
    }

    file_pairs = [
        (
            src_root / "kernels" / f"{src_tag}_sample_0_kernel.py",
            dst_root / "kernels" / f"{insert_tag}_sample_0_kernel.py",
        ),
        (
            src_root / "container_logs" / f"L{src_level}_P{src_problem_id}.log",
            dst_root / "container_logs" / f"L{insert_level}_P{insert_problem_id}.log",
        ),
    ]
    for src, dst in file_pairs:
        _remove_path(dst, dry_run=dry_run)
        if _copy_file(src, dst, dry_run=dry_run, remap=remap):
            copied.append(str(dst.relative_to(dst_root)))

    for subdir in ("logs", "workspaces"):
        src_dir = src_root / subdir / src_tag
        dst_dir = dst_root / subdir / insert_tag
        if not src_dir.is_dir():
            continue
        _remove_path(dst_dir, dry_run=dry_run)
        if dry_run:
            print(f"[dry-run] copytree {src_dir} -> {dst_dir}")
        else:
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dst_dir)
        if remap is not None:
            _remap_problem_references(
                dst_dir,
                src_level=src_level,
                src_problem_id=src_problem_id,
                dst_level=dst_level,
                dst_problem_id=dst_problem_id,
                dry_run=dry_run,
            )
        copied.append(str(dst_dir.relative_to(dst_root)))

    return copied


def _copy_checkpoint_kernels(
    src_ckpt: Path,
    dst_ckpt: Path,
    *,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool,
    dry_run: bool,
) -> bool:
    src = src_ckpt / "kernels" / f"level_{src_level}_problem_{src_problem_id}_kernel.py"
    insert_level, insert_problem_id = _insert_identity(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
    )
    dst = dst_ckpt / "kernels" / f"level_{insert_level}_problem_{insert_problem_id}_kernel.py"
    _remove_path(dst, dry_run=dry_run)
    remap = None if keep_source_keys else {
        "src_level": src_level,
        "src_problem_id": src_problem_id,
        "dst_level": dst_level,
        "dst_problem_id": dst_problem_id,
    }
    return _copy_file(src, dst, dry_run=dry_run, remap=remap)


def merge_final_eval_results(
    *,
    target_run: Path,
    source_run: Path,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool,
    dry_run: bool,
) -> None:
    target_path = target_run / "eval_results.json"
    source_path = source_run / "eval_results.json"
    target_data = _read_json(target_path)
    source_data = _read_json(source_path)

    src_entries = _pick_entries(source_data, src_level, src_problem_id)
    if not src_entries:
        raise FileNotFoundError(
            f"No source eval entry for L{src_level}P{src_problem_id} in {source_path}"
        )

    insert_level, insert_problem_id = _insert_identity(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
    )
    insert_key = format_result_key(insert_level, insert_problem_id)
    dst_key = format_result_key(dst_level, dst_problem_id)
    target_data[insert_key] = [_normalize_final_entry(src_entries[-1])]
    if insert_key != dst_key:
        target_data.pop(dst_key, None)
    sorted_results = _sorted_eval_results(target_data)

    if dry_run:
        print(f"[dry-run] write {target_path} ({insert_key} inserted, {dst_key} removed)")
        return
    _write_json(target_path, sorted_results, indent=4)


def merge_checkpoint_eval_results(
    *,
    target_run: Path,
    source_run: Path,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "merged": 0,
        "forward_filled": 0,
        "skipped_no_source": 0,
        "skipped_no_target": 0,
    }
    target_ckpts = target_run / "checkpoints"
    source_ckpts = source_run / "checkpoints"
    if not target_ckpts.is_dir():
        print(f"Warning: no checkpoints directory in target run: {target_ckpts}")
        return stats
    if not source_ckpts.is_dir():
        print(f"Warning: no checkpoints directory in source run: {source_ckpts}")
        return stats

    source_by_num = _load_source_checkpoint_entries(
        source_ckpts,
        src_level=src_level,
        src_problem_id=src_problem_id,
    )
    insert_level, insert_problem_id = _insert_identity(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
    )
    insert_key = format_result_key(insert_level, insert_problem_id)
    dst_key = format_result_key(dst_level, dst_problem_id)
    dst_tag = f"level_{dst_level}_problem_{dst_problem_id}"

    for target_node in sorted(target_ckpts.iterdir(), key=lambda p: p.name):
        if not target_node.is_dir():
            continue
        target_node_num = _checkpoint_node_number(target_node)
        if target_node_num is None:
            continue

        resolved = _resolve_source_for_target_node(target_node_num, source_by_num)
        if resolved is None:
            stats["skipped_no_source"] += 1
            continue

        source_node, src_entry, match_mode = resolved
        if match_mode != "exact":
            stats["forward_filled"] += 1

        target_eval_path = target_node / "eval_results.json"
        target_eval = _read_json(target_eval_path)

        target_eval[insert_key] = [
            _normalize_checkpoint_entry(
                src_entry,
                level=insert_level,
                problem_id=insert_problem_id,
            )
        ]
        if insert_key != dst_key:
            target_eval.pop(dst_key, None)
        sorted_results = _sorted_eval_results(target_eval)

        summary_path = target_node / "checkpoint_summary.json"
        prev_summary = _read_json(summary_path)
        checkpoint_node = target_node_num or prev_summary.get("checkpoint_node") or 0
        summary_doc = rebuild_checkpoint_summary(
            sorted_results,
            checkpoint_node=int(checkpoint_node),
            timestamp=prev_summary.get("timestamp"),
            elapsed_hours=prev_summary.get("elapsed_hours"),
        )

        _copy_checkpoint_kernels(
            source_node,
            target_node,
            src_level=src_level,
            src_problem_id=src_problem_id,
            dst_level=dst_level,
            dst_problem_id=dst_problem_id,
            keep_source_keys=keep_source_keys,
            dry_run=dry_run,
        )
        if keep_source_keys and insert_key != dst_key:
            _remove_path(target_node / "kernels" / f"{dst_tag}_kernel.py", dry_run=dry_run)

        if dry_run:
            print(f"[dry-run] write {target_eval_path} and {summary_path} ({target_node.name}, {match_mode})")
        else:
            _write_json(target_eval_path, sorted_results, indent=2)
            _write_json(summary_path, summary_doc, indent=2)
        stats["merged"] += 1

    return stats


def update_run_from_source(
    *,
    target_run: Path,
    source_run: Path,
    src_level: int,
    src_problem_id: int,
    dst_level: int,
    dst_problem_id: int,
    keep_source_keys: bool = True,
    dry_run: bool = False,
) -> None:
    if not target_run.is_dir():
        raise FileNotFoundError(f"Target run not found: {target_run}")
    if not source_run.is_dir():
        raise FileNotFoundError(f"Source run not found: {source_run}")

    insert_level, insert_problem_id = _insert_identity(
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
    )

    print(f"Target run: {target_run}")
    print(f"Source run: {source_run}")
    if keep_source_keys:
        print(
            f"Removing L{dst_level}P{dst_problem_id} from target; "
            f"inserting L{insert_level}P{insert_problem_id} with source keys"
        )
    else:
        print(
            f"Replacing L{dst_level}P{dst_problem_id} with data from "
            f"L{src_level}P{src_problem_id} (remapped to target slot)"
        )

    _purge_target_problem_from_run(
        target_run,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        dry_run=dry_run,
    )
    print("Purged existing target-problem artifacts and eval keys")

    copied = _copy_problem_tree(
        source_run,
        target_run,
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
        dry_run=dry_run,
    )
    if copied:
        print("Copied artifacts:")
        for path in copied:
            print(f"  - {path}")
    else:
        print("Warning: no top-level problem artifacts copied from source run")

    merge_final_eval_results(
        target_run=target_run,
        source_run=source_run,
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
        dry_run=dry_run,
    )
    print("Updated final eval_results.json")

    ckpt_stats = merge_checkpoint_eval_results(
        target_run=target_run,
        source_run=source_run,
        src_level=src_level,
        src_problem_id=src_problem_id,
        dst_level=dst_level,
        dst_problem_id=dst_problem_id,
        keep_source_keys=keep_source_keys,
        dry_run=dry_run,
    )
    print(
        "Checkpoint updates: "
        f"merged={ckpt_stats['merged']} "
        f"forward_filled={ckpt_stats.get('forward_filled', 0)} "
        f"skipped_no_source={ckpt_stats['skipped_no_source']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace one problem's artifacts in a batch run using a single-problem source run."
    )
    parser.add_argument(
        "--target-run",
        required=True,
        help="Target run folder name under run_integration/ (updated in place)",
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Source run folder name under run_integration/",
    )
    parser.add_argument(
        "--run-integration-root",
        type=Path,
        default=REPO_ROOT / "run_integration",
        help="Root directory containing run folders (default: repo run_integration/)",
    )
    parser.add_argument("--source-level", type=int, default=1, help="Source problem level")
    parser.add_argument("--source-problem-id", type=int, required=True, help="Source problem id")
    parser.add_argument("--target-level", type=int, default=1, help="Target problem level")
    parser.add_argument("--target-problem-id", type=int, required=True, help="Target problem id to replace")
    parser.add_argument(
        "--remap-to-target-slot",
        action="store_true",
        help="Rewrite source keys/paths to the target slot (e.g. L1P58 -> L1P38). "
        "Default: keep source keys (L1P58) and only remove the target slot.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args()

    target_run = args.run_integration_root / args.target_run
    source_run = args.run_integration_root / args.source_run

    update_run_from_source(
        target_run=target_run,
        source_run=source_run,
        src_level=args.source_level,
        src_problem_id=args.source_problem_id,
        dst_level=args.target_level,
        dst_problem_id=args.target_problem_id,
        keep_source_keys=not args.remap_to_target_slot,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
