"""Batch orchestrator for evolving-agent KernelBench integration (new governor path)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Support direct execution: python scripts_integration/new_evolving_agent/evolve_kb_batch.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
if str(SEA_ROOT) not in sys.path:
    sys.path.insert(0, str(SEA_ROOT))

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.performance_stats import (
    LIKELY_REWARD_HACK_SPEEDUP_THRESHOLD,
    SUSPICIOUS_SPEEDUP_WARN_THRESHOLD,
    build_baseline_lookup,
    classify_speedup_severity,
    min_non_outlier_runtime,
)
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench_integration import (
    KBGovernorConfig,
    cleanup_problem_build_artifacts,
    governor_result_to_dict,
    safe_run_kb_governor,
)
from evolving_common.memory_manager import (
    DEFAULT_ENABLE_L1_SKILL_UNIT_TEST_GC,
    DEFAULT_L1_SKILL_CONSECUTIVE_UNUSED_DELETE_AFTER,
    DEFAULT_L1_SKILL_DELETE_ON_UNIT_TEST_FAIL,
    DEFAULT_L1_SKILL_DELETION_GRACE_ITERATIONS,
    DEFAULT_L1_SKILL_UNIT_TEST_MAX_TOKENS,
    DEFAULT_L1_SKILL_UNIT_TEST_RUN_TIMEOUT_SEC,
    DEFAULT_SKILL_MERGE_INTERVAL,
    DEFAULT_SKILL_MERGE_SIMILARITY,
)


def _load_subset_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            level = int(row["level"])
            problem_id = int(row["problem_id"])
            rows.append({"level": level, "problem_id": problem_id, **row})
    return rows


def _read_json(path: Path, default: dict | list):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict | list) -> None:
    def _json_default(obj: Any) -> str:
        if isinstance(obj, BaseException):
            return f"{type(obj).__name__}: {obj}"
        if isinstance(obj, Path):
            return str(obj)
        return repr(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sum_timing_wall_seconds(path: Path) -> float:
    if not path.is_file():
        return 0.0
    total = 0.0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            try:
                total += float(obj.get("wall_time_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
    return total


def _to_kernelbench_eval_entry(run_entry: dict[str, Any], *, level: int, problem_id: str) -> dict[str, Any]:
    runtime_stats = run_entry.get("runtime_stats")
    if not isinstance(runtime_stats, dict):
        runtime_stats = {}

    raw_metadata = run_entry.get("metadata")
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}

    runtime = run_entry.get("runtime", -1.0)
    try:
        runtime = float(runtime)
    except Exception:
        runtime = -1.0

    merged_metadata = {
        "hardware": raw_metadata.get("hardware") or runtime_stats.get("hardware"),
        "device": raw_metadata.get("device") or runtime_stats.get("device"),
        "correctness_trials": raw_metadata.get("correctness_trials"),
        "source": "new_evolving_agent",
        "level": int(level),
        "problem_id": int(problem_id),
        "best_speedup": float(run_entry.get("best_speedup", 0.0) or 0.0),
        "backend": run_entry.get("backend"),
        "precision": run_entry.get("precision"),
        "iterations_run": int(run_entry.get("iterations_run", 0) or 0),
        "error": run_entry.get("error"),
    }

    return {
        "sample_id": 0,
        "compiled": bool(run_entry.get("best_compiled", False)),
        "correctness": bool(run_entry.get("best_correct", False)),
        "metadata": merged_metadata,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
    }


def _normalize_level_first_eval_doc(payload: dict | list | None) -> dict[str, dict[str, list]]:
    if not isinstance(payload, dict):
        return {}

    if payload and all(isinstance(v, dict) for v in payload.values()):
        return payload

    normalized: dict[str, dict[str, list]] = {}
    for problem_id, entries in payload.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            level_value = metadata.get("level")
            if level_value is None:
                continue
            level_key = str(level_value)
            pid_key = str(problem_id)
            normalized.setdefault(level_key, {})
            normalized[level_key].setdefault(pid_key, [])
            normalized[level_key][pid_key].append(entry)
    return normalized


def _level_eval_path(run_dir: Path, level: int) -> Path:
    return run_dir / f"eval_results_level_{level}.json"


def _problem_key(level: int, problem_id: str) -> str:
    return f"L{level}P{problem_id}"


def _remove_run_entry(
    runs: list[dict[str, Any]], *, level: int, problem_id: str
) -> list[dict[str, Any]]:
    pid = str(problem_id)
    return [
        entry
        for entry in runs
        if not (
            isinstance(entry, dict)
            and int(entry.get("level", -1)) == level
            and str(entry.get("problem_id")) == pid
        )
    ]


def _remove_eval_entry(
    eval_doc: dict[str, dict[str, list]], *, level: int, problem_id: str
) -> None:
    level_key = str(level)
    pid_key = str(problem_id)
    level_bucket = eval_doc.get(level_key)
    if isinstance(level_bucket, dict) and pid_key in level_bucket:
        del level_bucket[pid_key]


def _clear_problem_workspace(run_dir: Path, *, level: int, problem_id: str) -> None:
    workspace_dir = run_dir / "workspaces" / f"level_{level}_problem_{problem_id}"
    if workspace_dir.is_dir():
        shutil.rmtree(workspace_dir)


def _remove_kernel_export(run_dir: Path, *, level: int, problem_id: str) -> None:
    kernel_path = (
        run_dir
        / "kernels"
        / f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
    )
    if kernel_path.is_file():
        kernel_path.unlink()


def _purge_problem_state(
    *,
    run_dir: Path,
    runs: list[dict[str, Any]],
    eval_doc: dict[str, dict[str, list]],
    level_eval_docs: dict[int, dict[str, list]],
    level: int,
    problem_id: str,
) -> list[dict[str, Any]]:
    runs = _remove_run_entry(runs, level=level, problem_id=problem_id)
    _remove_eval_entry(eval_doc, level=level, problem_id=problem_id)
    level_bucket = level_eval_docs.get(level)
    if isinstance(level_bucket, dict):
        pid_key = str(problem_id)
        if pid_key in level_bucket:
            del level_bucket[pid_key]
    _clear_problem_workspace(run_dir, level=level, problem_id=problem_id)
    _remove_kernel_export(run_dir, level=level, problem_id=problem_id)
    cleanup_problem_build_artifacts(
        run_dir.parent,
        run_dir.name,
        level=level,
        problem_id=problem_id,
    )
    return runs


def _warn_resume_config_mismatch(
    *,
    run_dir: Path,
    subset_csv: Path,
    max_problems: int,
) -> None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return
    prior = _read_json(summary_path, default={})
    if not isinstance(prior, dict):
        return
    prior_subset = prior.get("subset_csv")
    if prior_subset and str(prior_subset) != str(subset_csv):
        print(
            f"[batch] warning: resume subset_csv differs from prior run "
            f"({prior_subset} vs {subset_csv})",
            file=sys.stderr,
        )
    prior_attempted = prior.get("total_attempted")
    if prior_attempted is not None and max_problems > 0 and int(prior_attempted) != max_problems:
        print(
            f"[batch] warning: resume --max-problems={max_problems} differs from "
            f"prior total_attempted={prior_attempted}",
            file=sys.stderr,
        )


def _extract_best_kernel_code(run_entry: dict[str, Any]) -> str | None:
    best_code = run_entry.get("best_code")
    if isinstance(best_code, str) and best_code.strip():
        return best_code

    records = run_entry.get("records")
    if isinstance(records, list):
        for record in reversed(records):
            if not isinstance(record, dict):
                continue
            candidate = record.get("candidate_code")
            if isinstance(candidate, str) and candidate.strip():
                return candidate

    return None


def _write_kernel_export(run_dir: Path, *, level: int, problem_id: str, code: str) -> Path:
    kernels_dir = run_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    export_path = kernels_dir / f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
    export_path.write_text(code, encoding="utf-8")
    return export_path


def _summarize_per_level_runs(
    runs: list[dict[str, Any]],
    *,
    likely_hack_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    likely_hack_keys = likely_hack_keys or set()
    per_level: dict[str, dict[str, Any]] = {}
    level_runtime_speedups: dict[str, list[tuple[float, float]]] = {}
    for entry in runs:
        level = str(entry.get("level"))
        problem_id = entry.get("problem_id")
        entry_key = f"L{level}P{problem_id}"
        bucket = per_level.setdefault(
            level,
            {
                "completed": 0,
                "correct": 0,
                "best_speedup": 0.0,
                "best_runtime": None,
            },
        )
        bucket["completed"] += 1
        if entry.get("best_correct"):
            bucket["correct"] += 1
        try:
            runtime_f = float(entry.get("runtime"))
        except Exception:
            runtime_f = -1.0
        if (
            entry.get("best_correct")
            and runtime_f >= 0
            and entry_key not in likely_hack_keys
        ):
            level_runtime_speedups.setdefault(level, []).append(
                (runtime_f, float(entry.get("best_speedup", 0.0) or 0.0))
            )

    for level, bucket in per_level.items():
        pairs = level_runtime_speedups.get(level, [])
        if not pairs:
            continue
        best_runtime = min_non_outlier_runtime([runtime for runtime, _ in pairs])
        if best_runtime is None:
            continue
        bucket["best_runtime"] = best_runtime
        for runtime_f, speedup_f in pairs:
            if runtime_f == best_runtime:
                bucket["best_speedup"] = speedup_f
                break
    return per_level


def _collect_suspicious_speedup_problems(
    runs: list[dict[str, Any]],
    *,
    baseline_results: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return audit records and keys excluded as likely_reward_hack."""
    suspicious: list[dict[str, Any]] = []
    likely_hack_keys: set[str] = set()
    baseline_by_level: dict[int, dict[int, float]] = {}
    if isinstance(baseline_results, dict):
        for level in (1, 2, 3):
            baseline_by_level[level] = build_baseline_lookup(baseline_results, level)

    for entry in runs:
        if not entry.get("best_correct"):
            continue
        try:
            speedup = float(entry.get("best_speedup", 0.0) or 0.0)
        except Exception:
            continue
        severity = classify_speedup_severity(speedup)
        if severity is None:
            continue

        level = int(entry.get("level"))
        problem_id = int(entry.get("problem_id"))
        entry_key = f"L{level}P{problem_id}"
        baseline_runtime = baseline_by_level.get(level, {}).get(problem_id)
        try:
            best_runtime = float(entry.get("runtime"))
        except Exception:
            best_runtime = None

        threshold = (
            LIKELY_REWARD_HACK_SPEEDUP_THRESHOLD
            if severity == "likely_reward_hack"
            else SUSPICIOUS_SPEEDUP_WARN_THRESHOLD
        )
        suspicious.append(
            {
                "workspace_id": f"level_{level}_problem_{problem_id}",
                "level": level,
                "problem_id": problem_id,
                "best_speedup": speedup,
                "best_runtime": best_runtime,
                "baseline_runtime": baseline_runtime,
                "severity": severity,
                "reason": f"speedup exceeds global threshold ({threshold:g}x)",
            }
        )
        if severity == "likely_reward_hack":
            likely_hack_keys.add(entry_key)

    return suspicious, likely_hack_keys


def _check_integration_dependencies(*, dry_run: bool) -> None:
    missing: list[str] = []
    if "kernelbench" not in sys.modules:
        try:
            __import__("kernelbench.dataset")
            __import__("kernelbench.prompt_constructor_toml")
        except Exception:
            missing.append("kernelbench package")

    try:
        __import__("evolving_common")
    except Exception:
        missing.append("Self-Evolving-Agent/evolving_common")

    if missing:
        raise RuntimeError(
            "missing integration dependencies: "
            + ", ".join(missing)
            + ". Ensure repository root is installed with `uv pip install -e .` and "
            "Self-Evolving-Agent deps are installed."
        )

    if not dry_run and not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY is required for non-dry runs because llm_client uses NVIDIA OpenAI-compatible API."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run new evolving-agent KernelBench batch.")
    parser.add_argument(
        "--subset-csv",
        type=str,
        default="subset_selection/selected_problems_50.csv",
        help="CSV from subset_selection/select_problems.py",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV parsing/output generation without LLM or GPU eval.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="runs_evolving/",
        help="Base directory for run artifacts",
    )
    parser.add_argument(
        "--time-sample-interval-sec",
        type=float,
        default=300.0,
        help="Recorder sampling interval for metrics_by_time.jsonl.",
    )
    parser.add_argument(
        "--no-l1",
        action="store_true",
        help="Disable L1 memory for this run (no promotion, no extractor).",
    )
    parser.add_argument(
        "--enable-skill-refinement",
        action="store_true",
        help="Enable the SkillRevise-style skill refinement add-on. When a skill "
        "fails to debug or improve the solution, the agent diagnoses and refines "
        "the blamed skills inline (default: disabled).",
    )
    parser.add_argument(
        "--skill-refinement-max-rounds",
        type=int,
        default=3,
        help="Maximum number of inline skill-refinement rounds per trigger "
        "(only used with --enable-skill-refinement).",
    )
    parser.add_argument(
        "--skill-deletion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable L1 skill deletion (unused-streak GC and optional unit-test GC). "
        "When disabled (--no-skill-deletion), the extractor catalog is capped "
        "to the most recent active skills (legacy behavior).",
    )
    parser.add_argument(
        "--skill-merging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable L1 skill merging (embedding cluster + LLM merge). "
        "Requires --skill-deletion (default: disabled).",
    )
    parser.add_argument(
        "--skill-merge-similarity",
        type=float,
        default=DEFAULT_SKILL_MERGE_SIMILARITY,
        help="Cosine similarity threshold for skill-merge clustering (default: 0.9).",
    )
    parser.add_argument(
        "--skill-merge-interval",
        type=int,
        default=DEFAULT_SKILL_MERGE_INTERVAL,
        help="Minimum global iterations between skill-merge passes (default: 50).",
    )
    parser.add_argument(
        "--l1-skill-consecutive-unused-delete-after",
        type=int,
        default=DEFAULT_L1_SKILL_CONSECUTIVE_UNUSED_DELETE_AFTER,
        help="Delete active skills unused for this many consecutive global iterations "
        "(only when --skill-deletion).",
    )
    parser.add_argument(
        "--l1-skill-deletion-grace-iterations",
        type=int,
        default=DEFAULT_L1_SKILL_DELETION_GRACE_ITERATIONS,
        help="Grace period before consecutive-unused deletion applies to new skills.",
    )
    parser.add_argument(
        "--enable-l1-skill-unit-test-gc",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_L1_SKILL_UNIT_TEST_GC,
        help="Re-run skill unit tests on every governor iteration (deletion GC pass). "
        "Default: only validate when a skill is first promoted/appended.",
    )
    parser.add_argument(
        "--l1-skill-delete-on-unit-test-fail",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_L1_SKILL_DELETE_ON_UNIT_TEST_FAIL,
        help="Mark skills deleted when post-append unit tests fail.",
    )
    parser.add_argument(
        "--l1-skill-unit-test-max-tokens",
        type=int,
        default=DEFAULT_L1_SKILL_UNIT_TEST_MAX_TOKENS,
        help="Max tokens for LLM unit-test artifact generation.",
    )
    parser.add_argument(
        "--l1-skill-unit-test-timeout-sec",
        type=float,
        default=60.0,
        help="Per-call LLM timeout for unit-test generation/validation.",
    )
    parser.add_argument(
        "--l1-skill-unit-test-run-timeout-sec",
        type=float,
        default=DEFAULT_L1_SKILL_UNIT_TEST_RUN_TIMEOUT_SEC,
        help="Subprocess timeout when executing generated skill_impl.py tests.",
    )
    parser.add_argument(
        "--baseline-timing-file",
        type=str,
        default=None,
        help="Fixed baseline timing JSON for speedup (default: results/timing/SONG_CPU2_A6000x2/baseline_time_torch.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run: use --run-name exactly (with timestamp suffix), "
        "reuse L1/run_dir, and replace results from --start-problem onward.",
    )
    parser.add_argument(
        "--start-problem",
        type=int,
        default=1,
        help="1-based subset index (after --max-problems trim) to start re-running. "
        "Only valid with --resume; earlier problems are kept unchanged.",
    )
    args = parser.parse_args()

    if args.start_problem < 1:
        raise ValueError("--start-problem must be >= 1")
    if args.start_problem > 1 and not args.resume:
        raise ValueError("--start-problem requires --resume")

    if not args.resume:
        # Append UTC timestamp to run-name to avoid collisions and make runs unique.
        # Format: YYYY_MM_DD_HH_MM (year_month_day_hour_minute)
        now_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M")
        args.run_name = f"{args.run_name}_{now_str}"

    _check_integration_dependencies(dry_run=args.dry_run)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.dry_run:
        raise RuntimeError(
            "CUDA GPU is required for this integration run. Use --dry-run for local validation."
        )

    subset_csv = Path(args.subset_csv)
    if not subset_csv.is_file():
        raise FileNotFoundError(f"Subset CSV not found: {subset_csv}")

    rows = _load_subset_rows(subset_csv)
    if args.max_problems > 0:
        rows = rows[: args.max_problems]

    if args.start_problem > len(rows):
        raise ValueError(
            f"--start-problem {args.start_problem} exceeds subset size {len(rows)}"
        )

    run_dir = Path(args.results_root) / args.run_name
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Resume run directory not found: {run_dir}. "
                "Pass --run-name with the full timestamped folder name."
            )
        _warn_resume_config_mismatch(
            run_dir=run_dir,
            subset_csv=subset_csv,
            max_problems=args.max_problems,
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    shared_l1_path = run_dir / "shared_l1.txt"
    if not shared_l1_path.exists():
        shared_l1_path.write_text(
            "# Shared L1 journal for evolving KernelBench batch\n",
            encoding="utf-8",
        )

    eval_path = run_dir / "eval_results.json"
    evolving_runs_path = run_dir / "evolving_runs.json"
    summary_path = run_dir / "run_summary.json"
    timing_path = run_dir / "batch_timing.jsonl"

    batch_session_started_at = datetime.now(timezone.utc).isoformat()
    batch_session_t0 = time.perf_counter()
    problems_timed_this_session = 0

    evolving_doc_raw = _read_json(evolving_runs_path, default={"runs": []})
    if isinstance(evolving_doc_raw, list):
        evolving_doc = {"runs": evolving_doc_raw}
    elif isinstance(evolving_doc_raw, dict):
        evolving_doc = evolving_doc_raw
    else:
        evolving_doc = {"runs": []}
    runs_raw = evolving_doc.get("runs", [])
    runs = runs_raw if isinstance(runs_raw, list) else []
    completed_keys = {
        f"L{entry['level']}P{entry['problem_id']}"
        for entry in runs
        if isinstance(entry, dict) and "level" in entry and "problem_id" in entry
    }
    eval_doc = _normalize_level_first_eval_doc(_read_json(eval_path, default={}))
    level_eval_docs: dict[int, dict[str, list]] = {}

    for idx, row in enumerate(rows, start=1):
        level = int(row["level"])
        problem_id = str(int(row["problem_id"]))
        key = _problem_key(level, problem_id)

        if args.resume and idx < args.start_problem:
            print(f"[batch] resume: keep prior result for {key} (index {idx})")
            continue

        if not args.resume and key in completed_keys:
            print(f"[batch] skip completed {key}")
            continue

        if args.resume and idx >= args.start_problem:
            print(f"[batch] resume: replacing prior state for {key} (index {idx})")
            runs = _purge_problem_state(
                run_dir=run_dir,
                runs=runs,
                eval_doc=eval_doc,
                level_eval_docs=level_eval_docs,
                level=level,
                problem_id=problem_id,
            )
            completed_keys.discard(key)

        row_backend = (row.get("backend") or "").strip().lower()
        backend = row_backend if row_backend else args.backend

        print(
            f"[batch] ({idx}/{len(rows)}) running L{level}P{problem_id} "
            f"backend={backend} precision={args.precision}"
        )

        problem_t0 = time.perf_counter()
        problem_started_at = datetime.now(timezone.utc).isoformat()

        if args.dry_run:
            entry = {
                "level": level,
                "problem_id": problem_id,
                "backend": backend,
                "precision": args.precision,
                "best_speedup": 0.0,
                "best_correct": False,
                "best_compiled": False,
                "best_code_path": None,
                "best_code": None,
                "iterations_run": 0,
                "records": [],
                "runtime": -1.0,
                "runtime_stats": {},
                "metadata": {},
                "error": "dry_run_no_gpu_execution",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        else:
            dataset = construct_kernelbench_dataset(level=level, source="local")
            problem = dataset.get_problem_by_id(int(problem_id))
            task_prompt = get_prompt_for_backend(
                ref_arch_src=problem.code,
                backend=backend,
                option="one_shot",
                precision=args.precision,
            )
            cfg = KBGovernorConfig(
                run_name=args.run_name,
                level=level,
                problem_id=problem_id,
                backend=backend,
                precision=args.precision,
                max_iterations=args.max_iterations,
                enable_promotion=(not args.no_l1),
                enable_l1_extractor=(not args.no_l1),
                shared_l1_path=shared_l1_path,
                results_root=Path(args.results_root),
                reference_code=problem.code,
                run_recorder_time_sample_interval_sec=args.time_sample_interval_sec,
                baseline_timing_file=Path(args.baseline_timing_file)
                if args.baseline_timing_file
                else None,
                enable_skill_refinement=bool(args.enable_skill_refinement),
                skill_refinement_max_rounds=int(args.skill_refinement_max_rounds),
                skill_deletion=bool(args.skill_deletion),
                skill_merging=bool(args.skill_merging),
                skill_merge_similarity=float(args.skill_merge_similarity),
                skill_merge_interval=int(args.skill_merge_interval),
                l1_skill_consecutive_unused_delete_after=int(
                    args.l1_skill_consecutive_unused_delete_after
                ),
                l1_skill_deletion_grace_iterations=int(args.l1_skill_deletion_grace_iterations),
                enable_l1_skill_unit_test_gc=bool(args.enable_l1_skill_unit_test_gc),
                l1_skill_delete_on_unit_test_fail=bool(args.l1_skill_delete_on_unit_test_fail),
                l1_skill_unit_test_max_tokens=int(args.l1_skill_unit_test_max_tokens),
                l1_skill_unit_test_timeout_sec=float(args.l1_skill_unit_test_timeout_sec),
                l1_skill_unit_test_run_timeout_sec=float(args.l1_skill_unit_test_run_timeout_sec),
                verbose=True,
            )
            result = safe_run_kb_governor(cfg, task_prompt=task_prompt)
            entry = governor_result_to_dict(result)
            entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        wall_time_sec = time.perf_counter() - problem_t0
        problem_finished_at = datetime.now(timezone.utc).isoformat()
        entry["subset_index"] = idx
        entry["started_at_utc"] = problem_started_at
        entry["finished_at_utc"] = problem_finished_at
        entry["wall_time_sec"] = round(wall_time_sec, 3)
        problems_timed_this_session += 1

        problem_status = "dry_run" if args.dry_run else (
            "error" if entry.get("error") else "ok"
        )
        _append_jsonl(
            timing_path,
            {
                "level": level,
                "problem_id": int(problem_id),
                "subset_index": idx,
                "started_at_utc": problem_started_at,
                "finished_at_utc": problem_finished_at,
                "wall_time_sec": round(wall_time_sec, 3),
                "status": problem_status,
            },
        )
        print(f"[batch] {key} done in {wall_time_sec:.1f}s ({idx}/{len(rows)})")

        export_code = _extract_best_kernel_code(entry)
        if export_code:
            export_path = _write_kernel_export(run_dir, level=level, problem_id=problem_id, code=export_code)
            entry["exported_kernel_path"] = str(export_path)
            if not entry.get("best_code_path"):
                entry["best_code_path"] = str(export_path)

        runs.append(entry)
        completed_keys.add(key)
        level_key = str(level)
        pid_key = str(problem_id)
        eval_entry = _to_kernelbench_eval_entry(entry, level=level, problem_id=problem_id)
        eval_doc.setdefault(level_key, {})
        eval_doc[level_key][pid_key] = [eval_entry]

        if level not in level_eval_docs:
            level_eval_docs[level] = _read_json(_level_eval_path(run_dir, level), default={})
        level_eval_docs[level][pid_key] = [eval_entry]

        evolving_doc["runs"] = runs
        _write_json(evolving_runs_path, evolving_doc)
        _write_json(eval_path, eval_doc)
        _write_json(_level_eval_path(run_dir, level), level_eval_docs[level])

        cleanup_problem_build_artifacts(
            args.results_root,
            args.run_name,
            level=level,
            problem_id=problem_id,
        )

    successful = [e for e in runs if e.get("best_correct")]
    baseline_results = (
        _read_json(Path(args.baseline_timing_file), default={})
        if args.baseline_timing_file
        else None
    )
    suspicious_speedup_problems, likely_hack_keys = _collect_suspicious_speedup_problems(
        runs,
        baseline_results=baseline_results if isinstance(baseline_results, dict) else None,
    )
    best_overall = 0.0
    best_runtime_overall: float | None = None
    runtime_candidates: list[float] = []
    for entry in successful:
        entry_key = f"L{entry.get('level')}P{entry.get('problem_id')}"
        if entry_key in likely_hack_keys:
            continue
        try:
            runtime_f = float(entry.get("runtime"))
        except Exception:
            continue
        if runtime_f >= 0:
            runtime_candidates.append(runtime_f)
    if runtime_candidates:
        best_runtime_overall = min_non_outlier_runtime(runtime_candidates)
        if best_runtime_overall is not None:
            for entry in successful:
                entry_key = f"L{entry.get('level')}P{entry.get('problem_id')}"
                if entry_key in likely_hack_keys:
                    continue
                try:
                    runtime_f = float(entry.get("runtime"))
                except Exception:
                    continue
                if runtime_f == best_runtime_overall:
                    best_overall = float(entry.get("best_speedup", 0.0) or 0.0)
                    break

    batch_session_wall_time_sec = time.perf_counter() - batch_session_t0
    total_problem_wall_time_sec = _sum_timing_wall_seconds(timing_path)

    summary = {
        "run_name": args.run_name,
        "resume": bool(args.resume),
        "start_problem": int(args.start_problem) if args.resume else None,
        "resumed_from_run_dir": str(run_dir) if args.resume else None,
        "subset_csv": str(subset_csv),
        "dry_run": bool(args.dry_run),
        "cuda_available": bool(has_cuda),
        "skill_deletion": bool(args.skill_deletion),
        "enable_l1_skill_unit_test_gc": bool(args.enable_l1_skill_unit_test_gc),
        "skill_merging": bool(args.skill_merging),
        "skill_merge_similarity": float(args.skill_merge_similarity),
        "skill_merge_interval": int(args.skill_merge_interval),
        "total_attempted": len(rows),
        "total_completed": len(runs),
        "total_correct": len(successful),
        "best_speedup_overall": best_overall,
        "best_runtime_overall": best_runtime_overall,
        "per_level_summary": _summarize_per_level_runs(
            runs,
            likely_hack_keys=likely_hack_keys,
        ),
        "suspicious_speedup_problems": suspicious_speedup_problems,
        "suspicious_speedup_count": len(suspicious_speedup_problems),
        "results_path": str(eval_path),
        "evolving_runs_path": str(evolving_runs_path),
        "batch_timing_jsonl": str(timing_path),
        "batch_started_at_utc": batch_session_started_at,
        "batch_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_session_wall_time_sec": round(batch_session_wall_time_sec, 3),
        "total_wall_time_sec": round(total_problem_wall_time_sec, 3),
        "avg_wall_time_sec": (
            round(total_problem_wall_time_sec / problems_timed_this_session, 3)
            if problems_timed_this_session
            else 0.0
        ),
        "problems_timed_this_session": problems_timed_this_session,
        "per_level_results": {
            str(level): str(_level_eval_path(run_dir, level))
            for level in sorted(level_eval_docs.keys())
        },
        "shared_l1_path": str(shared_l1_path),
    }
    _write_json(summary_path, summary)

    print(f"[batch] total wall time: {batch_session_wall_time_sec:.1f}s "
          f"(problems summed: {total_problem_wall_time_sec:.1f}s)")
    print("[batch] complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
