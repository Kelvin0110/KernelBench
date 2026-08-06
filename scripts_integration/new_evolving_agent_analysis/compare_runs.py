"""Compare evolving-agent KernelBench runs side by side as a markdown report.

Consumes ``aggregate_runs.json`` produced by ``aggregate_runs.py`` (or recomputes
it in-process with ``--recompute``) and emits:

- a run-overview table (mode, model, status, iterations, correctness, wall clock)
- a final-iteration performance table (speedup mean/median/geomean, fast-p)
- a skill-governance table (L1 catalog size, merges, deletions, refinements)
- optional delta tables versus ``--baseline-run`` (absolute and percent)
- per-iteration series compared at **matched iteration counts**: best-speedup
  geometric mean and fast-p@P, restricted to the iterations every compared run
  actually reached

Speedup aggregates and fast-p come from ``generate_run_performance_stats``, which
scores **correct, non-hack samples only**; failed problems are dropped from the
mean/median/geomean and penalized through the fast-p denominator instead.

Example:
    uv run python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
        --hardware NVIDIA_GH200x2 \
        --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
        --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52 \
        --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
SERVER_DIR = SEA_ROOT / "visualizations" / "kernelbench" / "server"
for _path in (str(SCRIPT_DIR), str(SEA_ROOT), str(SERVER_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kernelbench.performance_stats import (  # noqa: E402
    align_series_for_comparison,
    parse_fastp_values,
    read_json,
    resolve_threshold_key,
    safe_float,
)

from aggregate_runs import (  # noqa: E402
    DEFAULT_BASELINE_STEM,
    DEFAULT_HARDWARE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_ROOT,
    aggregate_runs,
    resolve_baseline_path,
)

DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_DIR / "comparison.md"
DEFAULT_FAST_P = 1.0
DEFAULT_ITERATION_STRIDE = 5

#: (label, dotted path into a run record, digits, higher_is_better)
DELTA_METRICS: tuple[tuple[str, str, int, bool], ...] = (
    ("total_correct", "outcomes.total_correct", 0, True),
    ("correct_rate", "outcomes.correct_rate", 4, True),
    ("best_speedup_overall", "outcomes.best_speedup_overall", 4, True),
    ("speedup_best_mean", "performance.speedup_best.mean", 4, True),
    ("speedup_best_median", "performance.speedup_best.median", 4, True),
    ("speedup_best_geomean", "performance.speedup_best.geometric_mean", 4, True),
    ("speedup_current_geomean", "performance.speedup_current.geometric_mean", 4, True),
    ("hack_iteration_count", "performance.hack_iteration_count", 0, False),
    ("problems_with_hack", "performance.problems_with_hack", 0, False),
    ("l1_entry_count", "governance.l1_entry_count", 0, False),
    ("total_wall_time_hours", "timing.total_wall_time_hours", 3, False),
    ("avg_wall_time_min", "timing.avg_wall_time_min", 3, False),
)


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _dig(record: dict[str, Any], dotted: str) -> Any:
    node: Any = record
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = safe_float(value)
    if number is None:
        return str(value)
    if digits <= 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def _fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if digits > 0 else f"{value:+.0f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}%"


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return None
    return (value - baseline) / abs(baseline) * 100.0


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(no data)_", ""]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return lines


def build_aliases(records: list[dict[str, Any]]) -> dict[str, str]:
    """Short column ids (``R1``, ``R2``, ...) so markdown tables stay readable."""
    return {record["run_name"]: f"R{index}" for index, record in enumerate(records, start=1)}


# --------------------------------------------------------------------------- #
# series helpers
# --------------------------------------------------------------------------- #
def speedup_series(record: dict[str, Any], key: str) -> list[dict[str, Any]]:
    series = _dig(record, f"series.speedup.{key}")
    return series if isinstance(series, list) else []


def fast_p_series(record: dict[str, Any], *, field: str, threshold: float) -> list[dict[str, Any]]:
    raw = _dig(record, f"series.{field}")
    if not isinstance(raw, dict):
        return []
    key = resolve_threshold_key(raw, threshold)
    if key is None:
        return []
    series = raw.get(key)
    return series if isinstance(series, list) else []


def _series_map(points: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            iteration = int(point.get("iteration"))
        except (TypeError, ValueError):
            continue
        value = safe_float(point.get("value"))
        if value is None:
            continue
        out[iteration] = float(value)
    return out


def matched_iterations(series_by_run: dict[str, dict[int, float]], *, stride: int) -> list[int]:
    """Iterations reached by every run, capped at the shortest run, then strided."""
    maps = [m for m in series_by_run.values() if m]
    if not maps:
        return []
    common: set[int] = set(maps[0].keys())
    for other in maps[1:]:
        common &= set(other.keys())
    if not common:
        return []
    cap = min(max(m.keys()) for m in maps)
    usable = sorted(i for i in common if i <= cap)
    if not usable:
        return []
    if stride <= 1:
        return usable
    sampled = [i for i in usable if i % stride == 0]
    if usable[0] not in sampled:
        sampled.insert(0, usable[0])
    if usable[-1] not in sampled:
        sampled.append(usable[-1])
    return sorted(set(sampled))


def _series_table(
    *,
    title: str,
    records: list[dict[str, Any]],
    aliases: dict[str, str],
    series_by_run: dict[str, dict[int, float]],
    stride: int,
    digits: int,
    baseline_run: str | None,
) -> list[str]:
    lines = [f"### {title}", ""]
    usable = {name: m for name, m in series_by_run.items() if m}
    if not usable:
        return lines + ["_(no per-iteration series available)_", ""]

    iterations = matched_iterations(usable, stride=stride)
    if not iterations:
        return lines + ["_(compared runs share no common iterations)_", ""]

    ordered = [r["run_name"] for r in records if r["run_name"] in usable]
    headers = ["iteration"] + [aliases[name] for name in ordered]
    if baseline_run in usable:
        headers += [
            f"delta({aliases[name]}-{aliases[baseline_run]})"
            for name in ordered
            if name != baseline_run
        ]

    rows: list[list[str]] = []
    for iteration in iterations:
        row = [str(iteration)]
        row += [_fmt(usable[name].get(iteration), digits) for name in ordered]
        if baseline_run in usable:
            base_value = usable[baseline_run].get(iteration)
            for name in ordered:
                if name == baseline_run:
                    continue
                value = usable[name].get(iteration)
                row.append(
                    _fmt_delta(value - base_value, digits)
                    if value is not None and base_value is not None
                    else "-"
                )
        rows.append(row)

    lines += _md_table(headers, rows)
    lines.append(
        f"_Matched over iterations {iterations[0]}..{iterations[-1]} "
        f"(intersection of all compared runs, stride {stride})._"
    )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def _legend_section(records: list[dict[str, Any]], aliases: dict[str, str]) -> list[str]:
    headers = ["id", "run_name", "status", "context_mgmt", "model", "endpoint"]
    rows = [
        [
            aliases[record["run_name"]],
            f"`{record['run_name']}`",
            str(record.get("status") or "-"),
            str(_dig(record, "config.context_management") or "-"),
            str(_dig(record, "config.model") or "-"),
            str(_dig(record, "config.nvidia_endpoint") or "-"),
        ]
        for record in records
    ]
    return ["## Runs", ""] + _md_table(headers, rows)


def _overview_section(records: list[dict[str, Any]], aliases: dict[str, str]) -> list[str]:
    headers = [
        "id",
        "context_mgmt",
        "itr",
        "problems",
        "completed",
        "correct",
        "correct_rate",
        "rate_basis",
        "wall_h",
        "avg_min/problem",
        "suspicious",
    ]
    rows = []
    for record in records:
        outcomes = record.get("outcomes", {})
        correct = outcomes.get("total_correct")
        rows.append(
            [
                aliases[record["run_name"]],
                str(_dig(record, "config.context_management") or "-"),
                str(record.get("max_iterations") or "-"),
                str(outcomes.get("total_attempted") or 0),
                str(outcomes.get("total_completed") or 0),
                "-" if correct is None else str(correct),
                _fmt(outcomes.get("correct_rate"), 3),
                str(outcomes.get("correct_rate_basis") or "-"),
                _fmt(_dig(record, "timing.total_wall_time_hours"), 2),
                _fmt(_dig(record, "timing.avg_wall_time_min"), 1),
                str(outcomes.get("suspicious_speedup_count") or 0),
            ]
        )
    return ["## Run overview", ""] + _md_table(headers, rows)


def _performance_section(
    records: list[dict[str, Any]], aliases: dict[str, str], thresholds: list[float]
) -> list[str]:
    headers = [
        "id",
        "final_itr",
        "problems",
        "best_mean",
        "best_median",
        "best_geomean",
        "best_n",
        "cur_geomean",
        "cur_n",
        "best_speedup_overall",
        "hack_itrs",
        "problems_with_hack",
    ] + [f"fast_p@{t}" for t in thresholds]
    rows = []
    for record in records:
        performance = record.get("performance", {})
        fast_p_best = performance.get("fast_p_best", {})
        best_n = _dig(record, "performance.speedup_best.n")
        current_n = _dig(record, "performance.speedup_current.n")
        row = [
            aliases[record["run_name"]],
            str(performance.get("final_iteration") or "-"),
            str(performance.get("problem_count") or 0),
            _fmt(_dig(record, "performance.speedup_best.mean")),
            _fmt(_dig(record, "performance.speedup_best.median")),
            _fmt(_dig(record, "performance.speedup_best.geometric_mean")),
            "-" if best_n is None else str(best_n),
            _fmt(_dig(record, "performance.speedup_current.geometric_mean")),
            "-" if current_n is None else str(current_n),
            _fmt(_dig(record, "outcomes.best_speedup_overall")),
            str(performance.get("hack_iteration_count") or 0),
            str(performance.get("problems_with_hack") or 0),
        ]
        row += [_fmt(fast_p_best.get(str(t)), 3) for t in thresholds]
        rows.append(row)
    lines = ["## Final-iteration performance (fast-p is `fast_p_best`)", ""]
    lines += _md_table(headers, rows)
    lines.append(
        "_Speedup aggregates use correct, non-hack samples only; `best_n`/`cur_n` are how "
        "many of the `problems` actually entered those aggregates. fast-p keeps the "
        "full-problem denominator so failures are penalized, and `fast_p_best` does **not** "
        "drop hack-flagged bests - a small `best_n` next to a high fast-p means most bests "
        "were hack-flagged._"
    )
    lines.append("")
    return lines


def _governance_section(records: list[dict[str, Any]], aliases: dict[str, str]) -> list[str]:
    headers = [
        "id",
        "deletion",
        "merging",
        "refinement",
        "l1_entries",
        "l1_active",
        "merges",
        "deleted",
        "refined",
        "deletion_events",
        "sidecars",
    ]
    rows = []
    for record in records:
        governance = record.get("governance", {})
        sidecars = governance.get("governance_sidecars_present") or []
        rows.append(
            [
                aliases[record["run_name"]],
                _fmt(_dig(record, "config.skill_deletion")),
                _fmt(_dig(record, "config.skill_merging")),
                _fmt(_dig(record, "config.enable_skill_refinement")),
                str(governance.get("l1_entry_count") or 0),
                str(governance.get("l1_active_count") or 0),
                str(governance.get("merge_count") or 0),
                str(governance.get("deleted_count") or 0),
                str(governance.get("refined_count") or 0),
                str(governance.get("deletion_event_count") or 0),
                str(len(sidecars)),
            ]
        )
    return ["## Skill governance", ""] + _md_table(headers, rows)


def _delta_section(records: list[dict[str, Any]], baseline_run: str) -> list[str]:
    baseline = next((r for r in records if r["run_name"] == baseline_run), None)
    lines = [f"## Deltas vs baseline run `{baseline_run}`", ""]
    if baseline is None:
        return lines + [f"_(baseline run `{baseline_run}` is not in the compared set)_", ""]

    others = [r for r in records if r["run_name"] != baseline_run]
    if not others:
        return lines + ["_(no other runs to compare)_", ""]

    for record in others:
        lines.append(f"### `{record['run_name']}`")
        lines.append("")
        rows = []
        for label, dotted, digits, higher_is_better in DELTA_METRICS:
            base_value = safe_float(_dig(baseline, dotted))
            value = safe_float(_dig(record, dotted))
            delta = None if value is None or base_value is None else value - base_value
            pct = _pct_delta(value, base_value)
            direction = "-"
            if delta is not None and delta != 0:
                improved = (delta > 0) if higher_is_better else (delta < 0)
                direction = "better" if improved else "worse"
            elif delta == 0:
                direction = "same"
            rows.append(
                [
                    label,
                    _fmt(base_value, digits),
                    _fmt(value, digits),
                    _fmt_delta(delta, digits),
                    _fmt_pct(pct),
                    direction,
                ]
            )
        lines += _md_table(["metric", "baseline", "run", "delta", "delta %", "direction"], rows)
    return lines


def _series_section(
    records: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    baseline_run: str | None,
    fast_p: float,
    stride: int,
) -> list[str]:
    lines = ["## Per-iteration comparison (matched iterations)", ""]

    geomean = {r["run_name"]: _series_map(speedup_series(r, "best_geometric_mean")) for r in records}
    lines += _series_table(
        title="Best-speedup geometric mean vs iteration",
        records=records,
        aliases=aliases,
        series_by_run=geomean,
        stride=stride,
        digits=4,
        baseline_run=baseline_run,
    )

    fastp = {
        r["run_name"]: _series_map(fast_p_series(r, field="fast_p_best", threshold=fast_p))
        for r in records
    }
    lines += _series_table(
        title=f"fast_p_best@{fast_p} vs iteration",
        records=records,
        aliases=aliases,
        series_by_run=fastp,
        stride=stride,
        digits=3,
        baseline_run=baseline_run,
    )

    if baseline_run and baseline_run in geomean and geomean[baseline_run]:
        lines.append(f"### Aligned final-iteration deltas vs `{baseline_run}`")
        lines.append("")
        rows = []
        for record in records:
            name = record["run_name"]
            if name == baseline_run:
                continue
            for label, series_by_run, digits in (
                ("best_geomean", geomean, 4),
                (f"fast_p_best@{fast_p}", fastp, 3),
            ):
                aligned = align_series_for_comparison(
                    evolving=[
                        {"iteration": i, "value": v} for i, v in sorted(series_by_run[name].items())
                    ]
                    if series_by_run.get(name)
                    else [],
                    aide=[
                        {"iteration": i, "value": v}
                        for i, v in sorted(series_by_run[baseline_run].items())
                    ]
                    if series_by_run.get(baseline_run)
                    else [],
                )
                if not aligned:
                    continue
                last = aligned[-1]
                delta = last["evolving"] - last["aide"]
                rows.append(
                    [
                        aliases[name],
                        label,
                        str(int(last["iteration"])),
                        _fmt(last["aide"], digits),
                        _fmt(last["evolving"], digits),
                        _fmt_delta(delta, digits),
                        _fmt_pct(_pct_delta(last["evolving"], last["aide"])),
                    ]
                )
        lines += _md_table(
            ["id", "metric", "matched_iteration", "baseline", "run", "delta", "delta %"], rows
        )
    return lines


def build_markdown(
    *,
    doc: dict[str, Any],
    records: list[dict[str, Any]],
    baseline_run: str | None,
    fast_p: float,
    stride: int,
    thresholds: list[float],
) -> str:
    lines: list[str] = [
        "# Evolving-agent cross-run comparison",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- aggregate_generated_at_utc: `{doc.get('generated_at_utc')}`",
        f"- runs_root: `{doc.get('runs_root')}`",
        f"- baseline_timing_file: `{doc.get('baseline_file')}`",
        f"- speedup_aggregate_policy: `{doc.get('speedup_aggregate_policy')}`",
        f"- runs compared: {len(records)}",
        "",
    ]

    partial = [r["run_name"] for r in records if r.get("status") != "complete"]
    if partial:
        lines.append(
            "> **Warning:** these runs are still partial (in flight or aborted) and are "
            "reported at whatever iteration/problem count they have reached: "
            + ", ".join(f"`{name}`" for name in partial)
        )
        lines.append("")

    aliases = build_aliases(records)
    lines += _legend_section(records, aliases)
    lines += _overview_section(records, aliases)
    lines += _performance_section(records, aliases, thresholds)
    lines += _governance_section(records, aliases)
    if baseline_run:
        lines += _delta_section(records, baseline_run)
    lines += _series_section(
        records, aliases, baseline_run=baseline_run, fast_p=fast_p, stride=stride
    )

    notes = [f"- `{r['run_name']}`: {w}" for r in records for w in (r.get("warnings") or [])]
    if notes:
        lines += ["## Notes", ""] + notes + [""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def load_aggregate(
    *,
    aggregate_path: Path,
    recompute: bool,
    runs_root: Path,
    output_dir: Path,
    baseline_file: Path,
    fast_p_thresholds: list[float],
    only_runs: list[str] | None,
    regenerate_stats: bool,
) -> dict[str, Any]:
    if not recompute and aggregate_path.is_file():
        try:
            doc = read_json(aggregate_path, default=None)
        except Exception as exc:
            print(f"[compare] cached aggregate unreadable ({exc}); recomputing", file=sys.stderr)
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("runs"), list):
            # A cache built from a different runs-root / baseline would silently
            # answer for the wrong data set, so only reuse a matching one.
            cached_root = str(doc.get("runs_root") or "")
            cached_baseline = str(doc.get("baseline_file") or "")
            if cached_root and cached_root != str(runs_root):
                print(
                    f"[compare] cached aggregate is for runs_root={cached_root} "
                    f"(requested {runs_root}); recomputing",
                    file=sys.stderr,
                )
            elif cached_baseline and cached_baseline != str(baseline_file):
                print(
                    f"[compare] cached aggregate used baseline={cached_baseline} "
                    f"(requested {baseline_file}); recomputing",
                    file=sys.stderr,
                )
            else:
                return doc
    result = aggregate_runs(
        runs_root=runs_root,
        output_dir=output_dir,
        baseline_file=baseline_file,
        fast_p_thresholds=fast_p_thresholds,
        only_runs=only_runs,
        regenerate_stats=regenerate_stats,
    )
    return result["doc"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help=f"aggregate_runs.json to consume (default: <output-dir>/aggregate_runs.json under {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rebuild the aggregate in-process instead of reading the cached JSON",
    )
    parser.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hardware", type=str, default=DEFAULT_HARDWARE)
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE_STEM)
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=None,
        help="Explicit baseline timing JSON path; overrides --hardware/--baseline",
    )
    parser.add_argument(
        "--runs",
        action="append",
        default=None,
        metavar="RUN_NAME",
        help="Restrict the comparison to this run name (repeatable)",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default=None,
        help="Run name to express the other runs as deltas against",
    )
    parser.add_argument("--fast-p", type=float, default=DEFAULT_FAST_P, help="fast-p threshold for the series table")
    parser.add_argument(
        "--iteration-stride",
        type=int,
        default=DEFAULT_ITERATION_STRIDE,
        help="Sample the per-iteration tables every N iterations (first/last always kept)",
    )
    parser.add_argument("--fast-p-values", type=str, default=None, help="Comma-separated fast-p thresholds")
    parser.add_argument(
        "--regenerate-stats",
        action="store_true",
        help="Force performance_stats.json regeneration when recomputing the aggregate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_MD),
        help=f"Markdown output path (default: {DEFAULT_OUTPUT_MD})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    aggregate_path = Path(args.aggregate) if args.aggregate else output_dir / "aggregate_runs.json"
    if not aggregate_path.is_absolute():
        aggregate_path = REPO_ROOT / aggregate_path
    output_md = Path(args.output)
    if not output_md.is_absolute():
        output_md = REPO_ROOT / output_md

    baseline_file = resolve_baseline_path(args)
    thresholds = parse_fastp_values(args.fast_p_values)

    will_recompute = bool(args.recompute) or not aggregate_path.is_file()
    if will_recompute and not baseline_file.is_file():
        print(
            f"[compare] no cached aggregate at {aggregate_path} and baseline file not found: {baseline_file}",
            file=sys.stderr,
        )
        return 2

    doc = load_aggregate(
        aggregate_path=aggregate_path,
        recompute=bool(args.recompute),
        runs_root=runs_root,
        output_dir=output_dir,
        baseline_file=baseline_file,
        fast_p_thresholds=thresholds,
        only_runs=args.runs,
        regenerate_stats=bool(args.regenerate_stats),
    )

    records = [r for r in doc.get("runs", []) if isinstance(r, dict) and r.get("run_name")]
    if args.runs:
        wanted = list(dict.fromkeys(args.runs))
        by_name = {r["run_name"]: r for r in records}
        missing = [name for name in wanted if name not in by_name]
        for name in missing:
            print(f"[compare] requested run not in aggregate: {name}", file=sys.stderr)
        records = [by_name[name] for name in wanted if name in by_name]

    if not records:
        print("[compare] no runs to compare; run aggregate_runs.py first or pass --recompute", file=sys.stderr)
        return 1

    doc_thresholds = doc.get("fast_p_thresholds")
    if isinstance(doc_thresholds, list) and doc_thresholds:
        thresholds = [float(t) for t in doc_thresholds]

    markdown = build_markdown(
        doc=doc,
        records=records,
        baseline_run=args.baseline_run,
        fast_p=float(args.fast_p),
        stride=max(1, int(args.iteration_stride)),
        thresholds=thresholds,
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"[compare] markdown={output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
