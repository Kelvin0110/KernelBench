#!/usr/bin/env python3
"""Early-checkpoint health check for an in-flight evolving-agent run.

Purpose
-------
After the CUDA toolchain fix (see ``env/install_cuda128_local.sh``), the runs
under ``runs_evolving/archived/with_NVCC_bug/`` are known-bad: ``nvcc`` was
absent, so ``load_inline(cuda_sources=...)`` could never build and the agent
learned to guard the build behind ``if os.getenv("CUDA_HOME")`` and silently
fall back to reference PyTorch ops.

This script lets you decide -- after a few GPU-hours rather than 30+ -- whether
a freshly launched run is actually exercising the CUDA path, by comparing the
completed prefix of the new run against the same problems in its archived
counterpart.

The headline signal is ``cuda_home_err``. Under the fix it must be ~0. If it is
still material, ``CUDA_HOME`` did not reach the eval subprocess and the run
should be killed immediately.

Usage
-----
    uv run python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py \
        --run runs_evolving/base_agent_gpt_oss_120b_itr30_GH200_nvcc_2026_08_07_14_00 \
        --baseline runs_evolving/archived/with_NVCC_bug/base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51

    # both arms at once, auto-pairing against the archive:
    uv run python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "runs_evolving" / "archived" / "with_NVCC_bug"

# Pairs a live run-name fragment with its known-bad archived counterpart.
AUTO_PAIRS = {
    "truncation": "base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51",
    "markov": "base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52",
    "selective": "base_agent_gpt_oss_120b_selective_itr30_GH200_2026_08_04_17_24",
    "folding": "base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_04_17_26",
}


def _iter_problem_metrics(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """problem name -> list of per-iteration ``metrics_iteration`` dicts."""
    out: dict[str, list[dict[str, Any]]] = {}
    pattern = str(run_dir / "workspaces" / "*" / "metrics_by_iteration.jsonl")
    for path in sorted(glob.glob(pattern)):
        problem = Path(path).parent.name
        rows: list[dict[str, Any]] = []
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A run being written to concurrently can leave a half line.
                    continue
                mi = rec.get("metrics_iteration")
                if isinstance(mi, dict):
                    rows.append(mi)
        if rows:
            out[problem] = rows
    return out


def _best_of(run_dir: Path, problem: str) -> dict[str, Any]:
    path = run_dir / "workspaces" / problem / "metrics_by_iteration.jsonl"
    best: dict[str, Any] = {}
    if not path.exists():
        return best
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mb = rec.get("metrics_best")
            if isinstance(mb, dict):
                best = mb
    return best


def _classify_kernel(src: str) -> str:
    """Label a kernel by whether its CUDA build can actually run.

    ``dead_cuda`` is the signature defect: CUDA sources exist, but the
    ``load_inline`` carrying them sits inside a conditional that is false when
    ``CUDA_HOME`` is unset, so the reference PyTorch path executes instead.
    """
    if "load_inline" not in src and "cpp_extension" not in src:
        return "pure_torch"

    has_cuda_src = bool(re.search(r"cuda_sources\s*=\s*(?!None|''|\"\")", src))
    if not has_cuda_src:
        return "cpp_only"

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "cuda_unknown"

    guarded = False
    unguarded = False

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.AST] = []

        def generic_visit(self, node: ast.AST) -> None:
            self.stack.append(node)
            super().generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in {"load_inline", "load"} and any(
                kw.arg == "cuda_sources" for kw in node.keywords
            ):
                nonlocal guarded, unguarded
                enclosing = [n for n in self.stack if isinstance(n, (ast.If, ast.Try))]
                if enclosing:
                    guarded = True
                else:
                    unguarded = True
            self.generic_visit(node)

    Visitor().visit(tree)
    if guarded and not unguarded:
        return "dead_cuda_guarded"
    if unguarded:
        return "cuda_unguarded"
    return "cuda_unknown"


def summarize(run_dir: Path, limit_to: set[str] | None = None) -> dict[str, Any]:
    per_problem = _iter_problem_metrics(run_dir)
    if limit_to is not None:
        per_problem = {k: v for k, v in per_problem.items() if k in limit_to}

    iters = cuda_home_err = compiled = correct = hack = 0
    for rows in per_problem.values():
        for mi in rows:
            iters += 1
            if mi.get("compiled"):
                compiled += 1
            if mi.get("correct"):
                correct += 1
            if mi.get("is_hack"):
                hack += 1
            if "CUDA_HOME" in (mi.get("error") or ""):
                cuda_home_err += 1

    speedups: list[float] = []
    best_correct = 0
    for problem in per_problem:
        mb = _best_of(run_dir, problem)
        if mb.get("correct"):
            best_correct += 1
            sp = mb.get("speedup")
            if isinstance(sp, (int, float)) and sp > 0:
                speedups.append(float(sp))

    kinds: dict[str, int] = {}
    for path in sorted(glob.glob(str(run_dir / "kernels" / "*_kernel.py"))):
        stem = Path(path).name.replace("_sample_0_kernel.py", "")
        if limit_to is not None and stem not in limit_to:
            continue
        try:
            src = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        kinds[_classify_kernel(src)] = kinds.get(_classify_kernel(src), 0) + 1

    return {
        "run": run_dir.name,
        "problems": len(per_problem),
        "iterations": iters,
        "cuda_home_err": cuda_home_err,
        "cuda_home_err_pct": 100.0 * cuda_home_err / iters if iters else 0.0,
        "compiled_pct": 100.0 * compiled / iters if iters else 0.0,
        "correct_iter_pct": 100.0 * correct / iters if iters else 0.0,
        "hack_iter_pct": 100.0 * hack / iters if iters else 0.0,
        "best_correct": best_correct,
        "speedup_geomean": (
            statistics.geometric_mean(speedups) if speedups else float("nan")
        ),
        "speedup_n": len(speedups),
        "speedup_gt_1_10": sum(1 for s in speedups if s > 1.10),
        "kernel_kinds": kinds,
    }


def _fmt(row: dict[str, Any]) -> str:
    return (
        f"  problems={row['problems']:<3d} iters={row['iterations']:<5d} "
        f"CUDA_HOME_err={row['cuda_home_err']:<4d} ({row['cuda_home_err_pct']:.1f}%)  "
        f"compiled={row['compiled_pct']:.1f}%  correct_iter={row['correct_iter_pct']:.1f}%  "
        f"hack_iter={row['hack_iter_pct']:.1f}%\n"
        f"  best_correct={row['best_correct']:<3d} "
        f"speedup_geomean={row['speedup_geomean']:.4f} (n={row['speedup_n']}) "
        f"sp>1.10={row['speedup_gt_1_10']}\n"
        f"  kernels={row['kernel_kinds']}"
    )


def compare(run: Path, baseline: Path | None) -> dict[str, Any]:
    # Only compare problems the live run has actually reached, otherwise the
    # baseline's full 50 problems make the new run look worse than it is.
    live_problems = set(_iter_problem_metrics(run))
    new = summarize(run)
    print(f"\n=== NEW: {run.name}")
    print(_fmt(new))

    old = None
    if baseline is not None and baseline.exists():
        old = summarize(baseline, limit_to=live_problems)
        print(f"\n=== BASELINE (same {len(live_problems)} problems): {baseline.name}")
        print(_fmt(old))

        print("\n=== VERDICT")
        if new["cuda_home_err"] == 0:
            print("  [OK]   CUDA_HOME errors eliminated (was "
                  f"{old['cuda_home_err']} on these problems).")
        else:
            print(f"  [STOP] {new['cuda_home_err']} CUDA_HOME errors remain -- "
                  "CUDA_HOME is not reaching the eval subprocess. Kill the run.")
        dead_new = new["kernel_kinds"].get("dead_cuda_guarded", 0)
        dead_old = old["kernel_kinds"].get("dead_cuda_guarded", 0)
        print(f"  dead-CUDA (guarded, never builds) kernels: {dead_old} -> {dead_new}")
        print(f"  compiled rate: {old['compiled_pct']:.1f}% -> {new['compiled_pct']:.1f}%")
        print(f"  best_correct:  {old['best_correct']} -> {new['best_correct']}")
        print("  NOTE: correctness and speedup are EXPECTED to drop -- kernels that "
              "previously 'passed' by falling back to reference PyTorch now have to "
              "actually compile.")
    return {"new": new, "baseline": old}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, help="Path to the in-flight run dir")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Archived with_NVCC_bug counterpart to compare against")
    ap.add_argument("--auto", action="store_true",
                    help="Find live runs under runs_evolving/ and auto-pair with the archive")
    ap.add_argument("--json", type=str, default=None, help="Write results to this JSON path")
    args = ap.parse_args()

    results = []
    if args.auto:
        # Runs live either directly under runs_evolving/ or one level down in a
        # per-model folder (e.g. runs_evolving/gpt-oss-120b/). Identify a run
        # directory by its workspaces/ child rather than by depth, so both
        # layouts work and intermediate folders are not mistaken for runs.
        live = []
        for depth in ("*", "*/*"):
            for p in sorted(glob.glob(str(REPO_ROOT / "runs_evolving" / depth))):
                path = Path(p)
                if not path.is_dir() or "archived" in path.parts:
                    continue
                if (path / "workspaces").is_dir():
                    live.append(path)
        if not live:
            print("No live runs found under runs_evolving/.")
            return
        for run in live:
            baseline = None
            for frag, arch in AUTO_PAIRS.items():
                # 'truncation' is the default arm: its dir name carries no mode token.
                token = frag if frag != "truncation" else None
                if token is None:
                    if not any(f in run.name for f in ("markov", "selective", "folding")):
                        baseline = ARCHIVE / arch
                        break
                elif token in run.name:
                    baseline = ARCHIVE / arch
                    break
            results.append(compare(run, baseline))
    else:
        if not args.run:
            ap.error("--run is required unless --auto is given")
        results.append(compare(Path(args.run), Path(args.baseline) if args.baseline else None))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
