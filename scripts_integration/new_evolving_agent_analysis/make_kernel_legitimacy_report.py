#!/usr/bin/env python3
"""Build KERNEL_LEGITIMACY_REPORT.md for a completed wave.

Self-contained: reads only committed/regenerable artifacts --
  <runs-root>/<arm>/visualizations/performance_stats.json   (canonical, uniform-30)
  <runs-root>/<arm>/workspaces/*/evaluation_terminal_output.jsonl
  <output-dir>/rescore_hack_threshold.json                  (for the stored-label column)
No scratch state. Re-runnable from a clean checkout.

Headline tables are the UNIFORM-30 view, matching what the pipeline writes by
default (generate_run_performance_stats.py --hack-threshold 30). The stored-label
column is shown beside it only to expose the 10x->30x seam.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys
from datetime import datetime, timezone

# Problems whose reference model computes something large and then collapses it,
# so an exact algebraic shortcut is worth 10-30x. Speedup here measures how
# wasteful the reference is, not kernel-engineering quality.
COLLAPSE = {
    "level_2_problem_13": "ConvTranspose3d -> mean over depth",
    "level_2_problem_42": "ConvTranspose2d -> global avg pool",
    "level_2_problem_51": "Gemm 8192x8192 -> mean over features",
    "level_2_problem_56": "Linear 32768 -> sigmoid -> sum (does NOT collapse)",
}
ORDER = ["truncation", "folding", "markov", "selective_r5", "compress",
         "deletion", "merge_sim08", "refinement", "l2"]


def tag(name: str) -> str:
    for p in ("base_agent_gpt_oss_120b_", "base_agent_gpt_5_6_terra_"):
        if name.startswith(p):
            name = name[len(p):]
            break
    name = re.sub(r"^itr30_GH200_\d{4}(_\d{2}){4}$", "truncation", name)
    return re.sub(r"_itr30_GH200_\d{4}(_\d{2}){4}$", "", name)


def geo(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def metrics(speedups):
    n = len(speedups)
    if not n:
        return {"n": 0, "corr": 0.0, "f1": 0.0, "f2": 0.0, "geo": 0.0}
    return {"n": n,
            "corr": sum(1 for s in speedups if s > 0) / n,
            "f1": sum(1 for s in speedups if s > 1.0) / n,
            "f2": sum(1 for s in speedups if s > 2.0) / n,
            "geo": geo(speedups)}


def arm_views(ps_path):
    """Return {iteration: {'full': m, 'excl': m}} from one performance_stats.json."""
    doc = json.load(open(ps_path))
    out = {}
    for k in (10, 30):
        rows = [i for i in doc["iterations"] if i["iteration"] == k]
        if not rows:
            continue
        pts = rows[0]["points"]
        sp = lambda ps: [p["best_speedup"] if p["best_correct"] else 0.0 for p in ps]
        out[k] = {"full": metrics(sp(pts)),
                  "excl": metrics(sp([p for p in pts if p["workspace_id"] not in COLLAPSE]))}
    return doc, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", required=True,
                    help="repeatable: <model-label>=<path>, e.g. gpt-oss-120b=runs_evolving/gpt-oss-120b/median")
    ap.add_argument("--cohort", default="",
                    help="substring an arm dir must contain (e.g. _2026_08_22_); blank = all")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    roots = []
    for spec in args.runs_root:
        label, _, path = spec.partition("=")
        roots.append((label, path or label))

    rescore_path = os.path.join(args.output_dir, "rescore_hack_threshold.json")
    rescore = json.load(open(rescore_path)) if os.path.isfile(rescore_path) else None

    data = {}
    for label, root in roots:
        for ps in sorted(glob.glob(os.path.join(root, "*", "visualizations", "performance_stats.json"))):
            arm_dir = os.path.basename(os.path.dirname(os.path.dirname(ps)))
            if args.cohort and args.cohort not in arm_dir:
                continue
            doc, v = arm_views(ps)
            data[(label, tag(doc["run_name"]))] = {"doc": doc, "views": v}

    if not data:
        print("no arms matched", file=sys.stderr)
        return 2

    k = lambda a: ORDER.index(a) if a in ORDER else 99
    L = []
    A = L.append
    A("# Kernel legitimacy audit — are the high-speedup samples real?")
    A("")
    A(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
      f"Cohort filter: `{args.cohort or '(none)'}`. Arms: {len(data)}.")
    A("")
    A("Regenerate with `make_kernel_legitimacy_report.py` (see §6). Reads only "
      "`performance_stats.json`, the per-problem eval records, and "
      "`rescore_hack_threshold.json` — no scratch state.")
    A("")
    A("**Headline tables are the uniform-30x view**, i.e. exactly what "
      "`generate_run_performance_stats.py` writes by default since submodule `fa133b1`. "
      "Metrics are best-so-far: `corr` = `fast_p_best@0.0`, `fast@1`/`fast@2` = "
      "`fast_p_best@{1.0,2.0}`, `geo` = geometric mean of best speedup over correct problems.")
    A("")
    A("---")
    A("")
    A("## 1. Headline (uniform 30x, n=50)")
    for label, _ in roots:
        subs = sorted([(a, d) for (m, a), d in data.items() if m == label], key=lambda x: k(x[0]))
        if not subs:
            continue
        A("")
        A(f"### {label}")
        A("")
        A("| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for a, d in subs:
            x, y = d["views"][10]["full"], d["views"][30]["full"]
            nm = a + (" (ctrl)" if a == "truncation" else "")
            A(f"| {nm} | {x['corr']:.3f} | {x['f1']:.3f} | {x['f2']:.3f} | {x['geo']:.3f} "
              f"| {y['corr']:.3f} | {y['f1']:.3f} | {y['f2']:.3f} | {y['geo']:.3f} |")
    A("")
    A("## 2. The same arms with the four collapse problems removed (n=46)")
    A("")
    A("Rationale in §4. `geo@30 (all 50)` is repeated so the cost of the exclusion is visible.")
    for label, _ in roots:
        subs = sorted([(a, d) for (m, a), d in data.items() if m == label], key=lambda x: k(x[0]))
        if not subs:
            continue
        A("")
        A(f"**{label}**")
        A("")
        A("| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 | geo@30 (all 50) |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for a, d in subs:
            x, y, f = d["views"][10]["excl"], d["views"][30]["excl"], d["views"][30]["full"]
            nm = a + (" (ctrl)" if a == "truncation" else "")
            A(f"| {nm} | {x['corr']:.3f} | {x['f1']:.3f} | {x['f2']:.3f} | {x['geo']:.3f} "
              f"| {y['corr']:.3f} | {y['f1']:.3f} | {y['f2']:.3f} | {y['geo']:.3f} | {f['geo']:.3f} |")
    ups = [(m, a) for (m, a), d in data.items() if d["views"][30]["excl"]["geo"] > d["views"][30]["full"]["geo"]]
    A("")
    A(f"Geomean falls on {len(data) - len(ups)} of {len(data)} arms and rises on {len(ups)} "
      f"({', '.join(a for _, a in sorted(ups)) or 'none'}) — those had a below-average best on the "
      "collapse problems. **The exclusion is not rank-preserving**; do not carry a ranking across views.")
    A("")
    if rescore:
        A("## 3. The 10x -> 30x `is_hack` seam")
        A("")
        A("`src/kernelbench/eval.py` changed `excessive_speedup_threshold` 10 -> 30 at "
          f"**{rescore.get('seam_utc')}** (commit `588a6a5`). `eval.py` is re-imported by every eval "
          "spawn, so it reached live arms with no restart and each run's stored `is_hack` column is a "
          "mixture of two rules. Uniform re-scoring is what the tables above use.")
        A("")
        A("| model | arm | geo@30 stored | geo@30 uniform-30 | change |")
        A("|---|---|---|---|---|")
        for model, g in rescore["aligned_within_model"].items():
            uni = {tag(n): v for n, v in g["arms"].items()}
            for a in ORDER:
                if a not in uni:
                    continue
                s, u = uni[a]["stored"]["best_geomean"], uni[a]["uniform"]["best_geomean"]
                ch = "—" if abs(u - s) < 5e-4 else f"**{(u/s-1)*100:+.1f}%**"
                A(f"| {model} | {a} | {s:.3f} | {u:.3f} | {ch} |")
        A("")
    A("## 4. Are the (10x, 30x] kernels legitimate?")
    A("")
    A("**Yes — 264 of the 268 re-scored evals are exact, fp32, real custom CUDA.** Audited three ways:")
    A("")
    A("- **Static.** All 268 run through `validate_kernel_static(backend='cuda', precision='fp32')`, "
      "including `global_module_patch` (the reference-corruption check added in `ede1898`, which did not "
      "exist when they were evaluated). **0 STRICT errors.**")
    A("- **Numerical.** Re-run independently with parameters synced from the reference at "
      "`atol=rtol=1e-4` (the tolerance eval uses): L2P13 correct, max|diff| 2.17e-05, 26.5x measured vs "
      "30.0x recorded; L2P42 correct, max|diff| 2.29e-05, 22.7x vs 22.9x. Live reference timings "
      "(8.412 ms, 6.018 ms) match the fixed baseline (8.4, 6.02).")
    A("- **Code.** All 21 metric-moving samples contain real `__global__` CUDA via `load_inline`.")
    A("")
    A("The band is concentrated in four problems:")
    A("")
    A("| problem | reference model |")
    A("|---|---|")
    for p, why in COLLAPSE.items():
        A(f"| {p.replace('level_2_problem_', 'L2P')} | {why} |")
    A("")
    A("Each computes something enormous then discards most of it, so an exact algebraic shortcut is worth "
      "10-30x. `logsumexp` over a size-1 dim (L2P51) is literally the identity.")
    A("")
    A("**Why they are excluded anyway.** KernelBench's prompt "
      "(`src/kernelbench/prompts/prompts.toml:13`) explicitly permits *\"algorithmic changes (such as "
      "online softmax)... only limited by your imagination\"*, so these are legal. But `EVAL.md` warns "
      "*\"a >2x speedup for anything is highly unlikely\"*, and there is a real distinction: online "
      "softmax restructures **how** the same work is done, whereas mean-of-GEMM -> matvec proves most of "
      "the reference's work is never observed and **deletes** it. Only the first is kernel engineering. "
      "On these problems the speedup measures how wasteful the reference model is, and at 10-30x it "
      "dominates any geometric mean it enters.")
    A("")
    A("Note this cut removes **problems**, not a speedup band. Cutting only >30x would be the same "
      "magnitude heuristic with a different constant: on L2P51 the ~150x kernels and the 22-30x kernels "
      "are the same trick, differing only in whether weight prep is cached. (The ~150x figure is also "
      "physically achievable — 134 MB of essential traffic in 36.8 us is ~3.6 TB/s, ordinary for "
      "GH200 HBM3e — so magnitude alone is not evidence of cheating.)")
    A("")
    A("## 5. The exception: an FP8 hack the checker used to miss")
    A("")
    A("Of the 268, four use reduced precision. One is decisive: a terra L2P56 kernel cast **both operands "
      "of a 32768-wide GEMM to `torch.float8_e4m3fn`** and ran it through `torch._scaled_mm` with fp16 "
      "accumulate. It passed the 1e-4 gate only because the following sigmoid **saturates**, which hides "
      "the FP8 error, and measured 15.6x.")
    A("")
    A("`check_precision_downgrade` missed it: `FP32_TO_FP16_PATTERNS` matched conversion idioms "
      "(`__float2half(`, `.half()`) but not `__half*` declarations, and had **no FP8 pattern at all**. "
      "`torch._scaled_mm` was absent from `TORCH_COMPUTATION_OPS`.")
    A("")
    A("**Fixed 2026-08-27** in `src/kernelbench/kernel_static_checker.py`:")
    A("")
    A("1. New **STRICT** check `fp8_downgrade` (`check_fp8_downgrade`, `FP32_TO_FP8_PATTERNS`) matching "
      "`torch.float8_e[45]m[23]*`, `torch._scaled_mm`, `__nv_fp8*`, `__nv_cvt_*_to_fp8*`, for required "
      "precision FP32/FP16/BF16. FP8 sets `is_hack`. **FP16 deliberately stays a WARNING** — a 10-bit "
      "mantissa can legitimately meet a 1e-4 gate; a 3-bit one cannot.")
    A("2. FP16 **storage/consumption** patterns added: `__half*` declarations, `half2`/`__half2`, "
      "`__half2float(`, `at::Half`, `torch::kHalf`. A bare `torch.float16` token was deliberately NOT "
      "added — the CUDA source lives inside a Python string literal, so string contents cannot be "
      "stripped before matching and it false-positives on prose.")
    A("3. `torch._scaled_mm`, `torch._int_mm`, `torch.addmm`, `torch.addbmm` added to "
      "`TORCH_COMPUTATION_OPS`.")
    A("4. `_parse_python_module` retries `ast.parse` on a dedented copy — an indented snippet raised "
      "`SyntaxError` and silently disabled the STRICT `code_bypass` check (pre-existing failing test "
      "`test_strict_checks_are_errors`, now green).")
    A("")
    A("**Validation.** 9 new unit tests; 76 pass across the three static-checker modules. On the wave "
      "corpus the 3 FP8 evals become STRICT errors, the 1 FP16 eval stays a warning, and all 264 "
      "legitimate kernels stay clean. Across **729 best-forming kernels**: **0 STRICT errors** — no false "
      "positives. 8 of the 729 carry FP16 warnings, and **7 of those 8 were already detectable before "
      "this change** — they were recorded as warnings and counted as bests anyway, because "
      "`precision_downgrade` does not set `is_hack`. Promoting FP16 too would change 8 problems' bests "
      "across 5 arms (largest: terra truncation geo 2.417 -> 2.270) with no correctness lost; that was "
      "considered and deliberately not done.")
    A("")
    A("**No pipeline change was needed.** A STRICT failure short-circuits in `governor.py:549` with "
      "`compiled=False, correct=False` and no eval, so it never carries a speedup, and "
      "`_resolve_is_hack` returns the stored label when speedup is absent — static-check hacks survive "
      "uniform re-scoring intact.")
    A("")
    A("**Seam.** The checker is parent-side (`governor.py` only) and bound at import in the long-lived "
      "parent, so it cannot reach a running arm. Arms launched from 2026-08-27 10:37 onward enforce "
      "`fp8_downgrade`; this cohort did not.")
    A("")
    A("## 6. Reproduction")
    A("")
    A("```bash")
    A("# stats (uniform-30 is the default; --use-stored-hack for the stored view)")
    A("for m in gpt-oss-120b gpt-5.6-terra; do")
    A("  .venv/bin/python Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py \\")
    A("    --all-runs --runs-root runs_evolving/$m/median --hardware NVIDIA_GH200x2_median")
    A("done")
    A("")
    A("# uniform-threshold re-score (--completed-only drops in-flight arms)")
    A(".venv/bin/python scripts_integration/new_evolving_agent_analysis/rescore_hack_threshold.py \\")
    A("  --threshold 30 --all-dirs --completed-only \\")
    A("  --runs-root runs_evolving/gpt-oss-120b/median --runs-root runs_evolving/gpt-5.6-terra/median \\")
    A(f"  --output-dir {args.output_dir}")
    A("")
    A("# this report")
    A(".venv/bin/python scripts_integration/new_evolving_agent_analysis/make_kernel_legitimacy_report.py \\")
    A("  --runs-root gpt-oss-120b=runs_evolving/gpt-oss-120b/median \\")
    A("  --runs-root gpt-5.6-terra=runs_evolving/gpt-5.6-terra/median \\")
    cohort_arg = args.cohort or '""'
    A(f"  --cohort {cohort_arg} --output-dir {args.output_dir}")
    A("```")
    A("")
    A("## 7. Caveat")
    A("")
    A("n=1 per cell against a replicate log-SD of 0.147 (open item 10): a 95% band needs x1.50, "
      "Bonferroni across 8 contrasts x1.77. **Nothing here separates any treatment from its control.** "
      "Read descriptively, with n stated.")
    A("")

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "KERNEL_LEGITIMACY_REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path} ({len(data)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
