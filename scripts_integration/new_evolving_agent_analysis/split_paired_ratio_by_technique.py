#!/usr/bin/env python3
"""Split an arm-vs-control paired speedup ratio by the technique its best kernel used.

Motivation: on the 2026-08-22 terra wave the compress arm's +19% geomean lead over the
truncation control was entirely attributable to CUDA Graphs (16 best kernels vs the
control's 6). Excluding those problems the ratio is x0.97. No aggregate metric shows this.

Usage:
  python3 split_paired_ratio_by_technique.py --runs-root runs_evolving/gpt-5.6-terra/median \
      --arm <arm_run_dir_name> --control <control_run_dir_name> [--hack-threshold 30]
"""
from __future__ import annotations
import argparse, json, math, os, re, statistics as st
from math import comb

TECHNIQUES = {
    "cuda_graph":   re.compile(r"CUDAGraph|cuda\.graph\("),
    "cudnn_bench":  re.compile(r"cudnn\.benchmark"),
    "tf32":         re.compile(r"allow_tf32|set_float32_matmul_precision"),
    "half":         re.compile(r"\.half\(\)|float16|bfloat16|__half|autocast"),
    "fp8":          re.compile(r"float8|_scaled_mm"),
}
# Problems whose reference model is algebraically collapsible; see KERNEL_LEGITIMACY_REPORT.md
COLLAPSE = {"L2P13", "L2P42", "L2P51", "L2P56"}


def geo(v: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in v) / len(v))


def load_arm(run_dir: str, threshold: float) -> dict[str, dict]:
    """Best clean sample per problem, recomputing is_hack uniformly as speedup > threshold."""
    data = json.load(open(os.path.join(run_dir, "evolving_runs.json")))
    out: dict[str, dict] = {}
    for r in data["runs"]:
        key = f"L{r['level']}P{r['problem_id']}"
        best, best_rec = None, None
        for rec in r["records"]:
            ev = rec.get("evaluation") or {}
            sp = ev.get("speedup")
            if not ev.get("correct") or sp is None or sp > threshold:
                continue
            if (ev.get("runtime") or 0) <= 0:
                continue
            if best is None or sp > best:
                best, best_rec = sp, rec
        techs = set()
        if best_rec is not None:
            code = best_rec.get("candidate_code") or ""
            techs = {n for n, p in TECHNIQUES.items() if p.search(code)}
        out[key] = {"best": best, "techs": techs}
    return out


def paired(arm, ctrl, keys, label):
    lr = [math.log(arm[k]["best"] / ctrl[k]["best"]) for k in keys]
    n = len(lr)
    if n < 2:
        print(f"  {label:44s} n={n} (too few)")
        return
    m, se = st.mean(lr), st.stdev(lr) / math.sqrt(n)
    wins = sum(1 for x in lr if x > 0)
    p = min(1.0, sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n * 2)
    print(f"  {label:44s} n={n:2d}  x{math.exp(m):.3f}  "
          f"95%CI [x{math.exp(m - 1.96 * se):.3f}, x{math.exp(m + 1.96 * se):.3f}]  "
          f"wins {wins}/{n}  sign p={p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--hack-threshold", type=float, default=30.0)
    a = ap.parse_args()

    arm = load_arm(os.path.join(a.runs_root, a.arm), a.hack_threshold)
    ctrl = load_arm(os.path.join(a.runs_root, a.control), a.hack_threshold)
    keys = sorted(k for k in arm if arm[k]["best"] and ctrl.get(k, {}).get("best"))

    print(f"arm     : {a.arm}\ncontrol : {a.control}\nthreshold: {a.hack_threshold}\n")
    print(f"geomean  arm={geo([arm[k]['best'] for k in keys]):.4f}  "
          f"control={geo([ctrl[k]['best'] for k in keys]):.4f}  aligned n={len(keys)}\n")

    print("Paired log-ratio, whole set and with each technique's problems removed:")
    paired(arm, ctrl, keys, "ALL")
    for tech in TECHNIQUES:
        used = [k for k in keys if tech in arm[k]["techs"]]
        if not used:
            continue
        print(f"    [{tech}] arm used it on {len(used)}/{len(keys)}; "
              f"control on {sum(1 for k in keys if tech in ctrl[k]['techs'])}")
        paired(arm, ctrl, used, f"only problems where arm used {tech}")
        paired(arm, ctrl, [k for k in keys if k not in used], f"excluding arm's {tech} problems")
    paired(arm, ctrl, [k for k in keys if k not in COLLAPSE], "excluding the 4 collapse problems")


if __name__ == "__main__":
    main()
