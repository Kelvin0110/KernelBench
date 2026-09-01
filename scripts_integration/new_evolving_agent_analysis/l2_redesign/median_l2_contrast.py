"""Compare shipped L2 vs redesigned L2 on the median wave, per CLAUDE.md 4."""
import json, math, statistics as st
from pathlib import Path

ROOT = Path("/localhome/local-tianzheng/KernelBench/runs_evolving/gpt-oss-120b/median")

def short(n):
    s = n.replace("base_agent_gpt_oss_120b_", "")
    i = s.find("_itr30_GH200")
    if i >= 0: return s[:i]
    return "truncation" if s.startswith("itr30_GH200") else s

def load(run: Path):
    d = json.loads((run / "visualizations" / "performance_stats.json").read_text())
    last = d["iterations"][-1]
    out = {}
    for p in last["points"]:
        sp = p.get("best_speedup")
        ok = bool(p.get("best_correct")) and sp is not None and float(sp) > 0
        out[p["workspace_id"]] = float(sp) if ok else None
    return out

runs = {}
for d in sorted(ROOT.iterdir()):
    if not (d / "visualizations" / "performance_stats.json").exists():
        continue
    ps = json.loads((d / "visualizations" / "performance_stats.json").read_text())
    if ps["iterations"][-1]["problem_count"] != 50:
        continue                      # only completed 50-problem arms
    runs[short(d.name)] = load(d)

print("arms loaded:", len(runs))
for k in runs: print("  ", k, " clean:", sum(1 for v in runs[k].values() if v))

# ---- treatment-agnostic lottery rule: max/min clean best >= 4 across ALL arms
keys = sorted({k for r in runs.values() for k in r})
lottery = []
for k in keys:
    vals = [r[k] for r in runs.values() if r.get(k)]
    if len(vals) >= 2 and max(vals) / min(vals) >= 4.0:
        lottery.append(k)
print(f"\nlottery problems ({len(lottery)}/{len(keys)}), spread>=4x across {len(runs)} arms:")
for k in lottery:
    vals = sorted(r[k] for r in runs.values() if r.get(k))
    print(f"   {k:26s} min {vals[0]:6.2f}  max {vals[-1]:7.2f}  ratio {vals[-1]/vals[0]:6.1f}x")

def geo(d, drop=frozenset()):
    v = [x for k, x in d.items() if x and k not in drop]
    return math.exp(sum(math.log(x) for x in v) / len(v)), len(v)

def fastp(d, thr=1.0, drop=frozenset()):
    ks = [k for k in d if k not in drop]
    return sum(1 for k in ks if d[k] and d[k] >= thr) / len(ks), len(ks)

def med(d, drop=frozenset()):
    return st.median([x for k, x in d.items() if x and k not in drop])

print("\n=== arm-level, uniform is_hack@30x, NVIDIA_GH200x2_median baseline ===")
print(f"{'arm':<18}{'geo(all)':>9}{'n':>4}{'geo(adj)':>9}{'n':>4}{'fastp@1':>9}{'fastp adj':>10}{'median':>8}")
order = ["truncation", "l2", "l2redesign_r1", "l2redesign_r2", "l2redesign_r3",
         "folding", "markov", "selective_r5", "compress", "deletion", "refinement", "merge_sim08"]
for a in order:
    if a not in runs: continue
    d = runs[a]
    g, n = geo(d); ga, na = geo(d, set(lottery))
    f, _ = fastp(d); fa, _ = fastp(d, drop=set(lottery))
    lbl = a or "truncation(ctl)"
    print(f"{lbl:<18}{g:>9.3f}{n:>4}{ga:>9.3f}{na:>4}{f:>9.3f}{fa:>10.3f}{med(d, set(lottery)):>8.3f}")

def paired(a, b, drop=frozenset()):
    """log-ratio of arm a vs arm b over problems both solved cleanly."""
    ls = [math.log(a[k] / b[k]) for k in a
          if k not in drop and a.get(k) and b.get(k)]
    n = len(ls); m = st.mean(ls); sd = st.stdev(ls)
    se = sd / math.sqrt(n)
    return math.exp(m), math.exp(m - 1.96 * se), math.exp(m + 1.96 * se), n

def mcnemar(a, b, thr=1.0, drop=frozenset()):
    ks = [k for k in a if k not in drop]
    aw = sum(1 for k in ks if (a.get(k) or 0) >= thr and (b.get(k) or 0) < thr)
    bw = sum(1 for k in ks if (b.get(k) or 0) >= thr and (a.get(k) or 0) < thr)
    n = aw + bw
    if n == 0: return aw, bw, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(aw, bw) + 1)) / 2 ** n * 2
    return aw, bw, min(1.0, p)

ctl = runs["truncation"]
print("\n=== paired per-problem vs the truncation control (same dir, Aug-22 wave) ===")
print(f"{'contrast':<26}{'ratio(all)':>11}{'ratio(adj)':>11}{'95% CI (adj)':>20}{'n':>4}  fastp win/loss/p")
for a in ["l2", "l2redesign_r1", "l2redesign_r2", "l2redesign_r3"]:
    r0 = paired(runs[a], ctl)
    r1 = paired(runs[a], ctl, set(lottery))
    w, l, p = mcnemar(runs[a], ctl, drop=set(lottery))
    print(f"{a+' / ctl':<26}{r0[0]:>11.3f}{r1[0]:>11.3f}{f'[{r1[1]:.3f}, {r1[2]:.3f}]':>20}{r1[3]:>4}   {w}/{l}/p={p:.3f}")

print("\n=== redesign replicates vs the shipped-gate l2 arm ===")
for a in ["l2redesign_r1", "l2redesign_r2", "l2redesign_r3"]:
    r0 = paired(runs[a], runs["l2"])
    r1 = paired(runs[a], runs["l2"], set(lottery))
    w, l, p = mcnemar(runs[a], runs["l2"], drop=set(lottery))
    print(f"{a+' / l2':<26}{r0[0]:>11.3f}{r1[0]:>11.3f}{f'[{r1[1]:.3f}, {r1[2]:.3f}]':>20}{r1[3]:>4}   {w}/{l}/p={p:.3f}")

print("\n=== replicate spread among the 3 redesign arms (identical config) ===")
gs = [geo(runs[a], set(lottery))[0] for a in ["l2redesign_r1", "l2redesign_r2", "l2redesign_r3"]]
raw = [geo(runs[a])[0] for a in ["l2redesign_r1", "l2redesign_r2", "l2redesign_r3"]]
for nm, v in (("raw", raw), ("adjusted", gs)):
    lg = [math.log(x) for x in v]
    print(f"  {nm:9s} {['%.3f' % x for x in v]}  max/min {max(v)/min(v):.3f}  log-SD {st.stdev(lg):.3f}")

print("\n=== cell test: redesign cell (n=3) vs the single shipped l2 arm ===")
for drop, nm in ((frozenset(), "raw"), (frozenset(lottery), "adjusted")):
    ds = [math.log(paired(runs[a], runs["l2"], drop)[0]) for a in
          ["l2redesign_r1", "l2redesign_r2", "l2redesign_r3"]]
    m = st.mean(ds); sd = st.stdev(ds); se = sd / math.sqrt(3)
    print(f"  {nm:9s} mean log-ratio {m:+.4f} -> {math.exp(m):.3f}x, "
          f"replicate SD {sd:.3f}, 95% CI "
          f"[{math.exp(m-4.303*se):.3f}, {math.exp(m+4.303*se):.3f}] (t.975,df2=4.303)")
