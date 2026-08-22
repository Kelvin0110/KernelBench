# Step 9 — Timing baselines and the hardware tag

*Part of the [2 × GH200 host setup guide](README.md).*

---

KernelBench computes

```
speedup = fixed_baseline / measured_runtime
```

where `fixed_baseline` is an **idle-GPU constant** read from
`results/timing/<hardware>/baseline_time_torch.json` — not re-measured per run. Every
number the experiment reports is divided by it, so getting it wrong silently rescales
an entire arm.

## The good news: the baseline already exists

Because the target is also GH200, the repo's shipped baseline applies directly:

```
results/timing/NVIDIA_GH200x2/baseline_time_torch.json     48 KB
  level1: 100 problems   level2: 100 problems   level3: 50 problems
  per entry: mean, std, min, max, num_trials=100,
             hardware="NVIDIA GH200 144G HBM3e", device="cuda:0", precision="fp32"
  meta: {precision: fp32, use_torch_compile: false}
```

It was generated on the source host on 2026-08-03. Nothing to install, and **no
launcher edits are needed** — `launch_run.sh:37,69`, `resume_run.sh:82`,
`launch_arm_reps.sh:145,231`, `launch_merge_reps.sh:148,230,278` and
`launch_nvcc_series.sh:29,77` already hardcode `NVIDIA_GH200x2`, which is now correct
rather than something to patch.

> Do **not** rely on the default. `evolve_kb_batch.py`'s `--hardware` default is
> `SONG_CPU6_A6000x4` — invoking the batch runner by hand without `--hardware` scores
> against A6000 baselines and produces nonsense. The launch scripts always pass it.

## Reuse or regenerate?

**Reuse the shipped baseline** if arms from the two hosts will ever appear in the same
analysis. Speedups are only comparable when the divisor is identical; regenerating on
the new host guarantees the two fleets are on different scales even if the hardware is
the same to within noise.

**Regenerate** only if [Verify the target matches](01-verify-target-matches.md) turned
up a real hardware difference — a different GPU SKU, a lower power cap, different max
clocks, MIG enabled, or ECC toggled. In that case the two hosts are not comparable and
you should say so in the run metadata rather than paper over it.

## Validate before trusting the reuse (recommended, ~1 h)

Measure a fresh baseline into a scratch hardware name on an **idle** GPU, then diff it
against the shipped one. This never touches `NVIDIA_GH200x2/`:

```bash
cd "$REPO"
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/generate_baseline_time.py \
  --hardware NVIDIA_GH200x2_validate --baseline baseline_time_torch \
  --precision fp32 --yes
```

Then compare:

```bash
uv run --no-sync python - <<'PY'
import json, statistics as st
A = json.load(open("results/timing/NVIDIA_GH200x2/baseline_time_torch.json"))
B = json.load(open("results/timing/NVIDIA_GH200x2_validate/baseline_time_torch.json"))
ratios, worst = [], []
for lvl in ("level1", "level2", "level3"):
    for name, a in A[lvl].items():
        b = B.get(lvl, {}).get(name)
        if not b or not a.get("mean") or not b.get("mean"):
            continue
        r = b["mean"] / a["mean"]          # new / shipped
        ratios.append(r); worst.append((abs(r - 1), r, lvl, name))
ratios.sort(); worst.sort(reverse=True)
n = len(ratios)
print(f"n={n}  median={st.median(ratios):.4f}  "
      f"p05={ratios[int(.05*n)]:.4f}  p95={ratios[int(.95*n)]:.4f}")
print("largest deviations (new/shipped):")
for _, r, lvl, name in worst[:10]:
    print(f"  {r:6.3f}  {lvl}/{name}")
PY
```

How to read it:

- **median within ~1.00 ± 0.03 and p05/p95 inside ~±0.10** → the hosts agree; reuse
  `NVIDIA_GH200x2` and delete the scratch dir.
- **a consistent shift** (median 1.08, say — the new host is uniformly slower) → a real
  hardware or clock difference. Chase it in
  [Verify the target matches](01-verify-target-matches.md) before accepting it; if it
  is genuine, the two hosts are not comparable.
- **a handful of wild outliers on otherwise-agreeing data** → almost always a busy GPU
  during measurement, not a hardware difference. Sub-millisecond level-1 kernels are
  the noisiest; check `nvidia-smi` was idle and re-measure.

Whatever you conclude, do it *before* launching arms. A baseline decided after the
fact cannot be applied retroactively without recomputing every speedup.

## Run naming across two hosts

Run directories are timestamped (`<run_name>_YYYY_MM_DD_HH_MM`), so the two hosts
cannot collide on disk. But if you pool both hosts' results into one
`runs_evolving/gpt-oss-120b/` tree for analysis, the arms become indistinguishable by
name. Add a host tag to `--run-name` on the second host, e.g.

```
base_agent_gpt_oss_120b_markov_itr30_GH200b
```

keeping the existing `<tag>` convention (encode any non-default parameter in the tag).
`--hardware` is also recorded in `run_summary.json` as `hardware_server`, so it stays
attributable there regardless.

---

[← Acceptance test](10-acceptance-test.md) · [Index](README.md) · [Multi-arm settings →](12-multi-arm-settings.md)
