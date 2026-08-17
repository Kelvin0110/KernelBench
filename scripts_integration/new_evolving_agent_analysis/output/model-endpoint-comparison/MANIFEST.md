# Cross-series synthesis manifest

Scope: provenance for [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).
Follows [ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

This synthesis covers **three** series on two axes — model (OSS-120B vs
Terra, matched on A6000) and hardware (OSS-120B on A6000 vs GH200). No series
is rescored onto another's baseline in either direction.

## Native baselines

| Series | Runs root | Baseline file |
|---|---|---|
| GPT-OSS-120B @ A6000 | `runs_evolving/inference_oss_120b` | `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json` |
| GPT-5.6-Terra @ A6000 | `runs_evolving/inference_gpt_56_terra` | `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json` |
| GPT-OSS-120B @ GH200 | `runs_evolving/gpt-oss-120b` | `results/timing/NVIDIA_GH200x2/baseline_time_torch.json` |

The measured accelerator in `SONG_CPU4_A6000x2` and `SONG_CPU6_A6000x4` is an
RTX A6000 in both cases; the folder names identify the *timing host*, not the
GPU.

## Baseline-comparison method (§7 of the report)

§7 compares the three `baseline_time_torch.json` vectors directly — torch
against torch, over the 249 problems present in all three files. It reports
per-problem ratio geomean, median, and the fraction of problems favouring each
host.

This is **not** a rescoring and does not violate ANALYSIS_RULES §2: no kernel
runtime is divided by a foreign baseline, and no run's
`visualizations/performance_stats.json` was touched. Its only use is to bound
how much of an apparent cross-host gap is attributable to the reference rather
than to the model or the design.

Key results: GH200 torch is 2.44× faster than A6000/CPU6 torch (geomean, faster
on 91% of problems); A6000/CPU6 torch is 1.14× faster than A6000/CPU4 torch
(faster on 100% of problems, 1.31× on Level 3).

## Source products (regenerated 2026-08-16)

- `output/gpt-oss-120b-inf-CPU6/{aggregate_runs.json,comparison.md,feature_evidence.json,EXPERIMENT_REPORT.md,MANIFEST.md}`
- `output/gpt-56-terra-inf-CPU4/{aggregate_runs.json,comparison.md,feature_evidence.json,EXPERIMENT_REPORT.md,MANIFEST.md}`
- `output/GH200x2_nvcc_fixed/{aggregate_runs.json,comparison.md,EXPERIMENT_REPORT.md,MANIFEST.md}`

Checkpoint tables in the synthesis are copied from those aggregates'
`series.fast_p_best` and `series.speedup.best_geometric_mean` at iterations 10
and 30, plus `outcomes.total_correct`.

## Verification (2026-08-16)

Both A6000 run sets were unpacked locally from
`runs_evolving/inference_gpt_56_terra.zip` and
`runs_evolving/inerence_oss_120b.zip` and re-aggregated from scratch. The
resulting iteration-10/30 tables matched the committed ones byte-for-byte
before the `n` fix, and the geomean at every checkpoint was independently
recomputed from the raw `performance_stats.json` points and matched to 1e-9.

## Required table

Every complete design variant must appear with `fast_p_best@0/1/2` and
`speedup_best` geomean (`n`) at iterations 10 and 30. `compare_runs.py` emits
that table into each series' `comparison.md`.

## Exclusions

- Terra folding (15/50), GH200 folding (49/50), GH200 deletion (30/31) — partial
- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/` — invalidated
  (missing NVCC/`CUDA_HOME`, learned PyTorch fallback)
- Old integrate runs directly under `runs_evolving/`
- Any cross-host `common-baseline/` rescoring (removed; do not regenerate)
