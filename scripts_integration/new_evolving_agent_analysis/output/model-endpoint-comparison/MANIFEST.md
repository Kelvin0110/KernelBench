# Inference Terra vs OSS comparison manifest

Scope: provenance for [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).
Follows [ANALYSIS_RULES.md](../../ANALYSIS_RULES.md). This redo uses only
`runs_evolving/inference_gpt_56_terra` and
`runs_evolving/inference_oss_120b`. Old integrate OSS and GH200 are out of
scope. Terra is not rescored onto CPU6.

## Native baselines

- OSS: `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- Terra: `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`

## Source products (regenerated 2026-08-16)

- `output/gpt-oss-120b-inf-CPU6/{aggregate_runs.json,comparison.md,feature_evidence.json,EXPERIMENT_REPORT.md,MANIFEST.md}`
- `output/gpt-56-terra-inf-CPU4/{aggregate_runs.json,comparison.md,feature_evidence.json,EXPERIMENT_REPORT.md,MANIFEST.md}`

Checkpoint tables in the synthesis are copied from those aggregates’
`series.fast_p_best` and `series.speedup.best_geometric_mean` at iterations
10 and 30, plus `outcomes.total_correct`.

## Required table

Every complete design variant must appear with `fast_p_best@0/1/2` and
`speedup_best` geomean (`n`) at iterations 10 and 30. `compare_runs.py`
emits that table into each series’ `comparison.md`.

## Exclusions

- Terra folding (15/50)
- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/`
- Old integrate runs directly under `runs_evolving/`
- Any Terra→CPU6 `common-baseline/` rescoring (removed; do not regenerate)
