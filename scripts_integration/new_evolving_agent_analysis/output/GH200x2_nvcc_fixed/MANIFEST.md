# GPT-OSS-120B GH200x2 (post-NVCC-fix) manifest

Scope: the GPT-OSS-120B runs under `runs_evolving/gpt-oss-120b/` executed on
GH200 **after** the NVCC / `CUDA_HOME` defect was repaired. Analysis follows
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

This folder is **not** `output/GH200x2/`. That folder and
`runs_evolving/archived/with_NVCC_bug/` remain invalidated under
ANALYSIS_RULES §7 and no number here is sourced from them.

## Runs and completion (as of 2026-08-22)

| Setting | Run name | Completed | Correct | Status |
|---|---|---:|---:|---|
| truncation | `..._itr30_GH200_2026_08_07_13_58` | 50/50 | 47/50 | complete |
| markov_report | `..._markov_itr30_GH200_2026_08_07_13_58` | 50/50 | 48/50 | complete |
| folding | `..._folding_itr30_GH200_2026_08_13_12_47` | 50/50 | 47/50 | complete |
| selective_retention | `..._selective_r5_itr30_GH200_2026_08_11_14_09` | 50/50 | 48/50 | complete |
| compress_trigger | `..._compress_itr30_GH200_2026_08_10_15_22` | 50/50 | 48/50 | complete |
| truncation+deletion | `..._deletion_itr30_GH200_2026_08_14_15_52` | 50/50 | 48/50 | complete |
| truncation+refinement | `..._refinement_itr30_GH200_2026_08_17_15_52` | 50/50 | 45/50 | complete |
| truncation+merge@0.8 | `..._merge_sim08_itr30_GH200_2026_08_19_17_29` | 50/50 | 48/50 | complete (rep 1/3) |
| truncation+merge@0.8 | `..._merge_sim08_itr30_GH200_2026_08_19_17_32` | 50/50 | 47/50 | complete (rep 2/3) |
| truncation+merge@0.8 | `..._merge_sim08_itr30_GH200_2026_08_19_17_35` | 50/50 | 46/50 | complete (rep 3/3) |
| truncation | `..._itr30_GH200_2026_08_20_16_32` | 33/50 | — | **in flight — excluded** |
| truncation | `..._itr30_GH200_2026_08_20_16_42` | 32/50 | — | **in flight — excluded** |
| markov_report | `..._markov_itr30_GH200_2026_08_20_16_35` | 32/50 | — | **in flight — excluded** |
| markov_report | `..._markov_itr30_GH200_2026_08_20_16_45` | 31/50 | — | **in flight — excluded** |
| folding | `..._folding_itr30_GH200_2026_08_20_16_39` | 30/50 | — | **in flight — excluded** |
| folding | `..._folding_itr30_GH200_2026_08_20_16_48` | 31/50 | — | **in flight — excluded** |

Runs root: `runs_evolving/gpt-oss-120b`. All run names are prefixed
`base_agent_gpt_oss_120b_`.

The six 2026-08-20 runs were confirmed live via `pgrep -af evolve_kb_batch` at
analysis time. They lack `run_summary.json`; `total_correct` is derived from
`workspaces/*/run_finished.json` over the finished subset only. They appear in
`aggregate_runs.{csv,json}` with warnings and are excluded from every headline
table and winner statement per ANALYSIS_RULES §3.

## Replicates

Only `truncation+merge@0.8` has completed replicates (3). Report rows for it are
averages: fast-p arithmetic, geomean in **log space** (geometric mean of the
per-replicate geomeans). Per-replicate values are printed alongside, because the
spread is wide enough to matter (I30 geomean 0.8379 / 0.8552 / 1.0919).

When the six in-flight runs land, truncation, markov_report, and folding each
gain 2 replicates and the tables must be recomputed.

## Verified confounds

- **`truncation+deletion` is deletion + unit-test GC.**
  `l1_skill_deletions.jsonl` in that run: 567 deletions, **326 `unit_test_fail`**
  + 241 `consecutive_unused`, while `run_summary.json` records
  `enable_l1_skill_unit_test_gc: false`. The gate reads the wrong config field
  (`gen3_stages.py`). The mechanisms are not separable from this run.
- **Merge arms did real work** (checked per ANALYSIS_RULES / CLAUDE.md §3.5):
  `l1_skill_embeddings.json` skill counts 700 / 728 / 679 and
  `l1_skill_merges.jsonl` line counts 124 / 171 / 180. No silent-failure case.

## Scoring baseline

`results/timing/NVIDIA_GH200x2/baseline_time_torch.json` (unmodified since
2026-08-03).

This is the **native** GH200 torch reference. Do not rescore these kernels onto
either A6000 vector (`SONG_CPU4_A6000x2`, `SONG_CPU6_A6000x4`), or the reverse.
Speedup is already relative; see ANALYSIS_RULES §2.

## Generated files

- `aggregate_runs.json` / `aggregate_runs.csv` (all 16 runs; in-flight flagged)
- `comparison.md` (10 complete runs; required iteration-10/30 table)
- `EXPERIMENT_REPORT.md`
- this `MANIFEST.md`

## Commands (2026-08-22)

```bash
# 1. regenerate per-run stats with the authoritative generator
.venv/bin/python Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py \
  --all-runs \
  --runs-root runs_evolving/gpt-oss-120b \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0
# -> discovered 16, generated 16, skipped 0

# 2. aggregate
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/gpt-oss-120b \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --regenerate-stats \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed

# 3. compare (10 complete runs; --runs repeated per run, control as --baseline-run)
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/gpt-oss-120b \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed/comparison.md
```

## Verification (2026-08-22)

Iteration-30 geomeans for truncation, markov_report, and selective_retention
were independently recomputed from raw `metrics_by_iteration.jsonl` (running min
over non-hack correct runtimes, divided into the native GH200 baseline) and
matched the generator to 1e-6: 0.905098 (n=47), 0.933166 (n=48), 0.954074
(n=48).

**Correction:** the 2026-08-16 edition reported 0.8746 (truncation) and 0.9830
(markov). Run artifacts and the baseline file are unchanged; the generator was
updated on 2026-08-16. Those two figures are superseded — see
`EXPERIMENT_REPORT.md` §8.

## Exclusions

- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/` (invalidated)
- The six 2026-08-20 replicate runs (in flight)
- A6000 numbers (different hosts and native baselines)
- Any GH200 → A6000 rescoring, in either direction
