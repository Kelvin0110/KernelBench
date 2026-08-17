# GPT-OSS-120B GH200x2 (post-NVCC-fix) manifest

Scope: the GPT-OSS-120B runs under `runs_evolving/gpt-oss-120b/` executed on
GH200 **after** the NVCC / `CUDA_HOME` defect was repaired. Analysis follows
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

This folder is **not** `output/GH200x2/`. That folder and
`runs_evolving/archived/with_NVCC_bug/` remain invalidated under
ANALYSIS_RULES §7 and no number here is sourced from them.

## Runs and completion

| Design | Run name | Completed | Correct | Status |
|---|---|---:|---:|---|
| truncation | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` | 50/50 | 47/50 | complete |
| markov_report | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` | 50/50 | 48/50 | complete |
| selective_retention | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` | 50/50 | 48/50 | complete |
| compress_trigger | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` | 50/50 | 48/50 | complete |
| folding | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_13_12_47` | 49/50 | 46/50 | **partial — excluded** |
| truncation+deletion | `base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_14_15_52` | 30/31 | 30/31 | **partial — excluded** |

Runs root: `runs_evolving/gpt-oss-120b`

Both partials lack `run_summary.json`; `total_correct` is derived from
`workspaces/*/run_finished.json` over the finished subset only. They appear in
`aggregate_runs.{csv,json}` with warnings and are excluded from every headline
table and winner statement per ANALYSIS_RULES §3.

## Scoring baseline

`results/timing/NVIDIA_GH200x2/baseline_time_torch.json`

This is the **native** GH200 torch reference. Do not rescore these kernels onto
either A6000 vector (`SONG_CPU4_A6000x2`, `SONG_CPU6_A6000x4`), or the reverse.
Speedup is already relative; see ANALYSIS_RULES §2.

## Generated files

- `aggregate_runs.json` / `aggregate_runs.csv` (all six runs; partials flagged)
- `comparison.md` (four complete runs; required iteration-10/30 table)
- `EXPERIMENT_REPORT.md`
- this `MANIFEST.md`

## Commands (2026-08-16)

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/gpt-oss-120b \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/gpt-oss-120b \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58 \
  --runs base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09 \
  --runs base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/GH200x2_nvcc_fixed/comparison.md
```

The previous contents of this folder were a two-run `comparison.md` (2026-08-11)
and a three-run aggregate that predated both the compress run finishing and the
`best_n` fix. Both were regenerated on 2026-08-16.

## Exclusions

- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/` (invalidated)
- Folding and deletion GH200 cells (partial)
- A6000 numbers (different hosts and native baselines)
- Any GH200 → A6000 rescoring, in either direction
