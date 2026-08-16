# GPT-OSS-120B inference / CPU6 manifest

Scope: the eight completed GPT-OSS-120B inference runs under
`runs_evolving/inference_oss_120b/`. Analysis follows
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

## Runs and completion

All eight runs have `run_summary.json`, 50/50 completed, and 50/50
`run_finished.json` markers. All record `resume=true`.

| Design | Run name | Correct |
|---|---|---:|
| truncation | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | 49/50 |
| truncation+deletion | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` | 49/50 |
| truncation+refine | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` | 47/50 |
| truncation+merge@0.7 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` | 49/50 |
| selective_retention | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | 48/50 |
| markov_report | `base_agent_oss120b_markov_itr30_2026_08_07_14_07` | 50/50 |
| folding | `base_agent_oss120b_folding_itr30_2026_08_09_13_47` | 48/50 |
| truncation+deletion+merge@0.7+refine | `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48` | 48/50 |

Runs root: `/home/kwtamai/KernelBench/runs_evolving/inference_oss_120b`

## Scoring baseline

`/home/kwtamai/KernelBench/results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`

Do not rescore these runs onto CPU4. Speedup is already relative.

## Generated files

- `aggregate_runs.json` / `aggregate_runs.csv`
- `comparison.md` (includes the required iteration-10/30 checkpoint table)
- `feature_evidence.json` / `feature_evidence.csv`
- `EXPERIMENT_REPORT.md` (this series’ narrative)
- this `MANIFEST.md`

## Commands (2026-08-16)

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/comparison.md

.venv/bin/python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root runs_evolving/inference_oss_120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_deletion_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49 \
  --runs base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56 \
  --runs base_agent_oss120b_markov_itr30_2026_08_07_14_07 \
  --runs base_agent_oss120b_folding_itr30_2026_08_09_13_47 \
  --runs base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48
```

Performance stats were rebuilt in-process (`regenerated_stale`) because run
artifacts were newer than the previous cache. The baseline file remained
CPU6. Feature evidence is read-only on run directories.

## Exclusions

- GH200 / `archived/with_NVCC_bug/` kernel-quality numbers
- Old integrate endpoint runs under `runs_evolving/` (not this series)
- Terra numbers (different model and native CPU4 baseline)
