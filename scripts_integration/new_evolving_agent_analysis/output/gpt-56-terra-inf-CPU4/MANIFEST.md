# GPT-5.6 Terra inference / CPU4 manifest

Scope: completed GPT-5.6 Terra inference runs under
`runs_evolving/inference_gpt_56_terra/`. Analysis follows
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md).

## Runs and completion

| Design | Run name | Completed | Correct | Status |
|---|---|---:|---:|---|
| truncation | `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | 50/50 | 49/50 | complete, resume=true |
| markov_report | `base_agent_terra_markov_itr30_2026_08_01_17_41` | 50/50 | 49/50 | complete, resume=true |
| compress_trigger | `base_agent_terra_compress_trigger_itr30_2026_08_10_15_24` | 50/50 | 49/50 | complete, resume=false |
| folding | `base_agent_terra_folding_itr30_2026_08_09_15_11` | 15/50 | 15/50 | **partial — excluded** |

Runs root: `/home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra`

Compress-trigger recorded `compress_hot_rounds=3`,
`compress_token_ratio=0.85`, `compress_every_n_iters=15`. That is the
governor default for hot rounds, not the hot=15 recipe in
`scripts_integration/new_evolving_agent/infer_api/RUN_WITH_UV_INFER.md`.

## Scoring baseline

`/home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`

Do not rescore these runs onto CPU6. Speedup is already relative. Source
`visualizations/performance_stats.json` files remain on this CPU4 file.

## Generated files

- `aggregate_runs.json` / `aggregate_runs.csv` (includes the partial folding
  row; headline tables drop it)
- `comparison.md` (complete runs only; required iteration-10/30 table)
- `feature_evidence.json` / `feature_evidence.csv` (three complete runs)
- `EXPERIMENT_REPORT.md`
- this `MANIFEST.md`

## Commands (2026-08-16)

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/inference_gpt_56_terra \
  --baseline-file results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/inference_gpt_56_terra \
  --baseline-file results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4 \
  --runs base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_terra_markov_itr30_2026_08_01_17_41 \
  --runs base_agent_terra_compress_trigger_itr30_2026_08_10_15_24 \
  --baseline-run base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4/comparison.md

.venv/bin/python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root runs_evolving/inference_gpt_56_terra \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4 \
  --baseline-run base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_terra_markov_itr30_2026_08_01_17_41 \
  --runs base_agent_terra_compress_trigger_itr30_2026_08_10_15_24
```

## Exclusions

- Folding (partial)
- GH200 / NVCC-bug archives
- OSS numbers in this folder (different model and CPU6 baseline)
- Any Terra→CPU6 rescoring
