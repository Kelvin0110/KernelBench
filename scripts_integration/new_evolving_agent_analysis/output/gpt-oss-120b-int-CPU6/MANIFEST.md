# Manifest: GPT-OSS-120B old-integrate CPU6/A6000 report

## Cohort and expected counts

- Runs root: `runs_evolving`
- Hardware recorded by runs: `SONG_CPU6_A6000x4` / `NVIDIA RTX A6000`
- Explicit baseline: `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- Expected complete runs: **10**
- Expected partial runs: **0**
- Expected finished workspaces: **500** (10 × 50)
- Expected final iteration per workspace: **30**
- Expected `cuda_home_err` rows across selected workspace metrics: **0**
- Expected feature-extractor warnings: **46**
- Aggregate failures / requested runs not found: **0 / 0**
- Speedup policy: `correct_only_exclude_hack`
- Headline comparison columns: final correctness; `fast_p_best@0.0`, `@1.0`, `@2.0`; and corresponding current fast-p
- Metric-leader policy: report observed maxima and ties, with n=1/no-control caveats; do not infer causal rankings

## Exact ten run names

Context management:

1. `base_agent_markov_report_itr30_2026_07_21_17_11`
2. `base_agent_folding_itr30_2026_07_28_01_09`
3. `base_agent_selective_retention_itr30_2026_07_24_17_17`
4. `base_agent_selective_retention_itr30_2026_07_26_15_43`

Governance/configuration:

5. `base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53`
6. `base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52`
7. `base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13`
8. `base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45`
9. `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48`
10. `base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24`

The two selective-retention names identify separate resumed campaigns. They are not automatically independent replicates.

## Generated files

Directory: `scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/`

- `aggregate_runs.json` — nested aggregate and per-iteration series
- `aggregate_runs.csv` — one row per run
- `comparison.md` — all ten runs
- `comparison-context.md` — four context runs
- `comparison-governance.md` — six governance/configuration runs
- `feature_evidence.json` — calls/tokens, errors, L0/L1/governance, provenance, deterministic case evidence
- `feature_evidence.csv` — one row per run/problem (500 rows plus header)
- `EXPERIMENT_REPORT.md` — evidence-first interpretation with explicit threshold headline columns and observed metric leaders
- `MANIFEST.md` — this inventory and reproduction record

## Provenance convention and schema caveat

The authoritative scope convention for this analysis is: qualifying runs directly under `runs_evolving/`, rather than inside endpoint-specific subfolders, use the old NVIDIA integrate endpoint. All ten listed runs meet that rule and are therefore classified as old-integrate GPT-OSS-120B.

These pre-August `run_summary.json` files omit both `nvidia_endpoint` and model metadata, so those aggregate fields remain null. The directory-layout classification is independently corroborated by:

- old runbooks using `NVIDIA_API_KEY` and the default integrate client path;
- sampled chat `model_id` values `openai/gpt-oss-120b` or `nvdev/openai/gpt-oss-120b`;
- the context runbook's explicit separation from newer inference-API instructions.

The schema caveat concerns absent per-run fields, not uncertainty about endpoint classification under the supplied convention.

`scripts_integration/new_evolving_agent_analysis/output/GH200x2/` is unrelated invalidated inference data. Its metrics are not inputs to this report.

## Exact commands

Run from `/home/kwtamai/KernelBench` with the repository virtual environment.

### Aggregate

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6 \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --runs base_agent_markov_report_itr30_2026_07_21_17_11 \
  --runs base_agent_folding_itr30_2026_07_28_01_09 \
  --runs base_agent_selective_retention_itr30_2026_07_24_17_17 \
  --runs base_agent_selective_retention_itr30_2026_07_26_15_43 \
  --runs base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53 \
  --runs base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24
```

### Compare all ten

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/aggregate_runs.json \
  --runs-root runs_evolving \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6 \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/comparison.md \
  --iteration-stride 5 \
  --fast-p 1.0 \
  --runs base_agent_markov_report_itr30_2026_07_21_17_11 \
  --runs base_agent_folding_itr30_2026_07_28_01_09 \
  --runs base_agent_selective_retention_itr30_2026_07_24_17_17 \
  --runs base_agent_selective_retention_itr30_2026_07_26_15_43 \
  --runs base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53 \
  --runs base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24
```

### Compare context runs

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/aggregate_runs.json \
  --runs-root runs_evolving \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6 \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/comparison-context.md \
  --iteration-stride 5 \
  --fast-p 1.0 \
  --runs base_agent_markov_report_itr30_2026_07_21_17_11 \
  --runs base_agent_folding_itr30_2026_07_28_01_09 \
  --runs base_agent_selective_retention_itr30_2026_07_24_17_17 \
  --runs base_agent_selective_retention_itr30_2026_07_26_15_43
```

### Compare governance/configuration runs

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/aggregate_runs.json \
  --runs-root runs_evolving \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6 \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/comparison-governance.md \
  --iteration-stride 5 \
  --fast-p 1.0 \
  --runs base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53 \
  --runs base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24
```

### Feature evidence

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root runs_evolving \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6 \
  --runs base_agent_markov_report_itr30_2026_07_21_17_11 \
  --runs base_agent_folding_itr30_2026_07_28_01_09 \
  --runs base_agent_selective_retention_itr30_2026_07_24_17_17 \
  --runs base_agent_selective_retention_itr30_2026_07_26_15_43 \
  --runs base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53 \
  --runs base_agent_with_merge_only_sim_08_itr30_2026_07_14_13_52 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_2026_07_14_14_13 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_2026_07_17_15_45 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_07_itr30_2026_07_17_15_48 \
  --runs base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_2026_07_18_05_24
```

No `--baseline-run` is supplied because there is no clean 50×30 old-integrate truncation control. The evidence output therefore correctly has `"comparisons": null`.

## Inputs inspected for the report

- All nine generated inputs listed above other than the two report files
- `scripts_integration/new_evolving_agent_analysis/README.md`
- `scripts_integration/new_evolving_agent/RUN_WITH_UV.md`
- `scripts_integration/new_evolving_agent/RUN_WITH_UV_CONTEXT.md`
- Each run's `run_summary.json`, generated performance stats, timing and workspace completion records
- Deterministic raw samples from `level_1_problem_100` chat, iteration snapshots, metrics, and selective milestones
- First/last merge and deletion ledger records plus final governance catalog sidecars

No source code, raw run, plan, unrelated report, or invalidated GH200 artifact was modified.
