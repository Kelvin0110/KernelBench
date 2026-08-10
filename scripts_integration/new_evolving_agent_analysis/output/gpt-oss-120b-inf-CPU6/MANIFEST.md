# OSS-120B CPU6 inference analysis manifest

This manifest defines the exact five-run analysis set used by `EXPERIMENT_REPORT.md`. Paths are repository-relative unless shown as absolute.

## Included completed runs

Runs root: `runs_evolving/inference_oss_120b`

| order | run name | run summary completion | finished workspace markers | final correctness |
|---:|---|---:|---:|---:|
| 1 | `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58` | 50/50 | 50 | 48/50 |
| 2 | `base_agent_oss120b_deletion_itr30_2026_08_02_17_57` | 50/50 | 50 | 48/50 |
| 3 | `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57` | 50/50 | 50 | 47/50 |
| 4 | `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49` | 50/50 | 50 | 47/50 |
| 5 | `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56` | 50/50 | 50 | 44/50 |

No other run is included in aggregate, comparison, feature-evidence, or report result tables.

## Baseline

Exact baseline timing file:

`results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`

The file records FP32 timings on `NVIDIA RTX A6000`. Aggregate and comparison commands below always pass both:

- `--runs-root runs_evolving/inference_oss_120b`
- `--baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`

Those explicit flags are required to prevent `compare_runs.py` from accepting or rebuilding against a cache from a different run root or baseline.

The merge-only and selective-retention `run_summary.json` files omit `hardware_server`. Their CPU6 series membership is inferred from the selected run series, the explicit baseline above, and A6000 hardware recorded in `eval_results*.json`.

## Source artifacts

### Pre-existing analysis inputs

Directory: `scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6`

- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `comparison-governance.md`
- `comparison-context.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Metric semantics and reporting style:

- `scripts_integration/new_evolving_agent_analysis/README.md`
- `scripts_integration/new_evolving_agent_analysis/EXPERIMENT_REPORT.md`

### Raw run artifacts

For each exact run named above:

- `run_summary.json`
- `batch_timing.jsonl`
- `eval_results.json` and `eval_results_level_*.json`
- `shared_l1.jsonl` and `shared_l1.txt`
- `visualizations/performance_stats.json`
- `workspaces/*/run_finished.json`
- `workspaces/*/metrics_by_iteration.jsonl`
- `workspaces/*/iteration_snapshots.jsonl`
- `workspaces/*/chat_history.jsonl`

Feature-specific sidecars when present:

- deletion: `l1_skill_usage.json`, `l1_skill_deletions.jsonl`, `l1_skill_unit_test_runs.jsonl`
- refinement: `skill_revisions.txt`
- merge-only: `l1_skill_usage.json`, `l1_skill_merges.jsonl`,
  `l1_skill_merge_clustering.jsonl`, `l1_skill_merge_state.json`,
  `l1_skill_catalog_stats.json`, `l1_skill_unit_test_runs.jsonl`,
  `skill_merges.txt`

### Targeted raw case inspection

The report's chat evidence is limited to deterministic `case_study_candidates` anchors:

- all five runs:
  `workspaces/level_1_problem_56/chat_history.jsonl`,
  `iteration_snapshots.jsonl`, and `metrics_by_iteration.jsonl`;
- refinement-only:
  `workspaces/level_3_problem_49/chat_history.jsonl`,
  `iteration_snapshots.jsonl`, and `metrics_by_iteration.jsonl`;
- merge-only:
  `workspaces/level_3_problem_34/chat_history.jsonl`,
  `iteration_snapshots.jsonl`, and `metrics_by_iteration.jsonl`;
- selective retention and control:
  `workspaces/level_1_problem_54/chat_history.jsonl`,
  `iteration_snapshots.jsonl`, and `metrics_by_iteration.jsonl`.

The cited iterations are recorded in `EXPERIMENT_REPORT.md`; long model responses and generated code were not copied.

## Generated files

Analysis tooling added for this inference series:

- `scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py`
- `scripts_integration/new_evolving_agent_analysis/test_analyze_feature_evidence.py`
- the extractor section in `scripts_integration/new_evolving_agent_analysis/README.md`

Pre-existing generated analysis products:

- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `comparison-governance.md`
- `comparison-context.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Deliverables created from those inputs and read-only raw inspection:

- `EXPERIMENT_REPORT.md`
- `MANIFEST.md`

Report authoring did not modify the plan file, source run artifacts, GH200
output, Terra output, or performance caches.

## Extractor and pipeline caveats

### `aggregate_runs.py` / `compare_runs.py`

- Speedup aggregates use correct, non-hack samples only; always retain `speedup_*.n`.
- Fast-p retains the full-problem denominator. `fast_p_best` does not apply the same hack exclusion as `speedup_best`.
- The recorded best-hack state is sticky after an earlier flagged iteration, so best geomean subsets can exclude later clean bests.
- `best_speedup_overall` is not a simple maximum; it follows the run-summary policy.
- `compare_runs.py` validates cached `runs_root` and `baseline_file` only when those requested values are supplied correctly.
- Aggregate/compare can regenerate `<run>/visualizations/performance_stats.json` if a cache is absent or stale. The recorded aggregate used existing caches for all five runs. Reproduction on a changed artifact tree may therefore write a refreshed cache unless run on a copy.

### `analyze_feature_evidence.py`

- It rejects incomplete selected runs before writing outputs.
- A chat turn is one valid JSON object in `chat_history.jsonl`.
- Token totals sum reported endpoint usage; missing usage is not imputed except that `total_tokens` may be derived from reported prompt plus completion tokens.
- M1 has 3 token-usage-missing chat rows and S1 has 2; their reported totals are lower bounds for those rows.
- Phase counts count valid chat rows. Action mix counts parsable action-selector calls, not unique iterations.
- Metrics rates use rows where the relevant `metrics_iteration` field is observed.
- Valid matched speedup requires finite positive speedup, `best_correct=true`, and `best_is_hack=false`.
- L0 values come from recorded snapshot counts. L1 and governance counts describe rows and sidecar events, not semantic skill quality.
- Error categories are heuristic; excerpts are sanitized and bounded.
- Case-study candidates are deterministic descriptive selections with workspace-name tie breaks. They are not causal evidence.
- Optional-artifact warnings are expected when a feature is disabled; the five-run evidence file contains 34 such warnings in total.

## Exact reproduction commands

Run from the repository root with the repository virtual environment:

```bash
cd /home/kwtamai/KernelBench
```

### 1. Aggregate the exact five runs

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_deletion_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49 \
  --runs base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56
```

Expected run-set metadata: `discovered=5`, `aggregated=5`, `complete_runs=5`, `partial_runs=0`, no missing requested run, and no failure.

### 2. Full five-run comparison

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/aggregate_runs.json \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --fast-p 1.0 \
  --iteration-stride 5 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_deletion_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49 \
  --runs base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/comparison.md
```

### 3. Governance-only comparison

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/aggregate_runs.json \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --fast-p 1.0 \
  --iteration-stride 5 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_deletion_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/comparison-governance.md
```

### 4. Context-only comparison

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/aggregate_runs.json \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --fast-p 1.0 \
  --iteration-stride 5 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/comparison-context.md
```

### 5. Feature evidence

`analyze_feature_evidence.py` has no timing-baseline option. Its `--baseline-run` is the matched run, not a hardware timing file.

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root runs_evolving/inference_oss_120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --runs base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --runs base_agent_oss120b_deletion_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57 \
  --runs base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49 \
  --runs base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58
```

Expected extractor summary: 5 runs, 250 workspaces, 34 warnings.
