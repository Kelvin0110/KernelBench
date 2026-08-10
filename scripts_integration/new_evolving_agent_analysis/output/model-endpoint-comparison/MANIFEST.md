# Model/endpoint comparison manifest

Scope: provenance for [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md). This
cross-series task reads frozen per-series aggregates and evidence products.
Source runs, source caches, code, plans, and repository history were not
modified. A prior Terra→CPU6 rescoring under `common-baseline/` was removed;
cross-model speed uses each series' native baseline only.

## 1. Classification and baseline rules

- Qualifying completed runs directly under `runs_evolving/` are classified as
  **old NVIDIA integrate endpoint** runs. This is the user's authoritative
  directory-layout rule. Legacy `run_summary.json` files may have null
  model/endpoint fields; those nulls do not override the classification.
- Runs under `runs_evolving/inference_oss_120b/` and
  `runs_evolving/inference_gpt_56_terra/` are **inference endpoint** runs.
- Native baselines retained for speed/fast-p:
  - OSS inference and old integrate OSS:
    `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
  - Terra inference:
    `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`
- Speedup is already relative (`torch_baseline / kernel`). Cross-model
  comparisons use those native relative scores; they do **not** recompute
  Terra kernels against the CPU6 torch vector.
- Fast-p thresholds retained: `0.0,0.5,0.8,1.0,1.5,2.0`.

## 2. Exact generated analysis inputs

### OSS-120B inference / CPU6

Directory:
`scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/`

- `EXPERIMENT_REPORT.md`
- `MANIFEST.md`
- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `comparison-context.md`
- `comparison-governance.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Raw root:
`runs_evolving/inference_oss_120b/`

Completed runs used:

1. `base_agent_gpt_oss_120b_itr30_2026_08_02_17_58`
   — truncation/no governance, T0.
2. `base_agent_oss120b_selective_recent5_itr30_2026_08_05_15_56`
   — selective retention.
3. `base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49`
   — merge-only similarity 0.7.

The completed deletion and refinement runs are used only for the original
component-report summary:

4. `base_agent_oss120b_deletion_itr30_2026_08_02_17_57`
5. `base_agent_oss120b_skill_refinement_itr30_2026_08_02_17_57`

### GPT-5.6 Terra inference / CPU4

Directory:
`scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4/`

- `EXPERIMENT_REPORT.md`
- `MANIFEST.md`
- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Raw root:
`runs_evolving/inference_gpt_56_terra/`

Completed runs used:

1. `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40`
2. `base_agent_terra_markov_itr30_2026_08_01_17_41`

Timing input:
`results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`.

Source run `visualizations/performance_stats.json` files remain on CPU4 and
were not overwritten.

### OSS-120B old integrate / CPU6

Directory:
`scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-int-CPU6/`

- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `comparison-context.md`
- `comparison-governance.md`
- `feature_evidence.json`
- `feature_evidence.csv`

The report does not depend on an old-series narrative report.

Raw root:
`runs_evolving/`

Completed runs used:

1. `base_agent_markov_report_itr30_2026_07_21_17_11`
   — old integrate Markov.
2. `base_agent_selective_retention_itr30_2026_07_24_17_17`
   — old integrate selective campaign A.
3. `base_agent_selective_retention_itr30_2026_07_26_15_43`
   — old integrate selective campaign B.
4. `base_agent_with_merge_only_sim_07_itr30_2026_07_14_13_53`
   — old integrate merge-only similarity 0.7.

The old aggregate was generated against:
`results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`.

## 3. Raw artifact classes

Generated analyses above derive from these raw artifacts under each selected run:

- `run_summary.json`
- `batch_timing.jsonl`
- `eval_results.json` and per-level evaluation JSON where present
- `shared_l1.jsonl` / `shared_l1.txt`
- `visualizations/performance_stats.json`
- `workspaces/*/run_finished.json`
- `workspaces/*/metrics_by_iteration.jsonl`
- `workspaces/*/iteration_snapshots.jsonl`
- `workspaces/*/chat_history.jsonl`
- governance sidecars where enabled

The cross-series report primarily reads frozen aggregate/evidence products.
Raw-artifact conclusions quoted from the source reports include the new
selective 404 case, Terra timeout/budget boundaries, resume state, and sticky
hack behavior.

## 4. Analysis semantics and runbooks

Metric/extractor documentation:

- `scripts_integration/new_evolving_agent_analysis/README.md`
- `scripts_integration/new_evolving_agent_analysis/output/README.md`

Old integrate launch/design documentation:

- `scripts_integration/new_evolving_agent/RUN_WITH_UV.md`
- `scripts_integration/new_evolving_agent/RUN_WITH_UV_CONTEXT.md`

Inference launch/design documentation:

- `scripts_integration/new_evolving_agent/infer_api/RUN_WITH_UV_INFER.md`
- `scripts_integration/new_evolving_agent/infer_api/RUN_WITH_UV_INFER_SKILL.md`
- `scripts_integration/new_evolving_agent/infer_api/RUN_SKILL_GOVERNANCE.md`

Important semantic rules:

- fast-p keeps the full aligned-problem denominator;
- `fast_p_best` uses running-best runtime and does not apply the same sticky
  hack exclusion as best-speedup aggregates;
- speedup geomeans include correct, non-hack selected samples only and must be
  read with `n`;
- endpoint-reported token totals may be lower bounds when usage fields are
  missing;
- error categories are heuristic;
- cumulative wall time is an operational metric, not a treatment effect;
- do not rescore one host's kernels onto another host's torch baseline for
  cross-model fast-p ranking.

## 5. Report derivations

### Cross-model inference truncation cell

- OSS values: T0 record in
  `gpt-oss-120b-inf-CPU6/aggregate_runs.json` and token/call/error values in
  `feature_evidence.json`.
- Terra fast-p, trajectory, correctness, tokens, calls, errors, and wall:
  `gpt-56-terra-inf-CPU4/{aggregate_runs.json,feature_evidence.json}`.
- Headline winner rules: larger correctness/fast-p wins; fewer reported
  tokens/calls/errors/wall wins only as an operational efficiency observation.

### Terra component comparison

- Native CPU4 fast-p and trajectories:
  `gpt-56-terra-inf-CPU4/aggregate_runs.json`.
- Correctness and operational metrics:
  `gpt-56-terra-inf-CPU4/aggregate_runs.json` and `feature_evidence.json`.
- Original-report interpretation:
  `gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md`.

### Old versus new endpoint cells

- Old fast-p/correctness/wall:
  `gpt-oss-120b-int-CPU6/aggregate_runs.{json,csv}`.
- New fast-p/correctness/wall:
  `gpt-oss-120b-inf-CPU6/aggregate_runs.{json,csv}`.
- Calls/tokens/errors and endpoint-failure evidence:
  corresponding `feature_evidence.json` files.
- Selective comparison uses both completed old selective campaigns and the one
  completed new selective run.
- Merge comparison uses only similarity-0.7 merge-only runs.

### Same-mode old OSS versus Terra Markov

- Old OSS outcomes/trajectory:
  `gpt-oss-120b-int-CPU6/aggregate_runs.json`.
- Old OSS calls/tokens/errors:
  `gpt-oss-120b-int-CPU6/feature_evidence.json`.
- Terra outcomes/trajectory:
  `gpt-56-terra-inf-CPU4/aggregate_runs.json`.
- Terra correctness/operations:
  original Terra aggregate/evidence.

All displayed decimal fast-p values are direct source values. Counts such as
fast-p qualifying problems are the exact fraction times the fixed denominator
50. Token differences and wall differences are arithmetic derived from the
displayed source totals.

## 6. Exclusions

- `scripts_integration/new_evolving_agent_analysis/output/GH200x2/INVALIDATED.md`
  is read as a tombstone. All GH200 kernel-quality values are excluded because
  the runs lacked a valid CUDA toolchain and learned fallback behavior. GH200
  remains invalidated inference data and is not reclassified as old integrate.
- OSS inference partial runs excluded:
  `base_agent_oss120b_markov_itr30_2026_08_07_14_07`,
  `base_agent_oss120b_folding_itr30_2026_08_09_13_47`, and
  `base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48`.
- Terra inference partial folding run excluded:
  `base_agent_terra_folding_itr30_2026_08_09_15_11`.
- Old integrate folding and unmatched governance combinations are excluded from
  endpoint-effect conclusions.
- No old truncation control exists in the selected old integrate series.
- Discarded approach: Terra→CPU6 rescoring under
  `model-endpoint-comparison/common-baseline/` (removed). Do not regenerate it
  for cross-model claims.
- No partial prefix, historical GH200 number, best-geomean-only ranking, or
  post-hoc case study is treated as a final comparative result.
