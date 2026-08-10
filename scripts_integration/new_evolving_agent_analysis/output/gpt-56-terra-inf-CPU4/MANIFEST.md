# GPT-5.6 Terra CPU4/A6000 analysis manifest

Scope: exactly two completed GPT-5.6 Terra inference runs. No GH200 or GPT-OSS
artifact is included.

## Runs and completion

| Run name | Mode | Completed / attempted | Finished workspaces | Correct | Status |
|---|---|---:|---:|---:|---|
| `base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40` | `truncation` | 50/50 | 50/50 | 49/50 | complete |
| `base_agent_terra_markov_itr30_2026_08_01_17_41` | `markov_report` | 50/50 | 50/50 | 47/50 | complete |

Runs root:

`/home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra`

Both `run_summary.json` files record `resume=true`. The final truncation
session starts at problem index 9 and times 42 problems; the final Markov
session starts at index 13 and times 38.

## Scoring baseline

Exact CPU4/A6000 timing file:

`/home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`

The file records NVIDIA RTX A6000, CUDA, FP32 timings. Replacing or
regenerating it changes aggregate/compare speedup and fast-p values.
Feature-evidence bests may instead retain the reference timing recorded in
`run_finished.json` or historical snapshots; see caveat 6.

## Source artifacts

Analysis inputs in this directory:

- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Raw run-level inputs, under each exact run directory:

- `run_summary.json`
- `batch_timing.jsonl`
- `eval_results.json` and available per-level `eval_results_level_*.json`
- `evolving_runs.json`
- `shared_l1.jsonl`
- `visualizations/performance_stats.json`

Raw workspace inputs:

- `workspaces/*/chat_history.jsonl`
- `workspaces/*/metrics_by_iteration.jsonl`
- `workspaces/*/iteration_snapshots.jsonl`
- `workspaces/*/run_finished.json`

Status-only excluded input:

- `runs_evolving/inference_gpt_56_terra/base_agent_terra_folding_itr30_2026_08_09_15_11`
  was inspected only to establish that folding was pending (no
  `run_summary.json`, 12 finish markers, next workspace in progress). No
  folding performance value enters the report.

The targeted case-study audit used those four workspace artifacts for:

- `level_3_problem_3`
- `level_3_problem_24`
- `level_2_problem_13`
- `level_1_problem_56`
- `level_2_problem_19`

Metric meanings and reporting style were taken from:

- `scripts_integration/new_evolving_agent_analysis/README.md`
- `scripts_integration/new_evolving_agent_analysis/EXPERIMENT_REPORT.md`

## Generated files

Analysis tooling added for this inference series:

- `scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py`
- `scripts_integration/new_evolving_agent_analysis/test_analyze_feature_evidence.py`
- the extractor section in `scripts_integration/new_evolving_agent_analysis/README.md`

Pre-existing generated evidence:

- `aggregate_runs.json`
- `aggregate_runs.csv`
- `comparison.md`
- `feature_evidence.json`
- `feature_evidence.csv`

Deliverables created from that evidence:

- `EXPERIMENT_REPORT.md`
- `MANIFEST.md`

Report authoring did not modify source run artifacts, the plan file, GH200
output, OSS output, or repository history.

## Extractor and reporting caveats

1. `analyze_feature_evidence.py` accepts only explicitly selected complete
   runs. It reads chat, metric, snapshot, finish, L1, summary, and optional
   governance artifacts; it writes only `feature_evidence.json` and CSV.
2. A chat turn is one valid JSONL object. Token totals sum endpoint-reported
   values; missing usage is not imputed. Truncation has no missing token
   totals. Markov has four calls without total-token usage: three
   evolving-report calls and one summarizer call.
3. Action counts cover parsable action-selector calls. Both runs have zero
   action parse errors.
4. Error categories are heuristic and based on
   `metrics_iteration.error`; they are descriptive, not authoritative root
   causes.
5. Valid matched speedup requires a finite positive best with
   `best_correct=true` and `best_is_hack=false`. Case-study candidates use
   deterministic rules and workspace-name tie breaks; they do not establish
   causality.
6. Compact best fields can come from `run_finished.json`/`metrics_best`.
   Historical snapshots can retain their original reference runtime. In
   Markov `level_2_problem_13`, the snapshot and finish marker report
   8.735632 from 38.0/4.35, while the current metric row reports 8.689655 from
   37.8/4.35. The feature candidate uses the finish-marker value.
7. `metrics_best.is_hack` is sticky after a hack-flagged iteration. It can
   invalidate an earlier clean best and reduce `speedup_best.n`; read every
   best aggregate with `n`. `fast_p_best` does not apply that sticky exclusion.
8. The extractor reports 18 warnings: nine missing optional governance
   sidecars per run. Governance is disabled in both runs, so these are
   expected missing-artifact warnings. Core chat/metric/snapshot/finish
   artifacts parsed successfully.
9. Wall-time summaries span resumed sessions. Recorded average minutes/problem
   divide cumulative total time by `problems_timed_this_session` (42 or 38),
   not by all 50 completed problems.
10. `aggregate_runs.py` can refresh a stale
    `visualizations/performance_stats.json`. The artifacts used here were
    reported as `cached`; the reproduction command below does not force a
    rebuild.

## Exact reproduction commands

Run from the repository root. All run-root, timing-baseline, comparison
baseline, thresholds, and output paths are explicit.

### 1. Aggregate

```bash
cd /home/kwtamai/KernelBench
uv run python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root /home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra \
  --baseline-file /home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --output-dir /home/kwtamai/KernelBench/scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --runs base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_terra_markov_itr30_2026_08_01_17_41
```

Expected completion summary: two discovered, two aggregated, two complete,
zero partial, zero failures.

### 2. Compare

```bash
cd /home/kwtamai/KernelBench
uv run python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --aggregate /home/kwtamai/KernelBench/scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4/aggregate_runs.json \
  --runs-root /home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra \
  --baseline-file /home/kwtamai/KernelBench/results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --output-dir /home/kwtamai/KernelBench/scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4 \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --fast-p 1.0 \
  --iteration-stride 5 \
  --runs base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_terra_markov_itr30_2026_08_01_17_41 \
  --baseline-run base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --output /home/kwtamai/KernelBench/scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4/comparison.md
```

### 3. Extract feature evidence

`analyze_feature_evidence.py` does not accept a timing-baseline file; its
explicit baseline is the matched comparison run supplied with
`--baseline-run`.

```bash
cd /home/kwtamai/KernelBench
uv run python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root /home/kwtamai/KernelBench/runs_evolving/inference_gpt_56_terra \
  --runs base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --runs base_agent_terra_markov_itr30_2026_08_01_17_41 \
  --baseline-run base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --output-dir /home/kwtamai/KernelBench/scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4
```

Expected extractor summary: two runs, 100 workspaces, 18 warnings.
