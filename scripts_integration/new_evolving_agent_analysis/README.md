# Cross-run analysis for evolving-agent KernelBench experiments

**Standing contract:** [ANALYSIS_RULES.md](ANALYSIS_RULES.md). Every series
report, `comparison.md`, and cross-model synthesis must include the required
iteration-10/30 table (`fast_p_best@0/1/2` and `speedup_best` geomean with `n`),
use each series' **native** timing baseline, and keep reason / root-cause
hypotheses / key insight / case study next to the numbers.

Two scripts that compare whole runs under `runs_evolving/` against each other
(context-management modes, skill-governance arms, models, iteration budgets).

Run everything with the repo venv: `/localhome/local-tianzheng/KernelBench/.venv/bin/python`
(or `uv run python`). Bare `python` will not find the `kernelbench` package.

---

## `aggregate_runs.py`

Discovers every run directory under `--runs-root` that has a `workspaces/` subdir
(`archived/` is skipped) and flattens each into one record:

| group | fields |
| --- | --- |
| identity | `run_name`, `run_dir`, `status`, `timestamp`, `run_name_timestamp`, `hardware`, `max_iterations` (+ `max_iterations_source`) |
| `config` | `context_management`, `model` / `coder_model` / `summarizer_model` / `extractor_model` / `action_selector_model`, `nvidia_endpoint`, `subset_csv`, `skill_deletion`, `skill_merging`, `enable_skill_refinement`, `enable_l1_skill_unit_test_gc`, `skill_merge_similarity`, `skill_merge_interval`, `skill_refinement_max_rounds`, `evolving_report_max_tokens`, `enable_static_check`, `dry_run`, `resume` |
| `outcomes` | `total_attempted`, `total_completed`, `total_correct`, `correct_rate`, `correct_rate_basis`, `outcomes_source`, `best_speedup_overall`, `best_runtime_overall`, `suspicious_speedup_count`, `workspace_count`, `workspaces_finished`, `per_level_summary` |
| `timing` | `batch_started_at_utc`, `batch_finished_at_utc`, `batch_session_wall_time_sec`, `total_wall_time_sec` / `_hours`, `avg_wall_time_sec` / `_min`, `problems_timed_this_session`, `batch_timing_rows`, `batch_timing_status_counts` |
| `governance` | `l1_entry_count`, `l1_active_count`, `l1_superseded_count`, `deleted_count`, `refined_count`, `merge_count`, `merge_events_total` / `_rejected` / `_skipped`, `skills_absorbed_by_merge`, `deletion_event_count`, `deletion_reasons`, `merge_passes_executed`, `unit_test_runs_total`, `catalog_compression_ratio`, `governance_sidecars_present` |
| `performance` | final-iteration `speedup_current` / `speedup_best` (`mean`, `median`, `geometric_mean`, `n`), `fast_p_best`, `fast_p_current`, `iteration_count`, `problem_count`, `final_iteration`, `final_aligned_count`, `hack_iteration_count`, `problems_with_hack` |
| `series` | per-iteration `speedup.{current,best}_{mean,median,geometric_mean}` and `fast_p_best` / `fast_p_current` per threshold |

Sources: `run_summary.json`, `batch_timing.jsonl`, `eval_results.json` (hardware
string), `workspaces/*/run_finished.json`, `shared_l1.jsonl` + governance
sidecars, and `<run>/visualizations/performance_stats.json`. The stats file is
**generated in-process** by importing `build_performance_stats` from
`Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py`
(no subprocess). `--regenerate-stats` forces a rebuild.

**Cache invalidation.** A cached `performance_stats.json` is only reused when it
is newer than every run artifact it derives from (`run_summary.json`, the
`workspaces/` tree, each `metrics_by_iteration.jsonl`). Otherwise it is rebuilt
and `performance_stats_source` reports `regenerated_stale` with a warning. This
matters for in-flight runs: a cache written when the run had 1 finished problem
would otherwise be reported verbatim for a run that now has 9.

**Correctness without a `run_summary.json`.** In-flight runs have no summary yet.
`total_correct` then falls back to counting `metadata.best_correct` across
`workspaces/*/run_finished.json` — the same predicate `evolve_kb_batch.py` uses
for `run_summary.total_correct` — and `correct_rate_basis` switches from
`total_attempted` to `workspaces_finished` (the rate is over finished problems,
not the full batch). `outcomes_source` records which path was taken; when no
marker exists at all, `total_correct` is `null`, never `0`.

### Commands

```bash
cd /localhome/local-tianzheng/KernelBench

# just the two finished GH200 runs
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52

# every discovered run (in-flight runs are reported as status=partial)
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2

# explicit baseline path, forced stats rebuild, custom output dir
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --regenerate-stats \
  --output-dir /tmp/kb_analysis
```

Flags: `--runs-root` (default `runs_evolving/`), `--output-dir` (default
`scripts_integration/new_evolving_agent_analysis/output/`), `--hardware`
(default `NVIDIA_GH200x2`), `--baseline` (default `baseline_time_torch`),
`--baseline-file` (overrides the two previous), `--runs` (repeatable run-name
filter), `--fast-p-values`, `--regenerate-stats`.

### Outputs

- `output/aggregate_runs.json` — the nested doc above plus run-set metadata
  (`runs_root`, `baseline_file`, `fast_p_thresholds`, `discovered`,
  `aggregated`, `complete_runs`, `partial_runs`, `requested_runs_not_found`,
  `failures`) and the per-iteration `series` for each run.
- `output/aggregate_runs.csv` — one flat row per run: config flags, outcomes,
  wall clock, governance counters, final-iteration aggregates, and one column
  per threshold (`fast_p_best@1.0`, `fast_p_current@1.0`, ...).

Exit code `2` when the baseline timing file is missing; otherwise `0` even if
individual runs fail (failures are listed in `failures` and on stderr).

---

## `compare_runs.py`

Consumes `output/aggregate_runs.json` (or recomputes it in-process with
`--recompute`, whenever the cached file is absent, or whenever the cached file
was built for a different `--runs-root` / baseline file) and writes a markdown
report:

1. **Runs** — legend mapping short ids `R1`, `R2`, ... to full run names, status, mode, model.
2. **Run overview** — iterations, problems, completed/correct, correct rate, wall hours, minutes per problem.
3. **Final-iteration performance** — best/current speedup mean, median, geomean, the sample sizes `best_n` / `cur_n` that produced them, `best_speedup_overall`, hack counters, and fast-p at every threshold.
4. **Skill governance** — governance flags plus L1 catalog size, merges, deletions, refinements, sidecar count.
5. **Deltas vs `--baseline-run`** — per metric: baseline, run, absolute delta, percent delta, and a `better`/`worse` direction that accounts for metrics where lower is better (wall time, hack counts, catalog size).
6. **Per-iteration comparison** — best-speedup geomean and `fast_p_best@<--fast-p>` versus iteration, restricted to the iterations **every compared run reached** (intersection capped at the shortest run) and sampled with `--iteration-stride`; delta columns appear when `--baseline-run` is set. A final table uses `align_series_for_comparison` for the pairwise matched last iteration, so a 10-iteration run is compared to a 30-iteration run at iteration 10, not 30.
7. **Notes** — per-run warnings (missing `run_summary.json`, hardware/baseline mismatch, unavailable stats).

### Commands

```bash
cd /localhome/local-tianzheng/KernelBench

# compare the two finished GH200 runs, truncation as the baseline
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51

# every run, recomputing the aggregate first, sampled every 10 iterations
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 \
  --recompute \
  --iteration-stride 10 \
  --fast-p 2.0 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

Flags: `--aggregate`, `--recompute`, `--runs-root`, `--output-dir`,
`--hardware`, `--baseline`, `--baseline-file`, `--runs` (repeatable),
`--baseline-run`, `--fast-p` (default `1.0`), `--iteration-stride` (default
`5`), `--fast-p-values`, `--regenerate-stats`, `--output` (default
`output/comparison.md`).

### Output

`output/comparison.md`, also printed to stdout (the `[compare] markdown=...`
line goes to stderr so stdout stays pipeable). Exit code `1` when no runs are
available to compare, `2` when a recompute is required but the baseline timing
file is missing.

---

## `analyze_feature_evidence.py`

Reads explicitly selected **completed** runs and writes compact behavioral
evidence without copying full chat histories or generated code:

- `feature_evidence.json` — per-run token/phase/action counts, observed
  compile/correct/hack rates, error categories, L0/L1 and governance summaries,
  per-problem outcomes, matched baseline comparisons, deterministic case-study
  candidates, parse diagnostics, and input provenance.
- `feature_evidence.csv` — one row per run/problem for filtering and independent
  checks.

The extractor rejects partial runs before writing either output. Token totals
use endpoint-reported usage, metrics rates use only rows where the relevant
`metrics_iteration` field is present, and matched speedup requires a positive,
correct, non-hack best in both runs. Case-study candidates are descriptive
anchors, not causal estimates.

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/analyze_feature_evidence.py \
  --runs-root runs_evolving/inference_oss_120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --runs BASELINE_RUN \
  --runs FEATURE_RUN \
  --baseline-run BASELINE_RUN
```

Paths are resolved from the repository root. Optional governance sidecars that
are absent because a feature is disabled are retained as explicit warnings.

---

## Metric semantics — read this before interpreting numbers

Authoritative rules, including native-baseline policy and the required
iteration-10/30 table, live in [ANALYSIS_RULES.md](ANALYSIS_RULES.md). The
notes below are the implementation details the scripts encode.

- **Speedup aggregates and fast-p use correct, non-hack samples only.** Per the
  `generate_run_performance_stats.py` docstring: incorrect/failed problems are
  *excluded* from mean/median/geometric mean rather than scored as 0 or -1,
  while fast-p still penalizes failures through the full-problem denominator
  (`aligned_count`). Consequence: `aggregates` are not directly comparable
  across runs with different failure counts unless you also read
  `problem_count` / `total_correct`; **fast-p is the comparable headline metric.**
- `fast_p_best` uses the running best runtime and does **not** filter the
  `is_hack` flag, whereas `speedup_best` aggregates do. A run whose best kernel
  is flagged as a hack can therefore show `fast_p_best > 0` with a
  `speedup_best.geometric_mean` of `0.0`.
- **Always read `speedup_*.n` (`best_n` / `cur_n` in the report) next to the
  aggregate.** Because hack-flagged bests are dropped, the headline
  `speedup_best.geometric_mean` can rest on a handful of problems: on
  `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51` the final-iteration
  geomean of `1.0767` is computed over `n=3` of `50` problems (47 have
  hack-flagged bests), while `fast_p_best@1.0 = 0.54` counts 27 of the 50. Deltas
  between two runs' `best_geomean` are comparisons of two different, small,
  self-selected subsets.
- `best_speedup_overall` from `run_summary.json` is *not* a maximum: it is the
  speedup of the problem with the minimum non-outlier runtime after excluding
  likely reward hacks.
- `status` is `complete` only when `run_summary.json` exists, `total_completed >=
  total_attempted > 0`, and every workspace has `run_finished.json`; anything
  else is `partial`. Partial runs are still aggregated and flagged with a
  warning banner in the comparison report.
- Governance counters are all `0` when the corresponding sidecars
  (`l1_skill_merges.jsonl`, `l1_skill_deletions.jsonl`, `l1_skill_usage.json`,
  ...) are absent — which is the case for every run recorded so far, since all
  of them ran with `skill_deletion` / `skill_merging` /
  `enable_skill_refinement` set to `false`. `l1_entry_count` comes from
  `shared_l1.jsonl` and is always populated.
- `--hardware` selects the baseline timing file used for every speedup **within
  a series**. Scoring a run against a baseline measured on different hardware
  produces meaningless *absolute* runtimes; speedup itself is already relative
  (`torch_baseline / kernel`). Cross-model comparisons therefore keep each
  series' native baseline rather than rescoring one host onto another. See
  [ANALYSIS_RULES.md](ANALYSIS_RULES.md). The aggregate still emits a warning
  when the run's recorded `metadata.hardware` does not appear in the baseline
  folder name.
