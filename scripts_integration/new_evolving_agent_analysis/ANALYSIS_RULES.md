# Analysis rules for evolving-agent KernelBench runs

This file is the standing contract for
`scripts_integration/new_evolving_agent_analysis/`. Reports, `comparison.md`,
aggregates, and canvases must follow it. If a number conflicts with these
rules, the number is wrong.

## 1. What a comparison must show

Every series report, `comparison.md`, and cross-model synthesis **must** include
a table with **one row per design variant** and these columns at **iteration 10
and iteration 30**:

| required column | metric |
|---|---|
| correctness / `fast_p_best@0` | fraction of all aligned problems whose running-best speedup is ≥ 0 |
| `fast_p_best@1` | same denominator, threshold 1.0 |
| `fast_p_best@2` | same denominator, threshold 2.0 |
| `speedup_best` geometric mean | correct, non-hack samples only; always print `n` next to it |

`compare_runs.py` emits this as **Required checkpoints: iterations 10 and 30**.
Do not replace it with a geomean-only ranking or a single final-iteration row.

Also report, in the same document:

- **reason** — what the design actually changed (L0 mode and/or governance flags);
- **possible root cause** — mechanism hypotheses grounded in artifacts, labeled
  as hypotheses;
- **key insight** — the smallest claim the numbers actually support;
- **case study** — deterministic extractor candidates plus a short artifact
  audit (`metrics_by_iteration.jsonl`, `chat_history.jsonl`,
  `iteration_snapshots.jsonl`, `run_finished.json`). Case studies are
  descriptive, not causal estimates.

## 2. Native baselines — do not rescore across hosts

Speedup is already `torch_baseline / kernel` on the host that evaluated the
run. Fast-p counts how often that **relative** bar is cleared. That relative
measure already absorbs hardware/baseline differences.

| series | runs root | native baseline file |
|---|---|---|
| GPT-OSS-120B inference | `runs_evolving/inference_oss_120b/` | `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json` |
| GPT-5.6 Terra inference | `runs_evolving/inference_gpt_56_terra/` | `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json` |

Rules:

- Keep one baseline file **within** a series.
- Across hosts/models, compare native relative speed/fast-p.
- **Do not** recompute Terra kernels onto the CPU6 torch vector (or OSS onto
  CPU4) to “harmonize” speedup. That changes the denominator without changing
  the kernels and distorts a metric that is already relative.
- **Do not** overwrite a run’s `visualizations/performance_stats.json` with a
  foreign baseline. Side copies under a comparison folder are also forbidden
  for cross-model ranking.
- `compare_runs.py` must be given the **same** `--runs-root` and
  `--baseline-file` that built that folder’s `aggregate_runs.json`. A mismatch
  silently rebuilds against the wrong root/baseline.

“CPU4” / “CPU6” name the *timing-folder* (`SONG_CPU4_A6000x2`,
`SONG_CPU6_A6000x4`). The measured accelerator in both is an RTX A6000.

## 3. Completion and inclusion

A run is **complete** only when all of these hold:

1. `run_summary.json` exists;
2. `total_completed >= total_attempted > 0`;
3. every workspace under `workspaces/` has `run_finished.json`.

Anything else is **partial**. Partial runs may appear in aggregates with a
warning; they **must not** enter headline tables, feature-evidence extraction,
or winner statements. Prefix scores from incomplete runs are not substitutes
for a 50-problem result.

Default problem set: `subset_selection/selected_problems_50.csv` (10 / 15 / 25
for levels 1 / 2 / 3). Default budget: 30 iterations.

## 4. Metric semantics

- **Headline ranking metric:** `fast_p_best@1.0` on the full aligned-problem
  denominator (50 when the subset is complete). State correctness and
  `fast_p_best@0/2` beside it. There is no single metric-independent “best
  component.”
- **`fast_p_best@p`** uses running-best runtime and the full denominator.
  Failures count against the score. It does **not** apply the sticky
  `metrics_best.is_hack` exclusion.
- **`fast_p_current@p`** is the last observed submission, not the running best.
  A large best–current gap is a retention fact, not a scoring bug.
- **`speedup_best` / `speedup_current` mean, median, geomean** include only
  correct, non-hack samples. Always print `n`. Geomean deltas across runs
  compare different self-selected subsets.
- **`metrics_best.is_hack` is sticky** inside a problem: after any hack-flagged
  iteration it can remain true even if a later clean kernel is the retained
  best. That drops the whole problem from geomean while `fast_p_best` can
  still count it.
- **`best_speedup_overall`** in `run_summary.json` is not a literal maximum; it
  is the selected non-outlier best-runtime summary.
- Tokens are endpoint-reported and may be lower bounds when usage fields are
  missing. Wall time is operational (resumes, contention, endpoint latency),
  not a treatment effect.
- Error categories from `analyze_feature_evidence.py` are heuristic.

## 5. Design variants and aliases

Name a variant from **context-management mode plus active governance flags**,
not from the raw directory timestamp:

| example label | meaning |
|---|---|
| `truncation` | L0 truncation; deletion/merge/refine off (control) |
| `markov_report` | evolving report replaces raw history |
| `folding` | L0 folding |
| `selective_retention` | keep a recent L0 window |
| `compress_trigger` | microcompact old L0 rounds on a token/iteration trigger |
| `truncation+deletion` | truncation plus skill deletion |
| `truncation+merge@0.7` | truncation plus merging at similarity 0.7 |
| `truncation+refine` | truncation plus skill refinement |
| `truncation+deletion+merge@0.7+refine` | combined governance on truncation |

Within a model/endpoint series, treat truncation with all governance off as
the control when it exists. Old integrate OSS has **no** clean truncation
control; do not invent one.

## 6. Interpretation boundaries (always state)

- **`n=1` per configuration.** Numbers are descriptive of that run, not a
  repeatable treatment effect.
- Problems in one run are coupled through sequential shared L1 memory.
- Resumed campaigns mix sessions, retained L1, and endpoint state.
- Remote serving (timeouts, 404s, budget-exceeded) is not a controlled factor.
- Matched case studies are deterministic selections from
  `analyze_feature_evidence.py`. They bound cherry-picking; they do not prove
  a mode caused the delta.

Label claims as **observation**, **hypothesis**, or **exclusion**. Do not
upgrade a hypothesis to a finding.

## 7. Hard exclusions

- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/` are
  **invalidated** (missing NVCC/`CUDA_HOME`, learned PyTorch fallback). Quote
  no kernel-quality number from them.
- Do not pool OSS and Terra into one speed ranking that mixes native
  baselines into a single “common baseline” rescoring.
- Do not use partial prefixes, best-geomean-only leaderboards, or post-hoc
  single-problem anecdotes as the final comparative result.

## 8. Commands (native baselines)

OSS inference, CPU6:

```bash
uv run python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6

uv run python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/inference_oss_120b \
  --baseline-file results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6 \
  --baseline-run base_agent_gpt_oss_120b_itr30_2026_08_02_17_58 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-oss-120b-inf-CPU6/comparison.md
```

Terra inference, CPU4:

```bash
uv run python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --runs-root runs_evolving/inference_gpt_56_terra \
  --baseline-file results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --fast-p-values 0.0,0.5,0.8,1.0,1.5,2.0 \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4

uv run python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --runs-root runs_evolving/inference_gpt_56_terra \
  --baseline-file results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4 \
  --baseline-run base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40 \
  --iteration-stride 5 \
  --output scripts_integration/new_evolving_agent_analysis/output/gpt-56-terra-inf-CPU4/comparison.md
```

Feature evidence is read-only on run dirs and rejects partials. Pass only
complete runs and a within-series `--baseline-run`.

## 9. Report structure

Each `output/<series>/EXPERIMENT_REPORT.md` should contain, in order:

1. Decision / executive summary (per-metric leaders, not one universal winner).
2. The required iteration-10/30 table for every **complete** design variant.
3. Design, aliases, held-constant settings, native baseline path.
4. Validity, completion, exclusions.
5. Metric semantics (or a pointer here).
6. Reason for each variant (what changed).
7. Observations at iter 10 vs 30, plus current-vs-best if it changes the story.
8. Possible root causes / mechanisms (hypotheses).
9. Key insights.
10. Deterministic case studies with artifact locators.
11. Limitations and provenance (`MANIFEST.md`).

The cross-series file `output/model-endpoint-comparison/EXPERIMENT_REPORT.md`
uses native relative scores only and repeats the same checkpoint table for
the matched cells (at least truncation, and Markov when both models have a
complete cell).
