# Evolving-agent experiment report — context management and skill governance

Status as of **2026-08-05 01:45 UTC**. Every number in §2 was produced by running the
commands quoted next to it; unfinished cells are marked **pending** and carry no
numbers. Metric definitions and caveats live in
[README.md](README.md) — read the "Metric semantics" section before quoting anything
from here.

Companion run books:
[RUN_WITH_UV_INFER.md](../new_evolving_agent/infer_api/RUN_WITH_UV_INFER.md) (context
arm), [RUN_SKILL_GOVERNANCE.md](../new_evolving_agent/infer_api/RUN_SKILL_GOVERNANCE.md)
(governance arm), [RUN_WITH_UV_CONTEXT.md](../new_evolving_agent/RUN_WITH_UV_CONTEXT.md)
(mode semantics).

---

## 1. Experiment design

### 1.1 Held constant

| Setting | Value | Source |
|---|---|---|
| Problem set | `subset_selection/selected_problems_50.csv` — 50 problems (L1: 10, L2: 15, L3: 25) | `config.subset_csv`, `outcomes.per_level_summary` |
| Iterations per problem | 30 (`--max-iterations 30`) | `config`/`max_iterations` |
| Model (all four roles: coder, summarizer, extractor, action selector) | `gpt-oss-120b` | `config.coder_model` etc. |
| Endpoint | `inference` (`--nvidia-endpoint inference`) | `config.nvidia_endpoint` |
| Hardware / baseline | `NVIDIA GH200 144G HBM3e`, scored against `results/timing/NVIDIA_GH200x2/baseline_time_torch.json` | `hardware`, `baseline_file` |
| Backend / precision | `cuda` / `fp32` (defaults) | batch CLI defaults |
| Static check | on (`enable_static_check: true`) | `config.enable_static_check` |
| L1 memory | on (shared `shared_l1.jsonl` per run) | `--no-l1` not passed |

One process per GPU; each run is pinned with `CUDA_VISIBLE_DEVICES`.

### 1.2 Two arms, one shared control

Both arms are one-factor-at-a-time deviations from the same control run,
`base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51`: `truncation` context
management with all three skill-governance features off.

**Arm A — L0 context management.** Varies `--context-management`; governance stays off
(`--no-skill-deletion`, merging and refinement omitted).

| # | Cell | `--context-management` | Run name | Status |
|---|---|---|---|---|
| A0 | **control** | `truncation` (default, flag omitted) | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51` | **done** |
| A1 | markov report | `markov_report` (+`--evolving-report-max-tokens 65536`) | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52` | **done** |
| A2 | selective retention | `selective_retention` | `base_agent_gpt_oss_120b_selective_itr30_GH200_2026_08_04_17_24` | **in flight** |
| A3 | folding | `folding` | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_04_17_26` | **in flight** |

**Arm B — L1 skill governance.** Holds context management at `truncation` — the control's
setting — and varies only the three governance flags. That is the whole point of pinning
the mode: every B cell differs from A0 by governance flags alone, so B-vs-A0 deltas are
attributable to governance and B cells are mutually comparable. A B cell run under
`markov_report` would confound the two factors and could only be compared to A1.

| # | Cell | `--enable-skill-refinement` | `--skill-deletion` | `--skill-merging` | Run name | Status |
|---|---|:--:|:--:|:--:|---|---|
| A0 | *(same control)* | ✗ | ✗ | ✗ | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51` | **done** |
| B1 | refinement only | ✓ | ✗ | ✗ | `base_agent_gpt_oss_120b_refine_itr30_GH200` | **pending** |
| B2 | merge only | ✗ | ✗ | ✓ | `base_agent_gpt_oss_120b_merge_itr30_GH200` | **pending** |
| B3 | deletion only | ✗ | ✓ | ✗ | `base_agent_gpt_oss_120b_delete_itr30_GH200` | **pending** |
| B4 | refine + merge | ✓ | ✗ | ✓ | `base_agent_gpt_oss_120b_refine_merge_itr30_GH200` | **pending** |
| B5 | refine + deletion | ✓ | ✓ | ✗ | `base_agent_gpt_oss_120b_refine_delete_itr30_GH200` | **pending** |
| B6 | deletion + merge | ✗ | ✓ | ✓ | `base_agent_gpt_oss_120b_delete_merge_itr30_GH200` | **pending** |
| B7 | all three | ✓ | ✓ | ✓ | `base_agent_gpt_oss_120b_refine_delete_merge_itr30_GH200` | **pending** |

Arm B is a full 2×2×2 factorial minus the (✗,✗,✗) corner, which A0 already supplies —
so main effects and two-way interactions are estimable from eight runs total, at one
replicate each (see §3.3 on what that does *not* buy).

Launch commands, per-cell verification, and scheduling are in
[RUN_SKILL_GOVERNANCE.md](../new_evolving_agent/infer_api/RUN_SKILL_GOVERNANCE.md).
Two mechanics worth restating because they can silently void a cell:

- `--skill-merging`'s help text claims it requires `--skill-deletion`. It does not: the
  runtime gate in `Self-Evolving-Agent/evolving_common/governor/gen3_stages.py:795` is
  `enable_skill_governance = enable_skill_deletion or enable_skill_merging`, and the
  merge pass at `gen3_stages.py:1034` only checks `enable_skill_merging`. B2 and B4 are
  real configurations.
- The merge pass embeds skills through `integrate.api.nvidia.com` regardless of
  `--nvidia-endpoint`, and `_maybe_run_skill_merge` swallows exceptions. A missing
  `NVIDIA_API_KEY` degrades B2/B4/B6/B7 to the control without failing the run. Grep the
  log for `skill merge skipped`.
- `--skill-merge-similarity` currently defaults to `DEFAULT_SKILL_MERGE_SIMILARITY = 0.7`
  (`Self-Evolving-Agent/evolving_common/memory_manager.py:54`); the argparse help string
  still says `0.9`, and the two finished runs recorded `skill_merge_similarity: 0.9` from
  an older submodule revision. Inert for A0/A1 (merging off), but B cells must record what
  they actually used — check `run_summary.json`, not the help text.

### 1.3 Out of matrix

`base_agent_gpt_56_terra_truncation_itr10_2026_07_31_17_43` (model `gpt-5.6-terra`,
10 iterations) is present under `runs_evolving/` and is picked up by the tooling. It
differs from the control in both model and iteration budget and is **not** part of either
arm; it appears in the full-inventory table in §5.1 only for completeness.

---

## 2. Results so far — control vs `markov_report`

Both runs are `complete` (50/50 problems finished, `run_summary.json` present).

Command that produced everything in §2.1–§2.4 (stdout is the markdown; stderr carries the
`[compare] markdown=...` line):

```bash
cd /localhome/local-tianzheng/KernelBench
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

`aggregate_runs.py` reported `discovered=2 aggregated=2 complete=2 partial=0`; both runs
used cached `performance_stats.json` (`stats=cached`). Artifacts:
`output/aggregate_runs.json`, `output/aggregate_runs.csv`, `output/comparison.md`.

### 2.1 Correctness, wall clock, hack counters

| Metric | A0 truncation | A1 markov_report | delta | source |
|---|---|---|---|---|
| `total_correct` / 50 | **49** | **48** | −1 (−2.0%) | `outcomes.total_correct` |
| `correct_rate` | 0.980 | 0.960 | −0.020 | `outcomes.correct_rate` |
| correct by level (L1/L2/L3) | 10/15/24 | 10/15/23 | −1 at L3 | `outcomes.per_level_summary` |
| `best_speedup_overall` | **6.0663** | **4.9410** | −1.1253 (−18.5%) | `outcomes.best_speedup_overall` |
| `best_runtime_overall` (ms) | 0.483 | 0.593 | +0.110 | `outcomes.best_runtime_overall` |
| `suspicious_speedup_count` | **0** | **0** | 0 | `outcomes.suspicious_speedup_count` |
| `hack_iteration_count` | **130** | **72** | −58 (−44.6%) | `performance.hack_iteration_count` |
| `problems_with_hack` | **47** / 50 | **36** / 50 | −11 (−23.4%) | `performance.problems_with_hack` |
| `total_wall_time_hours` | **34.273** | **31.888** | −2.385 (−7.0%) | `timing.total_wall_time_hours` |
| `avg_wall_time_min` per problem | 41.13 | 38.27 | −2.86 (−7.0%) | `timing.avg_wall_time_min` |
| batch window (UTC) | 2026-08-03 04:52 → 2026-08-04 15:08 | 2026-08-03 04:52 → 2026-08-04 12:45 | — | `timing.batch_*_at_utc` |
| `l1_entry_count` | 585 | 426 | −159 (−27.2%) | `governance.l1_entry_count` |
| governance flags (deletion / merging / refinement) | no / no / no | no / no / no | — | `config` |

`hack_iteration_count` counts iterations where `is_hack` was set; `problems_with_hack`
counts problems with at least one such iteration. `suspicious_speedup_count` is a
different, post-hoc audit (final best speedup above 10×/50×) and is 0 for both runs — the
`is_hack` flags here come from STRICT static-check categories, not from speedup outliers
(§4.1).

### 2.2 Final-iteration speedup aggregates

Correct, non-hack samples only. `n` is how many of the 50 problems entered each aggregate.

| Metric | A0 truncation | A1 markov_report | delta |
|---|---|---|---|
| `speedup_best.mean` | 1.0903 | 1.2315 | +0.1413 (+13.0%) |
| `speedup_best.median` | 1.1201 | 0.9476 | −0.1725 (−15.4%) |
| `speedup_best.geometric_mean` | 1.0767 | 0.7898 | −0.2868 (−26.6%) |
| `speedup_best.n` | **3** / 50 | **14** / 50 | +11 |
| `speedup_current.mean` | 2.0762 | 1.0182 | −1.0580 |
| `speedup_current.median` | 0.9948 | 0.5658 | −0.4290 |
| `speedup_current.geometric_mean` | 1.1551 | 0.5928 | −0.5623 (−48.7%) |
| `speedup_current.n` | **29** / 50 | **38** / 50 | +9 |

The `best_n` column is the headline caveat, not a footnote. A0's `best` aggregates rest on
3 problems, A1's on 14. Verified from the raw `metrics_by_iteration.jsonl`: final
`metrics_best.is_hack` is true for 47 problems in A0 and 36 in A1, and `50 − 47 = 3`,
`50 − 36 = 14` reproduce `best_n` exactly.

**The cause is a sticky flag, not a verdict on the best kernel.** `metrics_best.is_hack`
is not computed against the best kernel at all. In
`Self-Evolving-Agent/kernelbench_integration/governor.py` it is assigned the run-level
accumulator `run_had_hack`, which latches on the first hack-flagged iteration and never
resets:

```python
if eval_result.is_hack:
    run_had_hack = True          # never set back to False
...
metrics_best = {..., "is_hack": run_had_hack}
```

The latch is directly visible in the per-iteration traces — `H` marks a hack flag:

```
level_1_problem_100
  iter is_hack: ........H....HH........H......
  best is_hack: ........HHHHHHHHHHHHHHHHHHHHHH   <- latched at 9, never clears
```

Then `generate_run_performance_stats.py:294` does
`best_correct_flags.append(best_correct and not best_is_hack)`, so one hack anywhere in 30
iterations removes the problem from the best-speedup aggregate permanently.

This fully accounts for the observed numbers, with no appeal to detector false positives:

| | per-iteration hack rate | P(≥1 hack in 30 iters), i.i.d. | predicted excluded | observed excluded |
|---|---|---|---|---|
| A0 truncation | 130/1500 = 8.67% | 93% | 47/50 | **47/50** |
| A1 markov | 72/1500 = 4.80% | 77% | 39/50 | 36/50 |

A0 matches the binomial prediction exactly; A1 falls slightly below it because hack
iterations cluster within problems rather than arriving independently. Decisively:
**42 of A0's 47 excluded problems had a clean final iteration**, and 35 of A1's 36 did.
The exclusion is therefore uncorrelated with whether the kernel being scored is actually a
hack — it is an artifact of the accumulator.

Two consequences:

- `speedup_best.*` is unusable for cross-arm comparison at 30 iterations. The subsets are
  small, differently sized, and selected by a mechanism unrelated to the quantity being
  averaged. The disagreeing signs across mean (+13.0%), median (−15.4%) and geomean
  (−26.6%) on the same pair are the symptom.
- The **per-iteration** hack rate (8.67% vs 4.80%) *is* a sound comparable signal, because
  `metrics_iteration.is_hack` is per-iteration and does not latch. It is reported in §4.1.

`fast_p_best` is unaffected: it reads `best_correct` from the points dict, which does not
have `and not best_is_hack` applied. Together with `speedup_current` (which uses the
non-latching per-iteration flag), those are the metrics to compare arms on.

### 2.3 fast-p (all thresholds)

`fast_p` keeps the full 50-problem denominator, so failures are penalized. `fast_p_best`
does **not** drop hack-flagged bests.

| threshold p | A0 `fast_p_best` | A1 `fast_p_best` | delta | A0 `fast_p_current` | A1 `fast_p_current` | delta |
|---|---|---|---|---|---|---|
| 0.0 | 0.98 | 0.96 | −0.02 | 0.58 | 0.76 | +0.18 |
| 0.5 | 0.74 | 0.60 | −0.14 | 0.46 | 0.42 | −0.04 |
| 0.8 | 0.66 | 0.40 | −0.26 | 0.36 | 0.32 | −0.04 |
| **1.0** | **0.54** | **0.32** | **−0.22 (−40.7%)** | 0.28 | 0.26 | −0.02 |
| 1.5 | 0.38 | 0.12 | −0.26 | 0.26 | 0.12 | −0.14 |
| 2.0 | 0.26 | 0.10 | −0.16 | 0.20 | 0.10 | −0.10 |

At p=1.0 the difference is 27 vs 16 problems out of 50.

### 2.4 Trajectory (matched iterations, stride 5)

| iteration | A0 `best_geomean` | A1 `best_geomean` | delta | A0 `fast_p_best@1.0` | A1 `fast_p_best@1.0` | delta |
|---|---|---|---|---|---|---|
| 1 | 0.5261 | 1.0508 | +0.5247 | 0.000 | 0.020 | +0.020 |
| 5 | 0.3889 | 0.5094 | +0.1205 | 0.100 | 0.140 | +0.040 |
| 10 | 0.5313 | 0.4966 | −0.0347 | 0.180 | 0.160 | −0.020 |
| 15 | 0.4248 | 0.5909 | +0.1661 | 0.300 | 0.220 | −0.080 |
| 20 | 0.4309 | 0.6425 | +0.2117 | 0.420 | 0.280 | −0.140 |
| 25 | 0.5858 | 0.7127 | +0.1269 | 0.540 | 0.280 | −0.260 |
| 30 | 1.0767 | 0.7898 | −0.2868 | 0.540 | 0.320 | −0.220 |

`fast_p_best@1.0` separates monotonically from iteration ~15 onward. The `best_geomean`
column does not track it and is non-monotonic in both runs — again a `best_n` artifact:
the denominator of that geomean changes from iteration to iteration as bests get flagged
and unflagged. A0's jump from 0.5858 at iteration 25 to 1.0767 at 30 is a change in *which
three problems* survive the filter, not a 84% performance gain.

---

## 3. Reading of the results

### 3.1 What the comparison supports

- **`markov_report` is not a drop-in improvement over `truncation` on this workload.**
  Its `fast_p_best` is lower at every threshold ≥0.5, and the gap widens with iteration
  count. Nothing in the data suggests markov beats truncation on solution quality here.
- **Correctness is essentially unchanged** (49 vs 48 of 50; a single L3 problem). One
  problem is not a signal at n=1 per arm.
- **`markov_report` produced measurably fewer hack-flagged iterations** — 72 vs 130 over
  1,500 iterations, i.e. **4.80% vs 8.67%**. This is the largest relative difference in the
  run (−44.6%) and, being an unfiltered count over every iteration rather than a subset
  aggregate, it is the most robust of the observed differences. Quote the per-iteration
  rate, not the problem-level counts (36 vs 47): those are the same rate pushed through a
  sticky accumulator and carry no additional information (§2.2). It is *not* obviously good
  news — §3.2 explains how it partly re-explains the fast-p gap.
- **Wall clock is 7% lower** for markov (31.89 h vs 34.27 h). See §4.5 before treating
  that as a property of the mode.
- **L1 grew far less** under markov (426 vs 585 entries) with skill promotion policy
  identical. Consistent with markov's shorter prompts producing fewer promotable rounds.

### 3.2 What it does NOT support

- **It does not support a claim about `speedup_best.geometric_mean`.** A0's 1.0767 is over
  n=3 and A1's 0.7898 is over n=14. Those are not the same estimand. The −26.6% delta the
  tool prints is arithmetically correct and semantically close to meaningless; the
  disagreeing mean (+13.0%) and median (−15.4%) on the same pair of samples make that
  concrete.
- **It does not isolate a mechanism.** The `n=14 vs n=3` gap is not itself a mechanism —
  it is the sticky-flag artifact of §2.2 and carries no information about kernel quality.
  A real entanglement remains, though: the recomputed running best excludes hack-flagged
  iterations (`generate_run_performance_stats.py:189`), so a higher hack rate shrinks the
  pool of kernels eligible to *be* the best. A0 has the higher hack rate (8.67% vs 4.80%)
  and therefore the smaller eligible pool, yet still posts the higher `fast_p_best` at
  every threshold. That direction argues against the deflationary reading — A0 is not
  winning by being allowed to keep fast cheats, since those are excluded — but this data
  cannot fully separate "markov writes slower kernels" from "markov writes fewer
  static-check-violating kernels and therefore retains slower honest ones".
- **It does not support any claim about the two in-flight modes.** A2/A3 are ~8 of 50
  problems in; their numbers appear in §5.1 with a partial banner and must not be compared
  to the finished runs.
- **It does not say anything about skill governance.** Arm B has not started.

### 3.3 On n=1 per arm

There is **one run per configuration**. No repeat of any cell exists, so there is **no
estimate of run-to-run variance**, and consequently no threshold above which a difference
between two cells is distinguishable from noise. Everything in §3.1 is a description of
two specific runs, not an inference about the configurations.

Concretely, at 50 problems the binomial standard error on a fast-p estimate near 0.5 is
about `sqrt(0.25/50) ≈ 0.071`, so a 95% interval on A0's `fast_p_best@1.0 = 0.54` spans
roughly 0.40–0.68 from problem sampling alone — before adding LLM sampling
nondeterminism, timing jitter, and the sequential-L1 coupling described in §4.4. The
observed −0.22 gap is larger than that single-run interval, which is why §3.1 states the
direction; a 0.02 correctness difference or a 7% wall-clock difference is comfortably
inside it and should be read as "no detected difference".

The single most valuable next run is therefore **not** an eighth configuration but a
**repeat of A0** under an identical command. Without it, all eight Arm-B cells will be
compared to a control whose own dispersion is unknown, and the factorial design in §1.2
will yield main-effect estimates with no error term. The spare wave-4 slot in
[RUN_SKILL_GOVERNANCE.md](../new_evolving_agent/infer_api/RUN_SKILL_GOVERNANCE.md) is
reserved for exactly this.

---

## 4. Threats to validity

### 4.1 Reward hacking and the `is_hack` flag

`is_hack` is set by `resolve_is_hack`
(`Self-Evolving-Agent/kernelbench_integration/static_check.py:44`) when a STRICT
static-check error fires or `metadata.excessive_speedup` is true. Counted over the raw
`metrics_by_iteration.jsonl` for both runs, the flagged iterations are dominated by
static-check categories — `cuda_impl`, `code_bypass`, `pytorch_wrap`,
`torch_computation_ops`, sometimes with `workload_shrink` — and neither run has a single
speedup-threshold hack (`suspicious_speedup_count = 0` at the 10×/50× audit thresholds in
`src/kernelbench/performance_stats.py:15-16`). Two consequences:

- The flag is a **static-analysis verdict, not a measured cheat**. Any false-positive rate
  in those rules feeds the hack counts directly, so treat both the per-iteration rate and
  the derived exclusions as "tripped the static rules", not "provably cheated".
- The headline **per-iteration** rates are the comparable quantity: **8.67% (130/1500) for
  A0 vs 4.80% (72/1500) for A1** — markov roughly halves the rate. This uses
  `metrics_iteration.is_hack`, which is per-iteration and does not latch.
- The **problem-level** counts (47 vs 36 `problems_with_hack`) are *not* an independent
  signal. They are the per-iteration rate pushed through 30 draws and a sticky accumulator;
  A0's 47/50 is exactly what 8.67% over 30 iterations predicts (§2.2). Do not read
  "47 of 50 problems hacked" as an alarming behavioural finding — read the 8.67%.
- `--no-enable-static-check` would change `is_hack` semantics wholesale. Both runs had it
  on; any future cell must too, or it is not comparable.

> **Fix worth making before the governance cells run.** `metrics_best.is_hack` recording
> `run_had_hack` rather than the hack status of the actual best kernel is a defect, not a
> policy choice — it silently reduces `speedup_best` to a 3-of-50 sample. Recording the
> best kernel's own flag would restore `speedup_best` as a usable metric. This report does
> not change it: the two in-flight runs and the finished pair would then be scored under
> different semantics. If it is fixed, re-score all arms from the raw
> `metrics_by_iteration.jsonl` so every cell is comparable.

### 4.2 The correct-and-non-hack-only aggregation policy

`generate_run_performance_stats.py` excludes incorrect and hack-flagged problems from
mean/median/geomean rather than scoring them 0. `speedup_*` aggregates are therefore
computed over a self-selected subset whose size varies per run and per iteration, while
`fast_p` retains the full-problem denominator. Two rules follow:

1. **Quote `fast_p` as the headline; quote `speedup_*` only with its `n`.**
2. `fast_p_best` does not filter `is_hack`, while `speedup_best` does — the two metrics
   disagree by construction. The in-flight selective run in §5.1 shows the degenerate case:
   `fast_p_best@1.0 = 0.375` with `speedup_best.geometric_mean = 0.0` at `n = 0`.

### 4.3 Baseline-timing dependence

All speedups are ratios against
`results/timing/NVIDIA_GH200x2/baseline_time_torch.json`, a single measured PyTorch
baseline. Every fast-p and geomean in this report moves if that file is regenerated. Runs
scored against a baseline from different hardware produce meaningless speedups; the
aggregator warns when a run's recorded `metadata.hardware` does not match the baseline
folder, and no such warning fired for A0/A1 (both `NVIDIA GH200 144G HBM3e`). Cross-report
comparisons are only valid against the same baseline file — record its mtime
(`2026-08-03 04:36`) alongside results.

### 4.4 Run-to-run nondeterminism

Four independent sources, none controlled:

- **LLM sampling.** No seed is pinned for the endpoint; the coder, summarizer, extractor,
  and action-selector calls all sample. Two runs of an identical command produce different
  kernels.
- **Endpoint drift.** `gpt-oss-120b` is served remotely; weights/serving config can change
  between runs days apart. A0 and A1 started 32 seconds apart, which is the best-case
  scenario here and will *not* hold for Arm B cells run a week later.
- **Timing measurement.** Kernel runtimes are wall-clock measurements on a shared node;
  small speedups near 1.0 are within measurement noise, and fast-p thresholds at 0.8/1.0
  sit exactly where that matters.
- **Sequential L1 coupling.** L1 is shared and accumulates *across* the 50 problems within
  a run, so problem 40 is solved with a memory shaped by problems 1–39. Problems within a
  run are not independent samples, which makes the binomial interval in §3.3 an
  underestimate, and it means a single early divergence can propagate through the rest of
  the run.

### 4.5 Wall-clock confounds

The 7% wall-clock difference is the least trustworthy number in §2.1:

- A0 and A1 ran **concurrently on the same node**, one per GPU (`CUDA_VISIBLE_DEVICES=0`
  and `=1`), and shared the host, PCIe/NVLink, and the compile toolchain. Contention is
  not symmetric across the two.
- Both hit the **same remote inference endpoint concurrently**, so LLM latency is a shared,
  time-varying resource and a large fraction of per-iteration wall time.
- `total_wall_time_sec` is summed per problem from `batch_timing.jsonl`
  (A0: 123,383 s over 50 rows, all `ok`), so it includes queueing against the other run.
- A1's rewriter makes an **extra LLM call per iteration** (`--evolving-report-max-tokens
  65536`) yet still finished faster — evidence that the difference is dominated by
  contention and prompt length, not by mode cost.

Treat wall clock as a scheduling datum, not a result. A clean measurement needs the runs
serialized on an otherwise idle node.

### 4.6 Reporting-pipeline caveats

- `aggregate_runs.py` and `compare_runs.py` **write** `<run>/visualizations/performance_stats.json`
  into run directories when a cache is stale or absent. `runs_evolving/` is gitignored.
- If a cached aggregate is rejected for a runs-root mismatch *and* the baseline timing file
  is missing, `compare_runs.py` degrades (null performance + warnings, exit 0) instead of
  exiting 2. Check for a `baseline file not found` warning on stderr.
- `archived/` is skipped by discovery; three archived 50-workspace runs under
  `runs_evolving/archived/` are correctly excluded from every table here.

---

## 5. Pending cells

### 5.1 In flight — A2 selective_retention, A3 folding

Both launched 2026-08-04 ~17:25 UTC, still running as of 2026-08-05 01:45 UTC (confirmed
via `pgrep -af evolve_kb_batch.py`). **These are partial and must not be read as results.**
Snapshot from:

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 --recompute --iteration-stride 10 --fast-p 2.0 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

which reported `discovered=5 aggregated=5 complete=3 partial=2` and flagged both runs with
`run_summary.json missing (run still in progress or aborted)`.

| run | status | problems seen | finished | correct | rate basis | wall h so far | min/problem | `l1_entry_count` |
|---|---|---|---|---|---|---|---|---|
| A2 `..._selective_itr30_GH200_2026_08_04_17_24` | partial | 8 | 7 | 7 | `workspaces_finished` | 7.60 | 65.1 | 94 |
| A3 `..._folding_itr30_GH200_2026_08_04_17_26` | partial | 9 | 8 | 8 | `workspaces_finished` | 7.68 | 57.6 | 122 |

`correct_rate` for these two is over *finished* problems only (7/7 and 8/8), not over 50 —
it is not comparable to A0's 49/50. `context_mgmt` reads `-` because it is sourced from the
`run_summary.json` that does not exist yet; the launched commands used
`--context-management selective_retention` and `--context-management folding` respectively
(verified in `ps` output). Final speedup, fast-p, and correctness for A2/A3: **pending**.

Projected completion at the observed per-problem rate (50 × 65.1 min ≈ 54 h for A2,
50 × 57.6 min ≈ 48 h for A3, i.e. ~2026-08-06 to 08-07) — a projection from a
7–9 problem prefix, not a measurement, and both are currently slower per problem than the
finished runs.

**Refresh once they land** (the run names above are final; no timestamp guessing needed):

```bash
cd /localhome/local-tianzheng/KernelBench
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 --recompute \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52 \
  --runs base_agent_gpt_oss_120b_selective_itr30_GH200_2026_08_04_17_24 \
  --runs base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_04_17_26 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

Then confirm `complete=4 partial=0` on stderr before copying any number into §2.

### 5.2 Pending — Arm B, all seven governance cells

**Status: not started. No numbers exist for B1–B7.** Placeholder for the results table;
every cell is `pending` until the corresponding run reaches `status=complete`.

| # | Cell | `total_correct`/50 | `fast_p_best@1.0` | `speedup_best.geomean` (`n`) | `hack_iteration_count` | `l1_entry_count` | merges | deletions | refinements | wall h |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | control | 49 | 0.54 | 1.0767 (3) | 130 | 585 | 0 | 0 | 0 | 34.27 |
| B1 | refinement only | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B2 | merge only | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B3 | deletion only | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B4 | refine + merge | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B5 | refine + deletion | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B6 | deletion + merge | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B7 | all three | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| — | A0 repeat (variance) | pending | pending | pending | pending | pending | pending | pending | pending | pending |

The governance columns (`merges` / `deletions` / `refinements`) are 0 for A0 because its
sidecars are absent; for B cells a 0 there means either "the feature ran and did nothing"
or "the feature silently no-opped". Distinguish them with the per-cell verification block
in [RUN_SKILL_GOVERNANCE.md](../new_evolving_agent/infer_api/RUN_SKILL_GOVERNANCE.md)
within the first hour, not at the 34-hour mark.

**Refresh once Arm B lands** — substitute the real timestamped run names:

```bash
cd /localhome/local-tianzheng/KernelBench
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 --recompute \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_refine_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_merge_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_delete_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_refine_merge_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_refine_delete_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_delete_merge_itr30_GH200_<ts> \
  --runs base_agent_gpt_oss_120b_refine_delete_merge_itr30_GH200_<ts> \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

---

## 6. How to reproduce / refresh

Run from the repo root with the repo venv. Bare `python` will not resolve the
`kernelbench` package.

```bash
cd /localhome/local-tianzheng/KernelBench
```

**1. Regenerate §2 exactly** (aggregate then compare, truncation as baseline):

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 \
  --runs base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51 \
  --runs base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

**2. Inventory every discovered run** (in-flight runs come back `status=partial`;
`archived/` is skipped):

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2
```

**3. Full cross-run comparison** (recomputes the aggregate, samples every 10 iterations,
fast-p at 2.0) — this is what produced §5.1:

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 --recompute --iteration-stride 10 --fast-p 2.0 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51
```

**4. Force a stats rebuild against an explicit baseline, into a scratch dir** (use when a
run's `visualizations/performance_stats.json` is suspect; leaves `output/` untouched):

```bash
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --baseline-file results/timing/NVIDIA_GH200x2/baseline_time_torch.json \
  --regenerate-stats --output-dir /tmp/kb_analysis
```

**5. Check what is still running** before drawing conclusions from a `partial` row:

```bash
pgrep -af "evolve_kb_batch.py --run-name"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

### Outputs

| Path | Contents |
|---|---|
| `scripts_integration/new_evolving_agent_analysis/output/aggregate_runs.json` | Nested per-run record + per-iteration `series` + run-set metadata |
| `scripts_integration/new_evolving_agent_analysis/output/aggregate_runs.csv` | One flat row per run, 63 columns incl. one per fast-p threshold |
| `scripts_integration/new_evolving_agent_analysis/output/comparison.md` | The markdown report; also printed to stdout |

`compare_runs.py` prints the markdown to **stdout**; the `[compare] markdown=...` line
goes to **stderr**, so `... 2>/dev/null` gives a clean pipe. `output/` currently holds the
two-run comparison from step 1 — re-run step 3 to regenerate the five-run view.

### Checks before quoting a number

1. `status` must be `complete`. A `partial` run's `correct_rate` is over finished
   problems, not 50 (`correct_rate_basis` says which).
2. Read `best_n` / `cur_n` next to any `speedup_*` aggregate (§2.2, §4.2).
3. Confirm `performance_stats_source`; `regenerated_stale` means a cache was rebuilt —
   fine, but the pre-rebuild numbers in any older report are wrong.
4. Confirm no hardware/baseline-mismatch warning appears in the `Notes` section.
