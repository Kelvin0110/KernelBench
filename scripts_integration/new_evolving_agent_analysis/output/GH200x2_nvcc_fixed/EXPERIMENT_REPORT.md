# GPT-OSS-120B on GH200x2 — design variants, post-NVCC-fix

Status: **2026-08-16**. Four completed runs under `runs_evolving/gpt-oss-120b/`,
scored against the native GH200 baseline. Rules:
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md). Provenance:
[MANIFEST.md](MANIFEST.md).

## Decision summary

- **Headline (`fast_p_best@1.0` at iteration 30): a three-way tie at 0.46** —
  truncation, markov_report, and selective_retention are indistinguishable.
  compress_trigger is behind at **0.40**. No design separates on the headline
  metric on this host.
- **Correctness (`fast_p_best@0`):** markov, selective, and compress all reach
  **48/50 (0.96)**; truncation is last at **47/50 (0.94)**.
- **High bar (`fast_p_best@2.0`):** truncation **0.18**, selective 0.16, markov
  and compress 0.14. The spread is 2 problems wide.
- **Best-speedup geomean at iteration 30:** markov **0.9830 (n=48)**, selective
  **0.9541 (n=48)**, truncation **0.8746 (n=47)**, compress **0.7265 (n=48)**.
- **Every complete design has a geomean below 1.0.** The median retained kernel
  on GH200 is *slower* than GH200 torch. This is the defining fact of this
  series and it does not appear on either A6000 host.
- **Current retention:** markov and selective tie at **0.40**, truncation 0.28,
  compress 0.32.

## 1. Required checkpoints: iterations 10 and 30

Native baseline: `results/timing/NVIDIA_GH200x2/baseline_time_torch.json`.
`@0/@1/@2` are `fast_p_best`. Geomean is `speedup_best.geometric_mean`; `n`
equals `total_correct` (see ANALYSIS_RULES §4).

| design | correct | I10 @0 | I10 @1 | I10 @2 | I10 geomean (n) | I30 @0 | I30 @1 | I30 @2 | I30 geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (control) | 47/50 | 0.840 | **0.380** | **0.120** | 0.7352 (42) | 0.940 | **0.460** | **0.180** | 0.8746 (47) |
| markov_report | 48/50 | **0.900** | 0.340 | 0.040 | 0.7373 (45) | **0.960** | **0.460** | 0.140 | **0.9830 (48)** |
| selective_retention | 48/50 | **0.900** | 0.320 | 0.080 | **0.7657 (45)** | **0.960** | **0.460** | 0.160 | 0.9541 (48) |
| compress_trigger | 48/50 | 0.840 | 0.240 | 0.080 | 0.5624 (42) | **0.960** | 0.400 | 0.140 | 0.7265 (48) |

The same table is in [comparison.md](comparison.md). Folding (49/50) and
deletion (30/31) are omitted because they are partial.

`fast_p_best@1.0` trajectory:

| design | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | **0.08** | 0.26 | **0.38** | **0.40** | **0.42** | 0.42 | **0.46** |
| markov_report | 0.04 | **0.28** | 0.34 | 0.38 | 0.40 | 0.42 | **0.46** |
| selective_retention | 0.04 | 0.20 | 0.32 | 0.38 | **0.42** | **0.46** | **0.46** |
| compress_trigger | **0.08** | 0.20 | 0.24 | 0.28 | 0.36 | 0.38 | 0.40 |

Observation: the whole field converges to 0.46 and stops. Truncation leads
mid-run, selective catches at iteration 20–25, markov catches only at 30. The
spread between best and worst design at iteration 30 is **three problems**.

## 2. Design, aliases, held-constant settings

| Alias | Design | Exact run name |
|---|---|---|
| G0 | truncation | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58` |
| GM | markov_report | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58` |
| GS | selective_retention | `base_agent_gpt_oss_120b_selective_r5_itr30_GH200_2026_08_11_14_09` |
| GC | compress_trigger | `base_agent_gpt_oss_120b_compress_itr30_GH200_2026_08_10_15_22` |

Held constant: 50-problem subset, 30 iterations, `gpt-oss-120b`, inference
endpoint, GH200 144G HBM3e eval, native GH200 torch baseline, skill
deletion/merge/refine **off** in all four cells, per-run shared L1.

## 3. Reason for each variant

| Design | What actually changed |
|---|---|
| truncation | Control. Raw L0 history truncated to the context window. No skill GC. |
| markov_report | Bounded evolving report replaces raw history. Smallest L1 catalog (366). |
| selective_retention | Keep a recent L0 window. Largest L1 catalog (619). |
| compress_trigger | Microcompact older L0 rounds on a token/iteration trigger. L1 435. |

## 4. Possible root causes (hypotheses)

1. **The GH200 torch baseline is the binding constraint, not the context
   design.** All four geomeans sit at 0.73–0.98 and all four `fast_p_best@1.0`
   land within 0.06 of each other. Hypothesis: on Hopper, the cuBLAS/cuDNN
   paths the reference calls are strong enough that the agent's hand-written
   CUDA rarely wins, so the context-management treatment has little headroom to
   express itself. Design differences that are visible on A6000 compress into
   noise here.
2. **Hack exposure is materially higher on this host.** compress_trigger shows
   23/50 problems with a hack iteration (vs 8–17 across the whole A6000 OSS
   series). Hypothesis: when honest optimization cannot clear the bar, the
   search drifts toward shortcut kernels. This is a hypothesis — the hack
   detector is heuristic and `n=1`.
3. **compress_trigger is the one design that clearly loses.** It is behind at
   every checkpoint on @1 (0.24 at I10, 0.40 at I30) and has the worst geomean
   by a wide margin (0.7265). Hypothesis: the same over-compaction seen in the
   Terra compress cell, amplified by a host where fewer traces are recoverable.
4. **Markov's geomean lead is not a fast-p lead.** GM finishes +0.108 geomean
   over G0 but ties it at @1 (0.46) and loses at @2 (0.14 vs 0.18).
   Hypothesis: the report keeps the *median* kernel healthier while giving up
   the occasional outlier win — the same shape seen in OSS/A6000.

These are hypotheses. `n=1` per configuration prevents causal attribution.

## 5. Key insights

1. **On GH200, the design choice barely matters.** Three of four designs tie at
   the headline metric and the fourth is 0.06 behind. Any statement of the form
   "design X is best for OSS-120B" that was derived on A6000 does **not**
   transfer to this host.
2. **Sub-1.0 geomeans are the norm here.** Reporting a "speedup" of 0.87 means
   the retained best kernel is 13% *slower* than torch. Correctness stays high
   (0.94–0.96) while speed collapses — the two metrics decouple completely.
3. **Iteration 10 mis-ranks the field.** Truncation leads @1 at I10 (0.38) and
   merely ties at I30; selective is last at I10 (0.32) and ties at I30. The
   ordering at iteration 10 carries no information about iteration 30 here.
4. **Correctness and speed rank oppositely for the control.** Truncation is
   last on correctness (47/50) and first on `fast_p_best@2.0` (0.18).

## 6. Operational profile (not treatment effects)

| design | correct | wall h | avg min/problem | L1 active | hack itrs | problems w/ hack | best_overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| truncation | 47/50 | 74.2 | 88.0 | 571 | 16 | 11 | 5.45 |
| markov_report | 48/50 | 71.4 | 83.5 | **366** | **14** | 11 | 6.23 |
| selective_retention | 48/50 | 69.9 | 83.9 | 619 | 17 | 12 | **9.76** |
| compress_trigger | 48/50 | **64.2** | **77.1** | 435 | **32** | **23** | 2.96 |

Wall time includes endpoint latency and host contention.

## 7. Limitations

- `n=1` per design. No confidence interval.
- Sequential shared L1 couples problems inside a run.
- Two of six GH200 cells (folding, deletion) are still partial, so this series
  has no folding or governance comparison — unlike OSS/A6000, which has both.
- `best_speedup_overall` is a selected non-outlier summary, not a maximum.
- Native GH200 speedup only. Do **not** rescore onto an A6000 vector to compare
  hosts; compare native relative scores instead.
- The hack detector is heuristic; the elevated compress_trigger hack count is an
  observation, not a validated mechanism.
