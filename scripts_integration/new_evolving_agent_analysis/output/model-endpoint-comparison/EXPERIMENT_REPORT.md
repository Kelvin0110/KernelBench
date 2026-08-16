# Cross-series inference comparison — Terra vs OSS, native baselines

Status: **2026-08-16**. Redo of the inference-only comparison for
`runs_evolving/inference_gpt_56_terra` and
`runs_evolving/inference_oss_120b`. Rules:
[ANALYSIS_RULES.md](../../ANALYSIS_RULES.md) and
[output/ANALYSIS_RULES.md](../ANALYSIS_RULES.md).

Speedup is already `torch_baseline / kernel` on the host that evaluated the
run. Terra stays on CPU4; OSS stays on CPU6. Terra is **not** rescored onto
CPU6.

## Decision summary

1. **Matched truncation cell (native relative speed):** Terra leads
   `fast_p_best@1.0` **0.82 vs 0.72**, `fast_p_best@2.0` **0.26 vs 0.24**,
   geomean **1.780 (n=44) vs 1.385 (n=41)**, and current@1 **0.70 vs 0.46**.
   Correctness is tied at **49/50**. Terra uses fewer tokens (114.9M vs
   129.2M) and more wall time (104.1 h vs 66.2 h; both resumed).
2. **Matched Markov cell:** Terra again leads speed
   (best@1 **0.82 vs 0.60**, best@2 **0.30 vs 0.12**, geomean **1.815 vs
   1.030**). OSS Markov is the only **50/50** correctness run. Token
   efficiency is similar (Terra 55.0M, OSS 51.7M) because both emit an
   evolving report.
3. **Same component trend on both models?** **No.** On OSS, truncation
   uniquely leads best@1 (0.72) and Markov is a speed laggard (0.60) with
   perfect correctness. On Terra, truncation and Markov **tie** at 0.82
   and Markov is slightly ahead on geomean and best@2. Compress-trigger
   (Terra only) loses the 1.0 bar (0.70).
4. **No universal winner.** Fast-p@1, fast-p@2, correctness, current
   retention, tokens, and wall time pick different rows.

## 1. Required checkpoints — matched cells

`@0/@1/@2` = `fast_p_best`. Geomean = `speedup_best.geometric_mean`.

### Iteration 10

| model / design | baseline | correct | @0 | @1 | @2 | geomean (n) |
|---|---|---:|---:|---:|---:|---:|
| Terra truncation | CPU4 | 49/50 | 0.960 | **0.700** | 0.200 | **1.502 (45)** |
| Terra markov_report | CPU4 | 49/50 | 0.940 | **0.740** | 0.200 | 1.444 (43) |
| Terra compress_trigger | CPU4 | 49/50 | **0.980** | 0.680 | 0.200 | 1.405 (46) |
| OSS truncation | CPU6 | 49/50 | 0.880 | 0.520 | 0.140 | 1.170 (39) |
| OSS markov_report | CPU6 | **50/50** | 0.960 | 0.460 | 0.080 | 0.889 (42) |
| OSS selective_retention | CPU6 | 48/50 | 0.920 | 0.540 | 0.080 | 1.065 (39) |
| OSS folding | CPU6 | 48/50 | 0.940 | 0.500 | 0.160 | 1.017 (45) |

### Iteration 30

| model / design | baseline | correct | @0 | @1 | @2 | geomean (n) |
|---|---|---:|---:|---:|---:|---:|
| Terra truncation | CPU4 | 49/50 | 0.980 | **0.820** | 0.260 | 1.780 (44) |
| Terra markov_report | CPU4 | 49/50 | 0.980 | **0.820** | **0.300** | **1.815 (39)** |
| Terra compress_trigger | CPU4 | 49/50 | 0.980 | 0.700 | **0.300** | 1.644 (43) |
| OSS truncation | CPU6 | 49/50 | 0.980 | 0.720 | 0.240 | 1.385 (41) |
| OSS markov_report | CPU6 | **50/50** | **1.000** | 0.600 | 0.120 | 1.030 (36) |
| OSS selective_retention | CPU6 | 48/50 | 0.960 | 0.700 | 0.180 | 1.286 (37) |
| OSS folding | CPU6 | 48/50 | 0.960 | 0.600 | 0.220 | 1.224 (38) |
| OSS truncation+deletion | CPU6 | 49/50 | 0.980 | 0.640 | 0.180 | 1.252 (35) |
| OSS truncation+merge@0.7 | CPU6 | 49/50 | 0.980 | 0.640 | 0.160 | 1.239 (35) |
| OSS truncation+refine | CPU6 | 47/50 | 0.940 | 0.620 | 0.140 | 1.233 (32) |
| OSS all-gov | CPU6 | 48/50 | 0.960 | 0.660 | 0.260 | 1.397 (36) |

Full per-series tables, reasons, and case studies:
[OSS report](../gpt-oss-120b-inf-CPU6/EXPERIMENT_REPORT.md),
[Terra report](../gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md).

## 2. Why native baselines

A speedup of 1.0 means the kernel beat **that host’s** torch reference.
Fast-p counts how often that relative bar is cleared. Recomputing Terra
kernels against the CPU6 vector would change the denominator without
changing the kernels and would distort a metric that already absorbs
hardware. Source `visualizations/` were not overwritten with a foreign
baseline. A previous `common-baseline/` rescoring directory stays deleted.

## 3. Reason, root cause, insight — truncation pair

**Reason.** Same L0 truncation, governance off, 50×30, inference endpoint.
Model and host baseline differ (`gpt-5.6-terra` / CPU4 vs `gpt-oss-120b` /
CPU6).

**Possible root cause (hypothesis).** Terra produces faster relative
kernels earlier (I10 @1 0.70 vs 0.52) and keeps searching (I30 0.82, current
0.70). OSS T0 climbs later (I10 0.52 → I30 0.72) and keeps a weaker last
attempt (current 0.46). Terra also compiles more cleanly in the extractor
taxonomy (18 compile errors vs 107). Endpoint/model quality is confounded
with architecture.

**Key insight.** On native relative speed, Terra truncation is ahead of OSS
truncation at both checkpoints on @1, @2, and geomean, with tied
correctness. The 2026-08-10 synthesis that put OSS ahead used an invalid
CPU6 rescoring of Terra.

## 4. Reason, root cause, insight — Markov pair

**Reason.** Both replace raw L0 with an evolving report (1,500 report
calls each; ~6.6–6.8M report tokens). Terra tokens 55.0M vs OSS 51.7M.

**Possible root cause (hypothesis).** Markov’s compression helps Terra
reach the 1.0 bar (tie with truncation at 0.82) but, on OSS, biases the
search toward a correct-but-slow family (best@1 0.60, geomean 1.03, debug
actions 492 vs T0 390). The OSS `level_1_problem_54` collapse (2.96× →
0.16×) and Terra `level_3_problem_24` collapse (5.73× → 0.96×) are the same
kind of “report-anchored slower path” observation.

**Key insight.** Markov is **not** a portable upgrade. It is correctness-
and retention-friendly on OSS, and headline-competitive on Terra.

## 5. Case study (cross-model, descriptive)

The extractor cannot pair Terra vs OSS in one `runs_root`. These are the
within-series extremes that repeat across models:

| Pattern | Terra (vs truncation) | OSS (vs truncation) |
|---|---|---|
| Compressed context loses a fast specialized kernel | `level_3_problem_24`: 5.729 → Markov 0.964 / compress 0.803 | `level_1_problem_54`: 2.959 → Markov 0.157 |
| Compressed/report context also finds a large win | `level_3_problem_3`: 1.221 → Markov 3.360 | `level_2_problem_51`: 1.048 → Markov 4.857 |
| T0 uniqueness on one correctness miss | sticky-hack `level_2_problem_13` | `level_1_problem_56` failed on T0 and recovered on every other complete OSS arm |

Locators are in the per-series reports. These are extractor-selected
descriptive anchors, not a causal model effect.

## 6. Component-trend verdict

| Question | OSS inference | Terra inference |
|---|---|---|
| Does truncation win best@1? | **Yes** (0.72; next is selective 0.70) | **Tie** with Markov at 0.82; compress-trigger loses (0.70) |
| Does Markov raise current@1 vs truncation? | Yes (0.50 vs 0.46) | **No** after resume (0.64 vs 0.70) |
| Does Markov raise correctness? | Yes (50 vs 49) | Tie (49 vs 49) |
| Does extra L0 compression (folding / compress-trigger) help best@1? | Folding 0.60, below T0 | Compress-trigger 0.70, below T0 |
| Does governance beat truncation on best@1? | No (0.62–0.66) | n/a (no Terra governance cell) |

The same-trend generalization is **rejected** by the completed Markov
cells, not merely inconclusive.

## 7. Limitations

- `n=1` per configuration; sequential L1; all OSS runs and two Terra runs
  resumed.
- Terra compress-trigger used `compress_hot_rounds=3`, not the hot=15
  recipe.
- Terra folding still partial.
- Native relative speed is the correct cross-host comparison; it does not
  make the two torch references identical.
- Old integrate endpoint OSS is out of this redo.

## 8. Provenance

See [MANIFEST.md](MANIFEST.md). Numbers are copied from the 2026-08-16
series aggregates, not from the 2026-08-10 reports.
