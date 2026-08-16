# Evolving-agent model, feature, and endpoint analyses

Refreshed **2026-08-16** from eight completed OSS inference runs and three
completed Terra inference runs. Standing contract:
[ANALYSIS_RULES.md](ANALYSIS_RULES.md) and the folder-level
[../ANALYSIS_RULES.md](../ANALYSIS_RULES.md).

Every series `comparison.md` and experiment report includes the required
iteration-10/30 table: `fast_p_best@0` (correctness-like coverage), `@1`,
`@2`, and `speedup_best` geomean with `n`. Cross-model speed uses each
series’ **native** baseline. Terra is not rescored onto CPU6.

## Reports

- [Analysis rules](ANALYSIS_RULES.md)
- [GPT-OSS-120B on CPU6/A6000](gpt-oss-120b-inf-CPU6/EXPERIMENT_REPORT.md)
  — truncation versus deletion, refine, merge@0.7, selective, Markov,
  folding, and combined governance. [manifest](gpt-oss-120b-inf-CPU6/MANIFEST.md),
  [comparison](gpt-oss-120b-inf-CPU6/comparison.md).
- [GPT-5.6 Terra on CPU4/A6000](gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md)
  — truncation versus Markov and compress-trigger. Folding excluded
  (15/50). [manifest](gpt-56-terra-inf-CPU4/MANIFEST.md),
  [comparison](gpt-56-terra-inf-CPU4/comparison.md).
- [Cross-model synthesis](model-endpoint-comparison/EXPERIMENT_REPORT.md)
  — native relative speed, matched truncation/Markov cells, component-trend
  verdict. [manifest](model-endpoint-comparison/MANIFEST.md).
- [Invalidated GH200 analysis](GH200x2/INVALIDATED.md) — tombstone only.

## Strong observations (2026-08-16)

- OSS truncation still leads OSS `fast_p_best@1.0` at iteration 30
  (**0.72**). Selective is closest (0.70). Markov is 50/50 correct but
  **0.60** on best@1. Combined governance leads OSS best@2 (**0.26**).
- Terra truncation and Markov **tie at best@1 = 0.82**. Compress-trigger
  (hot rounds = 3, not the hot=15 recipe) is **0.70**. All three complete
  Terra runs are 49/50 correct.
- On native baselines, Terra truncation leads OSS truncation on best@1
  (0.82 vs 0.72), best@2 (0.26 vs 0.24), geomean, and current@1 (0.70 vs
  0.46). Correctness is tied 49/50.
- The component trend is **not** the same across models: OSS truncation
  uniquely wins best@1; Terra truncation and Markov tie.

## Tentative explanations

Labeled hypotheses, not findings. See the series reports for case studies
(`level_3_problem_24` / `level_1_problem_54` as compressed-context
regressions; `level_3_problem_3` / `level_2_problem_51` as compressed-
context wins).

Most configurations have one completed run. Problems are coupled through
sequential L1. All OSS runs and two Terra runs were resumed. Sticky
best-hack state changes geomean `n`.

## Pending

Terra folding remains excluded until it produces a complete 50-problem
summary.
