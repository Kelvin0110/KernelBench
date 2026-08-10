# Evolving-agent model, feature, and endpoint analyses

Generated 2026-08-10 from seven completed inference runs and ten completed
old-integrate GPT-OSS runs. The reports use explicit A6000 timing baselines,
exclude partial runs, and preserve the invalidated GH200 series only as a
tombstone. Cross-model speed uses each series' native baseline (speedup is
already relative); Terra is not rescored onto CPU6.

## Reports

- [GPT-OSS-120B on CPU6/A6000](gpt-oss-120b-inf-CPU6/EXPERIMENT_REPORT.md)
  — truncation control versus deletion-only, refinement-only, merge-only, and
  selective retention. Exact inputs and commands are in its
  [manifest](gpt-oss-120b-inf-CPU6/MANIFEST.md).
- [GPT-5.6 Terra on CPU4/A6000](gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md)
  — truncation versus Markov report. Exact inputs and commands are in its
  [manifest](gpt-56-terra-inf-CPU4/MANIFEST.md).
- [GPT-OSS-120B old integrate on CPU6/A6000](gpt-oss-120b-int-CPU6/EXPERIMENT_REPORT.md)
  — ten completed context and governance configurations, with observed leaders
  at fast-p thresholds 0, 1, and 2. Exact inputs and commands are in its
  [manifest](gpt-oss-120b-int-CPU6/MANIFEST.md).
- [Cross-model and endpoint synthesis](model-endpoint-comparison/EXPERIMENT_REPORT.md)
  — Terra versus OSS on native relative speed/fast-p, matched old/new endpoint
  cells, component-trend conclusions, and per-metric winners. Provenance is in
  its [manifest](model-endpoint-comparison/MANIFEST.md).
- [Invalidated GH200 analysis](GH200x2/INVALIDATED.md) — retained only as a
  tombstone; none of its performance numbers are reused.

## Strong observations

- On OSS-120B, truncation had the highest `fast_p_best@1.0` (0.72). The feature
  arms reached 0.60–0.62. Deletion matched truncation's 48/50 correctness while
  reducing the active L1 catalog from 549 to 31 entries.
- OSS deletion, refinement, and merging all executed and produced measurable
  catalog changes, but none improved the primary fast-p result in its single
  completed run.
- On Terra, truncation and Markov tied at `fast_p_best@1.0=0.78`. Markov ended
  with higher `fast_p_current@1.0` (0.50 versus 0.26), while truncation finished
  more problems correct (49/50 versus 47/50) with fewer calls and less recorded
  wall time.
- On native baselines, Terra truncation leads OSS truncation on best fast-p at
  1.0 (0.78 versus 0.72) and 2.0 (0.26 versus 0.20) and on correctness
  (49/50 versus 48/50); OSS leads final-current fast-p1 (0.46 versus 0.26).
  Terra remains much more token-efficient.
- The matched selective-retention and merge-only endpoint cells do not establish
  a systematic integrate-versus-inference effect. The old selective runs span
  the new run's fast-p1 score, while the old merge-only 0.7 run leads its new
  counterpart.

## Tentative explanations

Chat-history and sidecar evidence suggests mechanisms worth testing, not causal
conclusions:

- selective retention made fewer standard calls but carried substantially
  larger prompts;
- deletion and merging compressed retrieval state but may also have removed or
  blurred specialized knowledge;
- refinement added diagnosis/revision work without an aggregate final gain;
- Markov reports reduced reported prompt-token use and preserved optimization
  state in selected cases, but sometimes appeared to anchor search on slower
  strategy families.

Most configurations have one completed run, problems are coupled through
sequential L1 memory, launch dates and endpoint failures differ, and best-hack
state is sticky. Within a series keep one baseline file; across hosts compare
native relative speed/fast-p rather than recomputing one host onto another's
torch vector.

## Pending

OSS Markov, OSS folding, OSS combined deletion+merge+refinement, and Terra
folding remain excluded until they produce complete 50-problem summaries.
