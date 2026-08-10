# Inference feature analyses

Generated 2026-08-10 from seven completed inference runs. The reports use
hardware-matched A6000 timing baselines and exclude all partial runs and the
invalidated GH200 series.

## Reports

- [GPT-OSS-120B on CPU6/A6000](gpt-oss-120b-inf-CPU6/EXPERIMENT_REPORT.md)
  — truncation control versus deletion-only, refinement-only, merge-only, and
  selective retention. Exact inputs and commands are in its
  [manifest](gpt-oss-120b-inf-CPU6/MANIFEST.md).
- [GPT-5.6 Terra on CPU4/A6000](gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md)
  — truncation versus Markov report. Exact inputs and commands are in its
  [manifest](gpt-56-terra-inf-CPU4/MANIFEST.md).
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

Every configuration has one completed run, problems are coupled through
sequential L1 memory, launch dates and endpoint failures differ, and best-hack
state is sticky. OSS and Terra numbers must not be pooled because they use
different models and timing baselines.

## Pending

OSS Markov, OSS folding, OSS combined deletion+merge+refinement, and Terra
folding remain excluded until they produce complete 50-problem summaries.
