# Output-folder analysis rules

Canonical contract:
[`../ANALYSIS_RULES.md`](../ANALYSIS_RULES.md).

This directory’s reports (`*/EXPERIMENT_REPORT.md`, `*/comparison.md`,
`model-endpoint-comparison/`) must follow that file. Short form:

1. **Required table** for every complete design variant at **iteration 10 and
   30**: `fast_p_best@0` (correctness-like coverage), `@1`, `@2`, and
   `speedup_best` geometric mean with `n`.
2. **Native baselines only.** OSS inference uses
   `SONG_CPU6_A6000x4`. Terra inference uses `SONG_CPU4_A6000x2`. Do not
   rescore Terra onto CPU6 (or OSS onto CPU4). Speedup is already relative.
3. **Partial runs** (missing summary, `completed < attempted`, or missing
   `run_finished.json`) stay out of headline tables and feature evidence.
4. **GH200 / NVCC-bug archives are void.** Do not reuse their kernel-quality
   numbers.
5. Next to the table, write **reason**, **possible root cause** (hypothesis),
   **key insight**, and a **deterministic case study**.
6. `n=1` per configuration; sequential L1 coupling; no universal winner across
   all metrics.

`compare_runs.py` emits the required table as
“Required checkpoints: iterations 10 and 30”.
