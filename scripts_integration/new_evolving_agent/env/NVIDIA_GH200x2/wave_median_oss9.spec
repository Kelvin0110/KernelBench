# GPU 0 -- gpt-oss-120b, all 9 variants, median-baseline series (2026-08-22).
#
#   HARDWARE=NVIDIA_GH200x2_median MAX_ARMS_PER_GPU=9 \
#   RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
#     bash scripts_integration/new_evolving_agent/env/launch_wave.sh 0 \
#          scripts_integration/new_evolving_agent/env/wave_median_oss9.spec
#
# Full re-run under the corrected metric. Every pre-2026-08-22 speedup is void:
# eval_runner timed the candidate over 10 perf trials against a 100-trial
# baseline (submodule 7ba78c7 -- an identity kernel scored 0.12x, not 1.00x),
# and candidate median was divided by baseline mean (6a3e972). Both axes are
# therefore re-measured from scratch rather than compared against old arms.
#
# Merge uses similarity 0.8 (the code default), tagged merge_sim08 to line up
# with the existing merge_sim08 reps. CLAUDE.md open item 1 prefers 0.85; 0.8 is
# the deliberate choice here.
#
# tag          | context-mode         | extra flags
-              | truncation           |
folding        | folding              |
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
refinement     | truncation           | --enable-skill-refinement
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
l2             | truncation           | --enable-l2
