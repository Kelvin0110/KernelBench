# GPU 0 -- gpt-oss-120b, 6 cells, RESUMED from the wave killed 2026-08-23.
#
#   RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ MAX_ARMS_PER_GPU=6 \
#     bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/resume_wave.sh 0 \
#          scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/wave_median_oss6_resume.spec
#
# Six of the nine cells: 3 context modes + baseline, plus the three governance
# cells that have no prior clean measurement. folding / selective_r5 /
# refinement are deferred to a second wave.
#
# Resume points at the time of writing (auto-derived from batch_timing.jsonl):
#   -            3 done -> resume @4
#   markov       3 done -> resume @4
#   compress     3 done -> resume @4
#   l2           3 done -> resume @4
#   merge_sim08  4 done -> resume @5
#   deletion     0 done -> resume @1   (killed inside problem 1; nothing to reuse)
#
# TWO PROTOCOL SEAMS INSIDE THE RESUMED PREFIX, both inherited, neither fixable
# without discarding the prefix:
#   1. problem 1 was evaluated at num_perf_trials=100, problems 2+ at 25
#      (submodule 63bfc2b landed mid-wave and reached live arms via spawn
#      re-import). The compress arm is split mid-problem: 21 evals @100, 6 @25.
#   2. the prefix ran with the eval trims OFF under 9-arm contention; the
#      suffix runs with them ON. Contention only ever deflates speedup, so the
#      prefix is systematically penalised relative to the suffix.
#
# Also note: `--skill-merging` alone is not a pure merge cell. gen3_stages.py:889
# passes skill_deletion=enable_skill_governance into run_l1_skill_selection, so
# the merge arm inherits the deletion arm's L1-catalog visibility (the deletion
# GC itself is correctly gated). Consistent with the earlier merge reps.
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
l2             | truncation           | --enable-l2
