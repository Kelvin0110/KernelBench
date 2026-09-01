# GPU 1 -- qwen3.6-27b, 12 arms: the 9 standard cells + a 4-point merge-similarity
# sweep. Relaunched 2026-08-29 after the 2026-08-27 wave was voided by endpoint
# connection errors (see logs_archive/2026-08-27_qwen9_void/README.md).
#
#   MODEL=qwen3.6-27b RESULTS_ROOT=runs_evolving/qwen3.6-27b/ \
#   RUN_PREFIX=base_agent_qwen3_6_27b MAX_ARMS_PER_GPU=12 LAG_SEC=20 \
#   KB_GPU_EVAL_LOCK_SLOTS=3 KB_EVAL_MEM_GATE_FACTOR=7 \
#     bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/launch_wave.sh 1 \
#          scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/wave_median_qwen12.spec
#
# WHY 12 ARMS, ALL AT LAUNCH. 12/SLOTS=3/MEM_GATE=7 is exactly the sizing recipe in
# CLAUDE.md 3.4, and that section forbids adding arms mid-wave: it plants a
# contention seam in the running arms, and new arms would start at problem 1, i.e.
# inside the five problems carrying 74% of the benchmark's input-generation cost.
# All 12 share one GPU so every merge threshold is comparable to every other AND to
# the truncation control -- putting any of them on GPU 0 would compare across a
# contention boundary in their own favour.
#
# THE MERGE SWEEP settles CLAUDE.md open item 1, which records that the code default
# is 0.80 while 0.85 was the earlier validated operating point, "and neither is the
# settled default". At 0.80 on a completed 50-problem arm, 488 source skills
# collapsed into 41 merged skills (mean 11.9 sources each) -- the cluster chaining
# that item warns about -- for a paired ratio of 0.945 [0.787, 1.134], i.e. a ~72%
# smaller catalog at no measurable quality cost. The open question is whether a
# higher threshold preserves more distinctions for free, so this brackets 0.80 on
# both sides. Tags encode the parameter per CLAUDE.md 3.2.
#
# CAVEAT, n=1 per cell. Replicate noise on gpt-oss is log-SD 0.147 (open item 10), so
# a single contrast needs ~x1.50 to clear 95%. qwen has no measured replicate SD yet.
# Treat the sweep as a screen that picks what to replicate, not as a winner claim.
#
# NOT comparable to any gpt-oss or terra arm: qwen now packs L0 at 0.9 x 262144 =
# 235,929 tokens where every pre-2026-08-27 run packed at 115,200 (CLAUDE.md 3.7).
#
# tag          | context-mode         | extra flags
-              | truncation           |
folding        | folding              |
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
refinement     | truncation           | --enable-skill-refinement
merge_sim075   | truncation           | --skill-merging --skill-merge-similarity 0.75
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
merge_sim085   | truncation           | --skill-merging --skill-merge-similarity 0.85
merge_sim095   | truncation           | --skill-merging --skill-merge-similarity 0.95
l2             | truncation           | --enable-l2
