# gpt-5.6-terra -- BATCH 1 of 2, GPU 1, replicate r3.  Companion: wave_terra_b1_gpu0.spec
#
# 18 arms total across both GPUs, 9 per GPU:
#   8 cells x 2 replicates  -> the matrix, n=2, one COMPLETE replicate per GPU
#   1 merge threshold / GPU -> 0.9 on GPU0, 0.7 on GPU1, n=1 each
#
# BATCH 2 (deferred, not yet designed) is the L2-promotion tier. Everything L2 is
# held back -- including the plain `l2` cell at default parameters -- because the
# finished gpt-oss-120b l2 arm showed the current design is broken, not merely
# null: it promoted 9 standing rules of which 7 are the same idea ("don't write
# trivial kernels"), 12,012 of 15,176 chars, injected verbatim into EVERY coder
# system prompt (20,091 chars vs the control's 4,190 = 4.79x) for geomean 1.347
# vs the control's 1.389, paired 0.965 [0.816, 1.142], McNemar p=1.000.
# Running that config again would spend arms confirming a known defect.
# WHEN BATCH 2 RUNS IT MUST BRING ITS OWN `l2` CONTROL on the same GPU -- there is
# no L2 reference in this batch to compare it against, and a control from a
# different batch sits in a different contention and endpoint-latency window.
#
# The 8 cells are the gpt-oss 9-cell matrix minus l2. The first five are
# byte-identical to terra replicate 1 (the other server's wave launched
# 2026-08-22 21:08-21:23, env/NVIDIA_GH200x2/wave_median_terra6.spec); folding,
# refinement and merge_sim08 are copied verbatim from env/wave_gpu0.spec, so
# terra and gpt-oss are measured on the same matrix.
#
# NOTE ON REPLICATE 1: it is NOT poolable with these arms. ede1898 (2026-08-25
# 07:11) added the STRICT check_global_module_patch reference-corruption check,
# which is parent-side, so it is active for these arms and can never reach rep 1.
# A STRICT failure short-circuits before GPU eval (governor.py:549) and returns
# compiled=False correct=False is_hack=True, so it changes the SEARCH, not just
# the score. Measured: 0 of 13,500 gpt-oss submissions on this host trip it, so
# the 63 flags in that commit's validation corpus are almost certainly terra's --
# and L1P55, the problem it cites as passing 14 times, is subset problem 8.
#
# Launch:
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   HARDWARE=NVIDIA_GH200x2_median MODEL=gpt-5.6-terra MAX_ARMS_PER_GPU=9 \
#   RESULTS_ROOT=runs_evolving/gpt-5.6-terra/median/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra_r3 \
#     bash $HW/launch_wave.sh 1 $HW/wave_terra_b1_gpu1.spec
#
# HARDWARE=NVIDIA_GH200x2_median is REQUIRED and is NOT this host's default.
# hardware_env.sh derives $HARDWARE from the launcher's folder name, giving
# NVIDIA_GH200x2_2nd -- also median-bearing, so it would NOT fatal, it would
# silently score against a different file (14/50 subset problems differ >5%, all
# Level 3: L3 geomean 1.079, range 0.591-1.976; L1 1.007, L2 1.009).
#
# selective_r5 carries no flag on purpose: DEFAULT_SELECTIVE_RECENT_ROUNDS = 5 is
# hardcoded and there is no CLI flag -- inventing one aborts the run.
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
folding        | folding              |
refinement     | truncation           | --enable-skill-refinement
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
merge_sim07    | truncation           | --skill-merging --skill-merge-similarity 0.7
