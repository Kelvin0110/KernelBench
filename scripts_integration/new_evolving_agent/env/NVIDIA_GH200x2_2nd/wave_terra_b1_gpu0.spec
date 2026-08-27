# gpt-5.6-terra -- BATCH 1 of 2, GPU 0, replicate r2.  Companion: wave_terra_b1_gpu1.spec
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
# Launch -- pass NO baseline override; hardware_env.sh resolves this host's own
# folder name, NVIDIA_GH200x2_2nd:
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=gpt-5.6-terra MAX_ARMS_PER_GPU=9 \
#   RESULTS_ROOT=runs_evolving/gpt-5.6-terra/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra_r2 \
#     bash $HW/launch_wave.sh 0 $HW/wave_terra_b1_gpu0.spec
#
# BASELINE = NVIDIA_GH200x2_2nd, this host's own, measured on this silicon.
# An earlier launch of this spec (killed 2026-08-25 ~11:00 at problem 1/50, nothing
# lost) overrode it to NVIDIA_GH200x2_median to line up with terra replicate 1 on
# the other server. That override was DROPPED, because:
#   * rep 1 is not poolable with these arms anyway -- ede1898's STRICT check is
#     parent-side, active here and unreachable there (see above), so matching its
#     baseline bought nothing that the trajectory difference had not already cost;
#   * _median was measured on the OTHER server (it reached this clone by a
#     fast-forward pull from origin, not a local measurement);
#   * this host's completed gpt-oss wave is scored against _2nd
#     (hardware_server in all 9 run_summary.json), so _median would have put terra
#     on a different scale from this host's own corpus;
#   * neither GH200 Level-3 baseline is clean -- _median's meta note claims it
#     removed an up-to-3x batch-position artifact, yet on this subset _median L3 is
#     7.9% HIGHER than _2nd and they disagree in BOTH directions (per-problem ratios
#     0.591-1.976), each being the outlier on different problems. With no clean
#     choice, internal consistency on this host wins.
# The two differ on 14/50 subset problems, ALL Level 3 (L3 geomean 1.079; L1 1.007,
# L2 1.009). Being a per-problem constant it cancels in arm-vs-control ratios, so
# this changes the absolute level -- not the within-wave comparisons -- EXCEPT that
# the speedup is fed back into the coder prompt, so it does move the search. That
# is why it had to be fixed by relaunching rather than re-scored afterwards.
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
merge_sim09    | truncation           | --skill-merging --skill-merge-similarity 0.9
