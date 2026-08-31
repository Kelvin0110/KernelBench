# gpt-oss-120b -- SECOND 9-cell wave on this host. GPU 0 half (control + the L0 axis).
# Companion: wave_oss_r2_gpu1.spec (the governance axis). Runs alongside 8 resumed terra
# arms (4 per GPU), giving 9 arms on GPU 0 and 8 on GPU 1.
#
# The whole L0 axis sits with the control, so every context-management contrast is
# within-GPU; the governance cells cross to GPU 1, which is where a contention boundary
# does least damage (they were all flat nulls on terra).
#
# *** THIS IS n=1 OF A NEW REGIME, NOT AUTOMATICALLY A SECOND REPLICATE. ***
# A complete 9-cell gpt-oss wave already exists (Aug-22, all 9 arms finished, SAME
# NVIDIA_GH200x2_2nd baseline, results in output/GH200x2_2nd_aug22_wave). Pooling the
# two would take gpt-oss from n=1 to n=2 and drop the single-contrast floor from x1.50
# to ~x1.19 -- worth a lot. But FOUR protocol seams sit between them:
#
#  1. evolving_report_max_tokens 1536 -> 65536. Confirmed 1536 in all nine Aug-22
#     run_summary.json. gpt-oss truncated 0.176% of calls at the old budgets (worst:
#     preflight 1.16%, action_selector 0.31%) -- SMALL BUT NON-ZERO, unlike terra's
#     exact 0%. This one changes the agent's INPUTS, so it moves the search, and it is
#     the seam that cannot simply be re-scored away.
#  2. Eval trims. The Aug-22 arms ran with only KB_GPU_RESERVE_GB=0 and the lock
#     timeout -- no HOIST, no SKIP_DEAD_REF_TIMING, SLOTS=1 mutex, mem gate OFF. This
#     wave gets SLOTS=3 MEM_GATE=7 HOIST=1. Contention only ever DEFLATES a speedup,
#     so the old wave is if anything penalised relative to this one.
#  3. num_perf_trials 100 -> 25, mid-wave for Aug-22.
#  4. is_hack threshold 10x -> 30x, mid-wave for Aug-22. Re-derivable at analysis time
#     from the raw speedup, so this one is fixable.
#
# HOW TO DECIDE, rather than assuming: report this wave as n=1 on its own first. If its
# truncation control lands near the Aug-22 control's best_geomean of 1.3885, the seams
# are empirically inert and the two pool to n=2. If it does not, that IS the finding,
# and this wave still stands alone as a clean modern-regime measurement.
#
# Launch:
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=gpt-oss-120b MAX_ARMS_PER_GPU=12 RESULTS_ROOT=runs_evolving/gpt-oss-120b/ \
#   RUN_PREFIX=base_agent_gpt_oss_120b_r2 bash $HW/launch_wave.sh 0 $HW/wave_oss_r2_gpu0.spec
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
folding        | folding              |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
