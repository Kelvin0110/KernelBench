# gpt-5.6-terra -- ROUND 3 (replicate r4), GPU 0: the ENTIRE L0 CONTEXT-MANAGEMENT AXIS.
# Companion: wave_terra_b2_gpu1.spec (GPU 1 = the L1 governance axis).
#
# 9 arms TOTAL across both GPUs this round -- ONE replicate, not two. Batch 1 ran
# 9 cells x 2 GPUs = 18 arms (r2 on GPU0, r3 on GPU1); this round runs the same 9
# cells ONCE (split 6/6, plus three l2 replicates) with a SINGLE truncation control.
# That takes every matrix cell from n=2 to n=3.
#
# WHY THIS SPLIT, and not an arbitrary one:
# With one control there is exactly one contention boundary, so the only decision
# that matters is which comparisons sit on the control's side of it. All five L0
# modes (truncation control + markov + selective_r5 + compress + folding) are here,
# and so is l2_rep1 -- the L2 tier's own control requirement (8.10) is met for that
# one replicate; l2_rep2/rep3 sit on GPU 1 and cross the boundary.
# so the whole L0 axis -- including markov_report, the ONLY cell in either wave with
# support beyond noise -- is compared against a control on its own GPU. The four L1
# governance cells on GPU 1 cross the boundary; they are all flat nulls
# (0.958-1.039 at n=2), so that is where the bias can do least damage.
#
# HOW BIG IS THAT BIAS? Measured on batch 1's own phase logs, 9 arms/GPU:
#   GPU0 13,136 evals: lock wait p50 0.00s, mean 0.20s, 0.42% >=5s, held p50 1.40s
#   GPU1 13,155 evals: lock wait p50 0.00s, mean 0.22s, 0.68% >=5s, held p50 2.04s
# CLAUDE.md 3.4's "unequal arms-per-GPU biases the comparison" warning was collected
# under the MUTEX regime (SLOTS=1) where waits ran 300-685s. With KB_GPU_EVAL_LOCK_SLOTS=3,
# the armed mem gate and hoisted input generation, contention is essentially gone --
# and at 4-5 arms/GPU it is lower than the numbers above. The residual is far below
# the 0.0759 pooled replicate log-SD, so it cannot manufacture a cell result.
#
# PROTOCOL SEAMS vs BATCH 1 -- both audited against batch-1 artifacts, both INERT,
# so r4 POOLS with r2/r3 for all 9 cells:
#  (a) terra's context window was fixed 115,200 -> 945,000 (8.2x) on 2026-08-27
#      (superproject 5b195a1 + submodule b89c47f). Only folding/selective_retention/
#      compress_trigger read it. MEASURED: the 115,200 pack budget bound on 6 of
#      1,499 coder calls in r2_selective_r5 (0.4%) and 0 calls in every other
#      window-reading arm; compress fired 50/50 on reason="iters" (the periodic
#      schedule), never on the token ratio. token_limit in batch 1's
#      compression_events.jsonl reads 108000 = the gpt-oss window, confirming the clamp.
#  (b) every max_tokens budget went to 65536. Across ALL 97,964 recorded LLM calls in
#      batch 1, exactly ONE ceiling was ever reached: evolving_report at 1536, on
#      39/3,000 calls -- ALL 39 inside the two markov arms (r2 23, r3 16).
#      So markov is the one cell with a real seam. TESTED: problems whose report was
#      truncated scored geomean 0.758 vs control (n=24) against 0.747 for untruncated
#      (n=75) -- ratio 1.015, Welch t=0.10, and the two arms disagree in direction.
#      The 1536 ceiling does NOT explain markov's -27%; it is intrinsic to the mechanism.
#
# BASELINE = NVIDIA_GH200x2_2nd, resolved by hardware_env.sh from THIS folder's name.
# Pass NO --hardware and NO KB_DEFAULT_HARDWARE override -- see wave_terra_b1_gpu0.spec
# for why the _median override was dropped before batch 1 relaunched.
#
# selective_r5 carries no flag on purpose: DEFAULT_SELECTIVE_RECENT_ROUNDS = 5 is
# hardcoded and there is no CLI flag -- inventing one aborts the run.
#
# Launch (MAX_ARMS_PER_GPU default 6 is exactly enough for 6; do not raise it):
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=gpt-5.6-terra RESULTS_ROOT=runs_evolving/gpt-5.6-terra/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra_r4 \
#     bash $HW/launch_wave.sh 0 $HW/wave_terra_b2_gpu0.spec
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
folding        | folding              |
l2_rep1        | truncation           | --enable-l2
