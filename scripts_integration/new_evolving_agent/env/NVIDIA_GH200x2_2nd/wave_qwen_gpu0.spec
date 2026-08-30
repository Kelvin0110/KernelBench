# qwen3.6-27b -- FIRST qwen wave on this host. GPU 0 half (8 arms).
# Companion: wave_qwen_gpu1.spec (5 qwen). Alongside them run 5 RESUMED terra arms
# (1 on GPU0, 4 on GPU1), giving 9 arms per GPU.
#
# This GPU carries the CONTROL plus 7 of the 9 matrix cells, so every contrast that
# matters is within-GPU. refinement and the merge-threshold extras sit on GPU 1.
#
# MODEL NOTES that differ from gpt-oss and terra -- check these before reading results:
#  * alias resolves to nvidia/qwen/qwen3.6-27b; INFERENCE endpoint only (no integrate row).
#  * context window 262,144 -> L0 pack budget 235,929. Only folding / selective_retention
#    / compress_trigger read the window, so 6 of 9 cells are unaffected by it.
#  * qwen is the ONLY model here that VALIDATES max_tokens (max_model_len=262144);
#    gpt-oss and terra accept anything. With every budget now 65536, prompt+output must
#    stay under 262,144 -- fine at these prompt sizes, but a real ceiling, not advisory.
#  * MODEL_SAMPLING_PROFILES pins qwen's sampling (temp 0.6 / top_p 0.95 / top_k 20 /
#    min_p 0.0 / presence 0.0 / repetition 1.0) and OVERRIDES the caller's per-role
#    temperature. So role-level temperature settings are inert for qwen -- do not
#    attribute any qwen-vs-terra difference to them.
#
# NOT COMPARABLE ACROSS MODELS. Replicate noise is model-specific (gpt-oss log-SD 0.147,
# terra 0.0759); qwen's is UNMEASURED until it has closed replicate pairs. At n=1 per
# cell this wave is a SCREEN, not a test -- it cannot name a winner. Its first job is to
# establish qwen's own noise floor.
#
# Launch:
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=qwen3.6-27b MAX_ARMS_PER_GPU=12 RESULTS_ROOT=runs_evolving/qwen3.6-27b/ \
#   RUN_PREFIX=base_agent_qwen36_27b bash $HW/launch_wave.sh 0 $HW/wave_qwen_gpu0.spec
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
folding        | folding              |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
l2             | truncation           | --enable-l2
