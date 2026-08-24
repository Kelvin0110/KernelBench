# GPU 0 -- the three gpt-oss cells with no measurement under the corrected metric.
# Added 2026-08-24 alongside the six already running, taking GPU 0 to 9 arms.
#
#   KB_GPU_EVAL_LOCK_SLOTS=3 KB_EVAL_MEM_GATE_FACTOR=2.5 MAX_ARMS_PER_GPU=9 \
#   RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
#     bash .../env/NVIDIA_GH200x2/resume_wave.sh 0 .../wave_median_oss3_add.spec
#
# SLOTS MUST BE 3, matching the running six. The slot files are keyed by physical
# GPU UUID (gpu_lock.gpu_lock_key), so every arm on GPU 0 contends for the same
# three files and 9 arms still means at most 3 concurrent evals. Verified:
#   CUDA_VISIBLE_DEVICES=0 -> eval_uuid_d4d7964e...lock.slot{0,1,2}
# A different slot count would change the FILE NAMES and the two groups would not
# interlock -- 3 + N concurrent evals, straight back to the OOM regime.
#
# All three have 0 problems done (killed inside problem 1 on 2026-08-23), so
# resume `auto` starts them at problem 1. That means ~10 h inside subset problems
# 1-5, which carry 74% of the benchmark's input-generation cost and produced every
# one of the 18 OOMs. KB_EVAL_MEM_GATE_FACTOR=2.5 is what makes that safe: it caps
# concurrent evals by estimated bytes, so a 7.5 GB-input problem admits 2 rather
# than 3. The six running arms do not have the gate and do not need it -- they are
# past problem 5 and their evals are ~2 s holds on sub-GB inputs.
#
# tag          | context-mode         | extra flags
folding        | folding              |
selective_r5   | selective_retention  |
refinement     | truncation           | --enable-skill-refinement
