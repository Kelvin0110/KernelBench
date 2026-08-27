# GPU 1 -- qwen3.6-27b, all 9 variants. First wave for this model (2026-08-27).
#
#   MODEL=qwen3.6-27b RESULTS_ROOT=runs_evolving/qwen3.6-27b/ \
#   RUN_PREFIX=base_agent_qwen3_6_27b MAX_ARMS_PER_GPU=12 LAG_SEC=20 \
#     bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/launch_wave.sh 1 \
#          scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/wave_median_qwen9.spec
#
# Same 9 cells as wave_median_oss9.spec -- the spec carries no model, so the cell
# definitions are shared and only MODEL/RESULTS_ROOT/RUN_PREFIX differ.
#
# NOT comparable to any existing gpt-oss or terra arm. This is the first wave on
# the 2026-08-27 footing: every LLM budget is a uniform 65536 (was 512..16384) and
# the L0 context window now follows --model, so qwen packs at 0.9 x 262144 =
# 235,929 tokens where every earlier run packed at 115,200. Compare qwen arms
# against each other only. See CLAUDE.md 3.7.
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
