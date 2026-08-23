# GPU 1 -- gpt-5.6-terra, 6 variants, median-baseline series (2026-08-22).
#
#   HARDWARE=NVIDIA_GH200x2_median MAX_ARMS_PER_GPU=6 \
#   MODEL=gpt-5.6-terra RUN_PREFIX=base_agent_gpt_5_6_terra \
#   RESULTS_ROOT=runs_evolving/gpt-5.6-terra/median/ \
#     bash scripts_integration/new_evolving_agent/env/launch_wave.sh 1 \
#          scripts_integration/new_evolving_agent/env/wave_median_terra6.spec
#
# gpt-5.6-terra resolves to azure/openai/gpt-5.6-terra on the inference endpoint
# and takes reasoning_effort=high with no temperature/top_p (llm_client.py
# _is_reasoning_gpt). Smoke-tested 2026-08-22 on L1P100: iteration 1 compiled,
# was correct, and hit 8.17x -- but per-iteration latency is well above
# gpt-oss-120b, so expect this wave to trail GPU 0 substantially.
#
# Six of the nine cells, chosen to span both axes: 4 context modes + deletion
# + promotion. No merge/refinement/folding arm here.
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion       | truncation           | --skill-deletion
l2             | truncation           | --enable-l2
