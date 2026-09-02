# gpt-5.6-luna -- first wave on this model (2026-09-02). n=1 per cell: a SCREEN, not a test.
#
# Luna was registered in submodule f2f7221 (alias azure/openai/gpt-5.6-luna, inference
# endpoint only, 1,050,000-token window, in FLEET_INFERENCE_CHAT_MODELS). Verified live
# before launch: the endpoint serves it, returns content, and emits NO reasoning_content,
# so the max_tokens-eats-CoT hazard that affects gpt-oss does not apply here.
#
# MEASURED LATENCY at a coder-sized prompt (~2.1k in, 3 samples):
#   luna 32.2s median (14.5-43.3)  |  terra 22.1s  |  gpt-oss 2.6s
# The agent is LLM-bound, so budget ~1.5x a terra arm. Expect multiple days per arm.
#
# ALL FOUR ARMS ON ONE GPU, ON PURPOSE -- the opposite of the deletion/merge waves.
# Those had 3 replicates per cell and could be balanced across cards. This wave has FOUR
# DIFFERENT CELLS SHARING ONE CONTROL, so splitting would put the baseline arm in a different
# contention environment from the treatments it is the control for -- exactly the
# one-directional bias CLAUDE.md 3.4 warns about. Co-location is the correct choice here.
#
# *** CAVEAT ON the compress arm -- READ BEFORE ANALYSING IT ***
# compress_trigger fires on EITHER overflow OR a periodic every-N-iterations trigger
# (compress_trigger.py:216-218). At luna's window the overflow threshold is
# 0.85 x 1,030,000 ~= 875,500 tokens. The terra compress arm's MAXIMUM observed context
# was 121,055 tokens -- 7x below that -- so on luna the overflow path CAN NEVER FIRE.
# This cell therefore measures PERIODIC compaction only, and is NOT comparable to the
# gpt-oss compress cell (128k window), where overflow is the dominant path. There is no
# CLI window override, so this cannot be tuned away with a flag.
# For scale: on terra, compaction ran on only 50 of 1500 iterations (3.3%).
#
# the deletion arm runs BOTH deletion rules (--skill-deletion-rules defaults to 'both'),
# so per CLAUDE.md 3.8 report it as "deletion + unit-test admission gate", not "deletion".
# It also uncaps the extractor catalog (open item 6) -- report as "rule + catalog size".
#
# Params identical to every other live arm: SLOTS=3, MEM_GATE=7, HOIST=1, SKIP_REF=1,
# UNLOCK_CORR=0, RESERVE_GB=0, baseline NVIDIA_GH200x2_median.
#
# tag             | context-mode     | extra flags
baseline        | truncation       |
refinement      | truncation       | --enable-skill-refinement
compress        | compress_trigger | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
deletion        | truncation       | --skill-deletion
