# gpt-5.6-luna -- second batch, 3 more cells on GPU 0 (2026-09-02). n=1 each: a SCREEN.
#
# Companion to wave_luna4.spec (baseline / refinement / compress / deletion on GPU 1).
# Together these give luna 7 of the 9 standard cells; selective_retention and the
# merge sweep are still open.
#
# *** CROSS-GPU COMPARISON BOUNDARY -- THE MAIN CAVEAT FOR THIS BATCH ***
# luna's only control (luna_baseline, truncation) is on GPU 1. These three arms are on
# GPU 0. CLAUDE.md 3.4 measured that unequal arms-per-GPU biases speedups
# ONE-DIRECTIONALLY: waiting cannot deflate a speedup, but another arm's UNLOCKED GPU
# work (reference-model construction, nvcc/ninja, .to(device), empty_cache) can land
# inside your timing window, and the busier card exposes each window to more interferers.
# Since speedup = fixed_baseline / measured_runtime, the busier GPU is penalised.
# At launch GPU0=7 and GPU1=10, so these three sit on the QUIETER card and are therefore
# flattered relative to their control. Do not report folding/markov/l2_redesign vs
# baseline as a clean contrast without either (a) a baseline replicate on GPU 0, or
# (b) an explicit statement of the boundary. Option (a) is one extra arm and is the
# cheaper fix.
#
# *** CAVEAT ON folding -- SAME CLASS AS THE compress CAVEAT IN wave_luna4.spec ***
# folding is one of only three modes that READ the context window
# (gen3_stages.py::_resolve_folding_context_window); truncation and markov_report never
# do. At luna's 1,050,000-token window the folding archive budget is ~945,000 tokens,
# while the largest L0 context ever observed on this benchmark (terra) was 121,055
# tokens -- 8x below it. The omission path therefore CANNOT fire.
# Corroborated empirically: the terra folding arm
# (base_agent_gpt_5_6_terra_folding_itr30_GH200_2026_08_27_01_24) logged
# ZERO "folding archive pack: omitted" lines across a full 50-problem run.
# So this cell measures the L0 GLOBAL SUMMARY that folding always builds, NOT history
# omission. It is not comparable to the gpt-oss folding cell (128k window), where the
# budget can actually bind. There is no CLI window override to tune this away.
#
# markov_report does NOT read the window, so it is unaffected by the above.
# Its --evolving-report-max-tokens now defaults to 65536 (context_management.py:24).
# The launcher still does not pass the flag explicitly, but the DEFAULT is already the
# post-2026-08-27 value, so this arm lands in the "third group" described in CLAUDE.md
# section 4 -- not the old 1536 group. run_summary.json records the value; read it there
# rather than trusting any launcher default in future.
#
# l2_redesign = the redesigned promotion gate (hit_rate 0.60 + dedup 0.80, no standing
# cap), held at truncation like every other L2 arm so the axes stay separable. Verified
# against a completed terra l2redesign arm: context_management=truncation, enable_l2=True,
# redesign_l2=True, l2_use_hit_rate=True, l2_min_hit_rate=0.6, l2_dedup_similarity=0.8,
# l2_standing_cap=-1.
#
# Params identical to every other live arm: SLOTS=3, MEM_GATE=7, HOIST=1, SKIP_REF=1,
# UNLOCK_CORR=0, RESERVE_GB=0, baseline NVIDIA_GH200x2_median.
#
# tag          | context-mode  | extra flags
folding      | folding       |
markov       | markov_report |
l2_redesign  | truncation    | --enable-l2 --redesign-l2
