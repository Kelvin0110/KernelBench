# TERRA -- the 3 cells that exist for gpt-oss but not for terra, plus a fresh
# same-wave truncation control.
#
#   MODEL=gpt-5.6-terra RESULTS_ROOT=runs_evolving/gpt-5.6-terra/median/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra MAX_ARMS_PER_GPU=12 \
#     bash .../env/NVIDIA_GH200x2/launch_wave.sh 1 .../wave_median_terra_complete.spec
#
# WHY A NEW truncation CONTROL. CLAUDE.md 3.4: "put every arm you intend to compare
# on the same GPU as its control." The existing terra truncation arm
# (..._2026_08_22_21_08) ran in a 6-arm wave, before the STRICT global_module_patch
# check, and carries the 10->30 excessive-speedup seam mid-run. A control launched
# beside these three shares all of that, so the within-wave contrast is clean; the
# old arm stays usable as a cross-wave check.
#
# COMPARABILITY TO THE EXISTING 6 TERRA ARMS IS GOOD. The new static check is a
# no-op for terra: 0 of 8968 terra submissions ever rebound an nn/torch attribute
# (gpt-oss did it in 3 cells). So it cannot shift a terra number, and these arms
# remain directly comparable to the completed six. That is NOT true for gpt-oss.
#
# Cost: terra measured 70.7 h median summed problem wall-time per arm on this host
# (gpt-oss 58.3 h). Arms/GPU scaling is ~linear to 9 (~3% per-arm cost), so 4 or
# 10 arms on one GPU are both ~3 days calendar.
#
# tag          | context-mode         | extra flags
-              | truncation           |
folding        | folding              |
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
refinement     | truncation           | --enable-skill-refinement
