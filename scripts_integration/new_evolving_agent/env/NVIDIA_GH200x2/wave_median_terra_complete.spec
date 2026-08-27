# TERRA -- the 3 cells that exist for gpt-oss but not for terra.
# Completes the terra matrix to parity with gpt-oss (9 cells).
#
#   MODEL=gpt-5.6-terra RESULTS_ROOT=runs_evolving/gpt-5.6-terra/median/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra \
#     bash .../env/NVIDIA_GH200x2/launch_wave.sh 1 .../wave_median_terra_complete.spec
#
# COMPARABILITY TO THE EXISTING 6 TERRA ARMS IS GOOD -- this is why terra was the
# right matrix to complete first. The STRICT global_module_patch check added in
# `ede1898` is a NO-OP for terra: 0 of 8968 terra submissions ever rebound an
# nn/torch attribute (gpt-oss did it in 3 cells). A check that never fires cannot
# shift a number, so these arms are directly comparable to the completed six.
#
# Two seams these arms do NOT have, which the existing six DO -- so read a
# new-vs-old terra contrast with care:
#   1. The 10 -> 30 excessive_speedup_threshold change (2026-08-24T15:11:45) lands
#      MID-RUN in the old six; these run entirely at 30. Corrected offline by
#      rescore_hack_threshold.py -- use the re-scored numbers, not the stored ones.
#   2. The old six ran problems 1-3 un-trimmed under a 6-arm mutex and 4-50 trimmed
#      under SLOTS=3; these run trimmed under SLOTS=3 throughout. Contention only
#      ever deflates speedup, so the old prefix is the pessimistic end.
# There is no fresh truncation control in this wave: compare against the existing
# terra truncation arm (..._2026_08_22_21_08), across the two seams above.
#
# Cost: terra measured 70.7 h median summed problem wall-time per arm on this host
# (gpt-oss 58.3 h). 3 arms on one GPU is well inside the ~3 arms/GPU sweet spot.
#
# tag          | context-mode         | extra flags
folding        | folding              |
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
refinement     | truncation           | --enable-skill-refinement
