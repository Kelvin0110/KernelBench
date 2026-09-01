# gpt-5.6-terra -- the deletion-rule split at n=3, mirroring the gpt-oss cells
# launched 2026-08-31 (CLAUDE.md 3.8). Same two rules, same protocol, second model.
#
#   unused    = consecutive-unused streak GC only
#   unit_test = post-append LLM-authored pytest admission gate only
#
# WHY BOTH MODELS: open item 3 measures the unit-test gate at 30-58%% of all deletions
# on every arm on disk, and the split is 100/218 on the terra arm vs 291/232 on gpt-oss
# -- i.e. the two rules carry OPPOSITE weight per model. A gpt-oss-only decomposition
# cannot say whether that is a model property or noise, so terra gets the same n=3.
#
# CELLS ARE SPLIT ACROSS BOTH GPUs ON PURPOSE. CLAUDE.md 3.4: unequal arms-per-GPU
# biases speedups one-directionally, so a cell confined to one card would confound
# treatment with contention. 2/1 here, 1/2 on the other GPU.
#
# Replicates carry distinct tags (_r1/_r2/_r3) so the minute-resolution run-name
# collision in CLAUDE.md 3.2 cannot bite and every arm stays distinguishable downstream.
#
# Params match every live arm: SLOTS=3, MEM_GATE=7, HOIST=1, SKIP_REF=1,
# UNLOCK_CORR=0, RESERVE_GB=0. Never mix slot counts on one GPU.
#
# tag                  | context-mode | extra flags
deletion_unittest_r1 | truncation   | --skill-deletion --skill-deletion-rules unit_test
deletion_unused_r2   | truncation   | --skill-deletion --skill-deletion-rules unused
deletion_unittest_r3 | truncation   | --skill-deletion --skill-deletion-rules unit_test
