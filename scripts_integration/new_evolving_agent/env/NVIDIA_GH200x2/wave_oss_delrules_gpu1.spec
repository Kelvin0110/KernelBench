# gpt-oss-120b -- splitting the deletion cell per CLAUDE.md 3.8 (2026-08-31).
#
# --skill-deletion applies TWO rules and open item 3 measures the unit-test gate at
# 30-58%% of all deletions on every arm on disk. cef6d70 made them selectable, so this
# runs each rule ALONE at n=3. The historical `both` arms are the third cell and are
# already on disk, which is why they are not re-run here.
#
#   unused    = consecutive-unused streak GC only  (the usage rule, never before isolable:
#               --no-l1-skill-delete-on-unit-test-fail was a no-op, see open item 3)
#   unit_test = post-append LLM-authored pytest admission gate only
#
# CELLS ARE SPLIT ACROSS BOTH GPUs ON PURPOSE. CLAUDE.md 3.4 warns that unequal
# arms-per-GPU biases speedups one-directionally, so putting one cell entirely on one
# card would confound the treatment with the contention level. Each cell therefore has
# arms on both GPUs (2/1 here, 1/2 on the other), which is the closest to balanced that
# 3 replicates x 2 cells allows on 2 cards.
#
# Params match every other live arm: SLOTS=3, MEM_GATE=7, HOIST=1, SKIP_REF=1,
# UNLOCK_CORR=0, RESERVE_GB=0. Never mix slot counts on one GPU (CLAUDE.md 3.4).
#
# tag                  | context-mode | extra flags
deletion_unittest_r2 | truncation   | --skill-deletion --skill-deletion-rules unit_test
deletion_unused_r3   | truncation   | --skill-deletion --skill-deletion-rules unused
deletion_unittest_r3 | truncation   | --skill-deletion --skill-deletion-rules unit_test
