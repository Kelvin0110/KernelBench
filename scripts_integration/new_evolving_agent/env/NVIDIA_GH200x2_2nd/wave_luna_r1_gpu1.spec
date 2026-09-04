# gpt-5.6-luna, round 1 -- GPU 1 half: the L1 governance axis (4 of 10 free slots).
# Companion: wave_luna_r1_gpu0.spec (control + L0 axis + l2_redesign).
#
# The THREE-WAY DELETION DECOMPOSITION IS KEPT TOGETHER ON ONE GPU on purpose.
# --skill-deletion applies two independent rules (CLAUDE.md 3.8) and the unit-test
# admission gate is 30-58% of all deletions on every arm measured, so `both` cannot be
# read as "the usage rule". Splitting `both` / `unused` / `unit_test` across a contention
# boundary would put the finest-grained contrast in this wave across the one seam that
# does not cancel. `unused` still RUNS the unit test and only stops the eviction, so all
# three sub-cells hold LLM-call volume fixed and differ by rule, not by cost.
#
# STANDING CONFOUND on all four: --skill-deletion and --skill-merging both uncap the
# extractor candidate set (open item 6; gen3_stages.py:889 passes
# skill_deletion = deletion OR merging). It is common to the decomposition and cancels
# within it, but NOT against the control -- report those as "rule + catalog size".
#
# refinement is NOT here: it is the 11th cell and there are only 10 free slots. It sits
# in wave_luna_r1_backfill.queue and launches when a slot frees. It trips neither the
# deletion gate nor the catalog uncapping, so it is the one clean governance cell and
# loses least by starting late.
#
# tag              | context-mode | extra flags
deletion           | truncation   | --skill-deletion
deletion_unused    | truncation   | --skill-deletion --skill-deletion-rules unused
deletion_unittest  | truncation   | --skill-deletion --skill-deletion-rules unit_test
merge_sim08        | truncation   | --skill-merging --skill-merge-similarity 0.8
