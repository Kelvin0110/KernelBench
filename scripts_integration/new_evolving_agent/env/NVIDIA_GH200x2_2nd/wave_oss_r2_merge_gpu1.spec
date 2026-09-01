# gpt-oss-120b merge-threshold sweep, round 1 -- GPU 1 half (2 arms).
# Companion: wave_oss_r2_merge_gpu0.spec (merge_sim095_a).
#
# 0.75 and 0.85 bracket the r2 wave's existing merge_sim08 (0.8), which is ALREADY on
# GPU 1 -- so the three most-comparable thresholds share a GPU and their contrasts carry
# no contention term at all (CLAUDE.md 3.4).
#
# STANDING CONFOUND, common to every merge arm: --skill-merging also uncaps the
# extractor candidate set (open item 6; gen3_stages.py:889 passes
# skill_deletion = deletion OR merging). It cancels WITHIN the sweep but not against the
# truncation control -- report those contrasts as "threshold + catalog size".
#
# These start at problem 1 while every other live arm is deep in Level 3, so they enter
# the five problems carrying 74% of input-generation cost together. The mem gate at
# factor 7 binds there (L1P34 49 GiB and L1P22 42 GiB admit 2; L1P100 28 GiB admits 4).
# A slow first day is the gate working, not a stall.
#
# tag          | context-mode | extra flags
merge_sim075_a | truncation   | --skill-merging --skill-merge-similarity 0.75
merge_sim085_a | truncation   | --skill-merging --skill-merge-similarity 0.85
