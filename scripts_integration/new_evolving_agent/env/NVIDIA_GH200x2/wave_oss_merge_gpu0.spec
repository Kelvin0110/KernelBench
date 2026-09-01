# gpt-oss-120b -- merge-similarity sweep at 0.7 and 0.9, n=3 each.
#
# Open item 1: the code default is 0.8 and 0.85 was the earlier validated operating
# point; BOTH now have data and neither is settled. What 0.8 measurably did on a
# completed arm was collapse 488 source skills into 41 merged ones (mean 11.9 sources
# each) for a 168-entry catalog vs the control 600 -- cluster chaining -- at zero
# quality cost (paired 0.945 [0.787, 1.134], McNemar p=0.73).
#
# This wave brackets that: 0.7 should chain HARDER (fewer, broader skills) and 0.9
# should barely merge at all. Together with the 0.8/0.85 arms on disk that gives a
# four-point response curve for catalog size, which is the quantity open item 1 asks
# about -- not whether merging "works".
#
# CONFOUND TO CARRY INTO THE WRITE-UP (open item 6): --skill-merging also uncaps the
# extractor candidate catalog (50 -> uncapped). It is common to all four thresholds so
# it cancels WITHIN this sweep, but any comparison against a non-governance control
# must be reported as "rule + catalog size".
#
# Cells split across both GPUs (2/1 here, 1/2 on the other) for the same
# contention-balance reason as the deletion wave. Distinct tags per replicate.
#
# Params match every live arm: SLOTS=3, MEM_GATE=7, HOIST=1, SKIP_REF=1,
# UNLOCK_CORR=0, RESERVE_GB=0.
#
# tag                | context-mode | extra flags
merge_sim09_r1 | truncation   | --skill-merging --skill-merge-similarity 0.9
merge_sim07_r2 | truncation   | --skill-merging --skill-merge-similarity 0.7
merge_sim09_r3 | truncation   | --skill-merging --skill-merge-similarity 0.9
