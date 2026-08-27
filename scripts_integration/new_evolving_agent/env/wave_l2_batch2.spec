# L2 batch 2 -- gpt-oss-120b, GPU 0, alongside the four batch-1 arms.
#
# Batch 1 asked "does a better-calibrated rule-based gate promote a sane set?".
# The answer arrived before batch 1 finished, and it reframes the question:
# selection carries NO outcome signal (P(new best | selected) 0.1423 vs 0.1440
# for offered-but-not-selected on gpt-oss; -0.022 on terra; still zero within
# attempt-position strata), and min_new_bests is corr 0.87 with selections, so
# it is the same signal rescaled. No threshold on the ledger can work.
#
# So batch 2 leaves the ledger behind, and separates questions batch 1 conflates.
#
#   l2_judge      An LLM reads the rule TEXT and decides -- general, actionable,
#                 non-redundant, non-obvious. Floors are deliberately LOW here:
#                 they are only an admission bar ("was this exercised at all"),
#                 not the ranking. Leaving them at shipped values would hand the
#                 judge the same tiny, late-arriving candidate set the broken
#                 floors already produce.
#
#   l2_preseed    Install l2_hit's 5 standing rules at problem 1 and FREEZE.
#                 Every other L2 arm confounds "can the gate find rules" with
#                 "do rules help"; an arm that promotes nothing answers neither.
#                 This holds the rule set fixed and asks only the second.
#                 LEAKAGE: those rules were distilled from L1P22 and L1P50, both
#                 inside subset problems 1-10. Problems 11-50 are clean -- read
#                 the contrast there (40 problems, still enough to pair).
#
#   l2_extract    Same gate as l2_hit, different render. Standing text drops to
#                 ~28%, so the pair isolates prompt VOLUME from rule CONTENT.
#                 Listed in CLAUDE.md 8.9 and never run.
#
# Three arms, not four: the memory gate is armed and genuinely binding (7.5% of
# evals wait, max 671s) and its wait is charged against the 600s eval deadline,
# unlike the lock's. 1 of 1081 evals has already hit the gate timeout. More arms
# raise the chance of big-input evals coinciding, so this stays well under the
# 12-arm guidance and uses a long stagger.
#
# tag         | context-mode | extra flags
l2_judge      | truncation   | --enable-l2 --l2-judge --l2-min-tasks 2 --l2-min-selections 15 --l2-min-rate 0.05 --l2-standing-cap 6
l2_preseed    | truncation   | --enable-l2 --l2-preseed preseed/l2_hit_5rules.jsonl --l2-freeze
l2_extract    | truncation   | --enable-l2 --l2-render extract --l2-use-hit-rate --l2-min-hit-rate 0.70
