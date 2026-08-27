# Small-budget replicate design: does a FIXED standing rule set help?
#
# Two paired replicates, 15 problems each, on the level-2 block
# (subset_selection/selected_problems_L2block15.csv = problems 11-25 of the
# usual subset).
#
# Why this window
#   CLEAN. The pre-seeded rules were distilled from L1P22 and L1P50, both inside
#   problems 1-10, so a run over 1-25 is contaminated on its first 10. Problems
#   11-25 contain none of the source problems.
#   CHEAP. Problems 1-5 carry 74% of the benchmark's input-generation cost and
#   every OOM this project has seen. This window skips them entirely, so these
#   arms add little memory pressure to a GPU already running seven.
#
# Why replicates rather than one longer arm
#   The running 50-problem l2_preseed arm ALREADY covers problems 11-25, so a
#   single short arm would duplicate it. What is missing is a second and third
#   independent sample: every quality claim here is noise-limited (arm-level
#   log-SD 0.147-0.182), and n=1 per cell needs ~x1.5 to clear 95%. Replicates
#   are the only thing that moves that, and short clean-window pairs are the
#   cheapest replicates available.
#
# What it can and cannot show
#   Paired per-problem SE at n=15 is 1.035/sqrt(15) = 0.267, i.e. it detects
#   ~x1.68 or larger. It is a SCREEN for a large effect, not a test for a small
#   one. Pooled with the running arm's problems 11-25 it gets to n=45 pairs
#   across three samples, SE ~0.154, which is where it starts to be worth
#   quoting.
#
# Read it PAIRED per problem (q15_pre_rN vs q15_ctl_rN on matched problems),
# not as a difference of arm geomeans -- pairing is 1.8x tighter here.
#
# tag         | context-mode | extra flags
q15_ctl_r1    | truncation   | --subset-csv subset_selection/selected_problems_L2block15.csv
q15_pre_r1    | truncation   | --subset-csv subset_selection/selected_problems_L2block15.csv --enable-l2 --l2-preseed preseed/l2_hit_5rules.jsonl --l2-freeze
q15_ctl_r2    | truncation   | --subset-csv subset_selection/selected_problems_L2block15.csv
q15_pre_r2    | truncation   | --subset-csv subset_selection/selected_problems_L2block15.csv --enable-l2 --l2-preseed preseed/l2_hit_5rules.jsonl --l2-freeze
