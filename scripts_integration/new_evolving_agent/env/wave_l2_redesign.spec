# L2 redesign screen -- gpt-oss-120b, one GPU, four arms.
#
# gpt-oss is the right model for this: it is the arm where the shipped gate
# promoted ZERO rules, so it has the most headroom and is the case the redesign
# was derived from.
#
# The four arms decompose the change rather than only testing the bundle:
#
#   truncation   no L2 at all -- the base control. A fresh one is required
#                because the existing median-wave truncation arm ran in a
#                9-arm contention window, and CLAUDE.md 3.4 shows unequal
#                arms-per-GPU biases the comparison one-directionally.
#   l2           shipped defaults -- must reproduce the 0-promotion null.
#                CLAUDE.md 8.10: every L2 batch carries its own l2 control on
#                the SAME GPU.
#   l2_hit       hit-rate metric ONLY. Isolates the metric fix from the cap and
#                the dedup. Offline replay predicts 4 rules at hit>=0.70.
#   l2_redesign  the full proposal: hit>=0.60 + standing cap 6 + dedup 0.80.
#                Offline replay predicts 6 rules, 0 pairs at cosine >=0.80.
#
# n=1 per cell. Replicate noise is log-SD 0.147 (open item 10), so a single
# arm-vs-arm contrast needs ~x1.50 to clear 95%. This is a SCREEN: it confirms
# the mechanism live and gives a first quality reading. It cannot name a winner.
#
# tag         | context-mode | extra flags
truncation    | truncation   |
l2            | truncation   | --enable-l2
l2_hit        | truncation   | --enable-l2 --l2-use-hit-rate --l2-min-hit-rate 0.70
l2_redesign   | truncation   | --enable-l2 --l2-use-hit-rate --l2-min-hit-rate 0.60 --l2-standing-cap 6 --l2-dedup-similarity 0.80
