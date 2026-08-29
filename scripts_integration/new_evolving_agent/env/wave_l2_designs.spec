# Shipped L2 gate vs the redesign, with their own control, on one GPU.
#
# The point of this spec is that the two designs differ by exactly one flag, so
# the contrast is the design and nothing else. Every other L2 knob is left at its
# default and resolved by _resolve_l2_preset.
#
# Read it against the truncation arm IN THIS SPEC, not a historical one: a control
# from another wave sits in a different contention window and a different
# endpoint-latency window (open items 10-11), and unequal arms-per-GPU biases the
# comparison one-directionally (3.4).
#
# BEFORE quoting any arm-vs-arm result from this spec, read section 4's lottery
# warning. ~14 of the 50 subset problems are bimodal and set the geomean; on the
# 2026-08-27 wave the raw ranking was entirely an artifact of them, and every
# adjusted CI contained 1.0 including the identical-configuration null. n=1 per
# cell is a screen, not a test.
#
# tag         | context-mode | extra flags
truncation    | truncation   |
l2_shipped    | truncation   | --enable-l2
l2_redesign   | truncation   | --enable-l2 --redesign-l2
