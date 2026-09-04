# gpt-5.6-luna, round 1 -- GPU 0 half: the truncation CONTROL + the whole L0 axis + L2.
# Companion: wave_luna_r1_gpu1.spec (the L1 governance axis).
#
# GPU0 takes 6 of the 10 free slots (6 live -> 12); GPU1 takes 4 (8 live -> 12).
#
# The CONTROL sits here with the ENTIRE L0 axis, so every context-management contrast is
# within-GPU (CLAUDE.md 3.4). That axis carries the only effect either model has shown
# beyond noise -- markov_report, terra 0.731 [0.614,0.871] p=0.0033 -- so it is the one
# that must not cross a contention boundary. l2_redesign is here too because CLAUDE.md
# 8.10 requires an L2 arm to have its own control on the same GPU.
#
# L2: --enable-l2 --redesign-l2 ONLY. The shipped gate is deliberately NOT run: its
# promotion count is not reproducible (9 / 4 / 0 / 6 / 11 standing rules on byte-identical
# flags, CLAUDE.md 8.11-8.12) and dedup 0.80 is the one change with a measured effect.
#
# luna = azure/openai/gpt-5.6-luna: a REASONING model on the billing endpoint, context
# window 1,050,000 (same as terra). Probed before launch: 1.7s round-trip vs terra 7.0s.
#
# tag          | context-mode         | extra flags
-              | truncation           |
markov         | markov_report        |
folding        | folding              |
selective_r5   | selective_retention  |
compress       | compress_trigger     | --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
l2_redesign    | truncation           | --enable-l2 --redesign-l2
