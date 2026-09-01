# gpt-oss-120b merge-threshold sweep, round 1 -- GPU 0 half (1 arm).
# Companion: wave_oss_r2_merge_gpu1.spec (2 arms). Split 1/2 so the two GPUs land
# EVEN at 9 arms each (GPU0 was 8, GPU1 was 7).
#
# 0.95 is the far end of the sweep and the least informative neighbour of the existing
# merge_sim08 (0.8), so it is the one that pays the GPU-boundary cost. 0.75 and 0.85 --
# which bracket 0.8 -- stay on GPU 1 with it, giving three adjacent thresholds a
# zero-GPU-term contrast.
#
# tag          | context-mode | extra flags
merge_sim095_a | truncation   | --skill-merging --skill-merge-similarity 0.95
