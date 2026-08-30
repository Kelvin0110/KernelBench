# qwen3.6-27b -- GPU 1 half (5 arms). Companion: wave_qwen_gpu0.spec (8 arms + control).
# Shares GPU 1 with 4 RESUMED terra arms (l2_rep2, merge_sim07_a/b, merge_sim09_a),
# giving 9 arms on this GPU.
#
# refinement is here rather than beside the control because it is the cell least likely
# to need a contention-matched comparison: it closed as a flat null on terra (1.039) and
# it is the ONE governance cell that trips neither of CLAUDE.md's standing confounds
# (open item 3's unit-test admission gate, open item 6's uncapped extractor catalog).
#
# The four merge arms give qwen merge thresholds 0.7 and 0.9 at n=2 each; with
# merge_sim08 on GPU 0 that is a 3-point sweep. Every merge arm has merging on, so the
# uncapped-extractor-catalog confound is CONSTANT across them and cancels in
# threshold-vs-threshold contrasts -- but NOT against the truncation control.
#
# See wave_qwen_gpu0.spec for the qwen model notes (window, max_tokens validation,
# pinned sampling profile) and for why this wave is a screen rather than a test.
#
# Launch:
#   MODEL=qwen3.6-27b MAX_ARMS_PER_GPU=12 RESULTS_ROOT=runs_evolving/qwen3.6-27b/ \
#   RUN_PREFIX=base_agent_qwen36_27b bash $HW/launch_wave.sh 1 $HW/wave_qwen_gpu1.spec
#
# tag             | context-mode | extra flags
refinement        | truncation   | --enable-skill-refinement
merge_sim07_a     | truncation   | --skill-merging --skill-merge-similarity 0.7
merge_sim07_b     | truncation   | --skill-merging --skill-merge-similarity 0.7
merge_sim09_a     | truncation   | --skill-merging --skill-merge-similarity 0.9
merge_sim09_b     | truncation   | --skill-merging --skill-merge-similarity 0.9
