# gpt-oss-120b -- SECOND 9-cell wave, GPU 1 half (the L1 governance axis).
# Companion: wave_oss_r2_gpu0.spec (control + L0 axis). Shares GPU 1 with 4 resumed
# terra arms, giving 8 arms on this GPU.
#
# See wave_oss_r2_gpu0.spec for the four protocol seams against the Aug-22 gpt-oss wave
# and for why this is reported as n=1 of a new regime before any pooling is attempted.
#
# STANDING CONFOUNDS on these cells -- report the rule PLUS its side effects:
#  * deletion also enables an LLM-authored pytest admission gate no other arm has
#    (CLAUDE.md open item 3); on the completed gpt-oss deletion arm that gate supplied
#    92 of 153 deletions, i.e. the LARGER term.
#  * deletion AND merge both uncap the extractor candidate set (open item 6):
#    gen3_stages.py:889 passes skill_deletion = deletion OR merging, and
#    read_l1_extractor_catalog then returns the FULL active catalog instead of the last
#    50. Report as "rule + catalog size".
#  * refinement trips NEITHER, so that cell is clean on both dimensions.
#  * l2 at defaults is known-defective on gpt-oss: the completed l2 arm promoted 9
#    standing rules of which 6 are one idea, 69% of the standing text, for a null
#    (geomean 1.347 vs control 1.389). Expect the same shape; the value here is a
#    second observation of promotion COUNT, which is what CLAUDE.md 8.9's probe arms
#    need before they can be designed.
#
# Launch:
#   MODEL=gpt-oss-120b MAX_ARMS_PER_GPU=12 RESULTS_ROOT=runs_evolving/gpt-oss-120b/ \
#   RUN_PREFIX=base_agent_gpt_oss_120b_r2 bash $HW/launch_wave.sh 1 $HW/wave_oss_r2_gpu1.spec
#
# tag          | context-mode | extra flags
deletion       | truncation   | --skill-deletion
merge_sim08    | truncation   | --skill-merging --skill-merge-similarity 0.8
l2             | truncation   | --enable-l2
refinement     | truncation   | --enable-skill-refinement
