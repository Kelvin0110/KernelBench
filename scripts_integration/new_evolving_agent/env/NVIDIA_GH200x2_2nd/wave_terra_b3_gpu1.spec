# gpt-5.6-terra -- MERGE-THRESHOLD SWEEP, GPU 1 half. Companion: wave_terra_b3_gpu0.spec
#
# 12 NEW arms across both GPUs, launched ALONGSIDE the 12 resumed r4 arms, so each
# GPU carries 6 resumed + 6 new = 12 arms. That is the documented target for this
# host (CLAUDE.md 3.4: "Target 12 arms/GPU at SLOTS=3 with KB_EVAL_MEM_GATE_FACTOR=7").
#
# WHAT THIS FINALLY ANSWERS -- CLAUDE.md open item 1, the merge threshold. Until now
# the evidence was: code default 0.8, an earlier "validated" 0.85 with no closed pair,
# merge_sim08 at n=2 (flat null, 1.033), and merge_sim09 / merge_sim07 at n=1 each,
# which the file itself says are NOT mutually comparable. These 12 arms plus the
# resumed r4 merge arms give a full sweep with replicates at every threshold:
#
#   0.70 -> r3 (1) + 2 new  = 3
#   0.75 ->            3 new = 3
#   0.80 -> r2, r3, r4       = 3   (already have these)
#   0.85 ->            3 new = 3
#   0.90 -> r2, r4 (2) + 1 new = 3
#   0.95 ->            3 new = 3
#
# At n=3 per threshold the single-contrast 95% floor is ~x1.15 (pooled log-SD 0.0759),
# so a threshold effect has to be >15% to show -- state that when reporting, and use
# the n-vs-n cell test, never the per-problem CI alone (it treats PROBLEMS as the
# replication unit and overstates cell confidence).
#
# CONFOUND THAT DOES NOT CANCEL HERE: --skill-merging also UNCAPS the extractor
# candidate set (open item 6 -- gen3_stages.py:889 passes skill_deletion=
# enable_skill_governance = deletion OR merging, and read_l1_extractor_catalog then
# returns the FULL active catalog instead of the last 50). EVERY arm in this sweep has
# merging on, so the confound is CONSTANT ACROSS THE SWEEP and cancels in
# threshold-vs-threshold contrasts. It does NOT cancel against the truncation control,
# so report "merge@X vs merge@Y" cleanly but "merge vs control" as rule + catalog size.
#
# Distinct tags are mandatory: run names are minute-stamped, so two arms sharing a tag
# resolve to ONE directory and silently interleave (CLAUDE.md 3.2), and launch_wave.sh's
# duplicate guard only checks WITHIN one spec file -- it would not catch a collision
# across these two specs or against the resumed r4 arms. Hence the r5 prefix + _a/_b/_c.
#
# Launch:
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=gpt-5.6-terra MAX_ARMS_PER_GPU=12 RESULTS_ROOT=runs_evolving/gpt-5.6-terra/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra_r5 bash $HW/launch_wave.sh 1 $HW/wave_terra_b3_gpu1.spec
#
# tag              | context-mode | extra flags
merge_sim07_a      | truncation   | --skill-merging --skill-merge-similarity 0.7
merge_sim07_b      | truncation   | --skill-merging --skill-merge-similarity 0.7
merge_sim075_c     | truncation   | --skill-merging --skill-merge-similarity 0.75
merge_sim085_c     | truncation   | --skill-merging --skill-merge-similarity 0.85
merge_sim095_c     | truncation   | --skill-merging --skill-merge-similarity 0.95
merge_sim09_a      | truncation   | --skill-merging --skill-merge-similarity 0.9
