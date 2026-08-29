# gpt-5.6-terra -- ROUND 3 (replicate r4), GPU 1: the L1 GOVERNANCE AXIS + the L2 TIER.
# Companion: wave_terra_b2_gpu0.spec (GPU 0 = the L0 axis + the single control).
#
# 12 arms TOTAL this round across both GPUs, ONE replicate, split 6/6. There is NO
# truncation control on this GPU -- the only control is on GPU 0. That is a
# deliberate, accepted trade (it saves a 62-hour arm); see wave_terra_b2_gpu0.spec
# for the measurement showing the residual cross-GPU bias is negligible under
# SLOTS=3 (batch-1 lock wait: mean 0.20s GPU0 / 0.22s GPU1 at NINE arms/GPU).
# Four of the five cells here are flat nulls at n=2 (deletion 0.958, merge_sim08
# 1.033, refinement 1.039), which is why they are the ones placed across the boundary.
# Arms per GPU are now EQUAL (6 and 6), which removes the unequal-arms-per-GPU term
# of that bias outright -- what remains is only the single-control boundary itself.
#
# All six hold context at truncation, so the governance axis stays separable from
# L0 and the six arms are mutually comparable on this GPU.
#
# THE l2 CELL -- n=3, the first L2 arms ever run on terra. l2_rep1 sits on GPU 0
# beside the truncation control (CLAUDE.md 8.10 wants an L2 batch's control on its
# own GPU; rep1 is the one that gets it). rep2/rep3 are here. THE THREE TAGS MUST
# DIFFER: three arms tagged `l2` would all render to
# base_agent_gpt_5_6_terra_r4_l2_itr30_GH200, and since the run stamp is MINUTE
# resolution they would resolve to ONE directory and silently interleave (3.2).
# launch_wave.sh's duplicate guard only checks WITHIN one spec file, so it would
# not have caught a collision between the GPU-0 and GPU-1 specs.
#
# WHY THREE REPLICATES OF THE DEFAULT RATHER THAN THE 8.9 PROBE ARMS: L2 promotion
# is late-clustering and high-variance -- the single gpt-oss arm promoted 9 rules,
# and whether terra promotes 0, 3 or 15 is completely unmeasured. At n=1 a broken
# tier and an unlucky draw are indistinguishable, so there is nothing for
# l2_extract / l2_cap4 / l2_tasks4 to be measured AGAINST. n=3 gives the tier the
# same replication as every other cell this round and establishes the promotion-
# count spread the probe arms need. Run the 8.9 probes in the NEXT batch.
# Default parameters throughout,
# verified against l2_promotion.py: render=verbatim, min_tasks=3, min_selections=50,
# min_rate=0.70, min_new_bests=0 (disabled), max_entries=0 (unlimited). This is
# byte-identical to the config of the one completed gpt-oss l2 arm, so the two are
# directly comparable as a MODEL contrast on the same tier.
#
# Read CLAUDE.md 8 before analysing it. Three things to expect, all measured on the
# gpt-oss arm and none of them a malfunction:
#   * promotions cluster LATE (5 of 9 after global iteration 1110, i.e. problem ~37
#     of 50) because min_selections=50 needs accumulated selections. 16% of coder
#     calls saw no L2 text at all and the median call saw one rule, so a 50-problem
#     run barely exercises the tier. "L2 is a null" is UNTESTED at this length.
#   * the standing set duplicates: 6 of 9 gpt-oss rules were one idea ("don't write
#     trivial kernels"), 10,523 of 15,176 chars. Cause is compositional -- L1
#     accumulates near-duplicates, --skill-merging exists to collapse them, and this
#     arm runs with merging OFF. Terra's duplication rate is UNMEASURED; this arm
#     measures it.
#   * prompt burden is a STEP function, not a constant: 4.79x control was the gpt-oss
#     TERMINAL value, reached for only the last 10% of calls; the run mean was 2.21x.
#
# ANALYSIS BLOCKER, open item 7 -- aggregate_runs.py has no enable_l2 field and
# compare_runs.py's design_variant_label reads only the context mode plus
# skill_deletion/skill_merging/enable_skill_refinement, so THIS ARM AND THE GPU-0
# CONTROL BOTH RENDER AS DESIGN `truncation` in every CSV and delta table.
# run_summary.json does carry the flag (evolve_kb_batch.py:1771), so it is a small
# extraction fix -- but it MUST land before any report is generated from this wave.
#
# Health checks specific to this arm (neither is covered by the generic ones in 3.5;
# governor.py:1448 swallows every promotion-pass exception into a one-line
# `l2 promotion skipped:` print, so an arm that promotes nothing is indistinguishable
# from a truncation arm in every other artifact):
#   wc -l <run>/l2_promotions.jsonl <run>/l2_standing.jsonl
#   .venv/bin/python -c "import json;print(json.load(open('<run>/run_summary.json'))['l2_standing_count'])"
#
# NOTE the two standing confounds -- report these cells as the rule PLUS its
# side effects, never as the rule alone:
#   * deletion (CLAUDE.md open item 3): --skill-deletion ALSO switches on an
#     LLM-authored pytest admission gate that no other arm has
#     (DEFAULT_L1_SKILL_DELETE_ON_UNIT_TEST_FAIL = True). In the completed gpt-oss
#     deletion arm that gate was the LARGER term: 153 deletions = 92 unit_test_fail
#     + 61 consecutive_unused. Report as "deletion + unit-test admission gate".
#   * both deletion and merge (open item 6): gen3_stages.py:889 passes
#     skill_deletion=enable_skill_governance = (deletion OR merging), and
#     read_l1_extractor_catalog returns the FULL active catalog when that is true
#     instead of the last DEFAULT_L1_EXTRACTOR_CATALOG_MAX = 50. So either flag also
#     uncaps the extractor's candidate set. Report as "rule + catalog size".
#   refinement trips NEITHER of these, so that cell is clean on both dimensions.
#
# merge_sim09 is chosen over merge_sim07 for the third merge slot: it closes
# merge_sim09 to n=2 (r2 + r4), and 0.9 brackets the 0.85 validated operating point
# of open item 1 far more usefully than 0.7 does. merge_sim07 stays at n=1 (r3 only).
# The three merge thresholds are still NOT mutually comparable -- 0.8 is the only one
# with a closed pair on both replicates.
#
# BASELINE = NVIDIA_GH200x2_2nd, resolved by hardware_env.sh from THIS folder's name.
# Pass NO --hardware and NO KB_DEFAULT_HARDWARE override.
#
# Launch (MAX_ARMS_PER_GPU default 6 is enough for 5; do not raise it):
#   HW=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd
#   MODEL=gpt-5.6-terra RESULTS_ROOT=runs_evolving/gpt-5.6-terra/ \
#   RUN_PREFIX=base_agent_gpt_5_6_terra_r4 \
#     bash $HW/launch_wave.sh 1 $HW/wave_terra_b2_gpu1.spec
#
# tag          | context-mode         | extra flags
deletion       | truncation           | --skill-deletion
refinement     | truncation           | --enable-skill-refinement
merge_sim08    | truncation           | --skill-merging --skill-merge-similarity 0.8
merge_sim09    | truncation           | --skill-merging --skill-merge-similarity 0.9
l2_rep2        | truncation           | --enable-l2
l2_rep3        | truncation           | --enable-l2
