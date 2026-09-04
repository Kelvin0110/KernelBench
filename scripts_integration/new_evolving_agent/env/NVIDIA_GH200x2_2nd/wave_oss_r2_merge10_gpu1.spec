# gpt-oss-120b r2 merge-threshold sweep, top end: similarity 1.0 == NEVER MERGE.
# --skill-merging still uncaps the extractor candidate catalog (gen3_stages.py:889),
# so this is "catalog uncapped, merging off" -- the control that separates the
# open-item-6 catalog confound from merging itself. Read it against merge_sim08/085/095
# (all on GPU1, same baseline), NOT against the truncation control on GPU0.
# Distinct _a/_b/_c tags are REQUIRED: identical tags collide into one run dir.
merge_sim10_a | truncation | --skill-merging --skill-merge-similarity 1.0
merge_sim10_b | truncation | --skill-merging --skill-merge-similarity 1.0
merge_sim10_c | truncation | --skill-merging --skill-merge-similarity 1.0
