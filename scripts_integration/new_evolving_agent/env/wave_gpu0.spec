# GPU 0 -- governance axis (context held at truncation) + folding
#
#   bash scripts_integration/new_evolving_agent/env/launch_wave.sh 0 \
#        scripts_integration/new_evolving_agent/env/wave_gpu0.spec dry-run
#
# Merge runs at similarity 0.8 (the code default), tagged merge_sim08 to line up
# with the three existing base_agent_gpt_oss_120b_merge_sim08_itr30_GH200 reps.
# CLAUDE.md open item 1 prefers 0.85; 0.8 is the deliberate choice here.
#
# tag          | context-mode | extra flags
-              | truncation   |
merge_sim08    | truncation   | --skill-merging --skill-merge-similarity 0.8
deletion       | truncation   | --skill-deletion
refinement     | truncation   | --enable-skill-refinement
folding        | folding      |
