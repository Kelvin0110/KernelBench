# terra r4 wave -- HALTED 2026-08-29 15:50Z, resume plan

Cause: gpt-5.6-terra inference endpoint degraded from ~12:00Z 2026-08-29.
762 `coder_call_error: APIConnectionError` in 7,795 iterations (9.8% overall,
80.6% in the final 20 min). Verified UPSTREAM, not ours: with the whole wave
stopped and zero load, probes gave 0/2 ok at concurrency 1, 2/4 at 2, 3/24 at 12.
Not FD exhaustion (arms held 11-29 fds of a 1,048,576 limit).

DO NOT RESUME until an endpoint probe passes cleanly at concurrency 12.

Damage: 219 problems clean, 30 degraded, 214 DESTROYED (>50% of iterations failed).
The four arms that wrote run_summary.json (markov, deletion, refinement, l2_rep2)
are NOT valid 50/50 runs -- their whole Level-3 block is destroyed.

| arm | run dir | problems run | RESUME FROM |
|---|---|---|---|
| compress | base_agent_gpt_5_6_terra_r4_compress_itr30_GH200_2026_08_28_14_31 | 36 | **19** |
| deletion | base_agent_gpt_5_6_terra_r4_deletion_itr30_GH200_2026_08_28_14_32 | 50 | **15** |
| folding | base_agent_gpt_5_6_terra_r4_folding_itr30_GH200_2026_08_28_14_32 | 27 | **17** |
| trunc | base_agent_gpt_5_6_terra_r4_itr30_GH200_2026_08_28_14_29 | 33 | **18** |
| l2_rep1 | base_agent_gpt_5_6_terra_r4_l2_rep1_itr30_GH200_2026_08_28_14_32 | 36 | **18** |
| l2_rep2 | base_agent_gpt_5_6_terra_r4_l2_rep2_itr30_GH200_2026_08_28_14_34 | 50 | **18** |
| l2_rep3 | base_agent_gpt_5_6_terra_r4_l2_rep3_itr30_GH200_2026_08_28_14_34 | 24 | **19** |
| markov | base_agent_gpt_5_6_terra_r4_markov_itr30_GH200_2026_08_28_14_30 | 50 | **17** |
| merge_sim08 | base_agent_gpt_5_6_terra_r4_merge_sim08_itr30_GH200_2026_08_28_14_33 | 49 | **16** |
| merge_sim09 | base_agent_gpt_5_6_terra_r4_merge_sim09_itr30_GH200_2026_08_28_14_34 | 32 | **18** |
| refinement | base_agent_gpt_5_6_terra_r4_refinement_itr30_GH200_2026_08_28_14_33 | 50 | **16** |
| selective_r5 | base_agent_gpt_5_6_terra_r4_selective_r5_itr30_GH200_2026_08_28_14_31 | 26 | **20** |

Resume command per arm (run the EARLIER index first if doing several):

```bash
RESULTS_ROOT=runs_evolving/gpt-5.6-terra/ \
  bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/resume_run.sh \
  <gpu> <run_dir_name> <ctx_mode> <RESUME_FROM> -- <the arm's governance flags>
```

Flags per arm are in env/NVIDIA_GH200x2_2nd/wave_terra_b2_gpu{0,1}.spec.
resume_run.sh defect 3: it passes NO governance flags of its own -- you MUST
pass them after `--` or a deletion/merge/refinement/l2 arm silently resumes as
a plain truncation arm (evolve_kb_batch.py:675 returns [] when run_summary.json
is absent, so nothing catches it).
