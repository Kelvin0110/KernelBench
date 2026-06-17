# Run Docker Integration with uv

Run these commands from the repository root.

## Notes

- `scripts_integration/docker/docker_batch_run.py` launches containers with the host UID/GID, so mounted results are created with your user permissions instead of root.
- Use `build_image=False` only after the Docker image has already been built once.
- The new batch runner supports limiting work with `max_problems=<N>` or a specific `problem_ids="1,3,5"` list.

## Latest batch commands

### GPT-OSS 120B, step 5

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_1_inte_gpt_oss_120b_step5 \
  level=1 \
  num_workers=4 \
  gpus="0,1" \
  steps=5 \
  hours=0.5 \
  build_image=False \
  > docker_level_1_inte_gpt_oss_120b_step5.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_2_inte_gpt_oss_120b_step5 \
  level=2 \
  num_workers=2 \
  gpus="0,1" \
  steps=5 \
  hours=0.5 \
  build_image=False \
  > docker_level_2_inte_gpt_oss_120b_step5.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_3_inte_gpt_oss_120b_step5 \
  level=3 \
  num_workers=2 \
  gpus="0,1" \
  steps=5 \
  hours=0.5 \
  build_image=False \
  > docker_level_3_inte_gpt_oss_120b_step5.log 2>&1
```

### Kimi Thinking, step 5

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_1_inte_kimi_thinking_step5 \
  level=1 \
  num_workers=2 \
  gpus="1" \
  steps=5 \
  hours=0.5 \
  code_model=moonshotai/kimi-k2-thinking \
  feedback_model=moonshotai/kimi-k2-thinking \
  build_image=False \
  > docker_level_1_inte_kimi_thinking_step5.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_2_inte_kimi_thinking_step5 \
  level=2 \
  num_workers=2 \
  gpus="3" \
  steps=5 \
  hours=0.5 \
  code_model=moonshotai/kimi-k2-thinking \
  feedback_model=moonshotai/kimi-k2-thinking \
  build_image=False \
  > docker_level_2_inte_kimi_thinking_step5.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_3_inte_kimi_thinking_step5 \
  level=3 \
  num_workers=2 \
  gpus="1,3" \
  steps=5 \
  hours=0.5 \
  code_model=moonshotai/kimi-k2-thinking \
  feedback_model=moonshotai/kimi-k2-thinking \
  build_image=False \
  > docker_level_3_inte_kimi_thinking_step5.log 2>&1
```

### GPT-OSS 120B, step 20 with checkpointing

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_1_inte_gpt_oss_120b_step20 \
  level=1 \
  num_workers=2 \
  gpus="1" \
  steps=20 \
  hours=2.5 \
  checkpoint_distance=2 \
  build_image=False \
  > docker_level_1_inte_gpt_oss_120b_step20.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_2_inte_gpt_oss_120b_step20 \
  level=2 \
  num_workers=2 \
  gpus="3" \
  steps=20 \
  hours=2.5 \
  checkpoint_distance=2 \
  build_image=False \
  > docker_level_2_inte_gpt_oss_120b_step20.log 2>&1
```

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_3_inte_gpt_oss_120b_step20 \
  level=3 \
  num_workers=2 \
  gpus="1,3" \
  steps=20 \
  hours=2.5 \
  checkpoint_distance=2 \
  build_image=False \
  > docker_level_3_inte_gpt_oss_120b_step20.log 2>&1
```

## Subset examples

Run only the first 10 problems for a level:

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_1_subset_10 \
  level=1 \
  num_workers=2 \
  gpus="0,1" \
  max_problems=10 \
  steps=5 \
  hours=0.5 \
  build_image=False \
  > docker_level_1_subset_10.log 2>&1
```

Run specific problem IDs:

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=docker_level_1_selected \
  level=1 \
  num_workers=2 \
  gpus="0,1" \
  problem_ids="1,3,5" \
  steps=5 \
  hours=0.5 \
  build_image=False \
  > docker_level_1_selected.log 2>&1
```

### Run a mixed-level subset from a CSV file:

```bash
nohup uv run python scripts_integration/docker/docker_batch_run.py \
  run_name=aide_subset_gpt_oss_120b_step50 \
  level=1 \
  num_workers=2 \
  gpus="1,2" \
  subset_csv=subset_selection/selected_problems_50.csv \
  steps=50 \
  hours=6.0 \
  build_image=False \
  checkpoint_distance=1 \
  gpu_memory_fraction=0.0 \
  >> aide_subset_gpt_oss_120b_step50.log 2>&1
```

> Remark: gpu_memory_fraction set to be non 0 since initialize CUDA inside parent process leads to duplication of GPU memory usage. Hence, gpu reserver cannot be used in the aide setting.

# Merge and update the results run log
## Preview first
```bash
python scripts_integration/docker/update_run_from_source.py \
  --target-run aide_subset_gpt_oss_120b_step40_new_problem_set \
  --source-run aide_subset_gpt_oss_120b_step40_L1P58 \
  --source-problem-id 58 \
  --target-problem-id 38 \
  --dry-run
```

## Apply
```bash
python scripts_integration/docker/update_run_from_source.py \
  --target-run aide_subset_gpt_oss_120b_step40_new_problem_set \
  --source-run aide_subset_gpt_oss_120b_step40_L1P58 \
  --source-problem-id 58 \
  --target-problem-id 38
```