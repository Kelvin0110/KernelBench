# Command for background run
## Start process:
nohup ./run_evals.sh > eval.log 2>&1 &

## Check real time output:
tail -f eval.log

## Kill process:
pkill -f run_evals.sh
pkill -f eval_from_generations.py

## Check process:
ps aux | grep -E "run_evals.sh|eval_from_generations.py"


# Commands for generating samples
## Gpt oss 120b
```bash
uv run python scripts/generate_samples.py     run_name=test_hf_level_1_gpt_oss_120b     dataset_src=huggingface     level=1     server_type=nvidia     model_name=nvidia_nim/nvdev/openai/gpt-oss-120b     is_reasoning_model=True     verbose=True     num_workers=16 reasoning_effort=medium max_tokens=16384

uv run python scripts/generate_samples.py     run_name=test_hf_level_2_gpt_oss_120b     dataset_src=huggingface     level=2     server_type=nvidia     model_name=nvidia_nim/nvdev/openai/gpt-oss-120b     is_reasoning_model=True     verbose=True     num_workers=16 reasoning_effort=medium max_tokens=16384
uv run python scripts/generate_samples.py     run_name=test_hf_level_3_gpt_oss_120b     dataset_src=huggingface     level=3     server_type=nvidia     model_name=nvidia_nim/nvdev/openai/gpt-oss-120b     is_reasoning_model=True     verbose=True     num_workers=16 reasoning_effort=medium  max_tokens=16384
```

## Minimax m2
```bash
uv run python scripts/generate_samples.py     run_name=test_hf_level_1_minimax_m2     dataset_src=huggingface     level=1     server_type=nvidia     model_name=nvidia_nim/minimaxai/minimax-m2     is_reasoning_model=True     verbose=True     num_workers=16 max_tokens=16384

uv run python scripts/generate_samples.py     run_name=test_hf_level_2_minimax_m2     dataset_src=huggingface     level=2     server_type=nvidia     model_name=nvidia_nim/minimaxai/minimax-m2     is_reasoning_model=True     verbose=True     num_workers=16 max_tokens=16384

uv run python scripts/generate_samples.py     run_name=test_hf_level_3_minimax_m2     dataset_src=huggingface     level=3     server_type=nvidia     model_name=nvidia_nim/minimaxai/minimax-m2     is_reasoning_model=True     verbose=True     num_workers=16  max_tokens=16384
```

## DeepSeek R1 0528

```bash
uv run python scripts/generate_samples.py     run_name=test_hf_level_1_dsr1_0528    dataset_src=huggingface     level=1     server_type=nvidia     model_name=nvidia_nim/deepseek-ai/deepseek-r1-0528     is_reasoning_model=True     verbose=True     num_workers=16  max_tokens=16384

uv run python scripts/generate_samples.py     run_name=test_hf_level_2_dsr1_0528     dataset_src=huggingface     level=2     server_type=nvidia     model_name=nvidia_nim/deepseek-ai/deepseek-r1-0528    is_reasoning_model=True     verbose=True     num_workers=16  max_tokens=16384

uv run python scripts/generate_samples.py     run_name=test_hf_level_3_dsr1_0528     dataset_src=huggingface     level=3     server_type=nvidia     model_name=nvidia_nim/deepseek-ai/deepseek-r1-0528     is_reasoning_model=True     verbose=True     num_workers=16  max_tokens=16384

——————
CUDA_VISIBLE_DEVICES=2,3 nohup uv run python scripts/eval_from_generations.py   run_name=test_hf_level_1_dsr1_0528_2nd   dataset_src=local   level=1   num_gpu_devices=2 build_cache=False  timeout=300  gpu_arch='["Ampere"]'   verbose=True  num_cpu_workers=5 > eval_dsr1_0528_2nd_level1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup uv run python scripts/eval_from_generations.py   run_name=test_hf_level_2_dsr1_0528_2nd   dataset_src=local   level=2  num_gpu_devices=2  build_cache=False  timeout=600  gpu_arch='["Ampere"]'   verbose=True  num_cpu_workers=5 > eval_dsr1_0528_2nd_level2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup uv run python scripts/eval_from_generations.py   run_name=test_hf_level_3_dsr1_0528_2nd   dataset_src=local   level=3  num_gpu_devices=2  build_cache=False  timeout=600  gpu_arch='["Ampere"]'   verbose=True  num_cpu_workers=5 > eval_dsr1_0528_2nd_level3.log 2>&1 &

——————
uv run python scripts/benchmark_eval_analysis.py run_name=test_hf_level_1_dsr1_0528 level=1 hardware=SONG_CPU2_A6000x2 baseline=baseline_time_torch

uv run python scripts/benchmark_eval_analysis.py run_name=test_hf_level_2_dsr1_0528 level=2 hardware=SONG_CPU2_A6000x2 baseline=baseline_time_torch

uv run python scripts/benchmark_eval_analysis.py run_name=test_hf_level_3_dsr1_0528 level=3 hardware=SONG_CPU2_A6000x2 baseline=baseline_time_torch
```

CUDA_VISIBLE_DEVICES=0,1 nohup uv run python scripts/eval_from_generations.py   run_name=test_hf_level_1_kimi_thinking   dataset_src=local   level=1   num_gpu_devices=2 build_cache=False  timeout=300  gpu_arch='["Ampere"]'   verbose=True  num_cpu_workers=5 > eval_kimi_thinking_ori_level1.log 2>&1 &