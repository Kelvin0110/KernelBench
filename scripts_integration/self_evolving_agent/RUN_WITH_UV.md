# Run Self-Evolving-Agent on KernelBench (Subset Driven)

This guide runs the script set under `scripts_integration/self_evolving_agent/` and writes KernelBench-style level-first metrics.

## 1) Prerequisites

From repository root:

```bash
uv sync --extra gpu
```

For real LLM execution, ensure environment variables are set for NVIDIA integration used by `Self-Evolving-Agent/llm_client.py`:

```bash
export NVIDIA_API_KEY=...
export NVIDIA_CODER_MODEL=nemotron-ultra
export NVIDIA_SUMMARIZER_MODEL=gpt-oss-120b
```

## 2) Dry-run smoke test (no external LLM calls)

```bash
uv run python scripts_integration/self_evolving_agent/run_batch.py \
  --subset-csv subset_selection/selected_problems_50.csv \
  --output-path runs/sea_integration_run/eval_results.json \
  --backend cuda \
  --precision fp32 \
  --max-steps 2 \
  --dry-run
```

This command validates orchestration and output schema.

## 3) Real run with LLM calls

```bash
uv run python scripts_integration/self_evolving_agent/run_batch.py \
  --subset-csv subset_selection/selected_problems_50.csv \
  --output-path runs/sea_integration_run/eval_results.json \
  --dataset-source local \
  --prompt-option one_shot \
  --backend cuda \
  --precision fp32 \
  --max-steps 5
```

## 4) Output format

Results are written to:

- `runs/sea_integration_run/eval_results.json`

JSON schema is level-first:

```json
{
  "1": {
    "100": [
      {
        "sample_id": 0,
        "compiled": true,
        "correctness": false,
        "metadata": {
          "source": "self_evolving_agent",
          "level": 1,
          "problem_id": 100,
          "best_speedup": 0.0,
          "backend": "cuda",
          "precision": "fp32",
          "iterations_run": 2,
          "error": "..."
        },
        "runtime": 25.1,
        "runtime_stats": {
          "mean": 25.1,
          "std": 26.1,
          "min": 22.2,
          "max": 285.0,
          "num_trials": 100,
          "hardware": "NVIDIA RTX A6000",
          "device": "cuda:0"
        }
      }
    ]
  }
}
```
