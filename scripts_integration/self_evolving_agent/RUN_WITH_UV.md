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

# Provider configuration (new)

The Self-Evolving-Agent now uses an explicit provider registry. Configure providers and role mappings via environment JSON variables. Example entries (add to your `.env` or export in the shell):

```bash
export SEA_PROVIDERS='[{"id": "openai", "provider": "openai", "api_key_env": "OPENAI_API_KEY", "defaults": {"model": "gpt-4o"}} , {"id": "nvidia", "provider": "nvidia", "api_key_env": "NVIDIA_API_KEY", "defaults": {"model": "nemotron-ultra"}}]'

export SEA_ROLE_CODER='{"role":"coder","provider_id":"nvidia","model":"nemotron-ultra","max_retries":3}'
export SEA_ROLE_SUMMARIZER='{"role":"summarizer","provider_id":"openai","model":"gpt-4o","max_retries":2}'
```

Notes:
- `SEA_PROVIDERS` declares provider entries (id, provider type, which env var holds the API key, and optional defaults).
- `SEA_ROLE_<NAME>` maps a semantic role to a specific `provider_id` and model. This avoids auto-parsing model strings in scripts.

When `uv run` starts, ensure the env vars are loaded (for example by sourcing your `.env` file) so the ProviderRegistry can resolve API keys.

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
