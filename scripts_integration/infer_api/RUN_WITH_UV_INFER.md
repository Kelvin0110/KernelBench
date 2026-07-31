# Run KernelBench Evolving Agent on inference-api.nvidia.com (uv)

Companion to [RUN_WITH_UV_CONTEXT.md](../new_evolving_agent/RUN_WITH_UV_CONTEXT.md)
for runs using the **`inference-api.nvidia.com`** endpoint.

All context-management modes (`truncation`, `folding`, `markov_report`,
`selective_retention`) work identically. This file covers the env-var wiring and
CLI invocations needed to route calls to the inference endpoint with `gpt-5.6-terra`.

---

## Model specs

| Property | Value |
|----------|-------|
| Alias | `gpt-5.6-terra` |
| Full model ID (inference endpoint) | `azure/openai/gpt-5.6-terra` |
| Context window | 1,050,000 tokens |
| Max output tokens | 128,000 tokens |
| Endpoint | `https://inference-api.nvidia.com/v1` |

The 1,050,000-token context window is registered in
`Self-Evolving-Agent/evolving_common/context_management.py`; the 90% packing cap
(`CONTEXT_PACK_RATIO`) applies to that figure (~945K usable tokens for
folding / selective-retention prompt packing).

---

## Required environment variables

Set these in `.env` at the repo root (copied from `.env.example`):

```bash
# Route all LLM calls to inference-api.nvidia.com
NVIDIA_ENDPOINT=inference

# API key for the inference endpoint.
# Falls back to NVIDIA_API_KEY if NVIDIA_INF_API_KEY is not set.
NVIDIA_INF_API_KEY=<your inference API key>

# Point all four model roles at gpt-5.6-terra
NVIDIA_CODER_MODEL=gpt-5.6-terra
NVIDIA_SUMMARIZER_MODEL=gpt-5.6-terra
NVIDIA_EXTRACTOR_MODEL=gpt-5.6-terra
NVIDIA_ACTION_SELECTOR_MODEL=gpt-5.6-terra
```

> **Key resolution order for the inference endpoint:**
> `NVIDIA_INF_API_KEY` → `NVIDIA_API_KEY` → error.
>
> **Model ID resolution:** the alias `gpt-5.6-terra` maps to
> `azure/openai/gpt-5.6-terra` via `NVIDIA_INF_MODEL_ALIASES` in
> `Self-Evolving-Agent/evolving_common/llm_client.py`. Full IDs are accepted
> as-is if you prefer to set them directly in the env vars above.

Endpoint and model roles can be set via **CLI flags** (preferred) or env vars.
CLI flags override env vars; env vars set in `.env` are the fallback.
The batch script raises a clear error at startup if the required API key is missing.

| Purpose | CLI flag | Env var |
|---------|----------|---------|
| Endpoint | `--nvidia-endpoint inference` | `NVIDIA_ENDPOINT=inference` |
| All four model roles at once | `--model gpt-5.6-terra` | *(sets all four below)* |
| Coder model (overrides `--model`) | `--coder-model gpt-5.6-terra` | `NVIDIA_CODER_MODEL=gpt-5.6-terra` |
| Summarizer model (overrides `--model`) | `--summarizer-model gpt-5.6-terra` | `NVIDIA_SUMMARIZER_MODEL=gpt-5.6-terra` |
| Extractor model (overrides `--model`) | `--extractor-model gpt-5.6-terra` | `NVIDIA_EXTRACTOR_MODEL=gpt-5.6-terra` |
| Action-selector model (overrides `--model`) | `--action-selector-model gpt-5.6-terra` | `NVIDIA_ACTION_SELECTOR_MODEL=gpt-5.6-terra` |
| API key (inference) | *(env var only)* | `NVIDIA_INF_API_KEY=<key>` |

The resolved endpoint and model values are written to `run_summary.json` and
checked on resume — mismatched flags abort unless `--allow-resume-config-mismatch`
is passed.

---

## Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_inference_endpoint.py -q
```

Live smoke test (requires `NVIDIA_INF_API_KEY`):

```bash
uv run python Self-Evolving-Agent/tests/test_inference_endpoint.py inference
```

---

## Dry run (CLI plumbing only)

Validates CLI wiring and CSV parsing without GPU eval or real LLM calls.
`NVIDIA_INF_API_KEY` must be set in `.env`; the dry-run skips the actual key check.

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name terra_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --no-skill-deletion \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

Check `runs_evolving/<run_name>_*/run_summary.json` — it will contain
`"nvidia_endpoint": "inference"`, `"model": "gpt-5.6-terra"`, and the resolved
per-role model IDs.

---

## Small real CUDA run (smoke)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_terra_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --no-skill-deletion \
  >> base_agent_terra_smoke.log 2>&1 &
```

---

## Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_terra_itr10 \
  --max-iterations 10 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --evolving-report-max-tokens 65536 \
  >> base_agent_terra_itr10.log 2>&1 &
```

Substitute `--context-management` as needed (`truncation`, `folding`,
`markov_report`, `selective_retention`).

To use different models per role, add individual flags — they override `--model`:

```bash
  --model gpt-5.6-terra \
  --extractor-model gpt-oss-120b   # extractor uses a different model
```

---

## After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint`, `model`, `coder_model`, `summarizer_model`,
  `extractor_model`, `action_selector_model`, `context_management`, skill-governance flags
- `workspaces/level_*_problem_*/evolving_report.md` — evolving report (markov_report mode)
- `workspaces/level_*_problem_*/l0_milestones.json` — milestone index (selective_retention mode)
- `workspaces/level_*_problem_*/chat_history.jsonl` — all LLM turns

---

## Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_terra_itr10_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 10 \
  --start-problem 11 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --no-skill-deletion \
  >> base_agent_terra_itr10_resume.log 2>&1 &
```

Keep `--nvidia-endpoint`, `--model` (and any per-role overrides), `--context-management`,
and `--no-skill-deletion` identical to the original run — the batch script compares
them against `run_summary.json` and aborts on mismatch
(override with `--allow-resume-config-mismatch`).
Optional range resume: add `--end-problem M`.
