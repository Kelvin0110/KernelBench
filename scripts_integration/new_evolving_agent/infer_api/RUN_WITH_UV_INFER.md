# Run KernelBench Evolving Agent on inference-api.nvidia.com (uv)

Companion to [RUN_WITH_UV_CONTEXT.md](../new_evolving_agent/RUN_WITH_UV_CONTEXT.md)
for runs using the **`inference-api.nvidia.com`** endpoint.

This file mirrors that guide's structure: one section per context-management mode,
each with dry-run → smoke → full-50 → resume commands using `gpt-5.6-terra`.

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

## CLI flags vs env vars

Endpoint and model roles can be set via CLI flags (preferred) or `.env`. CLI flags
override env vars; per-role flags override `--model`.

| Purpose | CLI flag | Env var |
|---------|----------|---------|
| Endpoint | `--nvidia-endpoint inference` | `NVIDIA_ENDPOINT=inference` |
| All four model roles at once | `--model gpt-5.6-terra` | *(sets all four below)* |
| Coder model (overrides `--model`) | `--coder-model <id>` | `NVIDIA_CODER_MODEL=<id>` |
| Summarizer model (overrides `--model`) | `--summarizer-model <id>` | `NVIDIA_SUMMARIZER_MODEL=<id>` |
| Extractor model (overrides `--model`) | `--extractor-model <id>` | `NVIDIA_EXTRACTOR_MODEL=<id>` |
| Action-selector model (overrides `--model`) | `--action-selector-model <id>` | `NVIDIA_ACTION_SELECTOR_MODEL=<id>` |
| API key (inference) | *(env var only)* | `NVIDIA_INF_API_KEY=<key>` |

> **Key resolution:** `NVIDIA_INF_API_KEY` → `NVIDIA_API_KEY` → error.
> **Model alias resolution:** `gpt-5.6-terra` → `azure/openai/gpt-5.6-terra` via
> `NVIDIA_INF_MODEL_ALIASES` in `Self-Evolving-Agent/evolving_common/llm_client.py`.
>
> **⚠ Embedding always uses the integrate endpoint.**
> `embed_texts_nvidia` (L1 skill-merge embedding) is pinned to
> `integrate.api.nvidia.com` and keyed by `NVIDIA_API_KEY` regardless of
> `--nvidia-endpoint`. Set `NVIDIA_API_KEY` in `.env` even when all chat calls
> are routed to the inference endpoint.

The resolved endpoint and model values are written to `run_summary.json` and
checked on resume — mismatched flags abort unless `--allow-resume-config-mismatch`
is passed.

Minimum `.env` required for all modes below:

```bash
NVIDIA_INF_API_KEY=<your inference API key>
```

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

## Default / Truncation mode

Keeps only the latest N raw L0 rounds in prompts (default context-management mode).
No `--context-management` flag required.

Check `runs_evolving/<run_name>_*/run_summary.json` for `"nvidia_endpoint": "inference"`,
`"model": "gpt-5.6-terra"`, and `"context_management": "truncation"`.


### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_56_terra_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --no-skill-deletion \
  >> base_agent_gpt_56_terra_itr30_Aug_1.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --no-skill-deletion \
  --hardware NVIDIA_GH200x2 \
  >> base_agent_gpt_oss_120b_itr30_GH200_Aug_3.log 2>&1 &
```

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint`, `model`, per-role model IDs, `context_management: "truncation"`
- `workspaces/level_*_problem_*/chat_history.jsonl` — all LLM turns

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_gpt_56_terra_truncation_itr10_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 10 \
  --start-problem 11 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --no-skill-deletion \
  >> base_agent_gpt_56_terra_truncation_itr10_Jul_31.log 2>&1 &
```

---

## Markov report mode

Each iteration: **goal + evolving report + latest L0 only**. After eval, a dedicated
rewriter LLM updates `evolving_report.md`. Effective for long runs where a compact
evolving summary matters more than full L0 history.


### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_56_terra_markov_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --evolving-report-max-tokens 65536 \
  --no-skill-deletion \
  >> base_agent_gpt_56_terra_markov_itr30_Aug_1.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_markov_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --context-management markov_report \
  --evolving-report-max-tokens 65536 \
  --no-skill-deletion \
  --hardware NVIDIA_GH200x2 \
  >> base_agent_gpt_oss_120b_markov_itr30_GH200_Aug_3.log 2>&1 &
```

Optional rewriter knobs (defaults are usually fine):

- `--evolving-report-max-tokens` (default `1536`)
- `--evolving-report-timeout-sec` (default `90`)

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint`, `model`, `context_management: "markov_report"`,
  `evolving_report_max_tokens`, `evolving_report_timeout_sec`
- `workspaces/level_*_problem_*/evolving_report.md` — current evolving report per problem
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns incl. `phase=evolving_report`

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_terra_markov_itr10_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 10 \
  --start-problem 11 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management markov_report \
  --no-skill-deletion \
  >> base_agent_terra_markov_itr10_resume.log 2>&1 &
```

---

## Selective retention mode

Each iteration: **goal + milestone memory (selected past rounds, full detail) + latest L0
only**. Milestones are labeled per round (rules + additive LLM judge) and packed under
90% of the model context window. Non-milestone rounds are omitted from the prompt but
never deleted from disk.

With `gpt-5.6-terra`'s 1,050,000-token context window, the packing budget is ~945K
tokens — significantly larger than with `gpt-oss-120b`.

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_selective_retention.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_selective_retention_context_management -q
```


### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_terra_selective_itr10 \
  --max-iterations 10 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management selective_retention \
  --no-skill-deletion \
  >> base_agent_terra_selective_itr10.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_selective_itr10 \
  --max-iterations 10 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --context-management selective_retention \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_selective_itr10_Aug_1.log 2>&1 &
```

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint`, `model`, `context_management: "selective_retention"`
- `workspaces/level_*_problem_*/l0_milestones.json` — per-problem milestone index
  (`round_id`, `reasons`, `source`, `is_new_best`)
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns incl. `phase=milestone_judge`

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_terra_selective_itr10_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 10 \
  --start-problem 11 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management selective_retention \
  --no-skill-deletion \
  >> base_agent_terra_selective_itr10_resume.log 2>&1 &
```

---

## Folding mode

Each iteration keeps **recent N full L0 rounds** (default `N=15`: code, terminal,
reasoning) plus an **archived summary catalog** for older rounds. Before the coder,
an **unfold preflight** may request full detail for a few archived `round_id`s
(default: enabled; up to 2 attempts × 3 rounds). Archived summaries are packed
**newest-first** under ~90% of the context window (disk L0 unchanged).

With `gpt-5.6-terra`, the ~945K packing budget means significantly more archived
summaries fit in the prompt compared to `gpt-oss-120b`.

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_l0_rounds.py -q
```

### Dry run

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name terra_folding_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management folding \
  --no-skill-deletion \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

Check `runs_evolving/<run_name>_*/run_summary.json` for
`context_management: "folding"` and `skill_deletion: false`.

### Small real CUDA run (smoke)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_terra_folding_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management folding \
  --no-skill-deletion \
  >> base_agent_terra_folding_smoke.log 2>&1 &
```

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_terra_folding_itr30 \
  --max-problems 50 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management folding \
  --no-skill-deletion \
  >> base_agent_terra_folding_itr30.log 2>&1 &
```

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint`, `model`, `context_management: "folding"`
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns incl. `phase=preflight`
  (unfold) and `phase=l0_round_summary` (per-round archive summaries)
- Per-round `round_summary` lives on L0 in the problem workspace / snapshots (no
  separate milestones file)

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_terra_folding_itr30_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem 11 \
  --nvidia-endpoint inference \
  --model gpt-5.6-terra \
  --context-management folding \
  --no-skill-deletion \
  >> base_agent_terra_folding_itr30_resume.log 2>&1 &
```

Keep `--context-management folding` (and `--no-skill-deletion` if the original run
used it). Optional range resume: add `--end-problem M` as in
[RUN_WITH_UV.md](../new_evolving_agent/RUN_WITH_UV.md) §6.
