# Run KernelBench Evolving Agent — Skill Management on inference-api.nvidia.com (uv)

Companion to [RUN_WITH_UV.md](../RUN_WITH_UV.md) §4.1–§4.2 for **skill management**
runs using the **`inference-api.nvidia.com`** endpoint.

For context-management modes on this endpoint, see
[RUN_WITH_UV_INFER.md](RUN_WITH_UV_INFER.md).

This file records **full-run commands only** (no dry-run / smoke / unit-test
commands). Each section mirrors the skill-governance combinations from
`RUN_WITH_UV.md`, with `--nvidia-endpoint inference` and `--model gpt-oss-120b`.

---

## Model specs

| Property | Value |
|----------|-------|
| Alias | `gpt-oss-120b` |
| Full model ID (inference endpoint) | `nvidia/openai/gpt-oss-120b` |
| Context window | 128,000 tokens |
| Endpoint | `https://inference-api.nvidia.com/v1` |

The 128,000-token context window is registered in
`Self-Evolving-Agent/evolving_common/context_management.py`.

---

## CLI flags vs env vars

Endpoint and model roles can be set via CLI flags (preferred) or `.env`. CLI flags
override env vars; per-role flags override `--model`.

| Purpose | CLI flag | Env var |
|---------|----------|---------|
| Endpoint | `--nvidia-endpoint inference` | `NVIDIA_ENDPOINT=inference` |
| All four model roles at once | `--model gpt-oss-120b` | *(sets all four below)* |
| Coder model (overrides `--model`) | `--coder-model <id>` | `NVIDIA_CODER_MODEL=<id>` |
| Summarizer model (overrides `--model`) | `--summarizer-model <id>` | `NVIDIA_SUMMARIZER_MODEL=<id>` |
| Extractor model (overrides `--model`) | `--extractor-model <id>` | `NVIDIA_EXTRACTOR_MODEL=<id>` |
| Action-selector model (overrides `--model`) | `--action-selector-model <id>` | `NVIDIA_ACTION_SELECTOR_MODEL=<id>` |
| API key (inference) | *(env var only)* | `NVIDIA_INF_API_KEY=<key>` |

> **Key resolution:** `NVIDIA_INF_API_KEY` → `NVIDIA_API_KEY` → error.
> **Model alias resolution:** `gpt-oss-120b` → `nvidia/openai/gpt-oss-120b` via
> `NVIDIA_INF_MODEL_ALIASES` in `Self-Evolving-Agent/evolving_common/llm_client.py`.

The resolved endpoint and model values are written to `run_summary.json` and
checked on resume — mismatched flags abort unless `--allow-resume-config-mismatch`
is passed.

Minimum `.env` required for all modes below:

```bash
NVIDIA_INF_API_KEY=<your inference API key>
```

Skill-governance flags (same as [RUN_WITH_UV.md](../RUN_WITH_UV.md) §4.2):

| Flag | Default | Role |
|------|---------|------|
| `--skill-deletion` / `--no-skill-deletion` | on | Unused-streak GC + full active catalog for extractor |
| `--skill-merging` / `--no-skill-merging` | off | Embed → cluster → LLM merge (requires deletion) |
| `--skill-merge-similarity` | `0.9` | Cosine similarity threshold for merge clustering |
| `--skill-merge-interval` | `50` | Min global iterations between merge passes |
| `--enable-skill-refinement` | off | SkillRevise-style inline diagnosis/revision loop |
| `--skill-refinement-max-rounds` | `3` | Max refinement rounds per trigger |
| `--enable-l1-skill-unit-test-gc` | off | Unit-test GC every iteration (vs first-promote only) |

---

## Skill refinement only

Skill refinement is **off by default**. Pass `--enable-skill-refinement` to enable
the SkillRevise-style inline diagnosis/revision loop (requires the Gen3 staged path
with L1; this is the default).

### Full run

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_skill_refinement_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_oss120b_skill_refinement_itr30_Aug_3.log 2>&1 &
```

After a real run, check:

- `runs_evolving/<run_name>_*/shared_l1.jsonl` — versioned entries (`parent_id`, `version`, `status`)
- `runs_evolving/<run_name>_*/skill_revisions.txt` — human-readable revision log
- `workspaces/level_*_problem_*/chat_history.jsonl` — `skill_diagnosis` / `skill_revision` LLM turns

---

## Deletion only

(with unit test only when it is first promoted)

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  >> base_agent_oss120b_deletion_itr30_Aug_3.log 2>&1 &
```

---

## Merge only

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_merge_only_sim_07_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --no-skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.7 \
  --skill-merge-interval 50 \
  >> base_agent_oss120b_merge_only_sim_07_itr30_Aug_5.log 2>&1 &
```

---

## Deletion + merging

### Full 50 problems (with unit test every itrs)

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_merge_sim_08_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  --enable-l1-skill-unit-test-gc \
  >> base_agent_oss120b_deletion_merge_sim_08_itr30.log 2>&1 &
```

### Full 50 problems (with unit test only when it is first promoted)

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_merge_sim_08_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  >> base_agent_oss120b_deletion_merge_sim_08_itr30.log 2>&1 &
```

---

## Deletion + skill refinement

### Full 50 problems (with unit test only when it is first promoted)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_refine_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_oss120b_deletion_refine_itr30.log 2>&1 &
```

---

## Deletion + merging + skill refinement

### Full 50 problems (with unit test only when it is first promoted)

```bash
CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_merge_refine_sim_08_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_oss120b_deletion_merge_refine_sim_08_itr30.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_oss120b_deletion_merge_refine_sim_07_itr30 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.7 \
  --skill-merge-interval 50 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_oss120b_deletion_merge_refine_sim_07_itr30.log 2>&1 &
```

---

## After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `nvidia_endpoint: "inference"`, `model: "gpt-oss-120b"`,
  merge/deletion/refinement flags
- `batch_timing.jsonl` — per-problem `wall_time_sec`, `started_at_utc`, `finished_at_utc`
- `shared_l1.jsonl` — structured L1 catalog (versioned when skill refinement is enabled)
- `skill_revisions.txt` — present when `--enable-skill-refinement` wrote refined versions
- `l1_skill_usage.json` — usage streaks per skill (when deletion is on)
- `l1_skill_deletions.jsonl` — deletion audit events
- `l1_skill_merges.jsonl` — merge audit events (when `--skill-merging`)
- `l1_skill_directories/<entry_id>/` — generated unit-test sources

Open the KernelBench visualizer (`visualizations/kernelbench`) and select the run to
view the **Run L1 Skill Memory** panel (skills, merges, deletions, refinement version
chains, usage ledger).

---

## Resume

Keep the same skill-governance flags, `--nvidia-endpoint inference`, and
`--model gpt-oss-120b` as the original run. On resume, skill-governance flags in
`run_summary.json` are checked against the current CLI; mismatches **abort** unless
you pass `--allow-resume-config-mismatch`. See [RUN_WITH_UV.md](../RUN_WITH_UV.md) §6
for indexing / L1 purge details.

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_oss120b_deletion_merge_refine_sim_08_itr30_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem 28 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_oss120b_deletion_merge_refine_sim_08_itr30_resume.log 2>&1 &
```
