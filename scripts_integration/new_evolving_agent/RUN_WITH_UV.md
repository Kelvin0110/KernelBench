# Run New Evolving-Agent KernelBench Integration with uv

This guide runs the new `scripts_integration/new_evolving_agent` workflow on a Linux CUDA server.

For L0 context-management modes (`truncation` / `folding` / `markov_report`), see [RUN_WITH_UV_CONTEXT.md](RUN_WITH_UV_CONTEXT.md).

## 1) Prerequisites

- NVIDIA driver and CUDA runtime available
- Python 3.10+
- `uv` installed
- `NVIDIA_API_KEY` exported for coder/summarizer calls through `evolving_common.llm_client`

## 2) Create environment and install dependencies

From repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
uv pip install -r Self-Evolving-Agent/requirements.txt
uv pip install pytest
```

## 3) Dependency sanity checks

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
python -c "from kernelbench.dataset import construct_kernelbench_dataset; ds=construct_kernelbench_dataset(level=1, source='local'); print('level1_count=', len(ds))"
python -c "import evolving_common, kernelbench.prompt_constructor_toml; print('integration_imports_ok=True')"
python -c "import os; print('has_nvidia_api_key=', bool(os.getenv('NVIDIA_API_KEY')))"
```

## 4) Dry run / smoke test (no GPU eval / no LLM calls)

Validates subset parsing, timing artifacts, and output generation.

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name new_evolving_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.9 \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

After dry-run, check `runs_evolving/<run_name>_*/batch_timing.jsonl` (per-problem wall times) and
`run_summary.json` fields `total_wall_time_sec`, `batch_timing_jsonl`.

## 4.1) Skill refinement add-on (opt-in)

Skill refinement is **off by default**. Pass `--enable-skill-refinement` to enable the
SkillRevise-style inline diagnosis/revision loop (requires the Gen3 staged path with L1;
this is the default). Refined skill versions are written to `shared_l1.jsonl` and
mirrored in `skill_revisions.txt` next to the journal.

### Unit tests (no GPU / no API key)

From repository root:

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_skill_refinement.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_skill_refinement_flag -q
```

### Dry run with the flag (CLI plumbing only)

Validates that `--enable-skill-refinement` is accepted; does **not** call the LLM or
run skill refinement (dry-run skips GPU eval and governor execution).

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name skill_refinement_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 1 \
  --max-iterations 3 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

### Small real CUDA run with skill refinement

Use a short iteration budget first; refinement consumes main iterations inline (up to
`--skill-refinement-max-rounds` per trigger, default 3).

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_skill_refinement_ver2_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_with_skill_refinement_ver2_smoke.log 2>&1 &
```

### Full run

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_skill_refinement_ver2_itr50 \
  --max-iterations 50 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_with_skill_refinement_ver2_itr50_Jun_27.log 2>&1
```

After a real run, check:

- `runs_evolving/<run_name>_*/shared_l1.jsonl` — versioned entries (`parent_id`, `version`, `status`)
- `runs_evolving/<run_name>_*/skill_revisions.txt` — human-readable revision log
- `workspaces/level_*_problem_*/chat_history.jsonl` — `skill_diagnosis` / `skill_revision` LLM turns

Resume works the same as §6; add `--enable-skill-refinement` to the resume command if
the original run used it.

## 4.2) L1 skill deletion, merging & unit tests (default deletion on; merging off)

Skill deletion and executable unit tests are **enabled by default** on the Gen3 path
when L1 is on. Skill merging is **off by default** (`--skill-merging`). Both are
independent of `--enable-skill-refinement`.

CLI flags (replacing legacy `--enable-l1-skill-deletion`):

| Flag | Default | Role |
|------|---------|------|
| `--skill-deletion` / `--no-skill-deletion` | on | Unused-streak GC + full active catalog for extractor |
| `--skill-merging` / `--no-skill-merging` | off | Embed → cluster → LLM merge (requires deletion) |
| `--skill-merge-similarity` | `0.9` | Cosine similarity threshold for merge clustering |
| `--skill-merge-interval` | `50` | Min global iterations between merge passes |

Policies (see `evolving_common/governor/skill_deletion.py`, `skill_merge.py`):
- Unused-streak GC after `l1_skill_consecutive_unused_delete_after` global iterations
- Optional LLM-generated `skill_impl.py` / `test_skill_impl.py` under
  `l1_skill_artifacts/<entry_id>/` with subprocess validation
- Merge artifacts: `l1_skill_merges.jsonl`, `l1_skill_embeddings.json`, `l1_skill_merge_state.json`

### Unit tests (no GPU)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_skill_deletion.py -q
uv run python -m pytest Self-Evolving-Agent/tests/test_skill_merge.py -q
uv run python -m pytest Self-Evolving-Agent/tests/test_kb_skill_memory.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py -q
```

### Small real CUDA run — deletion (with unit test every itrs) + merging (no skill refinement)

```bash
CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_merge_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 20 \
  --enable-l1-skill-unit-test-gc \
  >> base_agent_with_deletion_merge_smoke.log 2>&1 &
```

### Full 50 problems — deletion only (with unit test only when it is first promoted)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_old_prompt_only_test_promoted_itr50 \
  --max-iterations 50 \
  --skill-deletion \
  --no-skill-merging \
  >> base_agent_with_deletion_old_prompt_only_test_promoted_itr50_Jul_2.log 2>&1 &
```

### Full 50 problems — merge only

```bash
CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_merge_only_sim_07_itr30 \
  --max-iterations 30 \
  --no-skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.7 \
  --skill-merge-interval 50 \
  >> base_agent_with_merge_only_sim_07_itr30_Jul_14.log 2>&1 &
```

### Full 50 problems — deletion + merging

#### (with unit test every itrs)
```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_merge_sim_08_itr30 \
  --max-iterations 30 \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  --enable-l1-skill-unit-test-gc \
  >> base_agent_with_deletion_merge_itr20_Jun_29.log 2>&1 &
```

#### (with unit test only when it is first promoted)
```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30 \
  --max-iterations 30 \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  >> base_agent_with_deletion_old_prompt_only_test_promoted_merge_sim_08_itr30_Jul_17.log 2>&1 &
```

### Full 50 problems — deletion + skill refinement
#### (with unit test only when it is first promoted)
```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30 \
  --max-iterations 30 \
  --skill-deletion \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_with_deletion_old_prompt_only_test_promoted_refine_itr30_Jul_14.log 2>&1 &
```

### Full 50 problems — deletion + merging + skill refinement
#### (with unit test only when it is first promoted)
```bash
CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30 \
  --max-iterations 30 \
  --skill-deletion \
  --skill-merging \
  --skill-merge-similarity 0.8 \
  --skill-merge-interval 50 \
  --enable-skill-refinement \
  --skill-refinement-max-rounds 3 \
  >> base_agent_with_deletion_old_prompt_only_test_promoted_merge_refine_sim_08_itr30_Jul_18.log 2>&1 &
```

To disable deletion (legacy capped extractor catalog):

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name kb_no_skill_deletion \
  --max-problems 2 \
  --max-iterations 10 \
  --no-skill-deletion \
  ...
```

After a real run, inspect under `runs_evolving/<run_name>/`:
- `batch_timing.jsonl` — per-problem `wall_time_sec`, `started_at_utc`, `finished_at_utc`
- `run_summary.json` — `total_wall_time_sec`, `avg_wall_time_sec`, merge/deletion flags
- `l1_skill_usage.json` — usage streaks per skill
- `l1_skill_deletions.jsonl` — deletion audit events
- `l1_skill_merges.jsonl` — merge audit events (when `--skill-merging`)
- `l1_skill_artifacts/<entry_id>/` — generated unit-test sources

Open the KernelBench visualizer (`visualizations/kernelbench`) and select the run to
view the **Run L1 Skill Memory** panel (skills, merges, deletions, refinement version
chains, usage ledger).

## 5) Real CUDA run

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name memory_evolving_agent_base_itr50_old_prompt --max-iterations 50 >> new_evolving_gpu_run_base_itr50_Jun_17.log 2>&1

CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name memory_evolving_agent_base_itr20_new_prompt --max-iterations 20 >> new_evolving_gpu_run_base_itr20_new_prompt_Jun_15.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name debug_memory_evolving_agent_base --max-problems 2 --max-iterations 5 >> debug_memory_evolving_agent_base.log 2>&1
```

## 6) Resume after failure (429, rate limits, etc.)

If a batch stops partway through (for example `coder_call_error: RateLimitError`), resume into the **same** run folder so shared L1 (`shared_l1.txt` / `shared_l1.jsonl`) is preserved.

1. Copy the exact directory name from `runs_evolving/` (includes the UTC timestamp suffix, e.g. `memory_evolving_agent_gen3_itr20_2026_06_03_14_05`).
2. Find the **1-based row index** of the first problem to re-run in the same subset CSV and `--max-problems` slice used originally (e.g. row 21 → `--start-problem 21`).
3. Re-run with `--resume` (no new timestamp is appended). Problems before `--start-problem` are left unchanged; from that index through the end, prior records are replaced and per-problem workspaces are cleared before re-run.
4. On resume, skill-governance flags in `run_summary.json` are checked against the current CLI (`skill_deletion`, `skill_merging`, merge knobs, unit-test GC, skill refinement). Mismatches **abort** unless you pass `--allow-resume-config-mismatch`.
5. Skills in `shared_l1.jsonl` / `shared_l1.txt` whose source is a problem at/after `--start-problem` are removed automatically (including refined children and merges that depend on those skills). Pass `--backup-l1-on-resume` to keep `*.resume.bak` copies first.

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name memory_evolving_agent_gen3_itr10_2026_06_03_11_30 \
  --max-problems 50 \
  --max-iterations 10 \
  --start-problem 11 \
  >> new_evolving_gpu_run_gen3_itr10_Jun_3.log 2>&1
```

Dry-run resume (validate indexing only):

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name memory_evolving_agent_gen3_itr10_2026_06_03_11_30 \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 50 \
  --start-problem 11 \
  --dry-run
```

## 7) Output files

Run artifacts are written to `runs_evolving/<run_name>/`:

- `shared_l1.txt`
- `shared_l1.jsonl` (structured L1 catalog; versioned when skill refinement is enabled)
- `skill_revisions.txt` (present when `--enable-skill-refinement` wrote refined versions)
- `l1_skill_usage.json` (when L1 skill deletion is enabled — default)
- `l1_skill_deletions.jsonl` (deletion audit log)
- `l1_skill_artifacts/<entry_id>/` (executable unit-test sources when unit tests are enabled)
- `eval_results.json` (level-first shape: `{level: {problem_id: [entries]}}`)
- `eval_results_level_<level>.json`
- `evolving_runs.json`
- `run_summary.json`
- `workspaces/level_<level>_problem_<problem_id>/`

Per-problem workspace recorder artifacts include:

- `run_manifest.json`
- `run_finished.json`
- `chat_history.jsonl`
- `iteration_snapshots.jsonl`
- `memory_evolution.jsonl`
- `metrics_by_iteration.jsonl`
- `metrics_by_time.jsonl`
