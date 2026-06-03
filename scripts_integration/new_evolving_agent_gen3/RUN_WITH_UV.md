# Run New Evolving-Agent KernelBench Integration with uv

This guide runs the new `scripts_integration/new_evolving_agent` workflow on a Linux CUDA server.

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

## 4) Dry run (no GPU eval / no LLM calls)

This validates subset parsing and output artifact generation.

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name new_evolving_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

## 5) Real CUDA run

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name memory_evolving_agent_gen2_itr20 --max-iterations 20 >> new_evolving_gpu_run_gen2_itr20_Jun_3.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --run-name debug_memory_evolving_agent --max-problems 2 --max-iterations 2
```

## 6) Resume after failure (429, rate limits, etc.)

If a batch stops partway through (for example `coder_call_error: RateLimitError`), resume into the **same** run folder so shared L1 (`shared_l1.txt` / `shared_l1.jsonl`) is preserved.

1. Copy the exact directory name from `runs_evolving/` (includes the UTC timestamp suffix, e.g. `memory_evolving_agent_gen3_itr20_2026_06_03_14_05`).
2. Find the **1-based row index** of the first problem to re-run in the same subset CSV and `--max-problems` slice used originally (e.g. row 21 → `--start-problem 21`).
3. Re-run with `--resume` (no new timestamp is appended). Problems before `--start-problem` are left unchanged; from that index through the end, prior records are replaced and per-problem workspaces are cleared before re-run.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts_integration/new_evolving_agent_gen3/evolve_kb_batch.py \
  --resume \
  --run-name memory_evolving_agent_gen3_itr20_2026_06_03_14_05 \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 50 \
  --max-iterations 20 \
  --start-problem 21
```

Dry-run resume (validate indexing only):

```bash
uv run python scripts_integration/new_evolving_agent_gen3/evolve_kb_batch.py \
  --resume \
  --run-name memory_evolving_agent_gen3_itr20_2026_06_03_14_05 \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 50 \
  --start-problem 21 \
  --dry-run
```

## 7) Output files

Run artifacts are written to `runs_evolving/<run_name>/`:

- `shared_l1.txt`
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
