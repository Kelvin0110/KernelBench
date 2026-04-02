# Run Evolving-Agent KernelBench with uv (CUDA Server)

This guide runs the prototype integration on a Linux server with an NVIDIA CUDA GPU.

## 1) Prerequisites

- NVIDIA driver and CUDA runtime available
- Python 3.10+
- uv installed
- API key available for the configured LLM provider used by Self-Evolving-Agent

## 2) Create environment and install dependencies

From repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
uv pip install -r Self-Evolving-Agent/requirements.txt
```

## 3) Quick sanity checks

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
python -c "from kernelbench.dataset import construct_kernelbench_dataset; ds=construct_kernelbench_dataset(level=1, source='local'); print('level1_count=', len(ds))"
```

## 4) Optional local wiring check (no kernel eval)

This checks CSV loading and output artifact generation without running GPU eval:

```bash
python scripts_integration/evolving_agent/evolve_kb_batch.py \
  --run-name evolving_proto_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

## 5) Real CUDA run (small prototype batch)

```bash
CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts_integration/evolving_agent/evolve_kb_batch.py --run-name evolving_proto_gpu_latest --max-iterations 5 >> initial_prototype_run.log 2>&1
```

## 6) Output files

Run artifacts are under:

- results/evolving_logs/<run_name>/shared_l1.txt
- results/evolving_logs/<run_name>/eval_results.json
- results/evolving_logs/<run_name>/evolving_runs.json
- results/evolving_logs/<run_name>/run_summary.json

Notes:
- `eval_results.json` uses KernelBench-style keyed-by-problem_id structure.
- `evolving_runs.json` stores richer per-iteration evolving-agent records.
