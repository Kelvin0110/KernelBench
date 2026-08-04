---
name: smoke-experiment
description: Run a small gpt-oss-120b smoke experiment (2-3 problems x 2-3 iterations) before committing or launching a full run, whenever a major feature lands in the evolving agent. Use after changing context-management modes, prompt construction, L0/L1 memory, skill governance (refine/delete/merge), governor stages, LLM client wiring, endpoints, or model selection. Also use when the user says "I changed X, is it working?", asks to validate a feature end-to-end, or is about to kick off a 50-problem run.
---

# Smoke experiment before full runs

Any non-trivial change to the evolving agent must be validated with a **short real
run** before a full 50-problem sweep or before declaring the feature done. Unit
tests and `--dry-run` do not exercise the governor loop, the LLM calls, or the GPU
eval path — the three places features actually break.

## Always smoke on `gpt-oss-120b` — never substitute another model

Smoke runs use `gpt-oss-120b` **without exception**, even when the feature targets a
different model. It is the known-good reference: fast, cheap, and every
context-window/packing default is tuned for it. Smoking on a new model conflates
"my feature is broken" with "this model behaves differently," which is exactly the
ambiguity the smoke run exists to eliminate.

**Testing a target model is a separate, opt-in step.** After the `gpt-oss-120b`
smoke run is clean, do **not** proceed to run the target model on your own
initiative. Stop and ask the user first — target-model runs cost real budget on a
different endpoint and are the user's call, not a default follow-up. Ask
explicitly, naming the model and endpoint, and wait for an answer.

## Check the GPU environment before any real run

Steps 1-2 need no GPU. Step 3 does, and the batch script's hardware defaults are
**not** correct on every host — a wrong `--hardware` silently produces speedups
computed against the wrong baseline, which looks like a working run.

Before launching step 3:

```bash
nvidia-smi --query-gpu=index,name,memory.used --format=csv
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
ls results/timing/
```

Then confirm all three, and **ask the user if any is uncertain** rather than
guessing:

- **`CUDA_VISIBLE_DEVICES`** — pick a GPU that is actually idle in `nvidia-smi`.
  Do not assume `0` is free.
- **`--hardware`** — must match this host's folder under `results/timing/`. The
  script defaults to `SONG_CPU6_A6000x4`, which is wrong on most machines (e.g. a
  GH200 host needs `--hardware NVIDIA_GH200x2`). Alternatively pass
  `--baseline-timing-file` directly.
- **`torch.cuda.is_available()` is `True`** — if it prints `False` while
  `nvidia-smi` shows GPUs, the installed PyTorch build does not match the driver's
  CUDA version. Report that to the user; it is an environment problem, not
  something to work around with `--dry-run`.

## Procedure

Run these in order. Stop and fix at the first failure — do not proceed to the next
step with a known failure.

### 1. Unit tests for the touched area

```bash
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py -q
uv run python -m pytest Self-Evolving-Agent/tests/ -q
```

Narrow to the relevant file when the full suite is slow (e.g.
`test_selective_retention.py`, `test_l0_rounds.py`, `test_markov_report.py`).

### 2. Dry run — CLI plumbing only

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name smoke_<feature>_dryrun \
  --max-problems 2 \
  --max-iterations 2 \
  --model gpt-oss-120b \
  --no-skill-deletion \
  --dry-run
```

Confirms new flags parse and `run_summary.json` records them. No LLM, no GPU.

### 3. Smoke run — 2-3 problems x 2-3 iterations, `gpt-oss-120b`

This is the step people skip. Don't. Run the GPU-environment checks above first.

```bash
CUDA_VISIBLE_DEVICES=<idle gpu> nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name smoke_<feature> \
  --max-problems 3 \
  --max-iterations 3 \
  --model gpt-oss-120b \
  --hardware <this host's folder under results/timing/> \
  --no-skill-deletion \
  >> smoke_<feature>.log 2>&1 &
```

Add the flags your feature actually exercises, e.g.
`--context-management selective_retention`, `--enable-skill-refinement`,
`--skill-merging`.

Keep `--max-problems` at 2-3 and `--max-iterations` at 2-3. Larger defeats the
purpose; the goal is a fast signal, not a result.

Do not add `--nvidia-endpoint inference` or a different `--model` here — that is
the target-model step, which requires asking the user first (see above).

### 4. Inspect the artifacts

Under `runs_evolving/smoke_<feature>_<UTC timestamp>/`:

- `run_summary.json` — every flag your feature added is present and correct;
  `total_correct > 0`; no unexpected `error` fields in `evolving_runs.json`
- `workspaces/level_*_problem_*/chat_history.jsonl` — the new phase actually fired
  (`phase=evolving_report`, `phase=milestone_judge`, `phase=preflight`,
  `phase=l0_round_summary`, …). **An empty or missing phase means the feature
  silently no-op'd** — the most common failure and one that a green test suite
  will not catch.
- Mode-specific files: `evolving_report.md` (markov_report),
  `l0_milestones.json` (selective_retention)
- The log — a run that "succeeded" while every iteration threw and got swallowed
  is a failure

## Reporting

Report what the smoke run showed, not just that it ran: problems attempted,
how many were correct, whether the new phase appears in `chat_history.jsonl`, and
any errors. State which GPU and `--hardware` baseline were used. If the feature did
not visibly change behavior, say so plainly rather than treating a clean exit as a
pass.

Once the `gpt-oss-120b` smoke run is clean, stop and hand the next decision to the
user rather than escalating on your own:

- **Target-model validation** (e.g. `--nvidia-endpoint inference --model
  gpt-5.6-terra`) — ask before running.
- **Full 50-problem sweep** — ask before launching; it is long and expensive.

Report the feature as smoke-validated, not as fully validated, until one of those
has actually run.
