# KernelBench — Evolving-Agent Experiments

Working notes for the `features/evolving-agent-final` branch.

---

## 1. What this project is

KernelBench evaluates LLM agents that write custom CUDA kernels for PyTorch
modules. This branch runs an **evolving agent** with two memory levels:

- **L0** — per-problem iteration history (context management applies here)
- **L1** — a skill catalog shared across all problems in a batch
  (`shared_l1.jsonl`), governed by deletion / merging / refinement

We are running a controlled experiment series measuring how **L0 context-management
mode** and **L1 skill-governance** affect kernel quality.

Two independent axes:

| axis | values |
|---|---|
| L0 context management | `truncation` (default/baseline), `folding`, `markov_report`, `selective_retention`, `compress_trigger` |
| L1 skill governance | `--skill-deletion`, `--skill-merging`, `--enable-skill-refinement` (7 non-empty combinations) |

Governance arms hold context at `truncation` so the two axes stay separable.

**Fixed protocol for every arm:** 50 problems (`subset_selection/selected_problems_50.csv`),
30 iterations, `gpt-oss-120b`, hardware `NVIDIA_GH200x2`. One arm ≈ **65–75 GPU-hours**.

Current results and cross-run comparisons live in
`scripts_integration/new_evolving_agent_analysis/output/GH200x2/`. Known defects and
experiment-design caveats are recorded in project memory
(`skill-governance-gotchas.md`) and `env/README.md`.

---

## 2. Environment — read this before running anything

### 2.1 Two mandatory exports

CUDA is a **userspace install** (no sudo). Without these, `nvcc` is absent and
every `load_inline(cuda_sources=...)` build fails — silently producing kernels that
fall back to plain PyTorch while still scoring `correct=True`:

```bash
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:/localhome/local-tianzheng/KernelBench/.venv/bin:$PATH
nvcc --version        # expect: release 12.8, V12.8.93 (matches torch 2.11.0+cu128)
```

`launch_run.sh` sets both itself. If you invoke `evolve_kb_batch.py` by hand, you must.
Reinstall with `scripts_integration/new_evolving_agent/env/install_cuda128_local.sh`.
Background: `scripts_integration/new_evolving_agent/env/README.md`.

### 2.2 Always use `uv run --no-sync`

A bare `uv run` **re-syncs the venv and prunes packages**. Verified:

```
uv sync --dry-run → Would uninstall 9 packages
  - scikit-learn - scipy - joblib - threadpoolctl - pytest - ruff - ...
```

Removing scikit-learn makes every `--skill-merging` iteration die with
`coder_call_error`. `launch_run.sh` uses `--no-sync` at all three call sites; keep it.

**Current state:** `pyproject.toml` declares `scikit-learn` in `[project] dependencies`
(promoted out of the `evolving-agent` extra), but `uv.lock` has **not** been
regenerated. They are intentionally out of sync, so `--no-sync` is load-bearing until
someone runs `uv lock && uv sync`. Do that only when no run is in flight — it shares
`.venv` with running jobs, drops pytest/ruff, and pins `scikit-learn==1.5.0` (the venv
currently has 1.7.2).

### 2.3 API keys and endpoints (`.env`)

| purpose | endpoint | key |
|---|---|---|
| chat (all LLM roles) | `inference-api.nvidia.com/v1` | `NVIDIA_INF_API_KEY` |
| embeddings (skill merge) | `inference-api.nvidia.com/v1` | `NVIDIA_INF_API_KEY` |

Model IDs differ per endpoint: `gpt-oss-120b` → `openai/gpt-oss-120b` (integrate) vs
`nvidia/openai/gpt-oss-120b` (inference). Use the aliases in `llm_client.py`, not raw IDs.

Embeddings choose their endpoint independently of chat via `NVIDIA_EMBED_ENDPOINT`
(default `inference`; model default `nvidia/qwen/qwen3-embedding-0.6b`). Probe both
endpoints with `scripts_integration/new_evolving_agent/env/probe_integrate_key.py`.

---

## 3. Running an experiment

### 3.1 The launcher (use this, don't hand-roll nohup)

```bash
bash scripts_integration/new_evolving_agent/env/launch_run.sh <gpu> <run_name> <ctx_mode> [extra flags...]
```

It preflights nvcc, ninja, the GH200 baseline dir, the API key, GPU-idleness, a live
`load_inline(cuda_sources=...)` compile probe, and (for merge arms) `import sklearn` —
then launches under `nohup` and prints the pid and log path. Every check exists because
its absence once silently corrupted a ~70 h run.

Fixed args it always supplies: `--max-problems 50 --max-iterations 30 --hardware
NVIDIA_GH200x2 --nvidia-endpoint inference --model gpt-oss-120b --coder-timeout-sec 600
--results-root runs_evolving/gpt-oss-120b/`.

### 3.2 Naming conventions

- **Run name:** `base_agent_gpt_oss_120b_<tag>_itr30_GH200`
  `<tag>` ∈ `{markov, folding, compress, selective_r5, deletion, refinement, merge_sim085, ...}`.
  **Encode any non-default parameter in the tag** (`selective_r5` = 5 recent rounds,
  `merge_sim085` = similarity 0.85). The runner appends `_YYYY_MM_DD_HH_MM`.
- **Log:** auto-derived, `<run_name>_<Mon>_<D>.log` in the repo root.
- **Results:** `runs_evolving/gpt-oss-120b/<run_name>_<timestamp>/`

### 3.3 Examples

```bash
# context-management arm
bash .../env/launch_run.sh 0 base_agent_gpt_oss_120b_markov_itr30_GH200 markov_report

# compress_trigger needs its tuning flags
bash .../env/launch_run.sh 0 base_agent_gpt_oss_120b_compress_itr30_GH200 compress_trigger \
  --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15

# governance arms — context held at truncation
bash .../env/launch_run.sh 1 base_agent_gpt_oss_120b_deletion_itr30_GH200   truncation --skill-deletion
bash .../env/launch_run.sh 1 base_agent_gpt_oss_120b_refinement_itr30_GH200 truncation --enable-skill-refinement
bash .../env/launch_run.sh 1 base_agent_gpt_oss_120b_merge_sim085_itr30_GH200 truncation \
  --skill-merging --skill-merge-similarity 0.85
```

### 3.4 Running several arms on one GPU

The agent is **LLM-bound, not GPU-bound** — an eval subprocess lives ~38 s but touches the
GPU for well under a second of that (GPU idle 98.75% of 1 s samples). So a GPU comfortably
hosts 2–3 arms. Measured on GH200, 3 level-3 problems × 5 iterations:

| arms on one GPU | throughput | wall clock per arm |
|---|---|---|
| 1 | 1.0× | 1453 s |
| 2 | 2.6× | 1127 / 1121 s |
| 3 | 2.9× | 1406 / 1195 / 1522 s (≈5% slower than solo) |

**Two settings are mandatory when sharing a GPU:**

```bash
KB_GPU_RESERVE_GB=0        # REQUIRED. Default 42 GB per arm.
KB_GPU_EVAL_LOCK=1         # default; leave on. Serialises the GPU phase of eval.
```

- **`KB_GPU_RESERVE_GB=0`** — each governor otherwise pins a 42 GB block while waiting on
  the LLM (`kernelbench_integration/governor.py`). Several reservers fight for headroom and
  can OOM whichever arm is mid-eval. Harmless to set: the block is released around eval
  anyway, so it never affected timing. Default stays 42 GB for single-arm runs.
- **`KB_GPU_EVAL_LOCK`** — cross-process `flock` keyed by GPU UUID
  (`evolving_common/governor/gpu_lock.py`), held only across the GPU phase of an eval.
  nvcc runs *before* the lock is taken, so the critical section is ~0.4–4 s rather than the
  full ~38 s eval. Free when uncontended (0.000 s wait across 129 solo evals), so it is on
  by default and costs single-arm runs nothing. Escape hatches: `KB_GPU_EVAL_LOCK=0` to
  disable, `KB_GPU_EVAL_LOCK_TIMEOUT_SEC` (default 1800) after which an arm logs loudly and
  proceeds **unlocked** rather than blocking forever. It cannot deadlock on a crashed arm —
  `flock` is released by the kernel on process death.

**Why the lock matters.** Speedup is `fixed_baseline / measured_runtime` where the baseline
is an idle-GPU constant from `results/timing/<hw>/baseline_time_torch.json`, so contention
can only ever *deflate* a speedup, never inflate it. The median hit is small (~3%), but the
**tail** is what the lock removes — with 3 unlocked arms, one 0.9 ms kernel showed CV 155%
and a worst sample of 22.3 ms; locked, CV 17% and worst 1.28 ms.

**Gotcha — the launcher refuses the 3rd arm.** `launch_run.sh:41` aborts when the GPU
reports >1000 MiB used. An idle arm holds ~558 MiB, so arm 2 passes but arm 3 reads ~1.1 GB
and is rejected. Raise that threshold (or bypass the guard) to launch three.

Expect ~25% of evals to wait at 3 arms (observed 12/45, 5–63 s, zero timeouts); heavy
level-3 problems have the longest critical sections because model construction and
correctness trials sit inside the lock too. Audit contention with:

```bash
grep -h "gpu-eval-lock" <log>          # waits ≥5 s; "proceeding UNLOCKED" = timeout, investigate
```

### 3.5 Health checks while running

```bash
grep -c CUDA_HOME <log>                     # must stay 0
grep -E "^\[batch\]" <log> | tail -2        # problem progress
uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
```

`torch._inductor` "No valid triton configs / OutOfMemoryError: triton_mm" tracebacks are
**benign** autotuner noise, not failures.

For merge arms specifically, confirm the merge pass is actually doing work — it swallows
its own exceptions when `verbose` is off, so a broken embedding path produces zero merges
with nothing in the log:

```bash
python3 -c "import json;print(len(json.load(open('<run>/l1_skill_embeddings.json'))['skills']))"
wc -l <run>/l1_skill_merges.jsonl
```

Both must be non-zero.

### 3.6 Resuming a damaged range

```bash
bash scripts_integration/new_evolving_agent/env/resume_run.sh <gpu> <run_dir_name> <ctx_mode> <start> [end]
```

Narrow ranges are safe — two mechanisms cooperate:

1. **Disk purge** removes only L1 entries sourced from problems inside `[start, end]`
   (plus refine/merge descendants).
2. **Causal prompt filter** (`collect_causal_l1_entry_ids`) restricts the visible catalog
   to entries with provenance strictly `< N` while replaying problem `N`.

A replayed problem never sees skills learned after it. Verified: replaying index 39 showed
267/344 entries, provenance 1..38, zero leakage. **For multiple resumes, run the earlier
index first.**

---

## 4. Analysis

```bash
# always pass --regenerate-stats; cached stats across runs were written by different
# code versions and are NOT comparable
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware NVIDIA_GH200x2 --runs-root runs_evolving/gpt-oss-120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2 --regenerate-stats

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware NVIDIA_GH200x2 --runs-root runs_evolving/gpt-oss-120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2 \
  --baseline-run base_agent_gpt_oss_120b_itr30_GH200_2026_08_07_13_58
```

Outputs land in `output/GH200x2/`: `aggregate_runs.{json,csv}`, `comparison.md`.

Primary metric is **`best_geomean`** (geometric mean of best speedup over correct
non-hack samples) plus **`fast_p@1.0`**. See `ANALYSIS_RULES.md` for the rules, and
`output/GH200x2/INVALIDATED.md` for which historical runs are void.

---

## 5. Repo layout

```
scripts_integration/new_evolving_agent/
  evolve_kb_batch.py              # the batch runner (all CLI flags)
  env/
    install_cuda128_local.sh      # userspace CUDA 12.8 (no sudo)
    launch_run.sh                 # ← launch arms with this
    resume_run.sh                 # narrow-range replay
    probe_integrate_key.py        # isolate key vs model failures
    eval_embed_duplicates.py      # rank embedding models by near-duplicate retrieval
    eval_embed_quality.py         # merge-outcome AUC (null result; kept as evidence)
    eval_embed_candidates.py      # fidelity-to-nv-embedcode (misleading; kept as evidence)
    README.md                     # nvcc postmortem
scripts_integration/new_evolving_agent_analysis/
  aggregate_runs.py  compare_runs.py  checkpoint_run.py  analyze_feature_evidence.py
  ANALYSIS_RULES.md  EXPERIMENT_REPORT.md  output/GH200x2/
Self-Evolving-Agent/               # git submodule
  evolving_common/llm_client.py               # endpoints, aliases, embeddings
  evolving_common/memory_manager.py           # governance defaults
  evolving_common/governor/gen3_stages.py     # staged governor; deletion/merge call sites
  evolving_common/governor/skill_merge*.py    # DBSCAN clustering + LLM merge
  evolving_common/governor/gpu_lock.py        # cross-process GPU-eval lock (concurrent arms)
  evolving_common/governor/gpu_reserver.py    # 42 GB idle reservation; KB_GPU_RESERVE_GB
  kernelbench_integration/eval_runner.py      # precompile-then-lock boundary
  kernelbench_integration/governor.py         # speedup computation
runs_evolving/gpt-oss-120b/        # current series
runs_evolving/inference_oss_120b/  # earlier series (only successful merge runs)
runs_evolving/archived/            # VOID (pre-nvcc-fix)
```

---

## 6. Open items

1. **Merge threshold** — code default is `0.8`, which clusters chain badly at realistic
   catalog sizes; `0.85` matches the validated operating point. Pass
   `--skill-merge-similarity 0.85` explicitly until the default is changed.
2. **`uv lock && uv sync`** — deferred until no run is in flight. Will install
   `scikit-learn==1.5.0`; re-run the merge-threshold calibration afterwards.
3. **`--enable-l1-skill-unit-test-gc` is a no-op** (`gen3_stages.py:893` reads the wrong
   config field), so every `--skill-deletion` arm is really deletion + unit-test GC. Fix
   before running more deletion cells or they stay confounded. Details in project memory.
4. **Rewrite `EXPERIMENT_REPORT.md`** — the current text is written against voided
   pre-nvcc-fix runs and its conclusions are reversed by the repaired data.
5. **Governance matrix is incomplete** — 5 of 7 cells untouched (~340 GPU-h at one arm
   per GPU; ~2 GPU-weeks of wall clock collapses to a few days at 3 arms per GPU, see §3.4).
6. **Commits are blocked** — git identity is unset in this repo; the user declined
   `git config`. Submodule changes to `llm_client.py` / `memory_manager.py` and the
   root `pyproject.toml` edit are all uncommitted.

---

## 7. Working preferences

- **Ask for confirmation before launching any `nohup` run** — they cost ~67 GPU-h each.
- Direct `nohup` from the tool layer gets denied; that is why launches go through
  `launch_run.sh` invoked via `bash`.
- Verify claims against artifacts before reporting. Several confident-sounding
  conclusions in this project turned out wrong until checked against the data.
