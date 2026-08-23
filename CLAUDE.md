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
Reinstall with `scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/install_cuda128_local.sh`.
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
bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_run.sh <gpu> <run_name> <ctx_mode> [extra flags...]
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
bash .../env/NVIDIA_GH200x2_2nd/launch_run.sh 0 base_agent_gpt_oss_120b_markov_itr30_GH200 markov_report

# compress_trigger needs its tuning flags
bash .../env/NVIDIA_GH200x2_2nd/launch_run.sh 0 base_agent_gpt_oss_120b_compress_itr30_GH200 compress_trigger \
  --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15

# governance arms — context held at truncation
bash .../env/NVIDIA_GH200x2_2nd/launch_run.sh 1 base_agent_gpt_oss_120b_deletion_itr30_GH200   truncation --skill-deletion
bash .../env/NVIDIA_GH200x2_2nd/launch_run.sh 1 base_agent_gpt_oss_120b_refinement_itr30_GH200 truncation --enable-skill-refinement
bash .../env/NVIDIA_GH200x2_2nd/launch_run.sh 1 base_agent_gpt_oss_120b_merge_sim085_itr30_GH200 truncation \
  --skill-merging --skill-merge-similarity 0.85
```

### 3.4 Running several arms on one GPU

The agent is **LLM-bound, not GPU-bound** — an eval subprocess lives ~38–45 s but touches
the GPU for well under a second of that. Sharing a GPU is therefore close to free, and the
limit is not the hardware but how much of each eval is serialised (see the lock below).

Full-scale measurements (50 problems × 30 iterations, matched per-problem against solo
baselines). *Throughput* = total iterations/hour across all arms, as a multiple of one solo
arm; N× means perfect linear scaling.

| arms/GPU | throughput | per-arm slowdown | efficiency |
|---|---|---|---|
| 1 | 1.00× | — | — |
| 3 | **3.07×** (steady state) | 0.98× (none) | ~100% |
| 6 | 3.31× *(wide lock)* | 1.69× | 55% |

Note the 3-arm figure is post-startup. Arms launched minutes apart begin in lockstep and
collide hard on the first ~9 problems (1.35× penalty, 2.37× throughput), then desynchronise
and the penalty disappears. Do not judge a configuration from its first problems.

**Three settings when sharing a GPU:**

```bash
KB_GPU_RESERVE_GB=0        # REQUIRED. Default 42 GB per arm.
KB_GPU_EVAL_LOCK=1         # default; leave on.
KB_GPU_EVAL_LOCK_TIMEOUT_SEC=1800   # default; raise if you run many arms
```

- **`KB_GPU_RESERVE_GB=0`** — each governor otherwise pins a 42 GB block while waiting on
  the LLM (`kernelbench_integration/governor.py`). Several reservers fight for headroom and
  can OOM whichever arm is mid-eval. Harmless to set: the block is released around eval
  anyway, so it never affected timing. Default stays 42 GB for single-arm runs.
- **`KB_GPU_EVAL_LOCK`** — cross-process `flock` keyed by GPU UUID
  (`evolving_common/governor/gpu_lock.py`). Free when uncontended (0.000 s wait across 129
  solo evals), so it is on by default and costs single-arm runs nothing. It cannot deadlock
  on a crashed arm — `flock` is released by the kernel on process death. On timeout an arm
  logs loudly and proceeds **unlocked** rather than blocking forever; `proceeding UNLOCKED`
  in a log means that eval's numbers are contended.
  **But before the 2026-08-22 fix that warning could never fire** — see the regimes box
  below. An empty `grep "proceeding UNLOCKED"` on a pre-fix log proves nothing.

**Why the lock exists.** Speedup is `fixed_baseline / measured_runtime` where the baseline
is an idle-GPU constant from `results/timing/<hw>/baseline_time_torch.json`, so contention
can only ever *deflate* a speedup, never inflate it. The median hit is small (~3%), but the
**tail** is what the lock removes — with 3 unlocked arms, one 0.9 ms kernel showed CV 155%
and a worst sample of 22.3 ms; locked, CV 17% and worst 1.28 ms.

#### The lock is narrow (2026-08-21) — and that is what sets the ceiling

The lock is held **only across correctness trials + the two timing windows**
(`kernelbench/eval.py::_gpu_timing_lock`). Reference-model construction, input generation
and nvcc/ninja loading run **unlocked** — they are the bulk of the wall time and do not
affect the numbers.

This matters because the lock is a single server, so it sets a hard concurrency ceiling:

| | wide lock (before) | narrow lock (now) |
|---|---|---|
| hold per eval | ~45 s | **~4.9 s** (max 9.3 s) |
| utilisation, 3 arms | ~92% | 26% |
| utilisation, 6 arms | ~91% | 46% |
| implied ceiling | **~3.9 arms/GPU** | see the two regimes below |

Under the wide lock, arms past ~4 bought nothing — they just queued. Adding arms cannot
beat the ceiling; only shrinking the critical section moves it.

**Trade-off, be aware:** because model construction is now unlocked, another arm's setup
work can touch the GPU during your timing window. Timing windows remain strictly mutually
exclusive (two arms can never time simultaneously), but this is a real fidelity trade that
has not yet been A/B'd against the wide-lock numbers above.

#### Two regimes: before and after the 2026-08-22 eval-deadline fix

The narrow lock did **not** raise the ceiling on its own. Until 2026-08-22 the binding
constraint was that **the eval deadline was shorter than the lock queue**. The lock is
acquired *inside* the eval child (`kernelbench/eval.py:650`) while the deadline is enforced
by the parent (`evaluate_in_subprocess` → `proc.join(evaluation_timeout_s)`, 600 s), so
queueing was charged against the *work* budget. Waiters were SIGTERM'd mid-wait and the
governor recorded a **fake compile failure** for a kernel that was never broken.

| | pre-fix (before 2026-08-22) | post-fix |
|---|---|---|
| evals killed mid-wait, 6 arms/GPU | **109** (17–20 per arm) | 0 by construction |
| eval-timeout rate, 6 arms | 3.9% | — |
| eval-timeout rate, 3 arms | 0.93% | — |
| safe concurrency | **3 arms/GPU** | higher; LLM API/CPU bind first |

Two traps in any pre-fix log:

- **`proceeding UNLOCKED` is unreachable.** Lock timeout is 1800 s (default) / 5400 s
  (launcher) but the eval deadline is 600 s, so the child always died first. Contention
  surfaced as `correct=False, compiled=False` instead — indistinguishable from a bad
  kernel. Audit by counting **unpaired `waiting` lines** (a `waiting` with no matching
  `acquired after`), not by grepping for UNLOCKED.
- **Any "max observed lock wait" is censored.** A wait exceeding 600 s never got to log an
  acquisition, so the distribution is truncated by construction and the observed max
  (~550 s) cannot approach the real tail.

The fix (submodule `7ac0e87`) has `gpu_lock` publish its running wait on every poll and the
parent extend its deadline by it, so `timeout_s` bounds work only. **It applies to newly
launched runs only** — `evaluate_in_subprocess` runs in the long-lived parent, which binds
the function object at import time and never re-imports. Arms already running when the fix
landed keep the old behaviour for their whole life.

Impact on pre-fix data is real but modest: of 17 problems that lost iterations across the
three completed merge reps, **15 (88%) still ended correct** — the metric is best-over-30
-iterations, so even 13/30 lost usually does not move it.

**`gpu_lock` is re-entrant** — nested acquisitions in one process pass through instead of
re-locking. Without this a wide+narrow transition self-deadlocks, since `flock` is per
open-file-description and a second `os.open()` blocks against the process's own lock.

**Gotcha — the launcher refuses the 3rd arm.** `launch_run.sh:41` aborts when the GPU
reports >1000 MiB used. An idle arm holds ~558 MiB, so arm 2 passes but arm 3 reads ~1.1 GB
and is rejected. Raise that threshold (or bypass the guard) to launch three or more.

**Never edit code while runs are live.** Eval uses `multiprocessing` **spawn**
(`execution.py:348`), so every eval re-imports `evolve_kb_batch.py` and `kernelbench/eval.py`
**from disk**. Nothing is frozen at launch. On 2026-08-20 an edit at 16:51 that imported a
module not created until 17:00 killed eval workers across all 9 live arms for 9 minutes —
32 iterations recorded as `compiled=False` fake compile failures, which the governor then
"debugged" as if the kernel were at fault. Young arms lost up to 45% of a problem's
iterations. If you must edit mid-run, write to a temp file, validate with `ast.parse`, then
`os.replace` it in atomically.

Audit contention with:

```bash
# WRONG -- always returns zero. eval_runner.py wraps eval_kernel_against_ref in
# redirect_stdout(), so the lock never prints to the arm log.
#   grep -h "gpu-eval-lock" <log>
# RIGHT -- the lines are captured into the per-iteration eval record:
python3 - <<'EOF'
import json,glob,re,statistics as st
pat=re.compile(r"acquired after ([\d.]+)s")
w=[float(m.group(1)) for f in glob.glob("runs_evolving/**/evaluation_terminal_output.jsonl",recursive=True)
   for l in open(f) for m in [pat.search(json.loads(l).get("terminal_output") or "")] if m]
print(len(w),"waits >=5s; median",st.median(w) if w else 0,"max",max(w) if w else 0)
EOF
```

#### Arms/GPU is the lever. The 100-trial change is a red herring (2026-08-23)

Submodule `7ba78c7` raised `num_perf_trials` 10 -> 100 and it is tempting -- I did it,
and so did a subagent -- to blame it for the contention. **It is not the cause.**
Non-lock-wait eval work on the *same* problem (L1P100), same GPU, same model:

| | p25 | median |
|---|---|---|
| Aug-07 solo, N=10, no contention | 67.0 s | 68.6 s |
| Aug-22 wave, N=100, 9 arms | 67.2 s | 71.8 s |

The lock hold is a subset of that work, so **the hold grew by at most ~3 s**. The
arithmetic agrees: at N=100 the two timing windows are 212 iterations, which for
L1P100 (ref 7.94 ms, candidate 1.25 ms) is 0.97 s of kernel time versus 0.15 s at
N=10 -- a 1.7 s delta, not a 10x critical section.

**What actually changed is arm count.** Same narrow lock, same code path:

| | evals waited >=5 s | median wait |
|---|---|---|
| Aug-20 reps, N=10, **3 arms** | 8% | 22 s |
| Aug-22 wave, N=100, **9 arms** | 76% | 298 s |

**The hold is ~20 s and always was** (busy-period inter-completion gap; consistent with
the Aug-20 median wait of 22 s at 3 arms). Only ~1 s of it is the timing loop -- the
rest is correctness trials, input generation, `empty_cache` and syncs. So:

- `num_perf_trials` was cut 100 -> 25 on 2026-08-23 (`eval_runner.py:108`). It buys back
  only ~2 s of a ~20 s hold, but it needed no restart -- see the propagation rule below.
- Skipping the discarded live reference timing is still correct (see below) but it is
  worth a few seconds, **not the "free 2x" an earlier version of this section claimed**.
- The earlier "~4.9 s hold, ~6 arms/GPU" figures came from a probe on a different
  problem mix (L3P4/9/42) and do not generalise. **Treat ~3 arms/GPU as the ceiling**
  until someone profiles the inside of the locked section.

Corollary: to actually raise the ceiling you must shrink the critical section.

#### The ~19 s that is not the timing loop is `get_inputs()` (2026-08-23) -- solved

Measured, not inferred. `get_inputs()` is called **three times inside the lock** --
`eval.py:887` (correctness trial), `:720` (candidate window), `:772` (reference window) --
and it is pure-CPU `torch.rand`. Wall-clock cost of those three calls, measured on this
box with `torch.set_num_threads(4)`:

| | per eval, 3 calls |
|---|---|
| **L1P34** (7.5 GB of inputs) | **19.2 s** |
| L2P19 (2.15 GB) | 5.5 s |
| L2 median | 1.5 s |
| L3 median | 0.7 s |

That accounts for essentially the whole unexplained hold. It also explains why lock wait
tracked the *problem list* rather than the calendar: see project memory
`per-problem-cost-is-input-volume.md`.

**The design rule this implies.** The lock is not protecting the holder's numbers -- it is
stopping the holder's GPU work from landing inside somebody else's timing window, where
contention can only deflate their speedup. So:

> Any GPU touch must be mutually exclusive with any timing window. Pure-CPU work must not be.

Under that rule `get_inputs()` (pure CPU) can leave the lock, but `_process_input_tensor`
(`eval.py:441`, ends in `.to(device=...)`) cannot. Correctness trials cannot either -- they
run two real model forwards. Only ~1/3 of the input cost is the CPU generation; the rest is
the transfer and cast, which stay.

**`get_inputs()` is not timed.** `timing.time_execution_with_cuda_event` receives
already-materialised tensors and brackets only `kernel_fn(*args)` between CUDA events, with
warmup outside. Moving generation therefore cannot change any reported runtime.

#### Two eval-lock switches (2026-08-23), both default OFF in code

`src/kernelbench/eval.py` is re-imported by every eval spawn, so an unconditional change
reaches live arms. Both trims are therefore gated, and turned on in
`env/NVIDIA_GH200x2_2nd/launch_wave.sh` -- i.e. for newly launched waves only.

| env var | what it does |
|---|---|
| `KB_EVAL_HOIST_INPUT_GEN` | builds `get_inputs()` tensors on the CPU before taking the lock; H2D stays locked. Falls back to the in-lock path on any failure, and skips correctness pregeneration unless `num_correct_trials == 1` (5-trial callers would hold five input sets in host RAM). |
| `KB_EVAL_SKIP_DEAD_REF_TIMING` | skips the reference *measurement* when a fixed baseline is supplied. **Only the measurement.** The excessive-speedup / reward-hack flag lives in the same `if` block but depends on `baseline_runtime` and the candidate runtime, never on the window -- gating it too would silently disable `is_hack`, the `is_new_best` veto, and hack filtering in `best_geomean`. That bug was caught in review; do not reintroduce it by skipping the whole block. |
| `KB_EVAL_PHASE_LOG` | path to append one JSON line per eval: `held_sec`, `waited_sec`, and a non-overlapping phase breakdown with an `other_sec` residual. Unset -> emits nothing. |

Neither trim changes a recorded number, but both shorten the hold, so a wave launched with
them sees less contention deflation than one launched without. Compare speedups across that
boundary with the same care as across a baseline change.

**Why the phase breakdown is a file and not a print.** Eval stdout is *not* a log here --
`eval_runner.py` captures it and `governor.py:1203` splices it into
`KERNEL_BENCH_EVAL_TERMINAL_OUTPUT`, which reaches the agent's prompt (78 of 93
`chat_history.jsonl` records on one problem carry it). Printing telemetry would mutate LLM
input mid-run. Anything added to eval stdout is an experiment change, not an observation.

**`gpu_lock` still records no hold time.** It yields `{acquired, waited_sec, timed_out}`
only, and `_SLOW_WAIT_LOG_SEC = 5.0` (`gpu_lock.py:41`) means sub-5 s waits are never even
counted -- so "lock wait ~= 0" only ever means "no single wait exceeded 5 s".
`KB_EVAL_PHASE_LOG` measures both uncensored from the `eval.py` side without touching the
lock's reporter contract.

**Not done: the shared/exclusive split.** Correctness must stay excluded from timing
windows, but it need not serialise against *other arms' correctness* -- neither is a
measurement. `flock` supports `LOCK_SH`/`LOCK_EX` directly. Deferred because `flock` has no
writer preference (a timing window can starve behind continuous correctness traffic; needs a
gate file) and N concurrent correctness runs raise peak GPU memory. Gate it on
`KB_EVAL_PHASE_LOG` data showing correctness actually dominates the hold.

#### Which edits reach a live run, and which need a relaunch

`evaluate_in_subprocess` uses `start_method="spawn"` (`execution.py:352`), so the child
pickles `kernelbench_eval_worker` **by reference** and re-imports its module from disk
every eval. That splits the codebase in two:

| runs in | example | live runs pick it up? |
|---|---|---|
| eval **child** | `kernelbench_integration/eval_runner.py`, `src/kernelbench/eval.py` | **yes**, on the next eval spawned |
| long-lived **parent** | `evolving_common/execution.py`, `evolve_kb_batch.py` governor loop | **no** -- bound at import, needs relaunch |

This is why `7ac0e87` (eval deadline) applied only to newly launched runs while the
`num_perf_trials` change took effect in ~3 minutes with no restart.

**Two traps when verifying an edit landed.** An eval *spawned* before the edit keeps the
old value and can complete long after it -- with 300-900 s lock waits, stale records keep
arriving for 10+ minutes. So (a) match on the new value (`trials 25`), never on "any
record newer than the edit", and (b) the value is only visible in
`evaluation_terminal_output.jsonl`, not the arm log.

**Always publish atomically:** temp file in the same directory -> `ast.parse` -> `os.replace`.
An in-place write is readable mid-flight by a spawning child; that is the 2026-08-20
nine-arm incident.

**Fixed 2026-08-23, behind `KB_EVAL_SKIP_DEAD_REF_TIMING`.** When `baseline_runtime` is
supplied the reference timing inside the lock is dead work: `governor.py:467` sets
`ref_for_speedup = self._baseline_runtime` for the speedup, `governor.py:490` *overwrites*
the measured `ref_runtime` with the same baseline before it reaches any metric or prompt
(verified: 209 correct iterations of L1P22 across nine arms all report exactly `2.96`, the
`baseline_time_torch.json` median), and `ref_runtime_stats` has no reader anywhere in the
repo. Do not expect a throughput jump -- it is worth a get_inputs + H2D + `empty_cache` +
31 reference executions per eval.

**How to verify a live edit is inert.** Constant-fold the flags to their defaults and diff
the ASTs of the affected functions, rather than eyeballing the patch. That is what caught
the `excessive_speedup` nesting bug above; a hand review had passed the same diff.

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
bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/resume_run.sh <gpu> <run_dir_name> <ctx_mode> <start> [end]
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
  evolving_common/governor/gpu_lock.py        # cross-process GPU-eval lock, re-entrant
  evolving_common/governor/gpu_reserver.py    # 42 GB idle reservation; KB_GPU_RESERVE_GB
  kernelbench_integration/eval_runner.py      # precompile-then-lock boundary
  kernelbench_integration/governor.py         # speedup computation
src/kernelbench/eval.py            # _gpu_timing_lock: locks correctness+timing only
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
5. **Governance matrix is incomplete** — 5 of 7 cells untouched. Six arms/GPU is only
   safe for runs launched **after** the 2026-08-22 eval-deadline fix (§3.4); before it,
   6 arms lost 3.9% of evals to mid-wait kills versus 0.93% at 3. Post-fix, 2 GPUs clear
   the remaining cells in roughly one wave. The fix is unvalidated under real contention —
   smoke-test two arms on an idle GPU with `KB_GPU_EVAL_LOCK_TIMEOUT_SEC` lowered before
   committing a full wave to it.
6. **Replicate noise is ~20%, and it bounds every conclusion.** Two runs with
   *identical* config on the same GPU (`itr30` rep1 vs rep2, 2026-08-20) differ by 22%
   in `best_geomean` (0.799 vs 1.028, n=31 matched problems). Any arm-vs-arm delta below
   that magnitude is indistinguishable from LLM stochasticity at n=1 run per arm. Several
   deltas in `output/GH200x2/comparison.md` are in that range. **Run replicates for any
   cell you intend to draw a conclusion from**, and report a paired log-ratio CI rather
   than a bare geomean difference.
7. **Solo baselines drift and must not be reused across dates.** The same 49 problems
   took 87.8 min/problem (Aug 07), 79.8 (Aug 13), 80.1 (Aug 14), 64.7 (Aug 17) — a 26%
   swing from inference-endpoint latency, not from anything in this repo. Because the
   agent is LLM-bound, any throughput or speedup figure normalised against a baseline
   from another date is inflated by that drift; efficiency >100% is the tell. Compare
   only within a time window, or measure a fresh solo baseline alongside the runs.

---

## 7. Working preferences

- **Ask for confirmation before launching any `nohup` run** — they cost ~67 GPU-h each.
- Direct `nohup` from the tool layer gets denied; that is why launches go through
  `launch_run.sh` invoked via `bash`.
- Verify claims against artifacts before reporting. Several confident-sounding
  conclusions in this project turned out wrong until checked against the data.
