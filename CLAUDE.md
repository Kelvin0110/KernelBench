# KernelBench — Evolving-Agent Experiments

Working notes for the `features/evolving-agent-final` branch.

> **This file is shared across every server that runs these experiments and is
> committed.** Keep it host-agnostic: protocol, mechanism, analysis rules.
> Anything true of only one machine — hostname, GPU count, driver, which
> `results/timing/` folder is *ours*, which runs are live right now, local venv
> drift, local incidents — belongs in **`CLAUDE.local.md`**, which is per-host
> and gitignored. Read that file first; it tells you which server you are on.
>
> Two servers currently run this series. They are distinguished only by their
> `$HARDWARE` name, which selects both a timing baseline and a launcher folder:
>
> ```
> results/timing/<HARDWARE>/                          # that host's baseline
> scripts_integration/new_evolving_agent/env/<HARDWARE>/   # that host's launchers
> scripts_integration/new_evolving_agent/env/common/       # shared by both
> ```
>
> `env/hardware_env.sh` derives `$HARDWARE` from the directory name of the
> launcher you invoked, so **run the launcher inside your own host's folder and
> never pass `--hardware` by hand.** Copying that folder and renaming it is *usually*
> all it takes to add a machine — but the name must also be a valid
> `results/timing/` folder carrying a `median`. When it is not, the launcher must
> pre-set `KB_DEFAULT_HARDWARE`, as `env/NVIDIA_GH200x2/launch_run.sh` and
> `resume_run.sh` already do to redirect onto a median-bearing baseline.

---

## 1. What this project is

KernelBench evaluates LLM agents that write custom CUDA kernels for PyTorch
modules. This branch runs an **evolving agent** with a tiered memory:

- **L0** — per-problem iteration history (context management applies here)
- **L1** — a skill catalog shared across all problems in a batch
  (`shared_l1.jsonl`), governed by deletion / merging / refinement
- **L2** — *(newer, off by default)* standing instructions: L1 skills with
  enough cross-task usage and new-best attribution are promoted to permanent
  rules injected into every coder system prompt (`--enable-l2`,
  `evolving_common/governor/l2_promotion.py`)

We are running a controlled experiment series measuring how **L0 context-management
mode** and **L1 skill-governance** affect kernel quality.

Axes:

| axis | values |
|---|---|
| L0 context management | `truncation` (default/baseline), `folding`, `markov_report`, `selective_retention`, `compress_trigger` |
| L1 skill governance | `--skill-deletion`, `--skill-merging`, `--enable-skill-refinement` (7 non-empty combinations) |
| L2 promotion | `--enable-l2` (+ `--l2-render`, `--l2-min-rate`, …) — a third axis, not yet part of the planned matrix |

Governance and L2 arms hold context at `truncation` so the axes stay separable.

**Fixed protocol for every arm:** 50 problems (`subset_selection/selected_problems_50.csv`;
10 × L1, then 15 × L2, then 25 × L3 — in that order), 30 iterations, `gpt-oss-120b`,
`$HARDWARE` per host.

**Run cost is not a constant — measure it, don't budget from this file.** Historical
arms were 65–75 GPU-hours; the Aug-22 wave projects **42–51 h/arm**. The difference is
inference-endpoint latency drift (open item 10) plus the mid-run `num_perf_trials`
100→25 change, not anything about the protocol.

Current results and cross-run comparisons live under
`scripts_integration/new_evolving_agent_analysis/output/`. Known defects and
experiment-design caveats are recorded in `env/README.md` and in project memory.

---

## 2. Environment — read this before running anything

### 2.1 Two mandatory exports

CUDA is a **userspace install** (no sudo). Without these, `nvcc` is absent and
every `load_inline(cuda_sources=...)` build fails — silently producing kernels that
fall back to plain PyTorch while still scoring `correct=True`:

```bash
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:<repo>/.venv/bin:$PATH
nvcc --version        # expect: release 12.8, V12.8.93 (matches torch 2.11.0+cu128)
```

Or source `env/common/activate.sh`, which sets both. Every launcher sets them itself;
if you invoke `evolve_kb_batch.py` by hand, you must. Reinstall with
`env/common/install_cuda128_local.sh` (each `env/<HARDWARE>/` symlinks it).
Background: `scripts_integration/new_evolving_agent/env/README.md`.
Host-specific paths and verified versions: `CLAUDE.local.md`.

### 2.2 Always use `uv run --no-sync`

A bare `uv run` **re-syncs the venv and prunes packages**, e.g.

```
uv sync --dry-run → Would uninstall N packages
  - scikit-learn - scipy - joblib - threadpoolctl - pytest - ruff - ...
```

Removing scikit-learn makes every `--skill-merging` iteration die with
`coder_call_error`. Every launcher uses `--no-sync` at all call sites; keep it.

**How far each host has drifted differs — check `CLAUDE.local.md` for this
machine's `uv sync --dry-run` count before trusting any number here.** The
divergence has been as small as 9 packages on one server and as large as 73 on
another, so treat the prune as potentially catastrophic, not merely annoying.

**Why it drifts:** `pyproject.toml` declares `scikit-learn` in `[project] dependencies`
(promoted out of the `evolving-agent` extra), but `uv.lock` has **not** been
regenerated. They are intentionally out of sync, so `--no-sync` is load-bearing until
someone runs `uv lock && uv sync`. Do that only when no run is in flight — it shares
`.venv` with running jobs and drops pytest/ruff.

### 2.3 API keys and endpoints (`.env`)

| purpose | endpoint | key |
|---|---|---|
| chat (all LLM roles) | `inference-api.nvidia.com/v1` | `NVIDIA_INF_API_KEY` |
| embeddings (skill merge) | `inference-api.nvidia.com/v1` | `NVIDIA_INF_API_KEY` |

Model IDs differ per endpoint: `gpt-oss-120b` → `openai/gpt-oss-120b` (integrate) vs
`nvidia/openai/gpt-oss-120b` (inference). Use the aliases in `llm_client.py`, not raw IDs.

Embeddings choose their endpoint independently of chat via `NVIDIA_EMBED_ENDPOINT`
(default `inference`; model default `nvidia/qwen/qwen3-embedding-0.6b`).

> `env/probe_integrate_key.py`, referenced by earlier revisions of this file, is
> **not in the repo** and was never tracked. To isolate a key failure from a model
> failure, call the endpoint directly with the alias table in `llm_client.py`.

---

## 3. Running an experiment

### 3.1 The launcher (use this, don't hand-roll nohup)

```bash
bash scripts_integration/new_evolving_agent/env/<HARDWARE>/launch_run.sh <gpu> <run_name> <ctx_mode> [extra flags...]
```

Use **your own host's** `<HARDWARE>` folder (`CLAUDE.local.md` names it). The folder
name is what selects the timing baseline, via `env/hardware_env.sh`.

It preflights nvcc, ninja, the baseline dir, the API key, GPU-idleness and a live
`load_inline(cuda_sources=...)` compile probe, then launches under `nohup` and prints
the pid and log path. (The `import sklearn` check for merge arms is in `launch_wave.sh`
only — `launch_run.sh` does not have it.) Every check exists because
its absence once silently corrupted a ~70 h run.

Fixed args it always supplies: `--max-problems 50 --max-iterations 30 --hardware
$HARDWARE --nvidia-endpoint inference --model gpt-oss-120b --coder-timeout-sec 600
--results-root runs_evolving/gpt-oss-120b/`.

**`launch_wave.sh` is the multi-arm form.** It takes a spec file (one arm per line:
`tag | context-mode | extra flags`) instead of a single arm, staggers launches by
`LAG_SEC`, writes a `wave_gpu<N>_<stamp>.manifest.tsv`, and supports
`dry-run` / `status` sub-modes. Unlike `launch_run.sh` it can express governance and
L2 arms. Specs live in `env/*.spec`.

#### Baseline resolution is a correctness check, not a path lookup

`kb_require_hardware` (`env/hardware_env.sh`) **fatals** when
`results/timing/$HARDWARE/baseline_time_torch.json` lacks a `median` field.
`get_timing_stats()` started recording a median in `6a3e972` and
`runtime_from_stats()` prefers it, so a pre-`6a3e972` baseline makes the run divide
candidate *median* by baseline *mean* — on the 50-problem subset that shifts 25 of 50
problems by >5% and inflates some ~4×. Silent metric error, not a crash.
`ALLOW_MEAN_BASELINE=1` downgrades it to a warning; don't.

**A baseline cannot be swapped under a running arm.**
`kernelbench_integration/baseline_timing.py::_load_baseline_json` is `@lru_cache`d on
the path, and `KBGovernor` (`governor.py:154`) lives in the long-lived parent — so the
JSON is read once per arm and held for its whole life. Editing the file on disk changes
nothing for a live run (unlike `src/kernelbench/eval.py`, which every eval re-imports;
see §3.4). Changing baselines means relaunching.

Corollary for analysis: a per-problem baseline error is a **common factor across all
arms scored against the same file**, so it cancels in arm-vs-arm ratios and moves only
the absolute level. It does, however, make those runs non-comparable to runs scored
against a different baseline file. Record which baseline a run used.

### 3.2 Naming conventions

- **Run name:** `base_agent_gpt_oss_120b_<tag>_itr30_GH200`
  `<tag>` ∈ `{markov, folding, compress, selective_r5, deletion, refinement, merge_sim085, ...}`.
  **Encode any non-default parameter in the tag** (`selective_r5` = 5 recent rounds,
  `merge_sim085` = similarity 0.85). The runner appends `_YYYY_MM_DD_HH_MM`.
- **Log:** auto-derived, `<run_name>_<Mon>_<D>.log` in the repo root.
- **Results:** `runs_evolving/gpt-oss-120b/<run_name>_<timestamp>/`

### 3.3 Examples

Substitute your own `$HW` (`env/<HARDWARE>/`, per `CLAUDE.local.md`).

```bash
HW=scripts_integration/new_evolving_agent/env/<HARDWARE>

# context-management arm
bash $HW/launch_run.sh 0 base_agent_gpt_oss_120b_markov_itr30_GH200 markov_report

# compress_trigger needs its tuning flags
bash $HW/launch_run.sh 0 base_agent_gpt_oss_120b_compress_itr30_GH200 compress_trigger \
  --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15

# governance arms — context held at truncation
bash $HW/launch_run.sh 1 base_agent_gpt_oss_120b_deletion_itr30_GH200   truncation --skill-deletion
bash $HW/launch_run.sh 1 base_agent_gpt_oss_120b_refinement_itr30_GH200 truncation --enable-skill-refinement
bash $HW/launch_run.sh 1 base_agent_gpt_oss_120b_merge_sim085_itr30_GH200 truncation \
  --skill-merging --skill-merge-similarity 0.85

# a whole wave from a spec file (governance + L2 arms; see env/wave_gpu0.spec)
bash $HW/launch_wave.sh 0 scripts_integration/new_evolving_agent/env/wave_gpu0.spec dry-run
bash $HW/launch_wave.sh 0 scripts_integration/new_evolving_agent/env/wave_gpu0.spec
bash $HW/launch_wave.sh 0 scripts_integration/new_evolving_agent/env/wave_gpu0.spec status
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

**Gotcha — the two launchers guard concurrency differently.** `launch_run.sh:41` aborts
when the GPU reports >1000 MiB used; an idle arm holds ~558 MiB, so arm 2 passes but arm 3
reads ~1.1 GB and is rejected. Raise that threshold (or bypass the guard) to launch three
or more. `launch_wave.sh` does **not** use that check — it gates on
`MIN_FREE_MIB=20000` plus `MAX_ARMS_PER_GPU=6` (counted from `CUDA_VISIBLE_DEVICES` in
`/proc/*/environ`), which is how 5-arm and 4-arm groups get launched. Use the wave
launcher when you want more than two arms on a GPU rather than editing the guard.

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
| Aug-22 wave, N=100, **9 arms on ONE GPU** | 76% | 298 s |

**The hold is ~20 s and always was** (busy-period inter-completion gap; consistent with
the Aug-20 median wait of 22 s at 3 arms). Only ~1 s of it is the timing loop -- the
rest is correctness trials, input generation, `empty_cache` and syncs. So:

- `num_perf_trials` was cut 100 -> 25 on 2026-08-23 (`eval_runner.py:108`). It buys back
  only ~2 s of a ~20 s hold, but it needed no restart -- see the propagation rule below.
- Skipping the discarded live reference timing is still correct (see below) but it is
  worth a few seconds, **not the "free 2x" an earlier version of this section claimed**.
- The earlier "~4.9 s hold, ~6 arms/GPU" figures came from a probe on a different
  problem mix (L3P4/9/42) and do not generalise. **~3 arms/GPU is the ceiling for
  *throughput*** until someone profiles the inside of the locked section.
  *Corrected 2026-08-25:* the critical section **was** shrunk (hoisted input gen, skipped
  dead reference window) and the mutex **was** replaced by a 3-slot semaphore, exactly as
  the corollary below demands. Measured at 9 arms/GPU the ceiling is gone — see
  "9 arms/GPU measured" below. Read this bullet as a description of the *mutex* regime.

Corollary: to actually raise the ceiling you must shrink the critical section.
(That has since been done; the corollary held.)

**Three arms/GPU is a throughput ceiling, not a safety limit — post-fix.**
*Corrected 2026-08-25: it is no longer a throughput ceiling either — 9 arms/GPU scales
linearly. See "9 arms/GPU measured" below. The rest of this paragraph is retained because
its warnings about cross-boundary comparison still apply.* Once the
eval deadline stopped charging lock queueing against the work budget (2026-08-22), more
arms stopped *corrupting* evals and merely stopped paying for themselves. A 5-arm +
4-arm wave (two GPUs, **not** 9 on one) has since run 19 h with `orph=0`, `unlock=0`
and a 0.84% eval-timeout rate over 5011 evals, at the cost of lock waits reaching 685 s.
So: exceed 3 when you need matrix coverage more than wall-clock, but do not report
throughput from such a wave, and never do it on a pre-2026-08-22 build. Do not read that
0.84% against the pre-fix "0.93% at 3 arms" figure — the comparison crosses the code
boundary and is not evidence that 5 arms beats 3.

**Unequal arms-per-GPU biases the comparison, one-directionally.** Measured on that
wave: GPU0 (5 arms) 18.2% of evals logged a wait, mean 11.1 s/eval; GPU1 (4 arms) 16.9%,
mean 8.0 s — GPU0 absorbs ~39% more. Waiting itself cannot deflate a speedup (timing
windows stay exclusive), but *unlocked* GPU work — reference-model construction,
nvcc/ninja, `custom_model.to(device)`, `empty_cache` on reserver release — can land
inside another arm's timing window, and GPU0 exposes each window to 4 concurrent
interferers versus GPU1's 3. Since `speedup = fixed_baseline / measured_runtime`, the
busier GPU is systematically penalised.

**So put every arm you intend to compare on the same GPU as its control.** If the only
truncation control sits on GPU0, every GPU1 arm is compared across a contention boundary
in its own favour. And the bias cannot be removed afterwards: `KB_EVAL_PHASE_LOG` gives
no hold data unless it was enabled at launch, the measured reference window — which
would be a perfect per-GPU contention probe — is overwritten by the fixed baseline
before being recorded (`governor.py:467,490`), and `gpu_lock` never reports hold time.

#### The ~19 s that is not the timing loop is `get_inputs()` (2026-08-23) -- solved

Measured, not inferred. `get_inputs()` is called **three times inside the lock** --
`eval.py:887` (correctness trial), `:720` (candidate window), `:772` (reference window) --
and it is pure-CPU `torch.rand`. Wall-clock cost of those three calls, measured on this
box with `torch.set_num_threads(4)` (numbers are host-specific — re-measure per server):

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

#### MEASURED: concurrent evals cost <3%, so the mutex was the wrong design (2026-08-23)

Everything above this line about "~3 arms/GPU" was inferred from throughput, never from a
controlled measurement of *fidelity*. Fidelity has now been measured directly, and **the
fidelity argument for a strict mutex was wrong** — the throughput ceiling itself stands.
(*Corrected:* an earlier revision of this paragraph read "the ceiling framing was wrong". The
probe below measured fidelity only; its own L1P34 wall-time swing, 21 s -> 425 s, is evidence
*for* the throughput ceiling, not against it.)
(*Corrected again, 2026-08-25:* the throughput ceiling does **not** stand once the mutex is
replaced by a 3-slot semaphore — that change had not been made when the sentence above was
written. Both this section's fidelity result and the ceiling's removal are now measured; see
"9 arms/GPU measured" below.)

Method: 6 saved kernels spanning input volume and kernel duration (L1P100 4.3 GB/2.4 ms,
L1P34 7.5 GB/5.4 ms, L2P19, L2P94 0.65 ms, L3P42 56 ms, L3P5), 5 repeats each, warm builds,
trims on, **`KB_GPU_EVAL_LOCK=0` so evals genuinely overlapped**, each kernel compared
against its own solo value on an otherwise idle GPU.

| | median inflation | worst |
|---|---|---|
| degree 2 | **1.2%** | 2.3% |
| degree 3 | **0.7%** | 2.8% |

Against the ~30% replicate noise (open item 9) that is an order of magnitude below the
noise floor. The decisive observation is L1P34: across repeats its **wall time swung
21 s -> 425 s while its measured runtime never left 5.36-5.52 ms**. Contention lands on H2D
and host memory, which sit *outside* the CUDA-event window, so it costs throughput and not
fidelity. The strict mutex was buying ~1-3% of a number while paying 250-320 s median waits.

Two design changes follow, both gated and both on for waves launched after this date:

- **`KB_GPU_EVAL_LOCK_SLOTS`** (`gpu_lock.py`) -- a counting semaphore built from the same
  crash-safe `flock`: one file per slot, acquiring any admits the eval, kernel still drops
  all on process death. `slots=1` is byte-identical to the old mutex (same single file).
  Verified peak concurrency 1/2/3/4 for slots 1/2/3/4; malformed values fail closed to 1;
  re-entrancy preserved. **Never mix slots=1 and slots>1 on one GPU** -- different file
  names, so they would not interlock.
- **`KB_EVAL_UNLOCK_CORRECTNESS`** (`eval.py`) -- **reverted to OFF on 2026-08-23, see the
  memory note below** -- correctness trials run outside the lock,
  which then covers only the timing window(s). A/B on L1P100, warm build: hold
  **1.96 s -> 0.78 s** with runtime unchanged (2.42/2.44 vs 2.38/2.45; solo control 2.41).
  This supersedes the shared/exclusive-split note below -- the probe above is strictly
  *more* aggressive than an SH/EX split, since it let timing windows overlap each other
  too, and still cost <3%.

  **But that probe measured runtime, not memory, and this flag was reverted hours later.**
  Concurrency of *device-resident* evals is bounded by the lock, and unlocking correctness
  removed that bound: each eval reserves ~30 GB (level-1 problems) to ~52 GB (L1P34) because
  the caching allocator retains input copy + output + intermediates + a second copy for the
  timing window. Six arms/GPU took a 146.8 GB card to 144.8 GB (~99%) and 1.8% of evals to
  CUDA OOM, each recorded as `compiled=True correct=False`. **Sizing rule with correctness
  locked: concurrent residents = SLOTS, so budget ~30-52 GB x SLOTS and do not raise SLOTS
  while arms are inside subset problems 1-5.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  would cut the retention but allocation happens inside the timed forward while the baselines
  were measured under the default allocator, so it is deliberately not set.

**The other thing that was wrong: problems 1-5 are not representative.** Measured
`get_inputs()` across all 50 subset problems -- the first five (`L1P100/22/26/33/34`, in
subset order) cost 12.5-21.7 s each and carry **74% of the entire benchmark's
input-generation cost**; problems 6-50 average 0.65 s (median 0.12 s). Every wave this
project has ever killed died inside problems 1-4, so **every contention number in this file
was collected in the worst 5 problems of the benchmark.** Both servers show it independently:
problems 1-5 run ~1.25-1.63x slower per problem than problems 6-17 (78 vs 48 min on one
server, 93.7 vs 74.8 min on the other). Do not extrapolate a wave's first days to its steady
state.

#### 9 arms/GPU measured — the throughput ceiling is gone; `SLOTS` is the real cap (2026-08-25)

Every "~3 arms/GPU" figure above was collected under the **mutex** (`SLOTS=1`), before the
critical section was shrunk. With `KB_EVAL_HOIST_INPUT_GEN=1`, `KB_EVAL_SKIP_DEAD_REF_TIMING=1`,
`KB_EVAL_UNLOCK_CORRECTNESS=0` and `KB_GPU_EVAL_LOCK_SLOTS=3`, a live 15-arm wave (9 gpt-oss on
GPU0, 6 terra on GPU1) was measured over 24 h. **Scaling is linear and nothing is binding.**

The clean arm-count contrast is one GPU against itself — same model, same GPU, same baseline:

| GPU0 | aggregate | per arm |
|---|---|---|
| 6 arms | 169 evals/h | **28.1 /h** |
| 9 arms | 245 evals/h | **27.2 /h** |

**50% more arms cost ~3% of per-arm throughput.** Host-wide the same holds: 12 -> 15 total arms
moved per-arm rate 27.0 -> 27.2 evals/h with aggregate 324 -> 408. Do **not** compare GPU0's
9-arm rate against GPU1's 6-arm rate to make this point — that crosses a *model* boundary
(gpt-oss vs terra have different endpoint latency) and cannot isolate arm count.

Resource utilisation at that degree, and the headroom in each:

| | GPU0 (9 arms) | GPU1 (6 arms) |
|---|---|---|
| lock util (of 3 slots) | 13-27% | 6-8% |
| *1-slot equivalent* | *39-81%* | *17-25%* |
| lock waits >5 s | 0.63% (6 h) | **0 / 3736** |
| GPU util median | 4% | 0% |
| device mem mean / p90 / max | 20.5 / 55.2 / **141.1** GB | 2.1 / 8.6 / 28.0 GB |
| host load1 | 11% of 144 cores | — |
| host mem | 6% of 1227 GB | — |

The italic row is the point: at `SLOTS=1` GPU0 would sit at **81%** and queue hard. **The
semaphore, not the arm count, is what made 9 arms viable.**

**The LLM endpoint is shared across both GPUs, so it is a host-wide budget, not a per-GPU one.**
It showed zero degradation from 12 to 15 arms. Coder call rate equals eval rate (27.2/h and
26.4/h), i.e. essentially every coder call reaches an eval, so an arm's iteration rate *is* its
coder rate — a convenient proxy when phase logs are absent.

**What actually caps this now is `SLOTS`, and it is already at the edge.** With correctness
locked, concurrent device residents = `SLOTS` **regardless of arm count**, so adding arms does
not raise the memory ceiling — it only raises the *probability* of hitting the worst case, by
making it likelier that all 3 slots hold big-input problems at once. At `SLOTS=3` the worst case
is 3 x ~52 GB = ~156 GB against a 143.4 GB card, and the observed 141.1 GB peak shows that tail
is real, not theoretical. **Raise arms, not slots.**

**Caveat on reading any lock number: utilisation tracks the problem list, not the arm count.**
Problems 1-5 carry 74% of the benchmark's input-generation cost, so a window inside them shows
several times the hold of a steady-state window. The 13% and 27% figures above differ mostly for
that reason, not because of arm count. Size a ceiling from problems 1-5, never from steady state.

**Corrected 2026-08-25: the memory gate at `factor=2.5` never fired.** This paragraph first
read "the memory gate works", citing the three arms carrying `KB_EVAL_MEM_GATE_FACTOR=2.5` that
traversed L1P34 (7.5 GB inputs, the problem behind every OOM) with **0 OOM in 90 evals** against
**16 / 360 = 4.4%** for the twelve ungated arms. The outcome is real; the attribution was not.
**Across 8,334 gated evals, zero waited more than 0.05 s** (p99 0.002 s) -- the gate never once
restricted admission, so the clean run was scheduling luck: those three arms crossed L1P34 while
the other twelve were deep in problems 20-40. Do not cite that 0/90 as evidence the gate works.

**Why it cannot bind at 2.5.** The reservation is `factor x input_bytes` against a budget of
`KB_EVAL_MEM_GATE_FRAC x card` (default 0.85). On L1P34 that is `2.5 x 7.0 GiB = 17.5 GiB`, so all
three slots together ask 52.5 GiB of a 121.3 GiB budget and are always admitted. **The gate binds
only when `factor x input_bytes` exceeds half the budget**, i.e.
`factor > budget / (2 x input_bytes)`.

*Corrected 2026-08-25 (same day, before the value was ever used): that 8.7 is the boundary above
which the gate admits **1**, not 2 -- and rounding it up to 9 crosses it.* The gate admits the Nth
eval iff `N x need <= budget`, so `max_conc = floor(budget / (factor x input_bytes))`.

*Corrected again 2026-08-25 (second pass, code read + 6-process probe): the boundaries were quoted
as 8.71 and 5.80; they are **8.667 and 5.778**, so 8.7 admits **1**, not 2, and 5.8 admits 2, not
3.* The formula was right; the **card size** was wrong. `budget` is
`frac x torch.cuda.mem_get_info(device)[1]` (`eval.py:503-506`), and on a GH200 144G that total is
`153,276,645,376 B = 142.75 GiB = 146,176 MiB` -- **not** the `146,831 MiB` nvidia-smi prints
(~655 MiB is not CUDA-visible). So `budget = 0.85 x 142.75 = 121.34 GiB`, not ~122, and
`budget / input = 121.34 / 7.00 = 17.334`, not 17.43. That ~0.6% is enough to flip 8.7 across a
band edge. Units, while we are here: L1P34's timing input is **exactly 7.000 GiB**
(112 x 64 x 512 x 512 fp32 = 7,516,192,768 B) = 7.516 GB decimal -- so this file's "7.5 GB" and
"7.0 GB" are the same tensor, and the band table below is GiB throughout.

| factor | need | admits | verdict |
|---|---|---|---|
| <= 5.778 | <= 40.4 GiB | 3 | inert -- `SLOTS` governs, this is what 2.5 did |
| **5.778 - 8.667** | **40.4 - 60.7 GiB** | **2** | **correct: 2 x 49.0 GiB = 98.0 GiB of a 142.75 GiB card** |
| > 8.667 | > 60.7 GiB | 1 | over-restrictive -- serialises the biggest problems |

Both edges are **verified empirically**, not just arithmetically: six spawned processes calling
`_device_memory_reservation` with an L1P34-sized need gave observed peak concurrency
3 / 3 / 2 / 2 / 2 / 2 / 1 / 1 at factor 5.0 / 5.7 / 5.78 / 7.0 / 8.6 / 8.66 / 8.67 / 9.0 -- exactly
`floor(budget/need)`, with the transitions falling between 5.7 and 5.78 and between 8.66 and 8.67.
(Probe it with `spawn`, never `fork`: after `fork` the child's `torch.cuda.mem_get_info` raises,
the gate's blanket `except` swallows it, and every process is admitted -- which looks like "the
gate does not work".)

Retention measured directly on this host (`max_memory_allocated` over a correctness copy + a
timing copy + both models, L1P34): **49.000 GiB = 7.00x input allocated; 50.75 GiB = 7.25x
reserved**. So **7 is the measured multiple and it sits mid-band**, which is why **`factor=7` is
both the physically accurate value and the correct one**. It is not a coincidence: the factor is
meant to *be* the retention multiple, so derive it from measured retention and then check it
against the band. Anything <= 5.778 is telemetry, not admission control.

**The gate is dead without `KB_EVAL_HOIST_INPUT_GEN=1` -- an undocumented precondition, and the
one most likely to silently un-arm your next wave.** `_need` is computed only when
`_cpu_inputs_timing is not None` (`eval.py:909`), and that variable is assigned in exactly one
place: inside `if _env_flag("KB_EVAL_HOIST_INPUT_GEN", default=False):` (`eval.py:859-868`, and it
is reset to `None` on any exception there). With the hoist off there are no host-side tensors to
size a reservation from, so `_need` stays 0, `_device_memory_reservation` takes its
`need_bytes <= 0` early return, and `KB_EVAL_MEM_GATE_FACTOR` does nothing at any value. **Both**
preconditions must hold: hoist on AND factor inside the binding band.

**Mem-gate waiting is charged against the eval deadline; lock waiting is not.** `gpu_lock`
publishes its running wait through `set_wait_reporter` and `execution.py:389-392` adds it to
`timeout_s` -- that is the 2026-08-22 fix. **Nothing publishes `_device_memory_reservation`'s
wait**, so a gate wait is spent out of the 600 s work budget, and a long enough one reproduces
exactly the failure that fix removed: SIGTERM mid-wait, recorded as a fake compile failure the
governor then "debugs". At `factor=7` a wait is a couple of queue rounds on problems 1-5 (tens of
seconds), so this is a caveat rather than a blocker -- but do not push the factor toward the
admit-1 band without also cutting `KB_EVAL_MEM_GATE_TIMEOUT_SEC` (default 600, i.e. the whole eval
deadline) and watching `mem_gate_waited_sec`. Note also that the gate's timeout branch `print()`s
to eval stdout, which `governor.py:1203` splices into the agent's prompt.

Wave health over 14,542 evals, for reference: OOM 0.165% (and **0 in the last 4,945**),
eval-timeout 0.743%, `proceeding UNLOCKED` 0, `worker_error` 0.

**Practical guidance -- the sizing recipe for the next wave.** Target **12 arms/GPU at
`SLOTS=3` with `KB_EVAL_MEM_GATE_FACTOR=7`**, fixed at launch.

- **Arms are the safe lever; slots are not.** Concurrent device residents = `SLOTS`, so arm count
  changes only the *probability* that three big-input evals coincide, never the ceiling. Lock
  utilisation at 9 arms was 13-27% of three slots, so 12 stays well inside capacity, and the
  measured price is ~3% of per-arm throughput per 50% of arms added.
- **Never raise `SLOTS` above 3 on a GH200-class card.** Both GPUs touched ~98% of the card at
  `SLOTS=3`, and GPU1 did it with only 6 arms -- which proves it is a slots effect, not an
  arm-count effect. `SLOTS=4` puts the worst case at 4 x ~52 GB = ~208 GB: a guaranteed OOM, and
  an OOM is recorded `compiled=True correct=False`, so the governor then debugs a kernel that was
  never broken. Slots also buy nothing here -- the lock sits at 13-27% of capacity, so it is not
  the constraint. *(Qualifier added 2026-08-25: that 4 x 52 GB worst case assumes the gate is
  OFF. With `factor=7` armed, the gate caps L1P34 at 2 residents no matter what `SLOTS` says, so
  above 3 the slots are simply inert on the problems that matter -- still no reason to raise
  them, but the failure mode is waste, not a guaranteed OOM.)*
- **Re-derive the factor for your own card** instead of copying 7:
  `max_conc = floor(frac x cuda_visible_total / (factor x largest_input_bytes))`, and take
  `cuda_visible_total` from `torch.cuda.mem_get_info()[1]`, **not** from nvidia-smi -- that
  discrepancy is what made the first two versions of the band table wrong. Then check *armed*
  before *fires*: the phase log's **`mem_need_gb`** field is the reservation each eval asked for,
  so an all-zero `mem_need_gb` column means the gate is off (factor 0 **or** hoist off) -- that
  is the check that would have caught the 2.5 setting. `mem_gate_waited_sec` is the second, weaker
  check: all-zero there is also consistent with "armed, correctly sized, no coincidence happened".
- **Size the worst case from a heterogeneous triple, not from 3 x the biggest.** The gate's real
  guarantee is `sum(reservations) <= budget`, and admission is order-dependent, so the worst
  admitted set is greedy: at `factor=7`/`FRAC=0.85` on a GH200 144G that is
  `L1P34 49.0 + L1P22 42.0 + L1P100 28.0 = 119.0 GiB` reserved (a second 42 GiB would be
  rejected at 133 > 121.34). Scaling by the measured 7.25/7.00 reserved-vs-allocated ratio and
  adding ~0.52 GiB of idle CUDA context per arm gives **~126 GiB of 142.75 = 88.5%** at 6 arms --
  safe, but with only ~16 GiB spare. If you want that margin back, drop `KB_EVAL_MEM_GATE_FRAC`
  to 0.80: `factor=7` still admits 2 on L1P34 (114.2 / 49.0 = 2.33), the triple above is refused,
  and the worst case falls to ~97 GiB = **68%** -- paid for only inside subset problems 1-5.

- Two reasons the degree is still not adjustable mid-wave: adding arms plants a contention seam
  inside the affected runs, and unequal arms-per-GPU biases comparisons one-directionally (see
  above). New arms also start at problem 1, i.e. straight into the five problems carrying 74% of
  input-generation cost and every OOM this project has seen. **Set the degree at launch.**

#### The eval-lock switches (2026-08-23), all inert unless exported

`src/kernelbench/eval.py` is re-imported by every eval spawn, so an unconditional change
reaches live arms. Every knob below is therefore gated, so it takes effect for newly launched
waves only.

**There are six, and four of them default to off.** The three boolean trims
(`KB_EVAL_HOIST_INPUT_GEN`, `KB_EVAL_SKIP_DEAD_REF_TIMING`, `KB_EVAL_UNLOCK_CORRECTNESS`)
default `False` via `_env_flag` (`eval.py:573-584`), which fails closed on unset, and
`KB_EVAL_MEM_GATE_FACTOR` defaults `0`. The other two are not switches:
`KB_GPU_EVAL_LOCK_SLOTS` defaults **`1`**, which is the historical mutex rather than "off",
and `KB_EVAL_PHASE_LOG` is a path that emits nothing until set.

**Which subset a launcher exports differs per host *and* per script**, so never infer it —
one host's `launch_wave.sh` sets both trims plus `KB_EVAL_UNLOCK_CORRECTNESS=1` and
`KB_GPU_EVAL_LOCK_SLOTS=3`, the other sets only the two trims, and `resume_run.sh`
deliberately sets `SLOTS=1`, `UNLOCK_CORRECTNESS=0`, `MEM_GATE_FACTOR=0`. No launcher in
this repo turns the mem gate on. Read the launcher you invoked, and `/proc/<pid>/environ`
for a run already in flight.

| env var | what it does |
|---|---|
| `KB_EVAL_HOIST_INPUT_GEN` | builds `get_inputs()` tensors on the CPU before taking the lock; H2D stays locked. Falls back to the in-lock path on any failure, and skips correctness pregeneration unless `num_correct_trials == 1` (5-trial callers would hold five input sets in host RAM). |
| `KB_EVAL_SKIP_DEAD_REF_TIMING` | skips the reference *measurement* when a fixed baseline is supplied. **Only the measurement.** The excessive-speedup / reward-hack flag lives in the same `if` block but depends on `baseline_runtime` and the candidate runtime, never on the window -- gating it too would silently disable `is_hack`, the `is_new_best` veto, and hack filtering in `best_geomean`. That bug was caught in review; do not reintroduce it by skipping the whole block. |
| `KB_EVAL_PHASE_LOG` | path to append one JSON line per eval: `held_sec`, `waited_sec`, `hoisted`, `unlocked_correctness`, `lock_slots`, `ref_window`, and a non-overlapping phase breakdown with an `other_sec` residual. Unset -> emits nothing. |
| `KB_EVAL_UNLOCK_CORRECTNESS` | runs the correctness trials outside the lock, leaving only the timing window(s) held. Hold 1.96s -> 0.78s on L1P100 with the recorded runtime unchanged. When on, the correctness trio is excluded from the `other_sec` residual -- otherwise it goes negative. **Leave this OFF on a shared GPU.** It removes the bound on how many evals are DEVICE-resident at once (correctness does the H2D of the whole input set plus two model forwards), and each eval reserves ~30 GB on a level-1 problem / ~52 GB on L1P34. With 6 arms/GPU it drove a 146.8 GB card to 144.8 GB and 1.8% of evals to CUDA OOM -- recorded as `compiled=True correct=False`, so the governor debugs a kernel that was never broken. Locking it again cut peak 141.4 -> 72.6 GB and OOM to 0, at the cost of ~5.7s of hold, with the lock still showing zero waits over 5s. **Check your own launcher before trusting the default:** the in-code default is OFF (`eval.py:897`), and `resume_run.sh` sets `0`, but at least one host's `launch_wave.sh` still defaults it to `1`. |
| `KB_GPU_EVAL_LOCK_SLOTS` | counting semaphore, default 1 (= the historical mutex, same lock file). See the measurement above. **Not every launcher sets it** -- one host's `launch_wave.sh` defaults it to 3, the other does not set it at all, and `resume_run.sh` defaults it to 1; `grep -n KB_GPU_EVAL_LOCK_SLOTS env/<HARDWARE>/*.sh` before launching. **The behavioural read is in the submodule, not `eval.py`** -- `gpu_lock.py:153` inside `lock_slots()`; `eval.py:568` reads it too, but only to stamp `lock_slots` into the phase log. That puts it on the **eval-child** side of the propagation table below, so a submodule checkout changes it for live arms. **Slot files are keyed by physical GPU UUID**, so every arm on a GPU shares the same N files -- 9 arms at slots=3 still means at most 3 concurrent evals. Verified by resolving `lock_paths()` under `CUDA_VISIBLE_DEVICES=0/1`. Corollary: at N>1 the files become `<base>.slotK` (`gpu_lock.py:159-164`), which do **not** interlock with a `slots=1` process, so two groups launched with different slot counts give 3+N concurrent and silently fail to exclude each other. Always pin the same value for every arm on a GPU. |
| `KB_EVAL_MEM_GATE_FACTOR` | device-memory admission on top of the slots, sized in bytes: reserves `factor x input_bytes` under a flock, so effective concurrency is `min(slots, floor(budget/need))`. 0 = off (default). **Requires `KB_EVAL_HOIST_INPUT_GEN=1` -- without it `_need` is 0 and this variable does nothing at any value** (`eval.py:909` gates on `_cpu_inputs_timing`, which only the hoist assigns). **Corrected 2026-08-25: use ~7, not the ~2.5 this row used to recommend (and not the ~9 an earlier revision said -- 9 admits only 1 on L1P34).** At 2.5 the gate provably never binds (0 of 8,334 evals waited >0.05 s); on a GH200 144G it is inert while `3 x factor x input_bytes <= budget` (**factor <= 5.778**, corrected from 5.8) and over-restrictive above `budget / (2 x input_bytes)` (**factor > 8.667**, corrected from 8.7 -- both edges re-derived from the CUDA-visible 142.75 GiB total and verified with a 6-process probe). Aim for the middle: the measured retention multiple, **7.00x allocated / 7.25x reserved on L1P34**. Turn it on for any arm that will traverse subset problems 1-5. Slots bound how many evals are resident, not how much they need -- 3 x ~49 GiB on L1P34 overruns a single GH200-class card; size against your own card. Companions: `KB_EVAL_MEM_GATE_FRAC` (budget as a fraction of the card, default 0.85) and `KB_EVAL_MEM_GATE_TIMEOUT_SEC` (default 600, then proceeds anyway **and prints to eval stdout, which reaches the agent's prompt**). Dead reservers are pruned via `/proc`; an eval needing more than the whole budget is still admitted when nothing else holds a reservation, so it cannot wedge with the GPU idle. Unlike the GPU lock's wait, **this wait is not published to the parent, so it is charged against the 600 s eval deadline**. |

None of these changes a recorded number. The two trims only shorten the hold and the mem
gate only ever admits fewer evals, so a wave launched with them sees *less* contention
deflation than one launched without. `KB_GPU_EVAL_LOCK_SLOTS>1` runs the other way — it
raises concurrency, which the table above prices at 0.7-1.2% median (2.8% worst) runtime
inflation. Either way you have crossed a boundary: compare speedups across it with the same
care as across a baseline change.

**Check whether the wave you are analysing actually had them on** — `/proc/<pid>/environ`,
not the launcher source. A wave launched before these flags were added to `launch_wave.sh`
runs with all of them off, and `_env_flag(..., default=False)` fails closed on unset.

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

**Superseded: the shared/exclusive split.** This was the planned next step -- `LOCK_SH` for
correctness, `LOCK_EX` for timing. It is no longer worth building. `KB_EVAL_UNLOCK_CORRECTNESS`
takes correctness out of the lock entirely, and `KB_GPU_EVAL_LOCK_SLOTS` admits N timing
windows concurrently, which together go further than an SH/EX split would have -- and the
2026-08-23 probe shows the fidelity cost of doing so is <3%. The `flock` writer-preference
starvation problem that made the split awkward does not arise for a counting semaphore.

#### Which edits reach a live run, and which need a relaunch

`evaluate_in_subprocess` uses `start_method="spawn"` (`execution.py:352`), so the child
pickles `kernelbench_eval_worker` **by reference** and re-imports its module from disk
every eval. That splits the codebase in two:

| runs in | example | live runs pick it up? |
|---|---|---|
| eval **child** | `kernelbench_integration/eval_runner.py`, `src/kernelbench/eval.py`, `evolving_common/governor/gpu_lock.py` | **yes**, on the next eval spawned |
| long-lived **parent** | `evolving_common/execution.py`, `evolve_kb_batch.py` governor loop, `gen3_stages.py`, `baseline_timing.py` cache | **no** -- bound at import, needs relaunch |

This is why `7ac0e87` (eval deadline) applied only to newly launched runs while the
`num_perf_trials` change took effect in ~3 minutes with no restart.

**The submodule counts.** `git -C Self-Evolving-Agent pull` (or any checkout that moves
its working tree) is a live code edit to every arm whose files land on the child side of
that table — `gpu_lock.py` above all. Check `git submodule status` for a `+` before
assuming the code you are reading is the code your runs started with, and re-check eval
health across the change's timestamp, not just afterwards.

**Two traps when verifying an edit landed.** An eval *spawned* before the edit keeps the
old value and can complete long after it -- with 300-900 s lock waits, stale records keep
arriving for 10+ minutes. So (a) match on the new value (`trials 25`), never on "any
record newer than the edit", and (b) the value is only visible in
`evaluation_terminal_output.jsonl`, not the arm log.

**Always publish atomically:** temp file in the same directory -> `ast.parse` -> `os.replace`.

#### Killed arms leave live-looking directories -- filter on processes, not dirs

A killed arm's run directory stays under the run's `--results-root` with all its evals
intact. Any glob over run directories therefore mixes its **pre-kill, higher-contention**
evals into what you think is the current state. This produced three separate wrong
numbers in one session, including a lock-wait p50 of 284 s where the true live figure
was 86 s. Always intersect with the live process list:

```bash
ps -eo cmd= | grep -oP '(?<=--run-name )\S+' | sort -u   # the only source of truth
```

**Lock-hold estimates from these artifacts are not trustworthy.** Busy-period
inter-completion gap, 1/throughput-assuming-saturation, and wait/(arms-1) disagree with
each other by 5x (20 s vs 83 s vs 112 s) and each is biased in a different direction.
Per-eval *work* (coder-turn -> eval-record, minus logged wait) is the one robust
quantity: 72-88 s median, and it barely moves with input size (0.2 GB problem 74.6 s,
6.4 GB problem 88.4 s). Do not repeat the claim that hold scales with tensor size --
that correlation was an arm-count confound. Settling the hold requires instrumenting
the locked region, which nobody has done.

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

# wave-era tooling (env/common/, symlinked into each env/<HARDWARE>/)
bash env/<HARDWARE>/launch_wave.sh <gpu> <spec> status   # per-arm pid/problem/lockmax/ORPHWAIT
bash env/common/wave_status.sh                           # same audit from the eval records
bash env/common/wave_watch.sh                            # 15-min watchdog -> wave_watch.log

uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
```

Read `ORPHWAIT` (a `waiting` with no matching `acquired after`) and `proceeding UNLOCKED`;
both must stay 0. **Caveat:** `wave_watch.sh` globs run directories without intersecting
the live process list, which is the exact trap documented below — a killed arm's directory
will keep contributing to its numbers.

`checkpoint_run.py --auto` degrades silently when `runs_evolving/archived/with_NVCC_bug`
is absent (not every host has it): the BASELINE block and the entire `=== VERDICT` section
are skipped, and any smoke-test directories sitting in the results root are reported as
arms. Check its output names against `ps`, not just its verdict.

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

**Refinement arms have a different sidecar, and a different trap.** `--enable-skill-refinement`
writes `skill_revisions.txt` (not a `*refine*` filename — a glob for "refine" misses it),
and no other arm has it. Confirm it is non-empty the same way.

Its `refinement_meta` block (`bug_solved`, `metric_before/after`, `metric_delta_pct`) is
computed **before** `revise_blamed_skills` is called (`skill_refinement.py:639-642`), so it
describes the **iteration that triggered** the refinement, not the effect of it. Every
record will show `bug_solved: false` and mostly negative deltas — that is the trigger
condition firing as designed, **not** evidence that refinement made things worse. There is
no field recording a refinement's outcome; you can only see it in the arm-level metric.

### 3.6 Resuming a damaged range

```bash
bash scripts_integration/new_evolving_agent/env/<HARDWARE>/resume_run.sh <gpu> <run_dir_name> <ctx_mode> <start> [end]
```

Narrow ranges are safe — two mechanisms cooperate:

1. **Disk purge** removes only L1 entries sourced from problems inside `[start, end]`
   (plus refine/merge descendants).
2. **Causal prompt filter** (`collect_causal_l1_entry_ids`) restricts the visible catalog
   to entries with provenance strictly `< N` while replaying problem `N`.

A replayed problem never sees skills learned after it. Verified: replaying index 39 showed
267/344 entries, provenance 1..38, zero leakage. **For multiple resumes, run the earlier
index first.**

#### Resuming a KILLED arm, and resuming a whole wave (2026-08-23)

`resume_run.sh` could not resume any of the nine arms killed on 2026-08-23 -- four separate
defects, each independently fatal, all now fixed:

1. `RESULTS_ROOT` was hardcoded to `runs_evolving/gpt-oss-120b/`, so it could not address
   anything under `median/` and exited `FATAL: no such run dir`.
2. It hard-required `run_summary.json`, which is written only at run *end* -- so every
   crash-resume, the one case resume exists for, was refused.
3. **It passed no governance flags.** `evolve_kb_batch.py` rebuilds a run's treatment from
   the CLI, and `_check_resume_config_mismatch` returns `[]` when the summary is absent
   (`evolve_kb_batch.py:675`) -- precisely the killed-arm case. So resuming
   `deletion`/`merge`/`refinement`/`l2` would have silently continued them as plain
   truncation arms, with no error anywhere. **Always pass the arm's flags after `--`.**
4. The GPU guard was `used > 1000 MiB`, but an idle arm holds ~558 MiB, so a multi-arm
   resume wave was impossible. Now `MIN_FREE_MIB` (default 20000), as in `launch_wave.sh`.

```bash
# one arm; `auto` derives the start from batch_timing.jsonl (completed + 1)
RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
  bash .../env/NVIDIA_GH200x2/resume_run.sh 0 <run_dir_name> truncation auto -- --skill-deletion

# a whole wave, one arm per spec line, flags taken from the spec (see defect 3)
KB_GPU_EVAL_LOCK_SLOTS=3 MAX_ARMS_PER_GPU=6 RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
  bash .../env/NVIDIA_GH200x2/resume_wave.sh 0 .../wave_median_oss6_resume.spec [dry-run]
```

`DRYRUN=1` (single arm) and `dry-run` (wave) resolve every run dir, resume point and flag
set without launching. Use them -- they are the only cheap check that defect 3 has not
recurred. `DO_BACKUP=0` skips the pre-resume tar for a barely-started arm.

**Resuming inherits protocol seams.** A resumed run's prefix keeps whatever protocol it was
measured under. The Aug-22 wave carries two: problem 1 was evaluated at
`num_perf_trials=100` and problems 2+ at 25 (`63bfc2b` reached live arms via spawn
re-import; the compress arm is split *mid-problem*, 21 evals at 100 and 6 at 25), and the
prefix ran with the trims off under 9-arm contention while the suffix runs with them on.
Contention only ever deflates speedup, so the prefix is systematically penalised relative to
the suffix **inside one run**. Restart instead of resume when the prefix is small.

---

## 4. Analysis

**`--hardware` is not optional here — the default is wrong.**
`aggregate_runs.py:64` sets `DEFAULT_HARDWARE = "NVIDIA_GH200x2"`, whose baseline has
**no `median` field on any entry**, so `build_baseline_lookup` silently falls back to
`mean`. Baseline resolution order is (1) `--baseline-file`, (2) `hardware_server` from
`run_summary.json`, (3) `--hardware`. **`run_summary.json` is written only after the
last problem** (`evolve_kb_batch.py`), so for any incomplete run step 2 is unavailable
and the wrong default is what you get. Always pass `--hardware` explicitly.

`--output-dir` is per-server too; keep one per host so the two servers never overwrite
each other. Existing dirs are `output/{GH200x2, GH200x2_nvcc_fixed,
gpt-oss-120b-inf-CPU6, gpt-56-terra-inf-CPU4, model-endpoint-comparison}` — the naming
is historical, not `$HARDWARE`-keyed; pick or create one deliberately.

```bash
# always pass --regenerate-stats; cached stats across runs were written by different
# code versions and are NOT comparable
.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware $HARDWARE --runs-root runs_evolving/gpt-oss-120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/<dir> --regenerate-stats

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware $HARDWARE --runs-root runs_evolving/gpt-oss-120b \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/<dir> \
  --baseline-run <the truncation arm from the same wave>
```

Outputs land in that dir: `aggregate_runs.{json,csv}`, `comparison.md`.

**~14 of the 50 subset problems are bimodal lotteries, and they dominate every
geomean.** Measured on the 2026-08-27 L2 wave (11 arms, one model, one GPU): on
these problems an arm either finds an algebraic collapse (7–23x) or does not (~1x),
with nothing in between. The cleanest proof is treatment-free — two arms with
**byte-identical configuration** landed on opposite sides of three of them:

| problem | `q15_ctl_r1` | `q15_ctl_r2` |
|---|---|---|
| L2P42 | 1.59x | **22.72x** |
| L2P51 | **7.17x** | 1.01x |
| L2P97 | **7.36x** | 0.99x |

Because they are also the *largest* speedups in the set, they set the geometric
mean, so an arm's headline number mostly records how many jackpots it happened to
hit. On that wave the effect was not subtle: raw arm geomeans spanned 1.058–1.696
and every L2 arm looked far worse than its control; with the lottery problems
removed the spread collapsed to 1.100–1.161 and **every paired CI contained 1.0,
including the identical-configuration null**. Per-problem *medians* were ≈1.0 for
every arm even before adjusting — the geomean gap was never a typical-problem
effect.

So: **identify them by a rule that never looks at arm identity** (e.g. max/min
clean speedup >= 4x across all arms in the wave), report adjusted and unadjusted
side by side, and never quote a raw 50-problem geomean as a comparative result.
Selecting outliers by their effect on the statistic under test would be circular;
selecting them by cross-arm spread is not. Project memory
`collapse-problems-inflate-terra-geomean` named L2P13/42/51/56 from a different
model and wave, which is independent corroboration.
`new_evolving_agent_analysis/l2_redesign/lottery_adjusted.py` implements the rule.

This is a *bigger* lever than replicates: adjusting took the paired CI from ~±40%
to ~±6% on the same data. It does not repeal open item 10 — it makes the residual
noise small enough to be worth fighting.

**Filter hacks per sample, not per problem.** Each `evolving_runs.json` run record
carries a top-level `best_speedup` / `best_is_hack` describing its *single best*
iteration. Filtering on that field drops the whole problem when its best iteration was
a hack — even though a clean, slower sample usually exists. On the completed truncation
control that mistake hit **14 of 50 problems** and moved the headline numbers a long
way: `best_geomean` 1.579 (wrong) vs **1.389** (correct), `fast_p_best@1.0` 0.520 vs
**0.640**. Walk `records[].evaluation` and take the best sample with
`correct and not is_hack`. `run_summary.json`'s `per_level_summary.correct` already
counts problems with at least one clean sample, so it agrees with the per-sample method.

**A non-empty `static_check_warnings` does NOT mean the sample is a hack.**
`resolve_is_hack` (`kernelbench_integration/static_check.py:44`) fires only on
STRICT static *errors* or `metadata.excessive_speedup`; the persisted field, built
by `collect_static_check_warnings`, merges those errors with purely advisory
warnings — `workload_shrink:`, `torch_computation_ops: Uses torch.matmul`,
`pytorch_wrap: ...`. Those are recorded for audit and are explicitly documented as
not setting the flag. Treating the field as a hack signal on the Aug-27 wave
discarded **383 of 756 clean samples** and cut the control from 47 problems to 27
while leaving its geomean almost unchanged (1.692 vs 1.696) — so the sanity check
that catches it is the problem **count**, not the speedup. Always reconcile your
per-sample walk against `aggregate_runs.py`'s `correct=N/50` before interpreting
anything. CLAUDE.md's own advice to keep static-check hacks flagged applies to
samples *already flagged*; it is not a licence to flag on the warning list.

**Recompute `is_hack` at analysis time; do not trust the recorded flag.** The
excessive-speedup threshold is a *default argument* in `src/kernelbench/eval.py`
(`excessive_speedup_threshold`), and `eval.py` is re-imported by every eval child — so
changing it re-classifies hacks for every live arm from that moment on, with no
relaunch. It went 10x -> 30x mid-wave in `588a6a5` (2026-08-24 15:11:45Z), which means
a run's `is_hack` column can be a mixture of two rules. Measured on the Aug-22 wave:
29 pre-cutover iterations sat in the disputed [10x, 30x) band and are flagged; 0
post-cutover iterations landed there. Re-scoring uniformly at 30x changes the best
sample on **3 problems out of ~412**, all in treatment arms, none in the control — small,
but not zero, and it favours the treatments.

Nothing is lost, because raw `speedup` is in the artifacts: re-derive the flag at one
threshold across the whole run. Only un-flag samples whose hack status came from
`metadata.excessive_speedup` — `resolve_is_hack` also fires on
`static_check_warnings`, and those must stay flagged.

**Headline metric is `fast_p_best@1.0`** on the full aligned-problem denominator, with
`fast_p_best@0/2` beside it (`ANALYSIS_RULES.md:81-85`). `best_geomean` is a *secondary*
figure — `ANALYSIS_RULES.md:158` explicitly forbids best-geomean-only leaderboards and
partial prefixes as a final comparative result. Earlier revisions of this file called
`best_geomean` primary; `ANALYSIS_RULES.md` wins.

For which historical runs are void, read `output/GH200x2_nvcc_fixed/MANIFEST.md`.
(`output/GH200x2/INVALIDATED.md` scopes itself to artifacts "dated 2026-08-05 or
earlier" but its own `comparison.md` is dated 2026-08-17, and `ANALYSIS_RULES.md:152`
voids that whole folder anyway.)

**Never compare across servers, baseline files, or dates without saying so.** Four
independent things break comparability and none is visible in the number:

- different `$HARDWARE` baselines;
- a baseline file regenerated under the same name;
- **median-vs-mean fallback** — every committed GH200 number in `output/GH200x2*` was
  computed against `results/timing/NVIDIA_GH200x2/` (mean fallback), while current runs
  use `_2nd` medians. Over the 50-problem subset those differ >5% on **25 of 50**
  problems, geomean ratio **1.149×**, worst ~4×. Historical `best_geomean` figures —
  including the ones quoted in the open items below — are on the old scale;
- inference-endpoint latency drift (open item 10).

Prefer the truncation arm *from the same wave* as the comparison baseline over any
historical run.

---

## 5. Repo layout

```
CLAUDE.md / CLAUDE.local.md       # shared doc / per-host doc (gitignored)
scripts_integration/new_evolving_agent/
  evolve_kb_batch.py              # the batch runner (all CLI flags)
  env/
    hardware_env.sh               # resolves $HARDWARE from the CALLING script's dir;
                                  #   fatals on a median-less baseline
    common/                       # shared by every server
      activate.sh                 #   CUDA_HOME + PATH exports
      install_cuda128_local.sh    #   userspace CUDA 12.8 (no sudo)
      wave_status.sh              #   per-arm progress / lock / orphan-wait audit
      wave_watch.sh               #   periodic watchdog -> wave_watch.log
      wave_collect.py             #   per-arm incident tracking / alerts
      wave_analyze.py             #   eval-record rollups
      verify_wave.sh  README.md
    <HARDWARE>/                   # one folder per server; name selects the baseline
      launch_run.sh               #   ← single arm
      launch_wave.sh              #   ← several arms from a .spec
      resume_run.sh               #   narrow-range replay; `auto` start; killed or finished
      resume_wave.sh              #   whole wave, same spec format  (some hosts only)
      wave_median_*.spec          #   that host's wave specs        (some hosts only)
      launch_arm_reps.sh  launch_merge_reps.sh  launch_nvcc_series.sh
      activate.sh, install_cuda128_local.sh, verify_wave.sh, wave_status.sh,
        wave_watch.sh             #   -> symlinks into common/
      # the host folders are NOT in sync: check yours before quoting a flag or a guard
    wave_gpu0.spec  wave_gpu1.spec  wave_locktest.spec
    gh200_setup/                  # host provisioning playbook, parts 00-14 + appendices
    eval_embed_duplicates.py      # rank embedding models by near-duplicate retrieval
    eval_embed_quality.py         # merge-outcome AUC (null result; kept as evidence)
    eval_embed_candidates.py      # fidelity-to-nv-embedcode (misleading; kept as evidence)
    README.md                     # nvcc postmortem
    PARALLEL_CAPACITY_AND_INCIDENT_2026_08_20.md
    INCIDENT_2026_08_20_venv_swap.md
scripts_integration/new_evolving_agent_analysis/
  aggregate_runs.py  compare_runs.py  checkpoint_run.py  analyze_feature_evidence.py
  ANALYSIS_RULES.md  EXPERIMENT_REPORT.md  output/<HARDWARE-ish>/
Self-Evolving-Agent/               # git submodule
  evolving_common/llm_client.py               # endpoints, aliases, embeddings
  evolving_common/memory_manager.py           # governance defaults; L1/L2 tiers
  evolving_common/prompt_context.py           # prompt assembly (L0 modes, L1, L2 blocks)
  evolving_common/governor/gen3_stages.py     # staged governor; deletion/merge call sites
  evolving_common/governor/skill_merge*.py    # DBSCAN clustering + LLM merge
  evolving_common/governor/l2_promotion.py    # L2 gate, rendering, standing-rule blocks
  evolving_common/governor/skill_usage_tracker.py  # usage ledger L2 promotion reads
  evolving_common/governor/gpu_lock.py        # GPU-eval lock: cross-process, re-entrant,
                                              #   N-slot semaphore (KB_GPU_EVAL_LOCK_SLOTS)
  evolving_common/governor/gpu_reserver.py    # 42 GB idle reservation; KB_GPU_RESERVE_GB
  kernelbench_integration/baseline_timing.py  # @lru_cache'd baseline load (see §3.1)
  kernelbench_integration/eval_runner.py      # precompile-then-lock boundary
  kernelbench_integration/governor.py         # speedup computation
src/kernelbench/eval.py            # _gpu_timing_lock; the trims, KB_EVAL_UNLOCK_CORRECTNESS
                                   #   and the mem gate (all inert unless exported)
results/timing/<HARDWARE>/         # per-server baselines; must carry `median`
                                   #   (enforced by hardware_env.sh, not by convention —
                                   #    most folders here do NOT comply)
runs_evolving/gpt-oss-120b/        # current series
runs_evolving/inference_oss_120b/  # earlier series -- present on some hosts only
runs_evolving/archived/            # VOID (pre-nvcc-fix) -- present on some hosts only
```

---

## 6. Open items

1. **Merge threshold** — code default is `0.8`; `0.85` was the earlier validated
   operating point (chains cluster badly at realistic catalog sizes below it). The
   Aug-22 wave deliberately ran `0.8` to line up with three existing `merge_sim08`
   reps, so **both thresholds now have data and neither is the settled default**.
   Decide from that data rather than re-asserting `0.85`; encode whichever you pick in
   the run tag.
   **What 0.8 actually did on a completed 50-problem arm:** 207 merge events (82
   accepted, 125 rejected), **488 source skills superseded into 41 active merged
   skills — a mean of 11.9 sources collapsed per surviving skill**. The active catalog
   ended at 168 entries (41 merged + 127 unmerged) against the control's 600. That is
   the cluster chaining this item warns about, now quantified. Its effect on quality
   was nil: paired ratio 0.945 [0.787, 1.134], McNemar p=0.73. So at 0.8 merging buys
   a ~72% smaller catalog for no measurable quality change — the question is whether
   0.85 preserves more distinctions without costing anything, not whether merging
   "works".
2. **`uv lock && uv sync`** — deferred until no run is in flight. Re-run the
   merge-threshold calibration afterwards. See `CLAUDE.local.md` for how far *this*
   host's venv has drifted from the lock.
3. **The deletion cell is confounded — but not by the mechanism this item used to name.**
   *Retracted:* "`--enable-l1-skill-unit-test-gc` is a no-op (`gen3_stages.py:893` reads
   the wrong config field), so every `--skill-deletion` arm is really deletion +
   unit-test GC." Per-iteration GC is genuinely **off**: `gen3_stages.py:953-966` sets
   `enable_unit_test_gc = enable_l1_skill_unit_tests AND enable_l1_skill_unit_test_gc`
   = `True AND False` (`memory_manager.py:52,54`), the CLI flag is wired through
   (`evolve_kb_batch.py:1616` → `config.py:91`), and the AND has been in place since
   submodule `bd92795` (2026-07-10). Artifact proof: `l1_skill_unit_test_runs.jsonl`
   has exactly one run per `entry_id` — GC would re-test every active skill every
   iteration.
   **The real confound is post-append unit testing.** `_run_post_append_unit_tests`
   (`memory_manager.py:692`) early-returns unless `is_l1_skill_deletion_enabled()`
   (`:705`), and `DEFAULT_L1_SKILL_DELETE_ON_UNIT_TEST_FAIL = True` (`:55`). So turning
   on `--skill-deletion` also turns on an LLM-authored pytest admission gate that no
   other arm has. In the live deletion arm that gate is the *larger* term: 153 deletions
   = **92 `unit_test_fail` + 61 `consecutive_unused`**. Pass
   `--no-l1-skill-delete-on-unit-test-fail` to isolate the usage rule, or report the
   cell as "deletion + unit-test admission gate".
4. **Rewrite `EXPERIMENT_REPORT.md`** — the current text is written against voided
   pre-nvcc-fix runs and its conclusions are reversed by the repaired data.
5. **Governance matrix — 4 of 7 cells untouched, and that was already true before the
   Aug-22 wave.** `{D}`, `{R}` and `{M}`×3 completed 2026-08-14 / 08-17 / 08-19
   (`output/GH200x2_nvcc_fixed/aggregate_runs.json`). The wave **re-runs the same three
   singletons on a different baseline and adds no new cell**; `D+M`, `D+R`, `M+R`,
   `D+M+R` remain open on GH200. (`gpt-oss-120b-inf-CPU6` has a `D+M+R` run, but on
   A6000/CPU6 against a different baseline.) The median-baseline series is a further
   re-run of the same cells, not new coverage.
   The old "N arms/GPU is safe" guidance is no longer a *safety* limit: per §3.4 the
   fidelity argument for a strict mutex was wrong (concurrent evals cost <3%) and the lock
   is now a counting semaphore. *Corrected 2026-08-25:* the ~3 arms/GPU **throughput**
   ceiling no longer stands either — 9 arms/GPU was measured at zero per-arm cost, so extra
   arms now buy matrix coverage **and** wall-clock. The cap that remains is `SLOTS`, not
   arms; see "9 arms/GPU measured" in §3.4.
   *Which arms are live, and where each one stopped or resumed from, is per-host live
   inventory — read `CLAUDE.local.md`, not this file.*
   *Retracted:* "`--skill-merging` requires `--skill-deletion`, so the merging cell is
   really D+M." That is only the help string at `evolve_kb_batch.py:1085-1086`; nothing
   in `evolve_kb_batch.py` or `KBGovernorConfig` validates it. The live merge arm runs
   merging alone — `l1_skill_catalog_stats.json` reports `"deleted": 0` and there is no
   `l1_skill_deletions.jsonl`. **Fix the help string**, don't repeat the claim.
6. **The governance axis is coupled to the extractor's candidate-set size — this is the
   biggest design defect currently known.** `read_l1_extractor_catalog`
   (`memory_manager.py:801-820`) returns **every** active skill when
   `enable_skill_deletion` is true and otherwise only the last
   `DEFAULT_L1_EXTRACTOR_CATALOG_MAX = 50` (`:48`). And `gen3_stages.py:889` passes
   `skill_deletion=enable_skill_governance`, which `:821` defines as
   `deletion OR merging`. So switching on *either* governance flag also removes the
   tail cap. Measured live from each arm's newest extractor prompt (2026-08-23): **merge 107
   candidates, deletion 28, every non-governance arm 50** (l2 51). A `deletion`-vs-control
   or `merging`-vs-control delta therefore mixes the governance rule with a change in
   how many skills the picker chooses among — and the three governance cells are not
   mutually comparable either. `--enable-skill-refinement` does not trip this, so the
   refinement cell is clean on this dimension.
   The absolute counts are a snapshot and grow as a run proceeds; re-measure before
   quoting them. The durable invariant is capped-at-50 versus uncapped.
   Until it is decoupled (give the cap its own flag, held fixed across arms), report
   governance results as "rule + catalog size", not as the rule.
7. **L2 is invisible to the analysis scripts. — FIXED 2026-08-27, see §8.11.**
   `aggregate_runs.py` now extracts the L2 config and `compare_runs.py`'s
   `design_variant_label` renders the L2 knobs, so an L2 arm no longer collapses to
   the same design string as its control. Original text follows.

   **L2 is invisible to the analysis scripts.** `aggregate_runs.py`'s config extraction
   has no `enable_l2` field, and `compare_runs.py`'s `design_variant_label` (`:147-159`)
   reads only the context mode plus `skill_deletion`/`skill_merging`/`enable_skill_refinement`
   — so an L2 arm and the truncation control both render as design `truncation` in the CSV
   and every delta table. `run_summary.json` does carry the flag
   (`evolve_kb_batch.py:1771`), so this is a small extraction fix, but it must land
   **before** any report is generated from a wave containing an L2 arm.
   This is a launch-blocker for any L2 wave — see §8.10.
8. **L2 promotion has no dedup gate — measured, and it is the tier's main defect.**
   The completed `l2` arm promoted **9 standing rules, 7 of which are the same idea**
   ("Fuse Compute Instead of Adding Trivial Kernels", "Avoid Trivial Custom Kernels in
   Hot Forward Paths", "Avoid Trivial Copy Kernels as Performance Boosts", …). Those 7
   are 12,012 of the 15,176 characters of standing text, and with `--l2-render verbatim`
   all of it goes into **every** coder system prompt: 20,091 chars vs the control's
   4,190, ~4.7x. It bought nothing — geomean 1.347 vs the control's 1.389, paired ratio
   0.965 [0.816, 1.142], fast_p@1 McNemar p=1.000.
   The cause is compositional: L1 accumulates near-duplicate skills, `--skill-merging`
   exists to collapse them, and the L2 arm ran with merging **off**, so one popular idea
   was expressed as 7 skills and promoted 7 times. Promotion then freezes their evidence
   and removes them from the extractor catalog, so nothing later reclaims the tokens.
   Fix before spending more arms on L2: either require `--skill-merging` with
   `--enable-l2`, or add a similarity gate to the promotion pass in `l2_promotion.py`.
   Also note promotions cluster late (5 of 9 after global iteration 1110) because the
   rate floor needs accumulated selections — a 50-problem run barely exercises the tier.
   *Corrected 2026-08-27 — §8 supersedes the counts in this item.* Two of them are wrong:
   the duplicate family is **6 of 9 rules and 10,523 of 15,176 chars (69%)**, not 7 /
   12,012 — the old figure folded in "Optimal Block Size & Shared-Memory Bias Caching",
   which is a genuinely distinct rule. And **4.79x is the terminal prompt ratio**, reached
   only for the last 10% of coder calls; the run mean is **2.21x**, the median 1.48x, and
   16% of calls saw no L2 text at all. The defect itself stands. §8.6 carries the measured
   response curve for every floor, §8.7 a cap-ranking degeneracy, and §8.8 three quantities
   that cannot be recovered after a run.
9. **L2 is unplanned scope.** `--enable-l2` is a third axis with its own knobs
   (`--l2-render`, `--l2-min-tasks/-selections/-rate/-new-bests`, `--l2-max-entries`).
   Decide whether L2 belongs in this paper's matrix before spending more arms on it.
   *Corrected:* this item used to add "its floors are calibrated for `--no-skill-deletion`
   runs and `l2_promotion.py` warns they should be relaxed when deletion is on, so an
   L2 × governance cell needs re-tuning first". That warning was removed by submodule
   `34edbcf` (2026-08-24); the recalibration to a single universal set was `114fd34`
   (2026-08-22), which left the stale warning behind for two days.
   `l2_promotion.py:51-68` now argues explicitly that `min_rate` is *regime-independent*
   because promotions fire early, while every regime's visible catalog is still ~40-100
   entries. Defaults today: MIN_TASKS 3 / MIN_SELECTIONS 50 / MIN_RATE 0.70 /
   MIN_NEW_BESTS 0 (disabled) / MAX_ENTRIES 0. An L2 × governance cell needs no
   re-tuning on that ground; it is still unplanned scope.
   §8 is the measured brief for this axis — read it before deciding whether L2 belongs
   in the matrix; §8.9 lists the candidate arms and the two code fixes that beat them.
10. **Replicate noise is ~30%, and it bounds every conclusion — this is the binding
   constraint on the whole series.** Three *identical-config* `merge_sim08` replicates
   (`2026_08_19_17_{29,32,35}`, `output/GH200x2_nvcc_fixed/aggregate_runs.csv`) give
   `best_geomean` (CSV column `speedup_best_geomean`) 0.838 / 0.855 / 1.092 — **log-SD
   0.147 (×1.16 per SD), max/min 1.303**. Quote the log-SD, not a percentage — on these
   three values `max/min − 1` is 30% and `(max − min)/max` is 23%, so any headline
   percentage is just a choice of framing. The older "~20%" figure is a *different,
   superseded* pair, not another framing of this one.
   *Supersedes* the earlier two-run figure (`itr30` rep1 vs rep2, 2026-08-20, ~22%): three
   replicates beat two, and that pair does not reproduce — today's committed CSV gives
   `0.775` (n=36) and `0.967` (n=31), both `status=partial`, and partial-run values move on
   every re-aggregation.

   The arithmetic that follows is unforgiving. At n=1 per cell the SD of a single
   arm-vs-control log ratio is `0.147 × √2 = 0.208`, so a 95% band needs a ratio above
   **×1.50** (or below ×0.66); across 8 contrasts, Bonferroni needs **×1.77**
   (*corrected* — this item previously said ×1.84). Detecting a +20% effect at 80% power
   needs ~10 replicates per arm. **No single-replicate wave in this series can support an
   arm-vs-arm winner claim** — only descriptive reporting with n stated. Spend arms on
   replicates of the one or two cells you intend to claim, launched on the *same GPU* as
   their control, and report a paired per-problem log-ratio CI.

   No script does that pairing today: `compare_runs.py` matches on iteration and has no
   per-problem or CI logic; `analyze_feature_evidence.py` pairs per problem but rejects
   partial runs (exit 2).
11. **Solo baselines drift and must not be reused across dates.** The same 49 problems
   took 87.8 min/problem (Aug 07), 79.8 (Aug 13), 80.1 (Aug 14), 64.7 (Aug 17) — a 26%
   swing from inference-endpoint latency, not from anything in this repo. Because the
   agent is LLM-bound, any throughput or speedup figure normalised against a baseline
   from another date is inflated by that drift; efficiency >100% is the tell. Compare
   only within a time window, or measure a fresh solo baseline alongside the runs.

---

## 7. Working preferences

- **Ask for confirmation before launching any `nohup` run** — tens of GPU-hours each
  (42–75 h depending on endpoint latency; see §1).
- Direct `nohup` from the tool layer gets denied; that is why launches go through
  `launch_run.sh` / `launch_wave.sh` invoked via `bash`.
- Verify claims against artifacts before reporting. Several confident-sounding
  conclusions in this project turned out wrong until checked against the data.
- **Sort every new fact into the right file.** Mechanism, protocol and analysis rules go
  here; hostname, GPU count, driver, `$HARDWARE`, live-run inventory, local venv drift
  and local incidents go in `CLAUDE.local.md`. When you catch yourself writing a literal
  hostname, absolute `/localhome/...` path, `NVIDIA_GH200x2*` folder name, or "the venv
  currently has ..." into *this* file, it belongs in the local one.
- **When you correct something here, say it is corrected** rather than deleting it. Open
  item 3 was re-reported twice because the retraction was silent.

---

## 8. L2 promotion — measured design brief

**Why this section exists.** `--enable-l2` has been run to completion exactly **once**
(`base_agent_gpt_oss_120b_l2_itr30_GH200_2026_08_22_20_34` — gpt-oss-120b, 50 problems ×
30 iterations, verbatim render, all floors at defaults). The next L2 wave should read this
rather than re-derive it. Split the contents accordingly:

- **§8.1–8.2, §8.7–8.8 are mechanism**, read out of `l2_promotion.py`. Host-agnostic and
  durable; they hold on any server and any model.
- **§8.3–8.6 are calibration from that single arm.** One sample, one model. **Re-measure
  before trusting any threshold on a different model** — §8.6 tells you how, and it is a
  two-minute job once an L2 arm exists.

Every number below was re-verified against artifacts on 2026-08-27.

### 8.1 What the tier is

L1 is **retrieved**: the extractor picks ~7 skills per iteration out of a candidate
catalog. L2 is **standing**: promoted rules are concatenated into the coder system prompt
for *every* iteration of *every* remaining problem, relevant or not.

Promotion is one-way and self-freezing:

- promoted skills are **removed from the extractor catalog**, so they stop accruing selections;
- their evidence is snapshotted into `l2_meta` at promotion and never re-derived. The module
  docstring gives the reason: recomputing `rate` afterwards grows the denominator against a
  frozen numerator, which would demote every L2 skill on a delay.

Promotion fires **only at a task boundary** — `governor.py:1430-1449`, after the last
iteration of a problem. So the standing set is immutable while a problem runs, and the
coder system prompt is constant across that problem's 30 iterations.

### 8.2 The gate: four floors (a conjunction), then an optional cap

| knob | default | what it screens | code |
|---|---|---|---|
| `--l2-min-tasks` | 3 | breadth — blocks skills that only ever fired on one problem | `l2_promotion.py:69` |
| `--l2-min-selections` | 50 | sample size — blocks small-n noise | `:70` |
| `--l2-min-rate` | 0.70 | **the load-bearing floor.** Extractor picks ~7 of ~50, so a random skill scores ~0.14; 0.70 is ~5× that | `:71` |
| `--l2-min-new-bests` | **0 — disabled** | outcome attribution. Still recorded, still feeds the ranking score | `:75` |
| `--l2-max-entries` | 0 — unlimited | top-N cap applied **after** the floors | `:76` |

`rate = total_selections / max(1, global_iter - created_at_global_iter)` (`:155`) —
normalized by opportunity, so the floor measures usefulness rather than seniority.

Ranking is consulted **only when `max_entries > 0`**:
`score = rate × log1p(tasks) × log1p(new_bests)` (`:138`), sorted `(-score, entry_id)`,
truncated at `:187-188`.

`--l2-render`: `verbatim` (default, the full L1 entry) · `extract` (only the
`Generalizable Rule` / `Anti-Pattern to Avoid` bullets — `_RULE_HEADINGS`, `:196`) ·
`distill` (LLM rewrite at the task boundary — **costs an extra LLM call per promotion**,
fails soft to `extract`).

### 8.3 What the one completed arm actually promoted

9 rules, 0 demotions, 15,176 chars of standing text. Ranked by score (= the order a cap
would keep):

| score | tasks | rate | new_bests | sel | promoted@iter | chars | title |
|---|---|---|---|---|---|---|---|
| 3.3654 | 5 | 0.732 | 12 | 93 | 780 | 1397 | Read‑Only Cache (`__ldg`) Boost for Vectorized CUDA Kernels |
| 2.7980 | 4 | 0.725 | 10 | 87 | 780 | 1489 | Optimal Block Size & Shared‑Memory Bias Caching |
| 2.2421 | 3 | 0.778 | 7 | 70 | 1290 | 1981 | Avoid Trivial Custom Kernels When Compiler Already Fuses Ops |
| 2.1356 | 3 | 0.792 | 6 | 57 | 240 | 1767 | Naïve Direct Conv2D Kernel Indexing Pitfalls |
| 2.0546 | 3 | 0.827 | 5 | 67 | 1230 | 1698 | Avoid Trivial Custom Kernels in Hot Forward Paths |
| 1.7570 | 3 | 0.787 | 4 | 63 | 1110 | 1784 | Fuse Compute Instead of Adding Trivial Kernels |
| 1.7570 | 3 | 0.787 | 4 | 63 | 1230 | 1739 | Fuse Memory‑Bound Ops – Skip Trivial Copy Kernels |
| 1.6381 | 3 | 0.734 | 4 | 58 | 1230 | 2057 | Avoid Trivial Copy Kernels as Performance Boosts |
| 1.5234 | 3 | 0.793 | 3 | 65 | 1350 | 1264 | Avoid Trivial Dummy Kernels for Performance Gains |

Outcome: geomean **1.347 vs the control's 1.389**, paired ratio 0.965 [0.816, 1.142],
`fast_p_best@1.0` McNemar p = 1.000 — **a null**.

Two structural facts visible in that table:

- **7 of 9 sit exactly at `min_tasks = 3`.** The breadth floor is what admits most of the
  set, not something they clear comfortably. `--l2-min-tasks 4` would cut 9 → 2.
- **Promotions cluster late** — 5 of 9 after global iteration 1110 (problem ~37 of 50),
  because `min_selections = 50` needs accumulated selections. A 50-problem run barely
  exercises the tier at all.

### 8.4 The defect: two-thirds of the standing set is one idea

Rules 4–9 above all say *"don't add a trivial kernel; fuse instead; amortize launch
overhead"*. Verified by reading each entry's `Generalizable Rule` bullet, not by title
similarity. Rules 1–3 are genuinely distinct (conv2d tiling/indexing · `__ldg` read-only
cache · block size ≤512 + shared-memory constant caching).

> **Corrected 2026-08-27.** Open item 8 says *"7 of which are the same idea … 12,012 of the
> 15,176 characters"*. The correct figures are **6 of 9, and 10,523 of 15,176 chars (69%)**.
> The 12,012 figure is exactly `10,523 + 1,489`, i.e. it wrongly folded *"Optimal Block Size
> & Shared-Memory Bias Caching"* into the trivial-kernel family. That entry is a distinct
> rule and would survive any dedup gate. The defect is real and the correction makes it no
> smaller in kind — but quote 6/69%, not 7/79%.

**The cause is compositional, not a bad threshold.** L1 accumulates near-duplicate skills;
`--skill-merging` exists to collapse them; this arm ran with merging **off**. So one popular
idea existed as 6 separate L1 skills, each independently cleared the floors, and each was
promoted. Promotion then froze their evidence *and* removed them from the extractor catalog,
so nothing downstream can ever reclaim the tokens.

### 8.5 The prompt burden is a step function — 4.79× is the terminal value, not the run

Measured from `chat_history.jsonl`, filtering on `phase == "coder"` (the other three phases
— `extractor` 542, `action_selector` 454, `summarizer` 1654 chars — are unaffected by L2 and
identical in both arms, so a whole-file median hides the effect entirely).

The control's coder system prompt is a **constant 4,190 chars** across all 1,500 calls. The
L2 arm's is a 7-level step function, one level per cumulative promotion:

| rules standing | chars | ×control | coder calls | share |
|---|---|---|---|---|
| 0 | 4,190 | 1.00× | 240 | 16.0% |
| 1 | 6,191 | 1.48× | 540 | 36.0% |
| 3 | 9,225 | 2.20× | 330 | 22.0% |
| 4 | 11,062 | 2.64× | 120 | 8.0% |
| 7 | 16,723 | 3.99× | 60 | 4.0% |
| 8 | 18,771 | 4.48× | 60 | 4.0% |
| 9 | 20,091 | **4.79×** | 150 | 10.0% |

> **Corrected 2026-08-27.** Open item 8's "20,091 chars vs the control's 4,190, ~4.7x" is
> the **end state**, reached only for the last 10% of coder calls. Over the whole run the
> mean is **9,242 chars = 2.21× control**, the median is **1.48×**, and **16% of calls saw
> no L2 text at all**. That materially weakens the "L2 bought nothing at 4.79× the prompt
> cost" framing: most of the run barely had a tier to pay for. It is the same point §8.3
> makes about late clustering — **a 50-problem run is too short to test L2 as designed**,
> and that, not prompt bloat, may be why the arm read as a null.

Counterfactual under `--l2-render extract`, same 9 rules, framing overhead held fixed:
standing text **15,176 → 4,196 chars (27.6%)**, terminal prompt **4.79× → 2.17×**, run
average **2.21× → 1.35×**.

> *Corrected:* an earlier note in this project quoted the extract terminal ratio as 2.00×.
> That omitted the per-entry framing the standing block adds; recomputed through the real
> `render_l2_entry`, it is **2.17×**.

### 8.6 Knob response curves — measured, and how to re-measure

Survival counts out of the 9 promoted, sweeping one floor at a time over the frozen
evidence in `l2_promotions.jsonl`:

| knob | response |
|---|---|
| `--l2-min-new-bests` | 0→9 · **1→9 · 2→9 · 3→9 (all inert)** · 4→8 · **5→5** · 6→4 · 7→3 · ≥8→2 |
| `--l2-min-rate` | 0.70→9 · 0.72→9 · 0.75→6 · **0.78→5** · 0.79→3 · **0.80→1** · 0.83→0 |
| `--l2-min-tasks` | 3→9 · **4→2** · 5→1 · 6→0 |
| `--l2-min-selections` | 50→9 · 60→7 · 70→3 · 80→2 · 90→1 |
| `--l2-max-entries` | 3 → `__ldg`, block-size, "…When Compiler Already Fuses" · **4 → all three distinct rules + one representative of the duplicate family** · 5 → adds a second duplicate |

Reading of those curves, for this model:

- **`--l2-min-new-bests` below 4 is a no-op.** Do not set it to 1 expecting an effect
  (except under a cap — see §8.7, where it becomes load-bearing for a different reason).
- **`--l2-min-rate 0.80` switches L2 off**, it does not tune it: rates cluster in 0.72–0.83,
  so one step past the default takes 9 → 1. Use 0.78 if you want ~half.
- **`--l2-min-tasks` is the sharpest knob** and nobody has tried moving it: 3 → 4 takes 9 → 2.
  It also attacks the defect most directly, since the duplicate family all sit at exactly 3.
- **`--l2-max-entries 4`, not 3**, is the best cap on this data — it is the smallest cap that
  keeps every distinct rule.

**Re-measure on your own model** (cheap; needs only a finished L2 arm):

```bash
.venv/bin/python - <<'EOF'
import json
R="runs_evolving/<model>/<the l2 arm>"
P=[json.loads(l) for l in open(f"{R}/l2_promotions.jsonl")]
for k,f in (("new_bests","new_best_attributions"),("rate","selection_rate"),
            ("tasks","distinct_tasks"),("selections","total_selections")):
    print(k, sorted(round(p[f],3) for p in P))
EOF
```

Then set each floor just below the value that keeps the fraction you want. A floor that
every promoted rule already clears is inert; a floor above the cluster switches the tier off.

### 8.7 The cap ranking degenerates at `new_bests = 0`

`score = rate × log1p(tasks) × log1p(new_bests)` and `log1p(0) = 0`, so **any candidate with
zero new-best attributions scores exactly 0 regardless of its rate or task count** — verified,
`score_candidate(0.99, 99, 0) == 0.0`. Such a candidate still *passes* the default floors,
because the new-bests floor ships **disabled at 0** — verified,
`passes_floors(sel=80, tasks=5, rate=0.80, new_bests=0) is True`. Ties then break on
`(-score, entry_id)`, i.e. **alphabetically**. Verified end to end: a cap of 3 over one real
candidate plus three synthetic zero-score ones returns the real one and then `aaa`, `bbb`.

**So a cap arm can silently select an arbitrary subset rather than the best one. Always arm
the new-bests floor alongside a cap** — `--l2-max-entries 4 --l2-min-new-bests 1` — so every
survivor has a non-zero score and the ranking means something.

**It could not have bitten on the one completed arm**, and the argument is worth keeping
because it is the only way to reason about eligibility after the fact:

1. that arm ran `l2_max_entries=0` (confirmed in `run_summary.json`), and
   `select_l2_promotions` truncates only under `max_entries > 0` (`:187-188`) — so it
   returned the eligible list unchanged;
2. the promotion loop's only other drops are the already-standing guard (`:417`, which
   cannot drop a first-time candidate) and `if not rendered.strip()` (`:426`) — and **0 of
   662 `shared_l1.jsonl` entries have empty content or produce an empty verbatim render**;
3. therefore **promoted ≡ eligible**, and since the minimum `new_bests` among the 9 was 3,
   **no skill was ever eligible with `new_bests = 0` in that arm**.

That is what makes the "inert below 4" reading in §8.6 a statement about the *eligible* set
and not merely the promoted one. It is conditional on no cap, one arm, one model.

### 8.8 Three things you cannot recover after the run — design your arm accordingly

1. **Do not replay eligibility from `l1_skill_usage.json`.** It stores a single *final*
   `global_iteration`, so replaying `rate = total_selections / (final_iter - created_at)`
   collapses to ~0.02 for every skill and reports **0 eligible in all 9 arms of the Aug-22
   wave — including the arm that demonstrably promoted 9.** This is exactly the frozen-evidence
   problem the module docstring describes: promotion fires at a task boundary when the
   denominator was small, and the denominator keeps growing afterwards against a frozen
   numerator. Eligibility is only replayable **boundary by boundary**, which is how the
   shipped floors were calibrated in the first place.
2. **`eligible_count` is computed and thrown away.** `l2_promotion.py:414` sets it on the
   summary; nothing persists it and even the verbose print at `governor.py:1443-1447` reports
   only `promoted`/`demoted`/`standing_count`. So **you cannot tell how many candidates
   cleared the floors but lost to a cap** — which is precisely the quantity a `--l2-max-entries`
   arm needs. If a cap arm matters, log it first (a one-line change) or the arm is
   uninterpretable.
3. **Rejected candidates leave no record at all.** `l2_promotions.jsonl` contains only
   `event: promote` / `demote`. There is no artifact of what nearly made it.

### 8.9 Candidate arms for an L2 batch, with what each is actually testing

None of these has been run. All four attack the §8.4 defect at a different point.

| arm | flags | attacks | measured expectation | caveat |
|---|---|---|---|---|
| `l2_merge` | `--enable-l2 --skill-merging --skill-merge-similarity 0.8` | **the source** — collapse the duplicates in L1 before anything can be promoted 6× | merging alone took a catalog 600 → 168 on a completed arm (open item 1) | **Confounded**: `--skill-merging` also uncaps the extractor catalog (open item 6). Read it against `merge_sim08`, **not** against `l2` — both have merging on, so the catalog confound cancels |
| `l2_extract` | `--enable-l2 --l2-render extract` | **the cost** — same rules, less text | terminal 4.79× → 2.17×, run-average 2.21× → 1.35× | Rule *count* and duplication are unchanged; this only separates prompt volume from rule content |
| `l2_cap4` | `--enable-l2 --l2-max-entries 4 --l2-min-new-bests 1` | **the count** — dedup as a side effect of ranking | keeps all 3 distinct rules + 1 representative | Ranking degenerates without the paired floor (§8.7); `eligible_count` is unlogged (§8.8) |
| `l2_tasks4` | `--enable-l2 --l2-min-tasks 4` | **the gate** — demand real cross-problem breadth | 9 → 2 | Sharpest knob, untested; 2 rules may be too few to move anything. Prefer this over `--l2-min-rate 0.80`, which switches the tier off |

The two **code** fixes are strictly better than any of these arms and cost no GPU time —
add a similarity gate to the promotion pass in `l2_promotion.py`, or make `--enable-l2`
require `--skill-merging`. Consider doing that *before* spending 4–5 arms measuring around
the defect.

### 8.10 Protocol for any L2 wave

- **Land the analysis fix first.** `aggregate_runs.py` / `compare_runs.py` cannot see
  `enable_l2` (open item 7), so an L2 arm and the control both render as design
  `truncation` in every CSV and delta table. This must land **before** a report is
  generated from a wave containing an L2 arm.
- **Every L2 batch must carry its own plain `l2` control on the same GPU.** A control from
  another batch sits in a different contention window and a different endpoint-latency
  window (open items 10–11), and per §3.4 unequal arms-per-GPU biases the comparison
  one-directionally.
- **n=1 per cell is a screen, not a test.** Replicate noise is log-SD 0.147, so a single
  contrast needs ≈×1.50 to clear 95% (open item 10). An L2 batch picks what to replicate;
  it cannot name a winner.
- **50 problems may be too short for this tier**, per §8.3/§8.5 — 16% of coder calls saw no
  L2 at all and the median call saw one rule. Treat "L2 is a null" as untested until either
  the run is longer or the floors are lowered enough to promote early.
- **Model matters more here than elsewhere.** The whole prompt-volume axis (`l2_extract`,
  `l2_cap4`) is premised on standing text being expensive. Check the context window of the
  model you are running before spending an arm on it.

```bash
HW=scripts_integration/new_evolving_agent/env/<HARDWARE>

# plain l2 control + the four probes, one spec, one GPU
cat > $HW/wave_l2.spec <<'SPEC'
l2            | truncation | --enable-l2
l2_merge      | truncation | --enable-l2 --skill-merging --skill-merge-similarity 0.8
l2_extract    | truncation | --enable-l2 --l2-render extract
l2_cap4       | truncation | --enable-l2 --l2-max-entries 4 --l2-min-new-bests 1
l2_tasks4     | truncation | --enable-l2 --l2-min-tasks 4
SPEC

bash $HW/launch_wave.sh 0 $HW/wave_l2.spec dry-run
bash $HW/launch_wave.sh 0 $HW/wave_l2.spec
```

Health checks specific to L2 arms — both must be non-zero, and neither is covered by the
generic checks in §3.5:

```bash
wc -l <run>/l2_promotions.jsonl <run>/l2_standing.jsonl
.venv/bin/python -c "import json;print(json.load(open('<run>/run_summary.json'))['l2_standing_count'])"
```

A zero here is silent: `governor.py:1448` swallows every promotion-pass exception into a
one-line `l2 promotion skipped:` print, and an arm that promotes nothing is
indistinguishable from a truncation arm in every other artifact.

### 8.11 Corrections and mechanism fixes (2026-08-27)

Everything in §8.1–8.10 above was derived from **one** arm,
`base_agent_gpt_oss_120b_l2_itr30_GH200_2026_08_22_20_34`. **That run directory no
longer exists on this host** — it survives only as rows in
`output/GH200x2_2nd_aug22_wave/`, so §8.6's re-measure recipe cannot be run against
it. The two L2 arms that *are* on disk change the picture:

| | gpt-oss-120b `…_21_32` | gpt-5.6-terra `…_21_23` |
|---|---|---|
| flags / protocol | identical | identical |
| problems | 50/50 | 50/50 |
| **rules promoted** | **0** | **4** |

Same gate, same defaults, 0 vs 4 — and 9 on the arm §8 is written from. **The
promotion count is not reproducible at defaults**, which is a prior question to
§8's "L2 is a null". A 0-promotion L2 arm is byte-identical to a truncation arm in
every metric, so §8's null may partly be measuring arms that had no tier at all.

**The gate is exactly replayable offline.** §8.8 says eligibility is replayable only
boundary-by-boundary; that is now implemented and validated to reproduce
`l2_promotions.jsonl` exactly on both arms (entry id, boundary, selections, tasks,
opportunity, rate). See `new_evolving_agent_analysis/l2_redesign/` — gate variants
can now be evaluated at zero GPU cost. Two traps it documents: both arms were
**resumed**, so the reconstructed `global_iteration` is offset by
`ledger_final − reconstructed` (26 terra, 42 oss) and mixing scales invents phantom
promotions; and the extractor prompt renders **two** `- id=` blocks, the second
being the standing-rules list, so a naive parse inflates a promoted skill's
visibility for the rest of the run.

**Corrected — `--l2-max-entries` is a PER-PASS cap, not a standing-set cap.**
§8.6 reads it as bounding the standing set ("4 → all three distinct rules + one
representative") and §8.9's `l2_cap4` arm is designed on that reading. Both are
wrong. `select_l2_promotions` runs at every task boundary and the standing set
accumulates across them: measured in replay, `--l2-max-entries 4` admits **19**
rules on the gpt-oss arm. Separately, already-standing skills were *counted against*
the cap — `compute_l2_candidates` reads `read_selectable_l1_jsonl`, which filters on
`status`, while `set_skill_tier` writes `tier` (`entry_tier` was imported into
`l2_promotion.py` and never used). Use the new `--l2-standing-cap` to bound the set.

**New defect — `selection_rate`'s denominator is not the support of its numerator.**
A skill can only be selected while it is inside the extractor's candidate set, and
`read_l1_extractor_catalog` returns `entries[-50:]` when governance is off
(`memory_manager.py:798`). Once a skill scrolls out of that tail its numerator
freezes while its denominator keeps ticking, so `rate` **decays monotonically**, at
a speed set by how fast the arm mints L1 skills — an arm-level property, not a
property of the skill. Measured with the exact per-iteration candidate sets:

| | gpt-oss | terra |
|---|---|---|
| L1 skills minted | 557 | 308 |
| median iterations a skill is *offered* | 118 | 231 |
| mean picks per extractor call | 6.81 | 3.85 |
| clearing tasks≥3 **and** selections≥50 | 49 | 35 |
| **best `selection_rate` ever reached** | **0.7059** | 1.0 |
| best `hit_rate` (selections / offers) | 0.9667 | 1.0 |

The gpt-oss arm's best candidate **missed the 0.70 floor by 0.0115** while being
picked in 96.7% of the iterations it was actually offered. This is why that arm
promoted nothing, and it is a threshold sitting on top of an arm-dependent ceiling.

**Three knobs added, all default OFF** (`--l2-use-hit-rate` / `--l2-min-hit-rate`,
`--l2-standing-cap`, `--l2-dedup-similarity`). Proven inert by replaying both arms
through the patched functions. *(Superseded by §8.13: they are now reachable
together as `--redesign-l2`, and `--l2-standing-cap` defaults to `-1` = no cap
rather than `0`. The default gate is still the shipped one.)* `hit_rate` needs a new `total_offers` counter in the
usage ledger; ledgers without it give `hit_rate` 0, which fails the floor rather
than scoring an unmeasured skill as a perfect hit.

Combined (`hit≥0.60`, standing cap 6, dedup 0.80) **both arms land on exactly 6
rules with zero pairs at cosine ≥0.80**, against 0-vs-4 shipped and 19-vs-8 under
the per-pass cap. **Be precise about what earns that:** `hit_rate` alone is *not*
arm-invariant — its floor response is only a shifted copy of `selection_rate`'s
(oss 19→4→2→0 across floors 0.60→0.80; terra 8→6→6→4). The arm-invariance comes
from the **standing cap**, by construction. `hit_rate` removes an unphysical decay
term and un-sticks the zero-collapse; it does not by itself make the metric
comparable across arms.

**§8.4's duplicate-family defect reproduces on a different arm and model.** Ranked
pairwise cosine over the gpt-oss candidate set at `hit≥0.60` shows a four-member
"enable TF32/cuDNN benchmark" family (0.82–0.87) and an "avoid trivial kernels"
family, 9 of 171 pairs at ≥0.80. τ=0.80 separates those from rules that merely share
vocabulary; it was chosen by reading the ranked list. Dedup is **greedy-pairwise
against already-kept rules, not transitive-closure clustering**, and fails **open and
loud** if embeddings are unavailable rather than silently promoting nothing.

**Fixed, both flagged in §8.8/open item 7 as blockers:**

- `eligible_count` is now persisted. A `pass` census row is appended to
  `l2_promotions.jsonl` at every boundary with `candidate_count`, `eligible_count`,
  `standing_after`, the full gate config and per-candidate drop reasons. §8.8 notes
  it was computed and discarded, leaving any capped arm uninterpretable.
- **Open item 7 is fixed.** `aggregate_runs.py` now extracts the L2 config and
  `compare_runs.py::design_variant_label` renders
  `truncation+l2:hit0.6:cap6:dedup0.8` instead of collapsing an L2 arm and its own
  control to the same design string. §8.10 calls this a launch blocker; it is
  cleared.

**Still true:** none of this shows L2 helps quality. It changes *which* rules get
promoted and makes the count reproducible. §8's null stands until a fresh arm says
otherwise, and at n=1 per cell no such arm can name a winner (open item 10).

### 8.12 The 11-arm redesign wave — measured (2026-08-29)

Eleven arms on one GPU, `gpt-oss-120b`, `NVIDIA_GH200x2_median`, all completed:
7 x 50 problems plus 4 x 15 problems on the level-2 block (problems 11-25) as
paired replicates. Health: `oom=0`, `proceeding UNLOCKED=0`, 1 mem-gate timeout in
~1200 evals. Full write-up and reproduction scripts:
`new_evolving_agent_analysis/l2_redesign/RESULTS.md`.

**Quality: a null, and the raw numbers were a lottery artifact.** Raw arm geomeans
spanned 1.058-1.696 and appeared to show the control beating every L2 variant. That
was 14 bimodal problems (see the lottery warning in section 4). Adjusted, every arm
sits in 0.971-1.057 with **every 95% CI containing 1.0**, and the
identical-configuration null contrast (0.961 [0.885, 1.043]) sits inside the same
band. Per-problem medians were ~1.0 for every arm even unadjusted. **No L2 variant
is distinguishable from its control in either direction, to within about +/-6%.**

**The shipped gate is still not reproducible.** Section 8.11 records 9 / 4 / 0 rules
from identical-flag runs; this wave's `l2` arm ran the same shipped defaults and
promoted **6** — a fourth distinct value.

**Dedup is the one change with a measured, reproducible effect.** Final standing
sets, embedded through the skill-merge path:

| arm | rules | pairs >= 0.80 cosine | max |
|---|---|---|---|
| **`l2_redesign`** (dedup 0.80) | 6 | **0** | **0.785** |
| `l2_judge` | 6 | 2 | 0.849 |
| `l2` (shipped) | 6 | 3 | 0.816 |
| `l2_preseed` | 5 | 3 | 0.911 |
| `l2_hit` | 6 | 4 | 0.912 |

Dedup fired 13 times in `l2_redesign`, and it is the only arm with no duplicate
pair. The section 8.4 defect reproduces plainly elsewhere: `l2_hit` promoted both
*"Avoid Hand-rolled Conv2d; Use cuDNN"* and *"Prefer cuDNN for Conv2d"* at 0.912.

**An LLM judge does NOT replace dedup — this is the answer to "can an agent make
the call instead of a rule".** The judge arm ran deliberately permissive floors
(`min_tasks 2`, `min_selections 15`, `min_rate 0.05`) so selectivity would come from
the judge rather than the gate. It accepted 6 rules with individually sensible
rationales and still left 2 near-duplicate pairs (max 0.849). The reason is
structural and will not be fixed by a better prompt: the judge scores each candidate
on its own merits, so it cannot see that two separately-plausible rules say the same
thing. **Redundancy is a set-level property; a per-item filter cannot detect it.**
Keep `--l2-dedup-similarity` regardless of whether the judge is on.

**Pre-seed freeze is exact** — `l2_preseed`, `q15_pre_r1`, `q15_pre_r2` all ended
`preseeded=5 standing=5 promoted=0 demoted=0`. The `preseeded_from` exemption is
load-bearing: without it a pre-seeded rule carries another run's `entry_id`, fails
the liveness check on the first pass, and the tier empties silently.

**Instrumentation defect, found and fixed here.** The pass-census key `dropped`
emitted *every* candidate carrying a `reasons` entry, including promoted ones — the
judge writes its acceptance rationale into the same field. Verified: `l2_judge`
entries 1 and 7 appear under `dropped` at `global_iteration 60` and were promoted at
that same boundary. Now emits `decisions[]` with an explicit `promoted` flag, and
`dropped` is filtered to genuine rejections. **Artifacts written before this fix
carry the misleading key** — cross-reference the `promote` events.

### 8.13 `--redesign-l2`: the merged design, and the no-cap default (2026-08-29)

The redesign landed on `features/evolving-agent-final` behind one switch. **The
shipped gate remains the default and is byte-unchanged**; the two designs are
directly comparable as `--enable-l2` versus `--enable-l2 --redesign-l2`.

```bash
--enable-l2                  # shipped gate (selection_rate, no dedup, no cap)
--enable-l2 --redesign-l2    # hit_rate 0.60 + dedup 0.80, no cap
```

**The preset is two knobs, both with measured support** (`L2_REDESIGN_PRESET` in
`l2_promotion.py` is the single source of truth, shared by the KernelBench and MLE
entry points so they cannot drift):

| knob | preset | why |
|---|---|---|
| `l2_use_hit_rate` / `l2_min_hit_rate` | on / 0.60 | `selection_rate`'s denominator counts iterations where selection was impossible, so it decays at a speed set by arm-level catalog growth |
| `l2_dedup_similarity` | 0.80 | the only change with a reproducible effect (section 8.12) |
| `l2_standing_cap` | **-1, no cap** | see below |

Deliberately excluded: `--l2-judge` (measured worst on quality, costs an LLM call
per pass, and does not dedup).

Precedence is **explicit flag > preset > shipped default**, implemented with
`None` argparse sentinels resolved in `_resolve_l2_preset` before validation --
without the sentinel, `--redesign-l2 --l2-min-hit-rate 0.70` would be silently
overridden. `--redesign-l2` without `--enable-l2` is a hard error rather than an
implicit enable. `run_summary.json` records `redesign_l2` alongside the *resolved*
knobs, and `compare_runs.py` renders `truncation+l2:redesign:hit0.6:dedup0.8`.

**Why the cap defaults to no cap (`-1`).** Measured on the 2026-08-27 wave: the
two arms that ran with **no cap at all** both ended at exactly **6** standing
rules (`l2` reached 6 at global iteration 1380, `l2_hit` at 1050). At ordinary run
lengths the *floors*, not a cap, are what bound the set -- so a cap in the default
preset would add a mechanism that does nothing while introducing the section 8.7
ranking degeneracy. Dropping it also drops the need for `--l2-min-new-bests 1`.

`-1` is the no-cap spelling; **any value `<= 0` means no cap**, and `0` is kept as
a silent alias because it was both the previous default and the previous
"unbounded" spelling. Re-purposing `0` to mean "promote nothing" would flip
behaviour under anything already passing it -- and "promote nothing" is already
expressible by omitting `--enable-l2`.

**A cap is still necessary in two regimes**, and the help text says so:

- **Loosened floors.** The judge arm (`min_tasks 2`, `min_selections 15`,
  `min_rate 0.05`) hit 6 rules at global iteration **90** -- problem 3 of 50 --
  and would have kept promoting for the remaining 47 problems uncapped. Any arm
  that relaxes the floors to let something else do the selecting needs a cap.
- **Long runs.** Promotion is one-way and the demote pass fires only on liveness,
  so the standing set is monotone and every rule sits in every remaining coder
  prompt. "Unbounded is fine" is a property of 50 problems, not of the design.

Pair any cap with `--l2-min-new-bests 1` (section 8.7).

**Census gap closed.** A cap *refusal* was previously recorded nowhere:
`select_l2_promotions` truncates with `eligible = eligible[:room]` and the census
binds `eligible_count` from that function's return, so a capped arm could not be
told apart from one whose floors simply admitted less -- the quantity section 8.8
says a capped arm needs. Refused candidates now carry a `standing cap: ...` reason
and the census gains `cap_refused_count`, so floors-eligible is recoverable as
`eligible_count + cap_refused_count + dedup drops`.

**Open, and cheap to settle:** no arm has run hit-rate 0.60 + dedup 0.80 with *no*
cap -- `l2_redesign` had a cap of 6, and because refusals were unrecorded at the
time, whether that cap bound in its tail is unknown. The bracketing evidence says
the risk is small (late promotions are rare: uncapped `l2` added one rule after
global iteration 1050, uncapped `l2_hit` none), but it is not a number. Replay it
offline at zero GPU cost with `new_evolving_agent_analysis/l2_redesign/` before
quoting an expected rule count for the preset.
