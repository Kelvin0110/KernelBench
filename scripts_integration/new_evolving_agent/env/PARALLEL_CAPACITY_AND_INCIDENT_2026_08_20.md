# 6 arms on one GPU: capacity verdict + a live-edit incident

Observed 2026-08-20 while running 6 context-management arms on GPU 0
(2x truncation, 2x markov_report, 2x folding, launched 16:32-16:48 UTC) alongside
3 merge arms on GPU 1 (launched 2026-08-19 17:29-17:35). 9 arms total.

---

## 1. Verdict: the GPU is not the constraint. 6 arms is comfortable.

| metric | GPU 0 (6 arms) | GPU 1 (3 arms, mature) |
|---|---|---|
| peak total GPU memory | 13186 MiB (12.9 GiB) = **9.0%** of the 142.8 GiB card | ~12.9 GiB |
| max concurrent eval subprocesses | **2** | 2 |
| concurrency histogram (184 samples) | `{1: 183, 2: 1}` | - |
| lock waits | n=32 median 37.3s p90 228.1s **max 519.0s** | n=559 median 30.9s p90 265.2s **max 524.9s** |
| waits exceeding the timeout | **0** | **0** |
| contended (`proceeding UNLOCKED`) evals | **0** | **0** |
| CUDA OOM | 0 | 0 |
| `coder_call_error` | 0 | 0 |
| rate-limit events | 0 | 1 (recovered, attempt 1/6) |

Doubling 3 arms -> 6 arms did **not** degrade lock contention: the two wait
distributions are statistically indistinguishable (median 37.3s vs 30.9s, max
519.0s vs 524.9s). Caveat: the 6-arm sample is young (n=32 vs n=559).

`KB_GPU_EVAL_LOCK_TIMEOUT_SEC` was raised to 5400s for the GPU-0 arms as
insurance. Given max observed wait is 519s, it was not needed - but the failure
mode it guards is silent (on expiry the lock logs `proceeding UNLOCKED` and
measures under contention, deflating that speedup), so the guard stays.

## 2. Why memory does not scale with arm count

**The arms' main processes hold ZERO GPU memory.** Verified: no arm pid ever
appears in `nvidia-smi --query-compute-apps`; only their children do.

Each candidate eval is a **short-lived `multiprocessing` spawn subprocess**
(~60-95s) that creates a CUDA context, runs, exits, and fully releases. Memory
is bursty, not cumulative - which is why 6 arms cost the same peak as 3.

`KB_GPU_RESERVE_GB=0` is essential here. The default 42 GB reservation
(`kernelbench_integration/governor.py:149`) would have meant 6 x 42 = 252 GB on
a 143 GB card. `GPUMemoryReserver.acquire()` returns early when
`reserve_bytes <= 0`, so zeroing it is a supported path, not a workaround.

## 3. Do arms queue *for memory*, or queue *holding* memory?

**They hold memory while queuing. Only the measurement is serialised.**

`eval_runner.py` ordering:
- `:74` `_precompile_candidate(...)` - nvcc compile + `load_custom_model`, which
  creates the CUDA context and allocates. **Outside the lock, by design** (nvcc is
  CPU work; holding the lock across it would turn a ~1s critical section into ~38s).
- `:78` `with gpu_eval_lock(...)` - wraps **only** `eval_kernel_against_ref`, the
  timed measurement.

So a subprocess allocates first, then queues. Directly observed: two eval
subprocesses co-resident at 548 MiB + 4738 MiB.

In practice the resident set stays at ~1 (183 of 184 samples) because the
precompile window is short next to the LLM-bound agent cycle. **Peak memory is
therefore governed by the largest single eval (~12.9 GiB), not by arm count** -
but the guarantee is statistical, not structural. The structural worst case is
N arms x peak-eval co-resident; at 6 x 12.9 GiB = 77 GiB that still fits, which
is the real reason 6 is safe.

---

## 4. INCIDENT: 38 failed eval spawns from live-editing the submodule mid-run

**Not a capacity problem.** Bounded and now stopped, but it hit all 9 arms.

- **Symptom:** 38 tracebacks - 25 across the 6 GPU-0 arms, 13 across the 3 merge
  arms. Every one identical:
  `ModuleNotFoundError: No module named 'evolving_common.governor.l2_promotion'`
- **Mechanism:** `gen3_stages.py:281` (also `:835`, `:1033`) does a function-level
  `from evolving_common.governor.l2_promotion import ...`. multiprocessing **spawn**
  re-imports the main module in every eval child
  (`spawn_main -> prepare -> _fixup_main_from_path -> runpy.run_path`), which
  re-executes `evolve_kb_batch.py` and re-imports `gen3_stages` **from disk**.
- **Trigger:** the submodule was edited at **17:00-17:03**, after all 9 arms were
  already running. `gen3_stages.py` gained the `l2_promotion` import and
  `l2_promotion.py` was created in the same minute (both mtime 17:00). Eval
  children spawning inside that window imported the new call site before the new
  module was readable, and died.
- **Blast radius:** each failed spawn is a lost eval. `run_kernelbench_eval`
  returns `{"ok": False, "worker_error": ...}`, which the governor records as a
  failed iteration - i.e. these surface as **false negatives**, indistinguishable
  from a genuinely bad kernel.
- **Status: stopped.** The last traceback in every log sits at 33-82% of the file
  with none in the tail; evals since then succeed with real speedups
  (merge rep1 `iter=26 correct=True speedup=0.3251`).

### The transferable lesson

**Running arms re-read code from disk on every eval spawn.** They do not run
against a snapshot of the code taken at launch. Editing `Self-Evolving-Agent/`
while runs are in flight changes the behaviour of those runs mid-experiment and
can kill evals outright.

The submodule is currently dirty (10 modified files) with `l2_promotion.py`
untracked - so more edits are presumably in progress.

**Do not edit the submodule while arms are running.** If a change is needed,
either wait for the runs to finish, or work in a separate checkout and only swap
it in between runs. A brief window of an inconsistent import graph is enough to
poison iterations across every concurrent arm at once.

### Worth considering

The 38 lost iterations are spread across 9 arms and land early in problem 1, so
the impact on final `best_geomean` is probably small - but it is not zero, and
it is not visible in any summary metric. If any of these arms produces a
surprising result, check its log for a traceback before trusting it.
