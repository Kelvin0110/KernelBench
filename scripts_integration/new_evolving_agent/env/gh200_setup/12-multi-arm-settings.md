# Step 10 — Multi-arm settings

*Part of the [2 × GH200 host setup guide](README.md).*

---

The agent is LLM-bound, not GPU-bound: an eval subprocess lives ~38-45 s but touches
the GPU for well under a second. Sharing a GPU across arms is close to free.
Measured on the source host (50 problems × 30 iterations):

| arms/GPU | throughput | per-arm slowdown |
|---|---|---|
| 1 | 1.00× | — |
| 3 | 3.07× (steady state) | none |
| 6 | 3.31× *(under the old wide lock)* | 1.69× |

Required when sharing a GPU:

```bash
export KB_GPU_RESERVE_GB=0            # REQUIRED. default 42.0 (governor.py:163)
export KB_GPU_EVAL_LOCK=1             # default; leave on (gpu_lock.py:60)
export KB_GPU_EVAL_LOCK_TIMEOUT_SEC=1800   # default; raise with many arms
```

- `KB_GPU_RESERVE_GB=0` — each governor otherwise pins a 42 GB block while waiting on
  the LLM. Several reservers fight for headroom and can OOM whichever arm is
  mid-eval. Harmless to set: the block is released around eval anyway; the default
  stays 42 GB for single-arm runs.
- `KB_GPU_EVAL_LOCK` — cross-process `flock` keyed by GPU UUID
  (`Self-Evolving-Agent/evolving_common/governor/gpu_lock.py`). Free when
  uncontended (0.000 s across 129 solo evals). Cannot deadlock: `flock` is released
  by the kernel on process death. On timeout the arm logs loudly and proceeds
  **unlocked** — `proceeding UNLOCKED` in a log means that eval's numbers are
  contended.

Other knobs: `KB_GPU_EVAL_LOCK_DIR`, `KB_GPU_EVAL_LOCK_LABEL`, `KB_ITER_CACHE`,
`KB_DATA`, `KB_READY_PROMISE`.

**Launcher gotcha:** `launch_run.sh:41` refuses a GPU reporting >1000 MiB used. An
idle arm holds ~550 MiB (548-558 observed), so arm 2 passes but arm 3 reads ~1.1 GB and is rejected.
Raise that threshold on the new host if you plan ≥3 arms per GPU.

**Never edit code while runs are live.** Eval uses `multiprocessing` **spawn**
(`Self-Evolving-Agent/evolving_common/execution.py:348`), so every eval re-imports `evolve_kb_batch.py` and
`kernelbench/eval.py` **from disk** — nothing is frozen at launch. On 2026-08-20 an
edit that briefly referenced a not-yet-created module killed eval workers across 9
live arms for 9 minutes; 32 iterations were recorded as fake `compiled=False`
failures which the governor then "debugged" as kernel bugs. If you must edit mid-run:
write to a temp file, `ast.parse` it, then `os.replace` atomically.

---

[← Timing baselines](11-timing-baselines.md) · [Index](README.md) · [First run →](13-first-run.md)
