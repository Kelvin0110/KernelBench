# GPU sharing vs. kernel speed — measured probe (2026-09-02)

**Question.** If N processes time a CUDA kernel on the *same* GPU, what happens to
(a) the *measured* kernel time KernelBench would score, and (b) *throughput*?

**Answer in one line.** Sharing a GPU costs **wall-clock, not measured kernel time**.
The CUDA-event median is flat to within ~4% out to 15 concurrent processes, while
aggregate throughput saturates at **~2x** and per-process throughput collapses as ~2/N.

## Setup

- Host `NVIDIA_GH200x2_2nd`, **GPU 1**, compute mode `Default`, **MPS not running**.
- Three small custom `load_inline` CUDA kernels, deliberately low-memory (12–256 MB of
  tensors each) so the probe could run against live arms without memory pressure:
  memory-bound `mish` (0.12 ms), `matmul 1024^3` (0.25 ms), `matmul 2048^3` (1.87 ms).
- Degrees **1, 3, 6, 9, 12, 15**. Workers rendezvous at a file barrier so they genuinely overlap.
- Two independent measurements:
  - **A. Saturating load** — tight launch loop for a fixed 15 s window, count launches
    (immune to straggler/ramp-down artifacts), with CUDA-event samples interleaved.
  - **B. KernelBench-style** — the repo's own `time_execution_with_cuda_event`
    (L2 thrash outside the event window, sync before each trial), 10 barrier-aligned blocks.
- **Caveat:** 8 live experiment arms shared GPU 1 throughout, so "degree 1" is not a
  pristine solo baseline. Their duty cycle is low and the background is common to every
  degree, so the *scaling* is sound; the absolute level carries a small offset.
  Health during the probe: 574 eval records, **0 OOM, 0 `proceeding UNLOCKED`**, 14/14 arms alive.

## A. Saturating load (tight launch loop, 15 s window)

**mish (memory-bound, 0.12 ms, 256 MB)**

| procs | aggregate launches/s | aggregate scaling | per-process | per-process vs solo | kernel median (ms) | median vs solo | p99 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 3605 | 1.00x | 3605 | 1.00x | 0.1224 | **1.000x** | 0.124 |
| 3 | 5409 | 1.50x | 1803 | 0.50x | 0.1249 | **1.021x** | 0.128 |
| 6 | 7492 | 2.08x | 1249 | 0.35x | 0.1236 | **1.010x** | 0.127 |
| 9 | 7444 | 2.06x | 824 | 0.23x | 0.1248 | **1.019x** | 0.127 |
| 12 | 7391 | 2.05x | 614 | 0.17x | 0.1246 | **1.018x** | 0.127 |
| 15 | 7291 | 2.02x | 486 | 0.13x | 0.1240 | **1.013x** | 0.127 |

**matmul 1024^3 (compute-bound, 0.25 ms, 12 MB)**

| procs | aggregate launches/s | aggregate scaling | per-process | per-process vs solo | kernel median (ms) | median vs solo | p99 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 1620 | 1.00x | 1620 | 1.00x | 0.2416 | **1.000x** | 0.253 |
| 3 | 2682 | 1.66x | 893 | 0.55x | 0.2514 | **1.041x** | 0.254 |
| 6 | 3333 | 2.06x | 555 | 0.34x | 0.2512 | **1.039x** | 0.254 |
| 9 | 3770 | 2.33x | 419 | 0.26x | 0.2505 | **1.037x** | 0.252 |
| 12 | 3493 | 2.16x | 291 | 0.18x | 0.2514 | **1.040x** | 0.253 |
| 15 | 3558 | 2.20x | 237 | 0.15x | 0.2511 | **1.039x** | 0.253 |

**matmul 2048^3 (compute-bound, 1.87 ms, 48 MB)**

| procs | aggregate launches/s | aggregate scaling | per-process | per-process vs solo | kernel median (ms) | median vs solo | p99 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 234 | 1.00x | 234 | 1.00x | 1.8685 | **1.000x** | 1.880 |
| 3 | 426 | 1.82x | 142 | 0.61x | 1.8657 | **0.998x** | 1.874 |
| 6 | 478 | 2.04x | 80 | 0.34x | 1.8660 | **0.999x** | 1.872 |
| 9 | 473 | 2.02x | 52 | 0.22x | 1.8650 | **0.998x** | 1.871 |
| 12 | 478 | 2.04x | 40 | 0.17x | 1.8656 | **0.998x** | 1.873 |
| 15 | 469 | 2.00x | 31 | 0.13x | 1.8668 | **0.999x** | 1.872 |


## B. KernelBench-style eval timing (barrier-aligned blocks, `time_execution_with_cuda_event`)

**mish (memory-bound, 0.12 ms, 256 MB)**

| procs | median (ms) | median vs solo | p90 vs solo | worst single trial (ms) | worst vs solo |
|---|---|---|---|---|---|
| 1 | 0.1313 | **1.000x** | 1.00x | 2.54 | **1.0x** |
| 3 | 0.1246 | **0.949x** | 0.84x | 1.83 | **0.7x** |
| 6 | 0.1244 | **0.947x** | 0.79x | 2.72 | **1.1x** |
| 9 | 0.1244 | **0.948x** | 0.79x | 2.24 | **0.9x** |
| 12 | 0.1242 | **0.945x** | 0.79x | 5.07 | **2.0x** |
| 15 | 0.1240 | **0.945x** | 0.79x | 3.23 | **1.3x** |

**matmul 1024^3 (compute-bound, 0.25 ms, 12 MB)**

| procs | median (ms) | median vs solo | p90 vs solo | worst single trial (ms) | worst vs solo |
|---|---|---|---|---|---|
| 1 | 0.2561 | **1.000x** | 1.00x | 1.54 | **1.0x** |
| 3 | 0.2511 | **0.980x** | 0.93x | 1.57 | **1.0x** |
| 6 | 0.2508 | **0.979x** | 0.92x | 2.39 | **1.6x** |
| 9 | 0.2510 | **0.980x** | 0.92x | 2.86 | **1.9x** |
| 12 | 0.2510 | **0.980x** | 0.93x | 9.25 | **6.0x** |
| 15 | 0.2508 | **0.979x** | 0.92x | 4.44 | **2.9x** |

**matmul 2048^3 (compute-bound, 1.87 ms, 48 MB)**

| procs | median (ms) | median vs solo | p90 vs solo | worst single trial (ms) | worst vs solo |
|---|---|---|---|---|---|
| 1 | 1.8958 | **1.000x** | 1.00x | 2.09 | **1.0x** |
| 3 | 1.8670 | **0.985x** | 0.97x | 6.15 | **2.9x** |
| 6 | 1.8656 | **0.984x** | 0.97x | 16.61 | **7.9x** |
| 9 | 1.8648 | **0.984x** | 0.97x | 11.10 | **5.3x** |
| 12 | 1.8651 | **0.984x** | 0.97x | 18.87 | **9.0x** |
| 15 | 1.8647 | **0.984x** | 0.97x | 30.11 | **14.4x** |


## Mechanism

Compute mode is `Default` and **MPS is not running**, so each process gets its own CUDA
context and the driver **time-slices** them rather than co-scheduling them. During its
slice a kernel has the whole GPU, so the CUDA-event window — which brackets only
`kernel_fn(*args)` — measures an essentially uncontended kernel. What the other processes
cost you is the *wait for your slice*, which lands outside the event window.

That is why the two tables disagree in the way they do:

- **median / p90 kernel time: immune** (0.945x–1.041x across all 3 kernels and all degrees).
- **worst single trial: badly hit** — up to **14.4x** (matmul 2048 at degree 15), because a
  context switch can land inside one trial's window.
- **aggregate throughput: hard ceiling at ~2x**, reached by degree 6 and flat to 15.
- **per-process throughput: ~2/N** — 0.50x at 3, 0.34x at 6, 0.22x at 9, 0.17x at 12, 0.13x at 15.

## Consequences for this project

1. **The scored number is safe; the schedule is not.** KernelBench scores
   `median` (`runtime_from_stats`), and the median is the statistic that survives sharing.
   This independently reproduces the 2026-08-23 result in `CLAUDE.md` §3.4 (concurrent evals
   cost <3% of *fidelity*) and extends it from degree 3 to degree 15.
2. **It does NOT contradict "9 arms/GPU scales linearly."** The agent is LLM-bound — an arm
   touches the GPU well under a second per iteration, so arms live at ~1–2% duty cycle and
   never reach the saturating regime probed here. This probe is the worst case, not the
   operating point. The ~2x ceiling binds only for GPU-saturating work.
3. **Never quote a `max`/tail runtime from a shared GPU.** The tail is the one statistic
   that moves, by up to 14x. Use median (as the harness does), and treat any per-trial
   worst-case from a multi-arm wave as meaningless.
4. **MPS would change this.** If aggregate throughput on one GPU ever becomes the binding
   constraint, enabling MPS is the lever — it lets contexts co-execute instead of
   time-slicing. It would, however, remove the isolation that currently protects the median,
   so it must not be turned on under a live measurement wave.

## Reproduce

```bash
cd scripts_integration/new_evolving_agent/env/gpu_sharing_probe
PROBE_GPU=1 PROBE_DEGREES=1,3,6,9,12,15 PROBE_DURATION=15 \
  ../../../../.venv/bin/python tput_orch.py      # table A (~6 min)
PROBE_GPU=1 PROBE_DEGREES=1,3,6,9,12,15 PROBE_TRIALS=100 PROBE_BLOCKS=10 \
  ../../../../.venv/bin/python orchestrate.py    # table B (~7 min)
```

Raw data: `results_tput.json` (A), `results.json` (B).
