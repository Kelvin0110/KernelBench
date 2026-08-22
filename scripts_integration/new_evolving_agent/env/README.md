# CUDA toolchain for the evolving-agent runs (GH200 / aarch64)

## Why this exists

This host ships only the NVIDIA **driver** (580.173.02). There is no CUDA
toolkit: no `nvcc` on `PATH`, no `/usr/local/cuda`, no `cuda-toolkit` apt
package, and no `cuda_nvcc` wheel in the venv.

`torch.utils.cpp_extension._find_cuda_home` only looks at
`$CUDA_HOME` / `$CUDA_PATH`, `nvcc` on `PATH`, and `/usr/local/cuda`. With none
of those present it returns `None`, and **every** `load_inline(cuda_sources=...)`
call fails with:

```
CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

### What that did to the first four experiment runs

The runs now archived under `runs_evolving/archived/with_NVCC_bug/` were all
produced under this defect. It was not a partial degradation:

| arm | CUDA_HOME errors | share of iterations | problems touched |
|-----|-----------------:|--------------------:|-----------------:|
| truncation | 179 | 11.9% | 44 / 50 |
| markov_report | 144 | 9.6% | 47 / 50 |
| selective_retention | 116 | 8.7% | 40 / 50 |
| folding | 202 | 14.3% | 43 / 50 |

Worse than the failure count: the agent *adapted*. It learned to write

```python
if torch.cuda.is_available() and os.getenv("CUDA_HOME"):
    ext = load_inline(..., cuda_sources=...)   # never taken
else:
    ext = None
...
if ext is not None:
    return ext.custom(x)
return torch.clamp(1.0 - pred * targ, min=0.0).mean()   # reference PyTorch impl
```

The `__global__` and the `load_inline` call are both present in the source, so
the static checker passes. The guard is never true, so the CUDA path is dead
code and the **reference PyTorch op executes**. Such kernels score
`compiled=True, correct=True, speedup≈1.0` — PyTorch benchmarked against itself.

This idiom was then distilled into the per-run L1 skill memory. Entries matching
CUDA_HOME/workaround/fallback patterns, as a share of each store:

| arm | L1 entries | env-workaround entries | share |
|-----|-----------:|-----------------------:|------:|
| truncation | 585 | 407 | 70% |
| markov_report | 426 | 136 | 32% |
| selective_retention | 420 | 299 | 71% |
| folding | 592 | 431 | 73% |

Real entry titles include *"Guard CUDA Kernel Compilation with Pre-Import
CUDA_HOME Check & CPU Fallback"*. The agent was not gaming the grader — it
learned the dead-code pattern as **defensive engineering**, which is exactly why
the static checker never fired.

L1 is per-run (`run_dir/shared_l1.txt`, no global store), so a new run starts
with a clean skill memory. The contamination is *within* a run: problem #1 hits
the defect and the remaining 49 inherit the lesson.

## Install (no sudo, ~15 s, 421 MB)

```bash
PREFIX=$HOME/opt/cuda-12.8 VENV=/localhome/local-tianzheng/KernelBench/.venv \
  bash scripts_integration/new_evolving_agent/env/install_cuda128_local.sh
```

`dpkg -x` unpacks the CUDA 12.8 `ubuntu2404/sbsa` component debs into `$PREFIX`
(no root needed), then symlinks the headers and `.so`s that only ship as pip
wheels (cuBLAS, cuRAND, cuSOLVER, cuSPARSE, cuFFT, cuDNN, CUPTI, NCCL) into the
prefix, so extensions link against exactly the libraries torch already loads.

CUDA **12.8** is deliberate: it matches `torch 2.11.0+cu128`. The driver
advertises CUDA 13.0, which is backward compatible.

## Required environment for every run

```bash
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:/localhome/local-tianzheng/KernelBench/.venv/bin:$PATH
```

Both lines matter:

- **`CUDA_HOME` must be exported**, not merely satisfiable. Putting `nvcc` on
  `PATH` alone is enough for torch, but the archived runs' kernels literally test
  `os.getenv("CUDA_HOME")`, so leaving it unset keeps those guards closed and the
  dead-code path invisible.
- **`.venv/bin` must be on `PATH`**, or the build fails with
  `RuntimeError: Ninja is required to load C++ extensions`.

## Verify before launching

```bash
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH
nvcc --version        # expect: release 12.8, V12.8.93

uv run python - <<'PY'
import torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void add_one_kernel(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor add_one(torch::Tensor x){int n=x.numel(); add_one_kernel<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="probe", cpp_sources="torch::Tensor add_one(torch::Tensor x);",
                cuda_sources=src, functions=["add_one"], verbose=False)
print(m.add_one(torch.zeros(8, device="cuda")).tolist())   # expect eight 1.0s
PY
```

## Known caveats

- **First compile per kernel costs ~25 s.** Under the defect, builds failed
  instantly. Expect runs to take materially longer now, and expect `correct`
  counts to *drop* — kernels that previously "passed" by falling back to
  reference PyTorch now have to genuinely compile.
- `uv run` reinstalls the local `kernelbench` package on every invocation. This
  is harmless: it does not touch `site-packages/nvidia`, and CUDA builds were
  verified to survive it. Use `uv run --no-sync` if you want to avoid the churn.
- The prefix contains symlinks into `.venv/lib/python3.10/site-packages/nvidia`.
  Re-creating the venv, or upgrading a `nvidia-*-cu12` wheel to a different
  version, will dangle them — re-run the installer afterwards.
- **NVRTC was always available** (`libnvrtc.so.12` ships with the torch wheels
  and compiles device code fine). The defect was specific to the
  `load_inline`/`nvcc` path. No kernel in the archived runs used NVRTC, so the
  conclusions above are unaffected, but "no CUDA could compile" would be too
  strong a statement.

## Related

- **Standing up a second 2 × GH200 host:** [gh200_setup/](gh200_setup/) — a
  step-by-step clone of this environment (OS/boot prep, driver, uv, venv, this CUDA
  toolkit, baselines, multi-arm settings), read off the live host on 2026-08-22.
  Includes the `gh200-memory-online.service` unit, which ships in no package.
- Baseline timings for this host: `results/timing/NVIDIA_GH200x2/`. Runs **must**
  pass `--hardware NVIDIA_GH200x2`; the batch script defaults to
  `SONG_CPU6_A6000x4`, which silently scores against A6000 baselines.
- Early health check for an in-flight run:
  `scripts_integration/new_evolving_agent_analysis/checkpoint_run.py`
