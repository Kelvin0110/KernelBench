# Step 6 — CUDA 12.8 toolkit (userspace, no sudo)

*Part of the [2 × GH200 host setup guide](README.md).*

---

### Why this exists (read before deciding to skip it)

`torch.utils.cpp_extension._find_cuda_home` looks only at `$CUDA_HOME`/`$CUDA_PATH`,
`nvcc` on `PATH`, and `/usr/local/cuda`. The source host has a driver but **no**
toolkit — `/usr/local/` is empty of CUDA and no `cuda-toolkit` apt package is
installed. With none of those present, every `load_inline(cuda_sources=...)` fails.

The failure is not loud. It taught the agent to emit

```python
if torch.cuda.is_available() and os.getenv("CUDA_HOME"):
    ext = load_inline(..., cuda_sources=...)   # branch never taken
else:
    ext = None
...
return torch.clamp(1.0 - pred * targ, min=0.0).mean()   # reference PyTorch runs
```

which passes the static checker (a `__global__` and a `load_inline` call are both
present in the source), scores `compiled=True, correct=True, speedup≈1.0`, and
poisoned four ~70 h runs before anyone noticed — up to 73% of the L1 skill catalog
became entries teaching this dead-code idiom. Full postmortem in
[README.md](../README.md).

**On the new host: install the toolkit before the first run, not after.**

### Installer

The target host is the **same architecture** as the source host (aarch64 / sbsa), so
the repo's own script runs unmodified:

```bash
cd "$REPO"
PREFIX=$HOME/opt/cuda-12.8 VENV="$REPO/.venv" \
  bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/install_cuda128_local.sh
```

Runs in ~15 s, downloads and installs ~421 MB. **Order matters: run this after
[Python environment](07-python-environment.md)**, because it symlinks against
`$REPO/.venv/lib/python3.10/site-packages/nvidia/`.

What it does, in order:

1. `curl`s ten CUDA 12.8 component debs from
   `developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa`
   (nvcc, crt, nvvm, cudart, cudart-dev, cccl, driver-dev, profiler-api, nvtx,
   nvrtc-dev). nvcc/crt/nvvm/nvrtc-dev are `.93`; the rest are `.90` — upstream
   versions its components separately and this is the real 12.8.1 combination.
2. `dpkg -x` each into a scratch root — **no root needed** — and copies
   `usr/local/cuda-12.8/` into `$PREFIX`.
3. Symlinks the headers and `.so`s that only ship as pip wheels (cuBLAS, cuRAND,
   cuSOLVER, cuSPARSE, cuFFT, cuDNN, CUPTI, NCCL, cuSPARSELt, nvJitLink, cuFile)
   from `site-packages/nvidia/` into the prefix, so extensions link against exactly
   the libraries torch already dlopens.
4. Adds unversioned `lib*.so` devlinks — pip wheels ship only versioned sonames, and
   `-lcublas` needs the devlink.

Verified on 2026-08-22: a clean run into a throwaway prefix reproduces the in-use
`$HOME/opt/cuda-12.8` exactly — 68 libs and 188 headers under
`targets/sbsa-linux/`, zero-line diff against the live prefix.

Expected result:

```
$HOME/opt/cuda-12.8/                      # 421 MB
├── bin/            nvcc, ptxas, cicc, cudafe++, nvlink, fatbinary
├── include  -> targets/sbsa-linux/include
├── lib64    -> targets/sbsa-linux/lib
├── nvvm/           cicc, libnvvm, libdevice.10.bc
└── targets/sbsa-linux/{include,lib}      # 188 headers, 68 libs
```

and

```
nvcc: NVIDIA (R) Cuda compiler driver
Built on Fri_Feb_21_20:26:18_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
```

**12.8 is deliberate** — it matches `torch 2.11.0+cu128`. Do not "upgrade" to 13.x to
match what `nvidia-smi` prints.

`gcc 13.3.0` (noble default) is a supported host compiler for nvcc 12.8; no
`gcc-12` sidegrade is needed. Verify with `gcc --version` before building.

---

[← Python environment](07-python-environment.md) · [Index](README.md) · [Environment exports →](09-environment-exports.md)
