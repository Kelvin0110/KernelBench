# Step 8 — Acceptance test

*Part of the [2 × GH200 host setup guide](README.md).*

---

Run all of it. Each check corresponds to a failure that once silently corrupted a
~70 h run.

```bash
cd "$REPO"
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH

nvcc --version | tail -2          # release 12.8, V12.8.93
which ninja                       # <repo>/.venv/bin/ninja
gcc --version | head -1           # 13.3.0
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
grep -q NVIDIA_INF_API_KEY .env && echo "api key present"
```

Then the real test — an end-to-end `load_inline(cuda_sources=...)` build and launch,
identical to `launch_run.sh`'s preflight:

```bash
uv run --no-sync python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="host_acceptance_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                cuda_sources=src, functions=["f"], verbose=False)
ok = m.f(torch.zeros(8, device="cuda")).sum().item() == 8.0
print("CUDA extension build+run:", "OK" if ok else "WRONG RESULT")
sys.exit(0 if ok else 1)
PY
```

First build takes **~25 s** (nvcc + ninja cold). `OK` is the only acceptable result.
If it prints a `CUDA_HOME environment variable is not set` traceback, stop and fix
[CUDA toolkit](08-cuda-toolkit.md)/[Environment exports](09-environment-exports.md) — do not launch anything.

---

[← Environment exports](09-environment-exports.md) · [Index](README.md) · [Timing baselines →](11-timing-baselines.md)
