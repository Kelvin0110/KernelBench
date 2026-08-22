# Appendix B — alternative nvcc sources (not used, not validated here)

*Part of the [2 × GH200 host setup guide](README.md).*

---

- `uv pip install nvidia-cuda-nvcc-cu12==12.8.93` puts an `nvcc` in the venv. It ships
  nvcc/crt/nvvm but **not** `cuda_runtime.h`, CCCL, or the driver headers, so
  `load_inline` still needs `nvidia-cuda-runtime-cu12`, `nvidia-cuda-cccl-cu12` and
  friends plus manual `CUDA_HOME` layout stitching. Cheaper to describe than to make
  work; the [CUDA toolkit](08-cuda-toolkit.md) deb path is what is proven on the source host.
- `sudo apt install cuda-toolkit-12-8` from NVIDIA's CUDA repo is the obvious answer
  **if you have root on the target**. It installs to `/usr/local/cuda-12.8`, which
  `_find_cuda_home` discovers via `/usr/local/cuda`. You still must `export CUDA_HOME`
  explicitly ([Environment exports](09-environment-exports.md)), because agent-written kernels test the variable directly. If you go
  this route, skip [CUDA toolkit](08-cuda-toolkit.md) entirely and set `CUDA_HOME=/usr/local/cuda-12.8`.
- **NVRTC was never the problem.** `libnvrtc.so.12` ships in the torch wheels and
  compiles device code fine even with no toolkit. The [CUDA toolkit](08-cuda-toolkit.md) defect is specific to the
  `load_inline`/`nvcc` path.

---

[← Appendix A: venv inventory](appendix-a-venv-inventory.md) · [Index](README.md)
