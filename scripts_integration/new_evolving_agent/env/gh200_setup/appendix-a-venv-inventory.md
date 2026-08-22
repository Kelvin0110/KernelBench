# Appendix A — exact venv contents (source host, 139 packages)

> **`-035` is wider.** It runs `--extra dev --extra vis --extra gpu --extra
> evolving-agent` = **209 distributions** (`aideml` cannot install on aarch64; see
> [Python environment](07-python-environment.md)). The pins below still all match
> exactly — spot-checked 12/12 on 2026-08-22, including torch 2.11.0+cu128,
> scikit-learn 1.5.0 and pandas 2.1.4. The extras are additive only: no package in
> this appendix was removed or changed version.

*Part of the [2 × GH200 host setup guide](README.md).*

---

CUDA-relevant pins, all `cu12`:

```
nvidia_cublas_cu12        12.8.4.1     nvidia_cufile_cu12        1.13.1.3
nvidia_cuda_cupti_cu12    12.8.90      nvidia_curand_cu12        10.3.9.90
nvidia_cuda_nvrtc_cu12    12.8.93      nvidia_cusolver_cu12      11.7.3.90
nvidia_cuda_runtime_cu12  12.8.90      nvidia_cusparse_cu12      12.5.8.93
nvidia_cudnn_cu12         9.19.0.56    nvidia_cusparselt_cu12    0.7.1
nvidia_cufft_cu12         11.3.3.83    nvidia_nccl_cu12          2.28.9
nvidia_nvjitlink_cu12     12.8.93      nvidia_nvshmem_cu12       3.4.5
nvidia_nvtx_cu12          12.8.90
cuda_bindings 12.9.7   cuda_pathfinder 1.3.3   cuda_toolkit 12.8.1 (bare metapackage,
                                                no `all` extra — ships no nvcc)
```

Core:

```
torch 2.11.0+cu128   triton 3.6.0     numpy 1.26.2      scipy 1.11.4
scikit_learn 1.5.0   pandas 2.1.4     ninja 1.13.0      setuptools 80.9.0
transformers 4.57.5  datasets 4.8.4   openai 2.15.0     litellm 1.41.1
modal 1.3.0.post1    pydra_config 0.0.17.post1          einops 0.8.1
tabulate 0.9.0       tomli 2.4.0      tqdm 4.66.3       python_dotenv 1.2.1
pytest 7.4.3         ruff 0.14.11     kernelbench 0.2.0.dev0 (editable, src/)
```

`kernelbench` is installed editable via
`__editable__.kernelbench-0.2.0.dev0.pth` → `<repo>/src`. `self-evolving-agent` is
**not** installed at all ([Repository](06-repository.md)).

---

[← Failure modes](14-failure-modes.md) · [Index](README.md) · [Appendix B: alternative nvcc →](appendix-b-alternative-nvcc.md)
