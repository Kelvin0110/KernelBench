# Step 5 — Python environment

*Part of the [2 × GH200 host setup guide](README.md).*

---

```bash
cd "$REPO"
uv sync --extra dev
```

`--extra dev` reproduces the source host exactly: base dependencies + `pytest` +
`ruff`. Do **not** add `--all-extras` — the `gpu` extra pulls `tilelang`,
`cupy-cuda12x`, `nvidia-cutlass-dsl` and the `evolving-agent` extra pulls `chromadb`;
none of these are installed on the source host (verified: no `chromadb` dist-info).

### The `--no-sync` rule

Every subsequent invocation must be `uv run --no-sync`, never bare `uv run`.
Current drift on the source host (read-only `uv sync --dry-run`, 2026-08-22):

```
Would uninstall 5 packages / Would install 1 package
 - iniconfig  - pluggy  - pytest  - ruff      ← the dev extra, silently pruned
 ~ nvidia-cusparselt-cu12==0.7.1              ← reinstalled
```

Two distinct hazards:

1. bare `uv run` prunes the dev extra;
2. reinstalling **any** `nvidia-*-cu12` wheel replaces files under
   `site-packages/nvidia/`, and [CUDA toolkit](08-cuda-toolkit.md)'s toolkit prefix contains **symlinks into that
   directory**. A reinstall dangles them and CUDA builds start failing. If it ever
   happens, just re-run the toolkit installer.

`launch_run.sh` uses `--no-sync` at all three call sites. Keep it.

> `CLAUDE.md [Prerequisites](02-prerequisites.md).2` warns that `uv sync` would remove **scikit-learn** and break
> `--skill-merging`. That is no longer true on the source host — the lock has since
> been regenerated and ships `scikit-learn==1.5.0`, which is installed. The
> `--no-sync` rule still stands for the two reasons above.

### Verify

```bash
.venv/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(torch.cuda.get_arch_list())"
```

Source-host reference:

```
2.11.0+cu128 12.8 91900
2 ['NVIDIA GH200 144G HBM3e', 'NVIDIA GH200 144G HBM3e']
['sm_80', 'sm_90', 'sm_100', 'sm_120']
```

The target should print exactly this. `sm_90` must be in the arch list.

---

[← Repository](06-repository.md) · [Index](README.md) · [CUDA toolkit →](08-cuda-toolkit.md)
