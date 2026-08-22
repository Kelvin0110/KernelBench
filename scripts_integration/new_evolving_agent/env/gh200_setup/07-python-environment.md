# Step 5 — Python environment

*Part of the [2 × GH200 host setup guide](README.md).*

---

```bash
cd "$REPO"
uv sync --extra dev
```

`--extra dev` reproduces the source host exactly: base dependencies + `pytest` +
`ruff` — 140 distributions.

### `--all-extras` does not work on aarch64

Measured on `-035`, 2026-08-22:

```
error: Distribution `torchtext==0.18.0` can't be installed because it doesn't have a
source distribution or wheel for the current platform
hint: You're on Linux (`manylinux_2_39_aarch64`), but torchtext (v0.18.0) only has
wheels for: `manylinux1_x86_64`, `macosx_11_0_arm64`, `win_amd64`
```

`torchtext` comes from the **`aideml`** extra and is the *only* blocker — torchtext
was archived upstream and never shipped aarch64 wheels. Per-extra resolution:

| extra | on aarch64 | adds |
|---|---|---|
| `dev` | OK | pytest, ruff |
| `vis` | OK | +7 (matplotlib) |
| `gpu` | OK | +22 (tilelang, cupy-cuda12x, nvidia-cutlass-dsl, nsight-python) |
| `evolving-agent` | OK | +47 (chromadb, pydantic) |
| `aideml` | **FAILS** | torchtext — no aarch64 wheel |

So the widest working set is everything except `aideml`:

```bash
uv sync --extra dev --extra vis --extra gpu --extra evolving-agent   # 209 packages
```

**`-035` currently runs this wider set; `-034` runs `--extra dev` only.** That is a
deliberate divergence — record it when comparing results across the two hosts.

Two things measured rather than assumed before doing it:

- **The toolkit survives.** Adding these extras is purely additive: 68 added, 0
  removed, **0 version changes**, and the only `nvidia-*` entries are *new*
  (`nvidia-cutlass-dsl`, `nvidia-ml-py`). No existing `nvidia-*-cu12` wheel is
  replaced, so the [CUDA toolkit](08-cuda-toolkit.md) symlinks do not dangle —
  verified 0 dangling afterwards, and `acceptance_test.sh` stayed 12/12.
- **The `evolving-agent` extra is a no-op for imports.** `self-evolving-agent`
  installs, but its editable install ships only a `.pth` and dist-info and exposes no
  top-level package, so `import evolving_common` *still* requires the `sys.path`
  insert in `evolve_kb_batch.py`. It does not shadow or change the mechanism in
  [Repository](06-repository.md) — it only adds weight.

The one real reason to want `gpu`: `eval_kernel_against_ref` accepts
`backend` in `{cuda, triton, tilelang, cute}`, and tilelang/cute need it. The evolving
agent runs `backend="cuda"`, so it is optional for these experiments.

**`--no-sync` matters more now**, not less: a bare `uv run` re-syncs to the default
extras and would prune all 69 of these packages.

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
