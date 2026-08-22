# Step 4 — Repository

*Part of the [2 × GH200 host setup guide](README.md).*

---

```bash
export REPO=$HOME/KernelBench
git clone git@github.com:Kelvin0110/KernelBench.git "$REPO"
cd "$REPO"
git checkout features/evolving-agent-final
git submodule update --init --recursive
```

Submodule pins at the time of writing:

```
Self-Evolving-Agent  015380b764bfb370b1e3aad4712e98c4bfe70603   (github.com/Kelvin0110/Self-Evolving-Agent)
aideml               ccdf85084bfea47be1e4358bf254a9ab5229caad   (github.com/Kelvin0110/aideml)
```

The KernelBench problem set (`KernelBench/level{1,2,3,4}/`) and
`subset_selection/selected_problems_50.csv` ship **inside the repo** — no HuggingFace
download step is required.

`Self-Evolving-Agent` is **not** pip-installed. `evolve_kb_batch.py:20-24` inserts the
repo root and `Self-Evolving-Agent/` onto `sys.path` at import time. Confirmed on the
source host: `import evolving_common` fails from a bare interpreter and that is
correct. Do not "fix" it by installing the `evolving-agent` extra.

### ⚠ Uncommitted work that a clone will NOT give you

The source host carries local modifications that are **not in any commit**. A fresh
clone silently lacks them — including the narrow GPU-eval lock described in
`CLAUDE.md [Driver](04-driver.md).4`, which is what makes >1 arm per GPU viable:

| file | change |
|---|---|
| `src/kernelbench/eval.py` | `_gpu_timing_lock` — narrows the lock to correctness + timing windows (+45 lines) |
| `Self-Evolving-Agent/evolving_common/governor/gpu_lock.py` | re-entrancy depth counter (+23/-1) |
| `Self-Evolving-Agent/kernelbench_integration/eval_runner.py` | precompile-then-lock boundary (+35/-19) |
| `CLAUDE.md` | docs |

Transfer them explicitly. On the **source** host:

```bash
cd /localhome/local-tianzheng/KernelBench
git diff -- src/kernelbench/eval.py CLAUDE.md    > /tmp/kb-root.patch
git -C Self-Evolving-Agent diff                  > /tmp/kb-sea.patch
```

On the **target** host, after cloning:

```bash
cd "$REPO"
git apply --check /tmp/kb-root.patch && git apply /tmp/kb-root.patch
git -C Self-Evolving-Agent apply --check /tmp/kb-sea.patch && \
  git -C Self-Evolving-Agent apply /tmp/kb-sea.patch
```

Skipping this does not error — it just means the target host is running the *wide*
lock, i.e. a hard ceiling of ~3.9 arms/GPU and different contention behaviour. If
you intend to compare results across the two hosts, they must match.

Also uncommitted on the source host: `pyproject.toml`'s `scikit-learn` promotion
(already reflected in `uv.lock` — see [Python environment](07-python-environment.md)) and untracked `build_*_ext/` scratch dirs
(safe to ignore).

---

[← uv](05-uv.md) · [Index](README.md) · [Python environment →](07-python-environment.md)
