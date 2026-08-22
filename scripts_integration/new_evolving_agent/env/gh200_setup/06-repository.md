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
Self-Evolving-Agent  995d35f0f401a012795935ee24d410764146afc6   (github.com/Kelvin0110/Self-Evolving-Agent)
aideml               ccdf85084bfea47be1e4358bf254a9ab5229caad   (github.com/Kelvin0110/aideml)
```

Always take the pin from the superproject rather than this file:

```bash
git rev-parse HEAD:Self-Evolving-Agent
```

### `Self-Evolving-Agent` is private — HTTPS will fail

`.gitmodules` records an **HTTPS** URL, but the repo is private, so a bare
`git submodule update --init` dies with:

```
fatal: could not read Username for 'https://github.com': No such device or address
```

(`aideml` is public and clones fine, which makes the failure look partial.) Point
just that submodule at SSH — this writes to `.git/config`, so it dirties no tracked
file and needs no edit to `.gitmodules`:

```bash
git config --local submodule."Self-Evolving-Agent".url \
  git@github.com:Kelvin0110/Self-Evolving-Agent.git
git submodule update --init --recursive
```

A repo-wide `url.<base>.insteadOf` rewrite does **not** work here — it is not applied
to the submodule clone.

### If init reports `upload-pack: not our ref`

The superproject pins a SEA commit that has never been pushed. Git then leaves the
submodule on `main`, **silently** — and SEA's `main` carries MLE-Bench UI/scoring
work, not the KernelBench evolving agent, so the tree looks plausible while being
wrong. Always assert the pin after init:

```bash
[ "$(git -C Self-Evolving-Agent rev-parse HEAD)" = "$(git rev-parse HEAD:Self-Evolving-Agent)" ] \
  && echo pin-ok || echo "PIN MISMATCH"
```

`acceptance_test.sh` checks this. The fix is to push the missing commit from the host
that created it; there is no way to reconstruct it from the remote.

The KernelBench problem set (`KernelBench/level{1,2,3,4}/`) and
`subset_selection/selected_problems_50.csv` ship **inside the repo** — no HuggingFace
download step is required.

`Self-Evolving-Agent` is **not** pip-installed. `evolve_kb_batch.py:20-24` inserts the
repo root and `Self-Evolving-Agent/` onto `sys.path` at import time. Confirmed on the
source host: `import evolving_common` fails from a bare interpreter and that is
correct. Do not "fix" it by installing the `evolving-agent` extra.

### The narrow GPU-eval lock is now committed (was: transfer-by-patch)

Earlier revisions of this guide told you to carry the narrow GPU-eval lock across as
`git diff` patches, because it lived only in uncommitted working-tree changes on
`lego-c2g2-smc-034`. **That is no longer necessary** — as of SEA `995d35f0` and
superproject `c7f9e52f` all three pieces are committed, and a correct
`git submodule update --init` gives you the lot:

| file | commit | change |
|---|---|---|
| `src/kernelbench/eval.py` | superproject | `_gpu_timing_lock` — narrows the lock to correctness + timing windows |
| `evolving_common/governor/gpu_lock.py` | SEA `f20d52a` | re-entrancy depth counter |
| `kernelbench_integration/eval_runner.py` | SEA `4a52c04` | precompile-then-lock boundary |

Verify all three landed rather than assuming:

```bash
grep -c _gpu_timing_lock src/kernelbench/eval.py                              # >= 1
grep -c _held_depth Self-Evolving-Agent/evolving_common/governor/gpu_lock.py  # >= 1
grep -c _precompile_candidate Self-Evolving-Agent/kernelbench_integration/eval_runner.py  # >= 1
```

**The re-entrancy counter is load-bearing, not a nicety.** `eval_runner.py` takes the
lock and then calls `eval_kernel_against_ref`, which takes it again in the same
process. `flock` is per open-file-description, so without the counter the second
`os.open()` blocks against the process's own lock: every eval stalls the full
`KB_GPU_EVAL_LOCK_TIMEOUT_SEC` (**default 1800 s**) and then runs its timing window
UNLOCKED. Measured on `-035` against a pin that predated `f20d52a`: nested acquisition
consumed the entire timeout; with the fix it returns in 0.000 s with
`reentrant=True`.

Still genuinely uncommitted on the source host: untracked `build_*_ext/` scratch dirs
(safe to ignore).

---

[← uv](05-uv.md) · [Index](README.md) · [Python environment →](07-python-environment.md)
