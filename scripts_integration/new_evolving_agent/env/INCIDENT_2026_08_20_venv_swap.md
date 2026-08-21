# Incident 2026-08-20 16:48 UTC — shared `.venv` re-synced under nine live arms

## What happened

At **16:48:26 UTC**, while nine arms were mid-run (3 merge arms on GPU 1 at
~23 h elapsed, 6 context-management arms on GPU 0 at ~16 min elapsed), something
ran a **bare `uv run` / `uv sync`** against the shared project environment. It:

| artifact | before | after |
|---|---|---|
| `scikit-learn` | 1.7.2 | **1.5.0** |
| `scipy` | 1.15.3 | **1.11.4** |
| `uv.lock` | 2026-07-31, no sklearn main dep | **rewritten**, `+ { name = "scikit-learn" }` ×2 |

Confirmed by mtimes: `scikit_learn-1.5.0.dist-info` 16:48:27, `sklearn/` 16:48:29,
`uv.lock` 16:48:26.

## What it was NOT

Not the launches. `launch_merge_reps.sh` and `launch_arm_reps.sh` both use
`uv run --no-sync`, and **zero** `Uninstalled`/`Installed` lines appear in any of
the nine arms' logs. Contrast the deletion arm's log (Aug 14), whose first two
lines are `Uninstalled 1 package` / `Installed 1 package` — that one *was* the
launcher, via the bare `uv run` at `launch_run.sh:63`.

## Root cause

Bare `uv run` invocations left in the repo's own documented tooling. `uv run`
re-syncs the project environment before executing. The most likely trigger is the
health-check one-liner, which was printed by `launch_run.sh`'s own footer and
documented in `CLAUDE.md` §3.4:

```bash
uv run python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
```

Anyone following the documented health-check procedure would silently re-sync the
venv under every running arm.

## Impact — assessed, not assumed

- **All 9 arms survived.** No process died.
- **Zero `sklearn`/`scipy`/`numpy` import failures** in any log
  (`grep -cE "(ImportError|ModuleNotFoundError).*(sklearn|scipy|numpy)"` → 0).
  The `ImportError` hits that do exist are in agent-written kernel code.
- **No L1 merge pass ran under the downgrade.** Last merges were 15:02 / 16:23 /
  16:10, all before 16:48. So no merge decision was made under 1.5.0 at the time
  of writing — but subsequent passes in these runs will be.
- `uv.lock`'s change is only +2 lines and is arguably *correct*: it records
  scikit-learn as a main dependency, which is what the pyproject edit in commit
  7a5a007 intended. This is CLAUDE.md open item #2 completing itself, at the
  worst possible moment.

## Open decision (needs a human)

The three merge arms **started** under scikit-learn 1.7.2 and will **finish**
under 1.5.0. DBSCAN is deterministic and the versions are unlikely to cluster
differently, but this is an unaudited intra-run change to the component that
defines the merge arm's independent variable.

- **Leave it**: 1.5.0 is the locked/intended version. CLAUDE.md open item #2
  already anticipated pinning 1.5.0 and calls for re-running the merge-threshold
  calibration afterwards.
- **Restore 1.7.2**: cleaner for these three runs, but means a *second* venv
  mutation under nine live arms — a new risk to fix an old one.

Deliberately **not** acted on: mutating the environment again while ~600 GPU-hours
are in flight is the larger hazard. Status quo is demonstrably working.

## Fix applied

Every bare `uv run` in the env tooling now uses `--no-sync`:

- `launch_run.sh` — the nohup launch site (line 63) **and** the footer health-check
- `resume_run.sh` — launch site and footer
- `launch_nvcc_series.sh` — launch site and health-check
- `CLAUDE.md` §3.4 — the documented health-check command

`launch_merge_reps.sh` / `launch_arm_reps.sh` were already correct, and both print
the pending `uv sync` diff during preflight so drift is visible before launch.

## Rule

**Never invoke `uv` without `--no-sync` while any run is in flight**, including for
read-only analysis. `uv run` is not read-only. Until `uv lock && uv sync` is done
deliberately with the box idle, `--no-sync` is load-bearing.
