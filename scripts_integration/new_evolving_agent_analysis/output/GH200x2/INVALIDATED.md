# INVALIDATION NOTICE — the four L0 context-management runs

**Status: the kernel-quality results in this directory are void.**
Written 2026-08-07. Applies to every artifact here dated 2026-08-05 or earlier
(`comparison.md`, `aggregate_runs.json`, `aggregate_runs.csv`) and to the
corresponding sections of `../../EXPERIMENT_REPORT.md`.

The four runs have been moved to `runs_evolving/archived/with_NVCC_bug/`.

| arm | archived run dir |
|-----|------------------|
| truncation (base) | `base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51` |
| markov_report | `base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_03_04_52` |
| selective_retention | `base_agent_gpt_oss_120b_selective_itr30_GH200_2026_08_04_17_24` |
| folding | `base_agent_gpt_oss_120b_folding_itr30_GH200_2026_08_04_17_26` |

---

## Defect 1 — no CUDA toolchain (affects all four arms)

The host had no `nvcc` and no `CUDA_HOME`, so `load_inline(cuda_sources=...)`
could never build. See [`../../../new_evolving_agent/env/README.md`](../../../new_evolving_agent/env/README.md)
for the full postmortem and the fix.

| arm | CUDA_HOME errors | share of iterations | problems touched |
|-----|-----------------:|--------------------:|-----------------:|
| truncation | 179 | 11.9% | 44 / 50 |
| markov_report | 144 | 9.6% | 47 / 50 |
| selective_retention | 116 | 8.7% | 40 / 50 |
| folding | 202 | 14.3% | 43 / 50 |

The failure count understates the damage, because the agent adapted to it. It
learned to guard the build behind a condition that is false in this environment
and fall through to the reference PyTorch implementation:

```python
if torch.cuda.is_available() and os.getenv("CUDA_HOME"):
    ext = load_inline(..., cuda_sources=...)   # never taken
else:
    ext = None
...
return torch.clamp(1.0 - pred * targ, min=0.0).mean()   # the reference impl
```

A `__global__` and a `load_inline` call are both present in the source, so the
static checker passes. Such kernels score `compiled=True, correct=True,
speedup≈1.0` — **PyTorch benchmarked against itself.**

The idiom then propagated into the per-run L1 skill memory:

| arm | L1 entries | env-workaround entries | share |
|-----|-----------:|-----------------------:|------:|
| truncation | 585 | 407 | 70% |
| markov_report | 426 | 136 | 32% |
| selective_retention | 420 | 299 | 71% |
| folding | 592 | 431 | 73% |

Verbatim entry titles include *"Guard CUDA Kernel Compilation with Pre-Import
CUDA_HOME Check & CPU Fallback"*. The agent was not gaming the grader; it learned
the dead-code pattern as **defensive engineering**, which is precisely why the
static checker never fired on it.

## Defect 2 — LLM endpoint failures (affects selective and folding only)

Independent of the toolchain problem, two arms lost work to API instability:

| arm | kernels | iterations | truncated problems | HTTP 404 | conn/timeout |
|-----|--------:|-----------:|-------------------:|---------:|-------------:|
| truncation | 50/50 | 1500/1500 | 0 | 1 | 13 |
| markov_report | 50/50 | 1500/1500 | 0 | 0 | 5 |
| **selective_retention** | 46/50 | 1338/1500 | **8** | 41 | 181 |
| **folding** | 49/50 | 1411/1500 | **5** | 15 | 66 |

`selective_retention` and `folding` are additionally non-comparable to the other
two arms on sample size alone.

---

## What survives

**Void — do not cite:** every kernel-quality number. `total_correct`
(49/48/46/45), all speedup means/medians/geomeans, `best_speedup_overall`,
`fast_p_*`, and any claim of the form "context policy X produces faster kernels."
These measured PyTorch fallback paths, not generated CUDA kernels.

**Void for a separate reason:** `metrics_best.is_hack`. It is assigned
`run_had_hack`, a sticky accumulator that latches on the first hack-flagged
iteration and never clears, so it means "some iteration in this problem ever
tripped the detector," not "the best kernel is a hack." One hack anywhere in 30
iterations permanently removes a problem from the best-speedup aggregate — which
is why `best_n` is 3 of 50 for truncation. 42 of those 47 excluded problems had a
clean final iteration. Use the *per-iteration* hack rate instead; see
`../../EXPERIMENT_REPORT.md` §2.2.

**Weakly survives:** the relative ordering of how heavily each context policy
leaned on the fallback idiom (markov highest at 14.6% of correct finals landing
in [0.95, 1.05], truncation lowest). Treat as a hypothesis, not a result — it was
not the intended measurement, and the two damaged arms confound any ranking.

**Never tested at all:** skill governance. `skill_deletion`, `skill_merging`,
`enable_skill_refinement`, and `enable_l1_skill_unit_test_gc` were `False` in all
four runs.

## Replacement

The series is being re-run on the repaired toolchain, starting with `truncation`
and `markov_report` at 50 problems x 30 iterations. Health of an in-flight run
can be checked against its archived counterpart without waiting for completion:

```bash
uv run python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
```

The gating signal is `cuda_home_err`, which must be 0. Expect `correct` counts
and speedups to **drop** relative to the archived numbers: kernels that
previously "passed" by falling back to reference PyTorch now have to genuinely
compile.
