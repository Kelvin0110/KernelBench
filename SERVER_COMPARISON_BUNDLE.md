# Cross-server throughput comparison — reference data from the fast server

Hand this to the slow server's agent. Every number below is measured, not estimated.
Run the **paired commands** on your side and diff.

- **This (fast) server:** 9 arms — 5 on GPU 0, 4 on GPU 1. 2× GH200 144G HBM3e.
- **Your (slow) server:** 5 arms on GPU 0, same GH200, same repo. ETA 7 days.
- **Gap:** 62.6 vs ~201.6 min/problem ⇒ **~3.2× slower**, with *fewer* arms per GPU.

Snapshot taken 2026-08-23T07:37Z, 11.2 h into a 50-problem × 30-iteration run.

---

## 0. The single most useful number

One full agent iteration, measured eval-to-eval:

| | fast server | yours |
|---|---|---|
| median iteration wall time | **114 s** | ~403 s (implied by 7 d ETA) |
| min/problem (median, n=86) | **62.6 min** | ~201.6 min |
| evals/hour/arm | **26.9** | ~8.9 |

```bash
# paired command — median seconds between consecutive evals in one arm
python3 - <<'PY'
import json,glob,statistics
from datetime import datetime
g=[]
for f in glob.glob('runs_evolving/*/*/workspaces/*/evaluation_terminal_output.jsonl'):
    ts=sorted(datetime.fromisoformat(json.loads(l)['wall_time_utc'].replace('Z','+00:00'))
              for l in open(f) if l.strip())
    g += [(ts[i]-ts[i-1]).total_seconds() for i in range(1,len(ts)) if 0<(ts[i]-ts[i-1]).total_seconds()<1800]
g.sort(); print('iteration wall median', statistics.median(g), 'n', len(g))
PY
```

Whatever accounts for your extra ~290 s per iteration is the whole story. The
hypotheses below are ordered by how much of that they could plausibly explain.

---

## 1. Code provenance — check these four commits FIRST

This is the highest-yield check. Three of the four directly govern throughput.

```bash
cd Self-Evolving-Agent
for c in 4a52c04 7ac0e87 63bfc2b a2b8749; do
  git merge-base --is-ancestor $c HEAD 2>/dev/null && echo "YES $c" || echo "NO  $c"
done
git rev-parse --short HEAD
```

Fast server: **all four present**, submodule at `63bfc2b`, main at `094652c`.

| commit | what it does | cost if MISSING |
|---|---|---|
| `4a52c04` | moves the GPU lock into `kernelbench/eval.py`, narrowing it to correctness+timing | **Wide lock: ~45 s held per eval instead of ~4.9 s.** Utilisation ~92% at 3 arms ⇒ hard ceiling **~3.9 arms/GPU**. Your 5 arms would spend most of their life queued. *This alone can explain 3×.* |
| `7ac0e87` | parent eval deadline excludes GPU-lock wait | Evals SIGTERM'd mid-wait at 600 s, recorded as `compiled=False`. The governor then "debugs" a kernel that was never broken — wasted iterations, and 3.9% of evals lost at 6 arms. |
| `63bfc2b` | `num_perf_trials` 100 → 25 | 4× longer *timed* window ⇒ 4× longer lock hold ⇒ far worse queueing at 5 arms. |
| `a2b8749` | median-based baseline | Correctness of the speedup metric, not speed. Check anyway. |

---

## 2. What our GPU lock actually covers

`src/kernelbench/eval.py` — acquired line **684**, released line **816**. Inside:

- `run_and_check_correctness(...)` — the correctness trials
- the **candidate** timing window (`time_execution_with_cuda_event`)
- the **reference** timing window
- the `torch.cuda.synchronize()` calls bracketing them

**Explicitly OUTSIDE the lock** (this is the point of `4a52c04`):

- `exec()` of the candidate source
- reference-model construction and input generation
- **nvcc / ninja compilation** ← the bulk of eval wall time
- the `_precompile_candidate()` pre-build in `eval_runner.py`

Measured hold ≈ 4.9 s (max 9.3 s). If your lock wraps the ninja build, hold is
~45 s and five arms cannot fit.

```bash
grep -n "_gpu_timing_lock\|_gpu_phase" src/kernelbench/eval.py
# fast server: 449 (def), 684 (__enter__), 816 (__exit__)
# If you instead find the lock in kernelbench_integration/eval_runner.py
# wrapping the whole eval_kernel_against_ref call, you have the WIDE lock.
```

---

## 3. Lock wait — read the RIGHT file

**The arm `.log` never contains lock messages.** `eval_runner.run_kernelbench_eval`
wraps the eval in `redirect_stdout`, so the child's `[gpu-eval-lock]` prints are
captured into `terminal_output` inside the jsonl. Grepping the log reports clean
no matter what happens — CLAUDE.md §3.4's documented command has this defect.

```bash
# WRONG (always empty):  grep "gpu-eval-lock" <arm>.log
# RIGHT:
grep -o "acquired after [0-9.]*s" <run>/workspaces/*/evaluation_terminal_output.jsonl | tail
```

Fast server, 2711 evals:

```
waits >=5s : 737 (27% of evals)     UNLOCKED: 0     orphaned: 0
p50 19.8s   p90 69.0s   max 685.1s   cumulative queued 11.0 h
```

Note `gpu_lock` only logs waits **≥5 s** (`_SLOW_WAIT_LOG_SEC`), so short waits
are invisible on both sides — the comparison is still apples-to-apples.

**If your p50 is in the hundreds of seconds, the wide lock (§1) is your answer.**

Caveat when comparing: lock wait is dominated by *which problem* the arms are on.
Subset indices 2–5 (`22_Tanh`, `26_GELU_`, `33_BatchNorm`, `34_InstanceNorm`) are
~1.6 B-element tensors with a stall tail up to 183× the median; nine arms in
lockstep there drove our p90 to 570 s, and it fell to **zero** two hours later on
the conv problems. Compare like-for-like subset indices or you will measure
problem weight, not configuration.

---

## 4. Environment settings

```bash
# read from a LIVE arm, not from your shell
p=$(pgrep -f evolve_kb_batch.py | head -1)
tr '\0' '\n' < /proc/$p/environ | grep -E "KB_GPU_RESERVE_GB|KB_GPU_EVAL_LOCK|CUDA_VISIBLE_DEVICES|NVIDIA_ENDPOINT"
```

Fast server:

```
KB_GPU_RESERVE_GB=0
KB_GPU_EVAL_LOCK_TIMEOUT_SEC=5400
KB_GPU_EVAL_LOCK=<unset → default on>
CUDA_VISIBLE_DEVICES=0
```

`KB_GPU_RESERVE_GB=0` is **required** when sharing a GPU. The default is 42 GB
*per arm*, pinned while the arm waits on the LLM. Five arms × 42 GB = 210 GB on a
144 GB card ⇒ reservers fight for headroom and can OOM whichever arm is mid-eval.

Fast discriminator:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

On the fast server this returns **empty** — arms hold no persistent GPU context
between evals. If yours lists 5 long-lived processes each holding tens of GB,
`KB_GPU_RESERVE_GB` is not 0 and that is a real problem.

---

## 5. LLM latency — the agent is LLM-bound, not GPU-bound

An eval subprocess lives ~38–45 s but touches the GPU for well under a second.
Most of the iteration is LLM round-trips. CLAUDE.md §7 documents a **26% swing in
min/problem from inference-endpoint latency alone**, with nothing in the repo
changing.

```bash
python3 - <<'PY'
import json,glob,statistics,collections
from datetime import datetime
per=collections.defaultdict(list)
for f in glob.glob('runs_evolving/*/*/workspaces/*/chat_history.jsonl'):
    rows=sorted((json.loads(l)['wall_time_utc'], json.loads(l).get('phase'))
                for l in open(f) if l.strip() and json.loads(l).get('wall_time_utc'))
    for i in range(1,len(rows)):
        dt=(datetime.fromisoformat(rows[i][0].replace('Z','+00:00'))
           -datetime.fromisoformat(rows[i-1][0].replace('Z','+00:00'))).total_seconds()
        if 0<dt<900: per[rows[i][1]].append(dt)
for ph,v in sorted(per.items(), key=lambda x:-len(x[1])):
    print(f'{str(ph):22} n={len(v):5} median {statistics.median(v):6.1f}s p90 {sorted(v)[int(len(v)*.9)]:6.1f}s')
PY
```

Fast server (9 arms all hitting the same endpoint concurrently):

```
coder                  n= 2715  median  19.3s  p90   36.6s
extractor              n= 2635  median   5.1s  p90    9.8s
action_selector        n= 2620  median  14.5s  p90  109.7s
summarizer             n=  913  median  62.8s  p90  129.8s
evolving_report        n=  302  median  82.2s  p90  134.4s
l0_round_summarizer    n=  290  median  69.1s  p90  128.1s
skill_diagnosis        n=  238  median  67.9s  p90  130.3s
preflight              n=  203  median   5.8s  p90    9.6s
milestone_judge        n=  179  median  79.2s  p90  134.4s
ALL inter-call gaps    n=10107  median  14.6s
```

If your `coder` median is 60 s+ against our 19.3 s, endpoint latency is your
bottleneck and no amount of GPU tuning will help. Confirm the endpoint and model:

```bash
grep -E "^NVIDIA_ENDPOINT|^NVIDIA_.*_MODEL" .env
tr '\0' ' ' < /proc/$(pgrep -f evolve_kb_batch.py | head -1)/cmdline | tr ' ' '\n' | grep -A1 -E "nvidia-endpoint|--model"
```

Fast server: `--nvidia-endpoint inference --model gpt-oss-120b`, `NVIDIA_ENDPOINT=inference`.
Note the `integrate` and `inference` endpoints use **different model IDs** and have
very different latency; a slow arm on `integrate` is a known trap.

---

## 6. CPU — nvcc/ninja is CPU work and runs unlocked

Because compilation is outside the lock, N arms compile *concurrently*. That is
free only if you have the cores.

```bash
echo "cores: $(nproc)  load: $(cut -d' ' -f1-3 /proc/loadavg)  MAX_JOBS=${MAX_JOBS:-unset}"
```

Fast server: **144 cores, load 9.03** — CPU is nowhere near saturated, which is
why 9 concurrent arms cost nothing.

If your load average is at or above your core count, your arms are serialising on
compilation and 5 arms is simply too many for the box. `MAX_JOBS` is unset on our
side (torch defaults to a per-build job count); if yours is set high, each of the
5 arms may be spawning many compiler processes and thrashing.

---

## 7. Sanity checks that rule out silent corruption

```bash
# must be 0 — nonzero means nvcc is missing and kernels silently fall back to
# plain PyTorch while still scoring correct=True
grep -c CUDA_HOME <arm>.log

# must be 0 — an eval that gave up waiting and ran contended (deflated speedup)
grep -c "proceeding UNLOCKED" <run>/workspaces/*/evaluation_terminal_output.jsonl

# timeout rate; ours is 14/1917 = 0.73%
grep -c "evaluation timeout after" <run>/workspaces/*/evaluation_terminal_output.jsonl
```

Fast server: `CUDA_HOME` 0, `UNLOCKED` 0, orphaned waits 0, timeout rate 0.73%
(reference: pre-`7ac0e87`, 6 arms/GPU produced **3.9%** and 3 arms **0.93%**).

A high timeout rate on your side, especially *without* the
`(excluding Ns GPU-lock wait)` suffix, means `7ac0e87` is missing and queue time
is being charged against the 600 s work budget.

---

## Ranked hypotheses

1. **Wide lock** (`4a52c04` missing) — hold 45 s vs 4.9 s, ceiling ~3.9 arms. Alone sufficient to explain 3.2×. *Check §1 and §2 first.*
2. **LLM endpoint latency / wrong endpoint** — agent is LLM-bound; a 3× coder latency is a 3× run. *Check §5.*
3. **`num_perf_trials=100`** (`63bfc2b` missing) — 4× the timed window, compounds #1.
4. **`KB_GPU_RESERVE_GB` unset** — 5 × 42 GB on a 144 GB card. *Check §4, one command.*
5. **Missing `7ac0e87`** — evals killed mid-wait become fake compile failures the governor wastes iterations "fixing".
6. **CPU saturation** — unlocked concurrent nvcc with too few cores. *Check §6.*

Cheapest-first order: §4 (one command) → §1 (four commands) → §6 (one command) →
§5 → §3. The first four take under a minute combined.
