# terra r4 + r5 -- HALTED 2026-08-30 ~15:56Z: LLM API BUDGET EXHAUSTED

Cause: the inference key's budget ran out. NOT an endpoint fault, NOT our code.

    coder_call_error: RateLimitError: Error code: 429 -
      {'error': {'message': 'Budget has been exceeded! Key=tianzheng-2ecb30f...

First seen 09:15:06Z, last 15:54:49Z. In the final 15 minutes **100% of iterations**
(7 of 7) were budget errors across ALL 24 arms, so every arm was dead in the water.
Stopped on the user's instruction; supervisor stopped first so it could not restart them.

is_rate_limit_error() already matches these (429 + 'budget'), so the client retried 6x
with backoff before surfacing them. Retrying cannot help an exhausted budget -- it only
slows the failure. NOTHING WILL RUN until the key is topped up or replaced.

## RESUME POINTS -- read the caveat, this is NOT the line count

A resume APPENDS replayed problems to batch_timing.jsonl, so the file can hold more
lines than the run has problems (r4_deletion: 61 lines, 50 distinct indices, last
index 25). The correct restart is **the LAST record's subset_index + 1**, since
entries are appended in execution order. Using `wc -l` would have restarted deletion
at 62 -- past the end of a 50-problem run. wave_supervisor.sh was fixed to match.

24 arms:

| arm | last idx done | distinct done | RESUME FROM | budget errs |
|---|---|---|---|---|
| r4_compress | 31 | 36/50 | **32** | 4 |
| r4_deletion | 25 | 50/50 | **26** | 4 |
| r4_folding | 28 | 28/50 | **29** | 3 |
| r4 | 30 | 33/50 | **31** | 4 |
| r4_l2_rep1 | 30 | 36/50 | **31** | 4 |
| r4_l2_rep2 | 29 | 50/50 | **30** | 4 |
| r4_l2_rep3 | 30 | 30/50 | **31** | 4 |
| r4_markov | 28 | 50/50 | **29** | 3 |
| r4_merge_sim08 | 27 | 49/50 | **28** | 4 |
| r4_merge_sim09 | 30 | 32/50 | **31** | 5 |
| r4_refinement | 27 | 50/50 | **28** | 5 |
| r4_selective_r5 | 32 | 32/50 | **33** | 3 |
| r5_merge_sim075_a | 7 | 7/50 | **8** | 4 |
| r5_merge_sim075_b | 6 | 6/50 | **7** | 4 |
| r5_merge_sim075_c | 10 | 10/50 | **11** | 4 |
| r5_merge_sim07_a | 10 | 10/50 | **11** | 4 |
| r5_merge_sim07_b | 10 | 10/50 | **11** | 4 |
| r5_merge_sim085_a | 7 | 7/50 | **8** | 3 |
| r5_merge_sim085_b | 7 | 7/50 | **8** | 4 |
| r5_merge_sim085_c | 10 | 10/50 | **11** | 4 |
| r5_merge_sim095_a | 6 | 6/50 | **7** | 4 |
| r5_merge_sim095_b | 6 | 6/50 | **7** | 4 |
| r5_merge_sim095_c | 10 | 10/50 | **11** | 4 |
| r5_merge_sim09_a | 10 | 10/50 | **11** | 4 |

## Restarting once budget is restored

1. VERIFY THE BUDGET FIRST with one cheap sequential coder call. If it returns
   429/budget, do NOT launch -- 24 arms burn through problems in minutes producing
   nothing, which is how the 2026-08-29 endpoint incident destroyed 214 problems.
2. resume_run.sh now exports the right eval env (RESERVE_GB=0 SLOTS=3 MEM_GATE=7
   HOIST=1) and takes governance flags after `--`; both were missing before.
3. wave_supervisor.sh recomputes restart points itself, so stale `start` values in
   wave_r4r5_supervise.tsv are harmless.

## Data status

Budget errors are CLEAN failures -- the iteration records an error and stops, it does
not silently produce a wrong answer. So no new corruption of the 2026-08-29 kind.
But a problem whose 30 iterations ran mostly-failing has a WEAK best-of-30, so
re-check per-problem failure rates for anything that ran between 09:15Z and 15:55Z
before scoring it.
