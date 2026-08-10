# Cross-series model, component, and endpoint report

Status: evidence frozen from the completed-run analyses listed in
[MANIFEST.md](MANIFEST.md), revised 2026-08-10. Cross-model speed and fast-p
use each series' **native** timing baseline. Speedup is already
`baseline_torch / kernel`, so it is a relative measure that absorbs host
baseline differences; Terra is **not** rescored onto CPU6.

## Decision summary

1. **Terra versus GPT-OSS-120B on the inference endpoint:** in the best-matched
   truncation/no-governance cell, Terra leads correctness (**49/50 versus
   48/50**) and best fast-p at 0.0 / 1.0 / 2.0 (**0.98/0.78/0.26 versus
   0.96/0.72/0.20**). OSS leads final-current retention at 0.0 / 1.0 / 2.0
   (**0.62/0.46/0.18 versus 0.38/0.26/0.04**). Terra uses far fewer reported
   tokens (**60,927,079 versus 126,271,726**). There is no metric-independent
   winner: Terra is stronger on best coverage and correctness; OSS is stronger
   on keeping a fast final kernel.
2. **Do both models show the same component-design trend?** **Inconclusive.**
   OSS inference truncation led its completed arms on `fast_p_best@1.0`
   (**0.72**). Terra truncation and Markov **tie** on best fast-p1 at **0.78**;
   truncation leads correctness (**49 versus 47**); Markov leads current
   fast-p1 (**0.50 versus 0.26**). OSS inference Markov is incomplete, so the
   same truncation-versus-Markov contrast does not exist on OSS.
3. **Old integrate versus new inference for OSS:** matched selective cells and
   the matched merge-only sim0.7 pair do not establish a systematic endpoint
   effect. Old integrate is descriptively ahead in the merge-only pair and in
   correctness for selective retention; new selective speed lies inside the two
   old selective outcomes. Endpoint 404 contamination, resumed old campaigns,
   and staggered code/launch dates prevent endpoint attribution.
4. **Old OSS Markov versus Terra Markov:** Terra leads best/current fast-p1
   (**0.78/0.50 versus 0.66/0.48**) and best fast-p2 (**0.26 versus 0.04**);
   old OSS leads correctness (**48 versus 47**). Terra uses fewer reported
   tokens (**47,399,320 versus 55,759,158**). Model and endpoint both change,
   so this pair isolates neither.

## 1. Scope and comparison rules

All included cells use the same 50-problem subset, 30 requested iterations, and
an RTX A6000 evaluation path. Each series keeps the baseline used when that
series was evaluated and aggregated:

- OSS inference and old integrate OSS:
  `results/timing/SONG_CPU6_A6000x4/baseline_time_torch.json`
- Terra inference:
  `results/timing/SONG_CPU4_A6000x2/baseline_time_torch.json`

Runs directly under `runs_evolving/` are classified as **old NVIDIA integrate
endpoint** runs by the user's authoritative directory-layout rule. Runs under
`runs_evolving/inference_oss_120b/` and
`runs_evolving/inference_gpt_56_terra/` are new inference-endpoint runs.

The main speed metric is `fast_p_best@p`: the fraction of all 50 problems whose
running best reaches threshold `p`. `fast_p_current@p` measures the last
observed submission. Correctness is reported separately. Best/current speedup
geomeans exclude incorrect and sticky-hack-flagged problems and therefore use
different selected `n`; they are secondary.

**Why native baselines are used for cross-model speed.** A speedup of `p`
means the kernel beat that host's torch reference by factor `p`. Fast-p counts
how often that relative bar is cleared. Recomputing Terra kernels against the
CPU6 torch vector would change the denominator without changing the kernels
and would distort the relative measure that already accounts for host
differences. Source run `visualizations/` and series aggregates were not
overwritten for a foreign baseline.

## 2. Inference endpoint: OSS versus Terra

### 2.1 Best-matched truncation cell (native baselines)

Both cells disable deletion, refinement, and merging and use truncation:

| Headline metric | OSS inference T0 (CPU6) | Terra inference truncation (CPU4) | Observed winner |
|---|---:|---:|---|
| correct | 48/50 | **49/50** | Terra, +1 problem |
| `fast_p_best@0.0` | 0.96 | **0.98** | Terra, +0.02 |
| `fast_p_best@1.0` | 0.72 | **0.78** | Terra, +0.06 |
| `fast_p_best@2.0` | 0.20 | **0.26** | Terra, +0.06 |
| `fast_p_current@0.0` | **0.62** | 0.38 | OSS, +0.24 |
| `fast_p_current@1.0` | **0.46** | 0.26 | OSS, +0.20 |
| `fast_p_current@2.0` | **0.18** | 0.04 | OSS, +0.14 |
| reported total tokens | 126,271,726 | **60,927,079** | Terra, 65,344,647 fewer |
| chat calls | 4,963 | **3,169** | Terra, 1,794 fewer |
| recorded wall time | 65.1474 h | **64.0659 h** | Terra, 1.0815 h lower; confounded |

Thus **Terra leads best coverage and correctness; OSS leads final-current
retention; Terra is much more token-efficient.** This must not be collapsed
into an unconditional winner.

The full best-fast-p profile is:

| threshold `p` | OSS T0 (CPU6) | Terra truncation (CPU4) | Winner |
|---:|---:|---:|---|
| 0.0 | 0.96 | **0.98** | Terra |
| 0.5 | 0.94 | 0.94 | tie |
| 0.8 | 0.84 | **0.92** | Terra |
| 1.0 | 0.72 | **0.78** | Terra |
| 1.5 | 0.28 | **0.48** | Terra |
| 2.0 | 0.20 | **0.26** | Terra |

### 2.2 Trajectory and operational profile

Iteration-aligned `fast_p_best@1.0` (native baselines):

| iteration | OSS T0 | Terra truncation |
|---:|---:|---:|
| 1 | 0.02 | **0.28** |
| 5 | 0.32 | **0.60** |
| 10 | 0.52 | **0.66** |
| 15 | 0.58 | **0.74** |
| 20 | 0.66 | **0.76** |
| 25 | 0.68 | **0.78** |
| 30 | 0.72 | **0.78** |

Observation: Terra stays ahead on best@1.0 at every aligned checkpoint and
plateaus at 0.78; OSS climbs later but does not catch Terra's best coverage.
Hypothesis, not finding: Terra may be more sample-efficient early while OSS
retains a stronger final-current kernel. One run per model cannot distinguish
model behavior from stochastic and sequential-memory divergence.

OSS T0 recorded 1,478 metric rows, 99 compilation errors, 295 output mismatches,
and 24 timeouts. Terra truncation recorded 1,025 rows, 10 compilation errors, 82
output mismatches, and 104 timeouts. These counts use different exposure totals
and endpoint histories; they are operational diagnostics, not normalized model
error rates. Terra's lower call/token burden coexists with fewer attempts.

## 3. Component-design trend

### 3.1 What each original report actually selected

- **OSS inference:** truncation T0 was the completed-arm
  `fast_p_best@1.0` leader at **0.72**. Deletion, refinement, and selective
  retention each reached 0.62; merge-only reached 0.60. T0 also led current
  fast-p1 at 0.46. Deletion tied T0 on correctness at 48/50, so even this report
  did not name one winner on every metric.
- **Terra inference (CPU4):** truncation and Markov tied on best fast-p1 at
  **0.78**; truncation led correctness 49/50 to 47/50; Markov led current
  fast-p1 0.50 to 0.26 and used fewer reported tokens. The report therefore
  identified metric-dependent leaders/ties, not one best component.

### 3.2 Terra Markov versus truncation (native CPU4)

| Metric | Terra truncation | Terra Markov | Observed winner |
|---|---:|---:|---|
| correct | **49/50** | 47/50 | truncation |
| best fast-p0 | **0.98** | 0.94 | truncation |
| best fast-p1 | 0.78 | 0.78 | tie |
| best fast-p2 | 0.26 | 0.26 | tie |
| current fast-p0 | 0.38 | **0.62** | Markov |
| current fast-p1 | 0.26 | **0.50** | Markov |
| current fast-p2 | 0.04 | **0.14** | Markov |
| reported tokens | 60,927,079 | **47,399,320** | Markov |
| calls | **3,169** | 5,349 | truncation |
| wall time | **64.0659 h** | 76.6912 h | truncation operationally; confounded |

Markov's speed profile is stronger for the current kernel, while truncation's
correctness and best@0 coverage are stronger; best@1 and best@2 are ties.

Trajectory at fast-p1:

| iteration | truncation | Markov | Leader |
|---:|---:|---:|---|
| 1 | 0.28 | **0.38** | Markov |
| 5 | 0.60 | **0.68** | Markov |
| 10 | 0.66 | **0.74** | Markov |
| 15 | 0.74 | **0.76** | Markov |
| 20 | 0.76 | **0.78** | Markov |
| 25 | 0.78 | 0.78 | tie |
| 30 | 0.78 | 0.78 | tie |

Direct observation: Markov leads early and preserves a much stronger current
state; both finish tied on best coverage, with truncation two problems ahead on
correctness. Hypotheses include report-based continuity and prompt compression
for Markov versus broader raw-history preservation for truncation; the data do
not establish those mechanisms causally.

### 3.3 Cross-model trend verdict

The same-trend generalization is **inconclusive**, because OSS inference Markov
is incomplete and the old integrate OSS series has no truncation control.
Available one-sided OSS findings are:

- selective retention: 0.62 best fast-p1 and 44/50 correct versus T0's
  0.72 and 48/50, with fewer standard calls but more reported tokens;
- deletion-only: unchanged 48/50 correctness but best fast-p1 down 0.10, while
  active L1 shrank 549→31;
- refinement-only and merge-only: no aggregate fast-p1 gain over T0 despite
  observed governance execution.

These are selective/governance observations within OSS, not evidence that OSS
responds to Markov as Terra does.

## 4. Old integrate versus new inference for OSS

Both sides of these matched cells use the CPU6 baseline (same model family /
same baseline file within the OSS series).

### 4.1 Selective retention

| Endpoint/run | best fast-p0 | best fast-p1 | best fast-p2 | correct | wall h |
|---|---:|---:|---:|---:|---:|
| old integrate selective campaign A | 0.98 | **0.66** | **0.16** | **49/50** | 57.6966 |
| old integrate selective campaign B | 0.98 | 0.52 | 0.14 | **49/50** | 59.5560 |
| new inference selective | 0.88 | 0.62 | 0.12 | 44/50 | 79.3914 |

Per metric: old A wins fast-p0, fast-p1, and fast-p2; both old runs win
correctness. New's fast-p1 **falls inside the old 0.52–0.66
range**, while its correctness is five problems lower. A deterministic new-run
case records model-group 404s and termination after five iterations; other new
workspaces also ended early on timeout. Both old selective campaigns were
resumed. Consequently this is not an endpoint effect estimate.

### 4.2 Merge-only sim0.7

| Metric | old integrate | new inference | Observed winner |
|---|---:|---:|---|
| correct | **49/50** | 47/50 | old |
| best fast-p0 | **0.98** | 0.94 | old |
| best fast-p1 | **0.68** | 0.60 | old |
| best fast-p2 | 0.12 | 0.12 | tie |
| current fast-p1 | **0.40** | 0.38 | old |
| wall time | **52.2123 h** | 84.6818 h | old operationally; confounded |

This pair shows a descriptive old advantage in correctness, best fast-p0/1,
current fast-p1, and wall time, with a tie at best fast-p2. It is one pair
across staggered code and service states, not a general endpoint effect.

### 4.3 Endpoint verdict

New OSS Markov and folding runs are incomplete; old OSS has no truncation
control; selective runs include 404 contamination and old resumptions. The
evidence therefore **does not establish a systematic old-integrate versus
new-inference difference**. It identifies pair-specific winners, not a stable
endpoint ordering.

## 5. Old OSS versus Terra in same-mode Markov

Native baselines (CPU6 for old OSS, CPU4 for Terra); model and endpoint both
differ:

| Metric | old integrate OSS Markov (CPU6) | inference Terra Markov (CPU4) | Observed winner |
|---|---:|---:|---|
| correct | **48/50** | 47/50 | OSS |
| best fast-p0 | **0.96** | 0.94 | OSS |
| best fast-p1 | 0.66 | **0.78** | Terra |
| best fast-p2 | 0.04 | **0.26** | Terra |
| current fast-p0 | **0.84** | 0.62 | OSS |
| current fast-p1 | 0.48 | **0.50** | Terra |
| current fast-p2 | 0.04 | **0.14** | Terra |
| reported tokens | 55,759,158 | **47,399,320** | Terra |
| calls | 6,369 | **5,349** | Terra |
| error rows / metric rows | 531/1,500 | 205/1,295 | Terra lower descriptively |
| wall time | **50.3341 h** | 76.6912 h | OSS operationally; confounded |

Terra has the stronger relative speed profile at the 1.0 and 2.0 thresholds;
OSS has a small correctness / best@0 / current@0 edge. Since **both model and
endpoint change**, this comparison cannot isolate either factor.

## 6. Validity threats and interpretation boundaries

### Direct observations

- Every quoted final result is from a completed 50-problem run; incomplete OSS
  Markov/folding and Terra folding results are excluded.
- Cross-model fast-p values come from each series' native aggregate
  (`gpt-oss-120b-inf-CPU6` and `gpt-56-terra-inf-CPU4`). No foreign-baseline
  rescoring directory is used.
- The old endpoint classification follows directory layout: qualifying direct
  children of `runs_evolving/` are old integrate; endpoint subfolders are
  inference.
- Fast-p uses the full 50-problem denominator. Best/current geomeans do not and
  are affected by the sticky best-hack latch.

### Limitations

1. **`n=1` per configuration.** No variance or causal treatment estimate exists.
2. **Sequential L1 coupling.** Problems within a run are not independent;
   earlier stochastic outcomes change memory used later.
3. **Sticky hack state.** One flagged iteration can exclude a later clean best
   from geomeans, while `fast_p_best` does not apply the same exclusion.
4. **Resumes and partial historical sessions.** Terra runs and old selective
   campaigns were resumed; cumulative wall time and retained L1 state cross
   sessions.
5. **Staggered code and launch dates.** Old and new campaigns did not run from a
   frozen simultaneous code/service state.
6. **Endpoint drift and failures.** New selective has direct 404 evidence;
   Terra has timeouts and a budget-exceeded boundary. Remote serving was not
   controlled.
7. **Operational accounting.** Tokens are endpoint-reported and can be lower
   bounds when usage fields are missing; calls and error rows reflect different
   attempt exposure; wall time includes contention and service latency.
8. **Host baseline identity.** Within a series, keep one baseline file. Across
   hosts, compare native relative speed/fast-p; do not recompute one host onto
   another's torch vector for cross-model ranking.
9. **Legacy metadata.** Old summaries omit model/endpoint fields. Endpoint is
   assigned by the authoritative directory-layout rule; model identity is
   supplied by the campaign scope/runbooks, not those null summary fields.

### Hypotheses, not findings

- Terra's early threshold lead and lower token use may reflect higher initial
  search efficiency.
- Markov reports may compress prompt state and preserve current-kernel quality,
  while also anchoring search or adding report-call overhead.
- OSS truncation may benefit from retaining broader late-run search context,
  which shows up more in current retention than in best coverage.
- Endpoint service quality may explain part of new selective's correctness
  deficit.

Each requires repeated, simultaneous, serialized runs with fixed code, endpoint,
native baseline, seeds where possible, and fresh L1 state.

## 7. Exclusions

- `output/GH200x2/` is an invalidated **inference** series with a missing CUDA
  toolchain and fallback contamination. It is not old integrate evidence and no
  kernel-quality number from it is used.
- Partial OSS inference Markov, folding, and combined-governance runs and partial
  Terra folding are excluded.
- Old integrate folding and unmatched governance combinations are not used to
  answer endpoint effects.
- Any prior Terra→CPU6 rescoring under `common-baseline/` is discarded and must
  not be reused for cross-model claims.
