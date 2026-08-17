# Cross-series synthesis — GPT-OSS-120B vs GPT-5.6-Terra, and OSS across two GPUs

Status: **2026-08-16**. Three completed series, native baselines throughout.
Rules: [ANALYSIS_RULES.md](../../ANALYSIS_RULES.md). Provenance:
[MANIFEST.md](MANIFEST.md).

| Series | Runs root | Native baseline | Complete designs |
|---|---|---|---:|
| GPT-OSS-120B @ A6000 | `runs_evolving/inference_oss_120b/` | `SONG_CPU6_A6000x4` | 8 |
| GPT-5.6-Terra @ A6000 | `runs_evolving/inference_gpt_56_terra/` | `SONG_CPU4_A6000x2` | 3 |
| GPT-OSS-120B @ GH200 | `runs_evolving/gpt-oss-120b/` | `NVIDIA_GH200x2` | 4 |

Speedup is `torch_baseline / kernel` **on the host that evaluated the run**. No
series is rescored onto another's baseline (ANALYSIS_RULES §2). Where a
cross-host statement is made below, §7 quantifies exactly how much of it is
hardware rather than model or design.

---

## 1. Decision summary

1. **No design wins every metric, in any series.** Correctness, the 1.0 bar, the
   2.0 bar, and geomean pick different rows in all three cells. Ranking on a
   single number is the main way to get this wrong.
2. **Truncation — the do-nothing control — is never beaten outright on the
   headline `fast_p_best@1.0`.** It wins on OSS/A6000 (0.72), ties on
   Terra (0.82) and ties on OSS/GH200 (0.46). Eight A6000 governance and
   context arms, three Terra arms, and four GH200 arms produced no design that
   clears it.
3. **Terra beats OSS on speed at every matched cell, but roughly half the gap
   is the host.** Terra's A6000/CPU4 torch baseline is easier than OSS's
   A6000/CPU6 baseline on **100% of 249 problems** (median 1.14×, Level 3
   1.31×). See §7.
4. **Markov is the clearest model-dependent design.** On OSS/A6000 it is the
   *worst* speed arm (@1 0.60) and the *only* perfect-correctness run (50/50).
   On Terra it *ties* the control at @1 and *wins* both @2 and geomean. The
   claim "component trends transfer across models" is **rejected**, not merely
   unproven.
5. **Hardware compresses the design signal to nothing.** On A6000 the OSS
   design spread at @1 is 0.12 (0.60→0.72). On GH200 it is 0.06, with a
   three-way tie at the top. Every GH200 geomean is **below 1.0** — the median
   retained kernel is slower than GH200 torch.

---

## 2. Best design per metric — GPT-OSS-120B @ A6000 (CPU6)

| design | I10 @0 | @1 | @2 | geomean (n) | I30 @0 | @1 | @2 | geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (control) | 0.880 | 0.520 | 0.140 | **1.1700** (44) | 0.980 | **0.720** | 0.240 | 1.3855 (49) |
| selective_retention | 0.920 | **0.540** | 0.080 | 1.0654 (46) | 0.960 | 0.700 | 0.180 | 1.2859 (48) |
| all-governance | 0.820 | 0.460 | **0.200** | 1.1531 (41) | 0.960 | 0.660 | **0.260** | **1.3966** (48) |
| truncation+deletion | 0.800 | 0.380 | 0.060 | 0.8606 (40) | 0.980 | 0.640 | 0.180 | 1.2518 (49) |
| truncation+merge@0.7 | 0.880 | 0.460 | 0.120 | 1.0173 (44) | 0.980 | 0.640 | 0.160 | 1.2387 (49) |
| truncation+refine | 0.840 | 0.520 | 0.100 | 1.1061 (42) | 0.940 | 0.620 | 0.140 | 1.2333 (47) |
| folding | **0.940** | 0.500 | 0.160 | 1.0169 (47) | 0.960 | 0.600 | 0.220 | 1.2243 (48) |
| markov_report | **0.960** | 0.460 | 0.080 | 0.8886 (48) | **1.000** | 0.600 | 0.120 | 1.0302 (50) |

| metric | iteration 10 | iteration 30 |
|---|---|---|
| `fast_p_best@0` (correctness) | **markov_report** 0.960 | **markov_report** 1.000 (50/50) |
| `fast_p_best@1` (headline) | **selective_retention** 0.540 | **truncation** 0.720 |
| `fast_p_best@2` | **all-governance** 0.200 | **all-governance** 0.260 |
| `speedup_best` geomean | **truncation** 1.1700 | **all-governance** 1.3966 (n=48 vs control n=49) |

**Reading it.** Truncation owns the 1.0 bulk. All-governance owns the tail — it
is the only arm above the control at both @2 and geomean, and with the corrected
`n` (48 vs 49) that geomean edge is a near-shared-sample result, not a subset
artifact. Markov buys perfect correctness by giving up speed: 50/50 correct with
the worst geomean in the series (1.03) and the worst @2 (0.12).

---

## 3. Best design per metric — GPT-5.6-Terra @ A6000 (CPU4)

| design | I10 @0 | @1 | @2 | geomean (n) | I30 @0 | @1 | @2 | geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (control) | 0.960 | 0.700 | 0.200 | **1.5023** (48) | 0.980 | **0.820** | 0.260 | 1.7796 (49) |
| markov_report | 0.940 | **0.740** | 0.200 | 1.4437 (47) | 0.980 | **0.820** | **0.300** | **1.8153** (49) |
| compress_trigger | **0.980** | 0.680 | 0.200 | 1.4046 (49) | 0.980 | 0.700 | **0.300** | 1.6438 (49) |

| metric | iteration 10 | iteration 30 |
|---|---|---|
| `fast_p_best@0` (correctness) | **compress_trigger** 0.980 | **three-way tie** 0.980 (all 49/50) |
| `fast_p_best@1` (headline) | **markov_report** 0.740 | **tie: truncation = markov_report** 0.820 |
| `fast_p_best@2` | **three-way tie** 0.200 | **tie: markov_report = compress_trigger** 0.300 |
| `speedup_best` geomean | **truncation** 1.5023 | **markov_report** 1.8153 (n=49, same 49 problems) |

**Reading it.** Markov is the best overall Terra design: it ties the control on
the headline metric, wins the 2.0 bar, and wins geomean **over the identical 49
problems** — a like-for-like +0.036. Compress-trigger (launched at
`compress_hot_rounds=3`, not the runbook's 15) is the cheapest cell in wall time
and L1 size and matches Markov at the 2.0 bar, but gives up 0.12 on the headline.
Only 3 of 4 Terra cells are usable; folding is 15/50 and excluded.

---

## 4. Best design per metric — GPT-OSS-120B @ GH200x2

| design | I10 @0 | @1 | @2 | geomean (n) | I30 @0 | @1 | @2 | geomean (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| truncation (control) | 0.840 | **0.380** | **0.120** | 0.7352 (42) | 0.940 | **0.460** | **0.180** | 0.8746 (47) |
| markov_report | **0.900** | 0.340 | 0.040 | 0.7373 (45) | **0.960** | **0.460** | 0.140 | **0.9830** (48) |
| selective_retention | **0.900** | 0.320 | 0.080 | **0.7657** (45) | **0.960** | **0.460** | 0.160 | 0.9541 (48) |
| compress_trigger | 0.840 | 0.240 | 0.080 | 0.5624 (42) | **0.960** | 0.400 | 0.140 | 0.7265 (48) |

| metric | iteration 10 | iteration 30 |
|---|---|---|
| `fast_p_best@0` (correctness) | **tie: markov = selective** 0.900 | **three-way tie** 0.960 (48/50) |
| `fast_p_best@1` (headline) | **truncation** 0.380 | **three-way tie** 0.460 |
| `fast_p_best@2` | **truncation** 0.120 | **truncation** 0.180 |
| `speedup_best` geomean | **selective_retention** 0.7657 | **markov_report** 0.9830 |

**Reading it.** The design signal has essentially collapsed. Three of four
designs tie at the headline metric and the fourth is one problem behind. Only
compress_trigger separates, and downward. Every geomean is < 1.0.

---

## 5. Model comparison — OSS-120B vs Terra, matched designs on A6000

| matched cell | metric | OSS (CPU6) | Terra (CPU4) |
|---|---|---:|---:|
| truncation | correct | 49/50 | 49/50 |
| | I30 @1 | 0.720 | **0.820** |
| | I30 @2 | 0.240 | **0.260** |
| | I30 geomean | 1.3855 (49) | **1.7796** (49) |
| | current@1 | 0.460 | **0.700** |
| | tokens | 129.2M | **114.9M** |
| markov_report | correct | **50/50** | 49/50 |
| | I30 @1 | 0.600 | **0.820** |
| | I30 @2 | 0.120 | **0.300** |
| | I30 geomean | 1.0302 (50) | **1.8153** (49) |
| | current@1 | 0.500 | **0.640** |
| | tokens | **51.7M** | 55.0M |

**What is consistent across the two models**

- Truncation is never beaten outright at `fast_p_best@1.0`.
- Extra L0 compression loses the 1.0 bar (OSS folding 0.60, Terra compress 0.70;
  both below their control).
- Iteration 10 does not predict iteration 30. The I10 @1 leader fails to hold on
  both models (OSS selective 0.54 → loses; Terra markov 0.74 → ties).
- Compressed context both **loses a fast specialized kernel and finds a big
  win** in the same run: OSS `level_1_problem_54` 2.96×→0.16× against
  `level_2_problem_51` 1.05×→4.86×; Terra `level_3_problem_24` 5.73×→0.96×
  against `level_3_problem_3` 1.22×→3.36×.

**What reverses between the two models**

| question | OSS-120B | Terra |
|---|---|---|
| Does truncation win @1 outright? | **Yes** (0.72; next 0.70) | **No** — ties markov at 0.82 |
| Is markov a speed win? | **No** — worst arm (0.60, gm 1.03) | **Yes** — wins gm and @2 |
| Does markov raise correctness? | **Yes** (50/50 vs 49/50) | **No** — tie at 49/50 |
| Does markov raise current@1? | **Yes** (0.50 vs 0.46) | **No** (0.64 vs 0.70) |

The same-trend generalization is **rejected** by the completed Markov cells.

**Caveat that materially shrinks the model gap:** the two A6000 hosts do not
present the same bar. See §7.

---

## 6. Hardware comparison — the same model, two GPUs

Matched designs, GPT-OSS-120B, governance off, 50×30, inference endpoint:

| design | metric | A6000/CPU6 | GH200x2 | change |
|---|---|---:|---:|---|
| truncation | correct | 49/50 | 47/50 | −2 |
| | I30 @1 | 0.720 | 0.460 | **−0.26** |
| | I30 @2 | 0.240 | 0.180 | −0.06 |
| | I30 geomean | 1.3855 | 0.8746 | **−0.51** |
| | current@1 | 0.460 | 0.280 | −0.18 |
| markov_report | correct | 50/50 | 48/50 | −2 |
| | I30 @1 | 0.600 | 0.460 | −0.14 |
| | I30 geomean | 1.0302 | 0.9830 | −0.05 |
| selective_retention | correct | 48/50 | 48/50 | 0 |
| | I30 @1 | 0.700 | 0.460 | **−0.24** |
| | I30 geomean | 1.2859 | 0.9541 | −0.33 |

Three things change together:

1. **Correctness barely moves** (0.96–1.00 → 0.94–0.96). The agent still writes
   kernels that produce the right answer.
2. **Speed collapses.** Every GH200 geomean is < 1.0; the A6000 control's 1.39
   becomes 0.87. `fast_p_best@1.0` falls by 0.14–0.26.
3. **The design ranking flattens and partly inverts.** On A6000 truncation leads
   @1 by 0.12 over markov; on GH200 they tie. On A6000 markov has the *worst*
   geomean; on GH200 it has the *best*.

---

## 7. Why the hardware differs — the baseline is the explanation

Speedup is relative to *that host's* torch. Comparing the three baseline
vectors over the 249 shared problems:

| comparison | geomean ratio | median | fraction of problems |
|---|---:|---:|---|
| A6000/CPU6 torch ÷ GH200 torch | **2.44×** | 2.65× | GH200 faster on **91%** |
| A6000/CPU4 torch ÷ GH200 torch | **2.82×** | 2.87× | GH200 faster on 93% |
| A6000/CPU6 torch ÷ A6000/CPU4 torch | **0.87×** | 0.88× | CPU6 faster on **100%** |

**GH200 (observation + hypothesis).** GH200's torch reference is ~2.4× faster
than A6000/CPU6's on the same problems. The agent must therefore beat a bar that
is 2.4× higher in absolute terms to score the same 1.0. That is sufficient on its
own to explain a geomean falling from 1.39 to 0.87 with correctness unchanged —
the kernels did not get worse, the reference got much better. *Hypothesis* for
the mechanism: on Hopper the cuBLAS/cuDNN paths torch dispatches to are
substantially better tuned than the hand-written CUDA the agent produces, and
HBM3e removes the memory-bandwidth headroom that naive fusion exploits on
Ampere. This also explains the flattened design ranking: with almost no headroom
above the reference, context management has little left to express.

**The two A6000 hosts are also not equivalent (observation).** CPU6's torch is
faster than CPU4's on **every single one of the 249 problems** — median 1.14×,
and 1.31× on Level 3. OSS ran on CPU6 (the harder bar); Terra ran on CPU4 (the
easier one). An identical kernel therefore scores a materially higher speedup in
the Terra series than in the OSS series.

**Consequence for §5, stated carefully.** ANALYSIS_RULES §2 forbids rescoring
one host's kernels onto another's vector, and this report does not do it. But
the direction and rough size of the bias are known and one-directional: it
inflates Terra relative to OSS. Terra's apparent geomean lead in the truncation
cell (1.78 vs 1.39, +28%) is therefore an **upper bound** on the model
difference; a baseline gap of ~14% accounts for roughly half of it. Terra
plausibly still leads after that, but **"Terra is ~28% faster than OSS" is not a
supportable claim** — the honest statement is "Terra leads on native relative
speed, by an amount the host confound inflates." The `fast_p_best` counts carry
the same bias, since a problem near the 1.0 bar is easier to clear on CPU4.

The correctness comparison is **not** affected: correctness is baseline-free, and
there the two models are tied at 49/50 on truncation, with OSS ahead 50/49 on
markov.

---

## 8. What survives all three series

- **Truncation is the right default.** Across 15 completed design cells on three
  hosts, nothing beats it on the headline metric; several arms cost 20–65% more
  wall time to finish behind it.
- **Pick the design from the metric you actually care about.** Correctness →
  markov (OSS) or anything (Terra, GH200: all tied). Median kernel speed →
  all-governance (OSS/A6000), markov (Terra, GH200). Tail performance →
  all-governance (OSS/A6000), markov/compress (Terra), truncation (GH200).
- **Aggressive compression is the one consistent loser.** Compress-trigger and
  folding are below their control at the 1.0 bar in every series they appear in.
- **Report the host with the number.** A speedup figure without its baseline is
  uninterpretable across these series — the same kernel scores ~2.4× differently
  on GH200 vs A6000 and ~1.14× differently between the two A6000 hosts.

---

## 9. Limitations

- **`n=1` per configuration.** Every number is descriptive of one run. No
  confidence intervals; small deltas (≤ 0.04 on fast-p ≈ 2 problems) are not
  separable from noise.
- Problems are coupled through sequential shared L1 within a run.
- All 8 OSS/A6000 runs and 2 of 3 Terra runs are resumes. Terra compress_trigger
  is the only fresh cell, confounding mode with resume.
- Terra compress_trigger ran `compress_hot_rounds=3`, not the runbook's 15.
- Terra has **no** governance cell, so the OSS governance findings
  (deletion/merge/refine) have no cross-model check at all.
- GH200 has no governance or folding cell (both partial), so its design coverage
  is 4 of the 8 OSS/A6000 arms.
- The hack detector and the error taxonomy in `analyze_feature_evidence.py` are
  heuristic. `metrics_best.is_hack` is a run-level latch and does **not** gate
  geomean eligibility (ANALYSIS_RULES §4).
- Wall time is operational (endpoint latency, contention, resumes), not a
  treatment effect. Tokens are endpoint-reported and may be lower bounds.
- `output/GH200x2/` and `runs_evolving/archived/with_NVCC_bug/` are invalidated
  and contribute nothing here.

## 10. Provenance

Per-series detail, reasons, and case studies:
[OSS/A6000](../gpt-oss-120b-inf-CPU6/EXPERIMENT_REPORT.md),
[Terra/A6000](../gpt-56-terra-inf-CPU4/EXPERIMENT_REPORT.md),
[OSS/GH200](../GH200x2_nvcc_fixed/EXPERIMENT_REPORT.md).

All numbers regenerated 2026-08-16 from run artifacts unpacked locally and
re-aggregated against native baselines. The iteration-10/30 tables reproduce
byte-for-byte from the raw `performance_stats.json` on every complete run.

**Correction applied in this pass.** `aggregate_runs.py:552,601` had been ANDing
`metrics_best.is_hack` into the geomean sample count. That field is the
run-level `run_had_hack` latch, and `generate_run_performance_stats.py` (module
docstring, line 369) forbids using it as an eligibility gate. Every `n` in the
2026-08-16 first-pass reports was understated by roughly `problems_with_hack`;
corrected values now equal `total_correct`. **No geomean, fast-p, or correctness
value changed.** The narrative claims that had rested on the bad `n` — chiefly
"Markov's geomean edge uses five fewer samples" (Terra) and "AllGov and T0 are
different selected subsets" (OSS) — were retracted and rewritten in the
per-series reports.
