# Aug-22 median-baseline wave — performance report

Generated 2026-08-27T01:46:11+00:00 on `NVIDIA_GH200x2` (host `lego-c2g2-smc-034`). Baseline: `results/timing/NVIDIA_GH200x2_median/baseline_time_torch.json` (median-bearing, resolved per-run from `run_summary.json` `hardware_server`).

**Scope.** 9 `gpt-oss-120b` arms + 6 `gpt-5.6-terra` arms, all 50/50 problems x 30 iterations, `runs_evolving/{gpt-oss-120b,gpt-5.6-terra}/median/`.

```
.venv/bin/python Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py \
  --all-runs --runs-root runs_evolving/<model>/median --hardware NVIDIA_GH200x2_median
```

Output: `<run_dir>/visualizations/performance_stats.json` (15 files).

> **Cohort warning — this report covers the 15 arms stamped `2026_08_22` only.** Three further terra
> cells (`folding`, `merge_sim08`, `refinement`) were launched at `2026_08_27_01_2x` while this analysis
> was running and are at problem 1-2. They must be excluded: `rescore_hack_threshold.py --all-dirs`
> computes a **within-model aligned intersection**, so including them collapses terra's aligned set from
> **50 problems to 1** and reports every terra arm at geomean ~8.5 (that is just L1P100). Same family as
> the documented "killed arms leave live-looking directories" trap — glob results must be intersected
> with the intended cohort, not merely with what is on disk. Re-run the rescore only against the
> completed cohort.


Metrics are **best-so-far** (running non-hack best): `correctness` = `fast_p_best@0.0` = fraction of problems with a clean correct best; `fast@1`/`fast@2` = `fast_p_best@{1.0,2.0}`; `geo` = geometric mean of best speedup over correct problems. Per `ANALYSIS_RULES.md`, `fast_p_best@1.0` is the headline and `best_geomean` is secondary.

---

## 1. Headline tables (as generated — stored `is_hack` labels, n=50)

### gpt-oss-120b

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 |
|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.820 | 0.480 | 0.080 | 1.036 | 0.880 | 0.640 | 0.140 | 1.297 |
| folding | 0.800 | 0.420 | 0.100 | 1.106 | 0.920 | 0.660 | 0.180 | 1.443 |
| markov | 0.820 | 0.540 | 0.080 | 1.072 | 0.960 | 0.700 | 0.100 | 1.189 |
| selective_r5 | 0.920 | 0.460 | 0.140 | 1.160 | 1.000 | 0.660 | 0.200 | 1.402 |
| compress | 0.900 | 0.620 | 0.080 | 1.119 | 1.000 | 0.780 | 0.160 | 1.336 |
| deletion | 0.900 | 0.460 | 0.020 | 0.982 | 0.980 | 0.680 | 0.160 | 1.379 |
| merge_sim08 | 0.780 | 0.440 | 0.040 | 0.940 | 0.940 | 0.640 | 0.140 | 1.309 |
| refinement | 0.840 | 0.420 | 0.140 | 1.275 | 0.980 | 0.680 | 0.220 | 1.525 |
| l2 | 0.900 | 0.460 | 0.060 | 0.949 | 0.960 | 0.720 | 0.220 | 1.471 |

### gpt-5.6-terra

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 |
|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.960 | 0.760 | 0.220 | 1.666 | 1.000 | 0.860 | 0.480 | 2.417 |
| markov | 0.980 | 0.740 | 0.200 | 1.503 | 0.980 | 0.840 | 0.300 | 1.758 |
| selective_r5 | 0.960 | 0.780 | 0.280 | 1.664 | 0.980 | 0.820 | 0.320 | 2.049 |
| compress | 0.940 | 0.800 | 0.440 | 2.197 | 0.940 | 0.840 | 0.600 | 2.901 |
| deletion | 0.980 | 0.840 | 0.280 | 1.803 | 1.000 | 0.980 | 0.460 | 2.479 |
| l2 | 0.960 | 0.700 | 0.240 | 1.497 | 1.000 | 0.800 | 0.360 | 1.966 |

**Validation.** `fast_p_best@0.0` reproduces `run_summary.json` correctness exactly on 12 of 15 arms. The 3 that differ do so by exactly one problem (0.020) and are precisely the arms with a contaminated-cell replay in flight — their `run_summary.json` is stale and still counts the reference-corruption hack as correct.

**Caveat — three gpt-oss cells were mid-replay when this was generated.** The contaminated-cell reruns (truncation/L1P56, deletion/L1P55, merge_sim08/L1P50) had 3, 1 and 1 of 30 iterations and no clean correct sample. Those three arms are effectively scored on 49 real problems against a denominator of 50, so their `corr`/`fast@1`/`fast@2` are **floors** and can each gain up to 0.020. Regenerate when the replays finish.

---

## 2. The `is_hack` threshold seam, and whether the affected kernels are legitimate

`src/kernelbench/eval.py` changed `excessive_speedup_threshold` **10x -> 30x** at **2026-08-24T15:11:45Z** (commit `588a6a5`). `eval.py` is re-imported by every eval spawn, so the new rule reached all live arms with no restart and each run's `is_hack` column is a **mixture of two rules**. Re-scoring uniformly at 30x flips **268 of 22,517** eval labels, all in one direction (hack@10x -> clean@30x).

### 2.1 The band is four problems, and that is the whole story

| problem | flips | reference model | collapses because |
|---|---|---|---|
| L2P42 | 130 | ConvTranspose2d -> global avg pool | pooling is linear in the conv output |
| L2P13 | 111 | ConvTranspose3d -> mean over depth | depth mean collapses to a 2D conv |
| L2P51 | 24 | Gemm 8192x8192 -> mean over features | `mean_j(W_j.x+b_j-s_j) = (mean_j W_j).x + const` |
| L2P56 | 3 | Linear 32768 -> sigmoid -> sum | **does not collapse** (sigmoid is nonlinear) |

**248 of 268 flips are terra, 20 are gpt-oss.** That asymmetry — not any difference in kernel quality — is why terra's geomean moves 8-18% under re-scoring and gpt-oss's barely moves. Every one of these reference models computes something enormous and then discards most of it, so an exact algebraic shortcut is worth 10-30x. `logsumexp` over a size-1 dim (L2P51) is literally the identity.

### 2.2 Audit: are they legitimate?

**Static — 0 of 268 fail.** Ran `kernel_static_checker.validate_kernel_static(backend='cuda', precision='fp32')` over every flipped kernel, including `global_module_patch` — the reference-corruption check added in `ede1898`, which did **not** exist when these were evaluated. Zero STRICT errors. The only hits are `torch_computation_ops` (128) and `pytorch_wrap` (111), and per `static_check.py:44-63` warnings never set `is_hack`; only STRICT errors and `metadata.excessive_speedup` do.

**Numerical — independently re-run** (params synced from the reference, `atol=rtol=1e-4`, the same tolerance eval uses):

| kernel | correct @1e-4 | max abs diff | measured speedup | recorded |
|---|---|---|---|---|
| terra compress L2P13 it12 | yes | 2.17e-05 | 26.5x | 30.0x |
| terra compress L2P42 it12 | yes | 2.29e-05 | 22.7x | 22.9x |

Live reference timings (8.412 ms, 6.018 ms) land on the fixed baseline (8.4, 6.02), so the baseline is sound for these problems. L2P51's numerical check was queued behind newly-launched arms and abandoned; its algebra is exact by inspection but is **not** empirically confirmed here.

**Code.** All 21 metric-moving samples contain real `__global__` CUDA via `load_inline`. L2P42's is pure CUDA with no PyTorch ops. L2P13's delegates the lowered 2D conv to `F.conv2d` (hence `pytorch_wrap`) — a purity question, not a correctness one.

**Verdict: 264 of 268 are legitimate, exact, fp32 custom CUDA.**

### 2.3 The exceptions — a hack class the checker misses

Scanning all 268 for reduced-precision signatures: **264 are fp32 throughout, 3 use FP8+FP16 (all L2P56), 1 uses FP16 (L2P51)**. Two of them are metric-moving:

| sample | speedup | what it does |
|---|---|---|
| terra truncation L2P56 it28 | 15.6x | casts **both operands to FP8 E4M3**, runs the GEMM through `torch._scaled_mm` (a PyTorch op) with `out_dtype=float16, use_fast_accum=True`, polynomial sigmoid |
| oss markov L2P51 it9 | 10.8x | consumes the GEMM output in **FP16** (`__half2` loads) |

The L2P56 kernel passes correctness at 1e-4 only because the sigmoid **saturates** — the sum of 32,768 saturated sigmoids is insensitive to FP8 GEMM error. `check_precision_downgrade` misses it: `FP32_TO_FP16_PATTERNS` matches *conversion idioms* (`__float2half(`, `(__half)x`, `static_cast<__half>(`, `.half()`) but not `__half*` pointer declarations, and **contains no FP8 pattern at all** (`float8_e4m3fn`, `_scaled_mm`, `__nv_fp8`). `check_torch_computation_ops` does not list `torch._scaled_mm` either.

So re-scoring at 30x **admits two genuine hacks** that the 10x rule caught by accident. This is worth a checker PR independent of anything in this wave.

### 2.4 Why 30x is not the right line either

Above 30x on L2P51 there are kernels at **~150x** (runtime 0.0368 ms) that are the *same* exact lowering, differing only in caching the weight statistics under proper version invalidation (`data_ptr` + `_version` + device + dtype). That kernel passes the static checker with **zero** warnings. Traffic check: x (67 MB) + out (67 MB) = 134 MB in 36.8 us = **~3.6 TB/s**, ordinary bandwidth-bound behaviour on GH200 HBM3e. So `150x` is *physically achievable*, not prima facie impossible — the honest reason to exclude it is not its magnitude.

The threshold therefore discards real work above the line and admits real hacks below it. It is a magnitude proxy standing in for a correctness test.

---

## 3. Does this class belong in KernelBench at all?

**The prompt explicitly permits it.** `src/kernelbench/prompts/prompts.toml:13`:

> You may replace multiple operators with custom implementations, consider operator fusion opportunities ..., **or algorithmic changes (such as online softmax). You are only limited by your imagination.**

So an exact algebraic lowering is within the letter of the task, and the README frames the task as transpiling operators *"at whatever level of granularity they desire"*.

**But the resulting number does not measure what the benchmark is for.** `EVAL.md` warns:

> be always paranoid about suspiciously good results — kernel engineers and existing compilers are already pretty good, so **a >2x speedup for anything is highly unlikely**

There is a real distinction inside "algorithmic changes":

- **Online softmax** restructures *how* the same work is done to cut memory traffic. The work still happens.
- **Mean-of-GEMM -> matvec** proves most of the reference's work is never observed in the output and **deletes** it.

Both are exact. Only the first is kernel engineering. On L2P13/42/51 the measured speedup is dominated by *how wasteful the reference model is*, not by how well the kernel was written — so it answers a different question from the rest of the benchmark, and because it is 10-30x it dominates any geometric mean it enters.

**Consistency point.** Excluding only the >30x samples is the magnitude heuristic again with a different constant: the 150x and the 22-30x kernels on L2P51 are *the same trick*, differing only in whether weight prep is cached. A defensible cut removes the **problems**, not a speedup band.

---

## 4. Scoring views

Three ways to score the same artifacts. **View C is the recommended headline** for claims about kernel design quality; view A is what the visualization pipeline currently produces.

| view | rule | effect |
|---|---|---|
| **A** stored labels | mixture of 10x (pre-seam) and 30x | what `performance_stats.json` contains today |
| **B** uniform 30x | one threshold over the whole run | removes the ~150x L2P51 kernels; admits 2 precision hacks |
| **C** collapse problems excluded | drop L2P13/42/51/56, n=46 | removes the algebraic-collapse class outright |

### 4.1 View B — uniform 30x, geomean@30 (aligned n=50, 15-arm cohort)

| model | arm | stored | uniform 30x | change |
|---|---|---|---|---|
| gpt-oss-120b | truncation | 1.297 | 1.332 | **+2.7%** |
| gpt-oss-120b | folding | 1.443 | 1.443 | — |
| gpt-oss-120b | markov | 1.189 | 1.196 | **+0.6%** |
| gpt-oss-120b | selective_r5 | 1.402 | 1.402 | — |
| gpt-oss-120b | compress | 1.336 | 1.336 | — |
| gpt-oss-120b | deletion | 1.379 | 1.379 | — |
| gpt-oss-120b | merge_sim08 | 1.309 | 1.309 | — |
| gpt-oss-120b | refinement | 1.525 | 1.525 | — |
| gpt-oss-120b | l2 | 1.471 | 1.471 | — |
| terra | truncation | 2.417 | 2.773 | **+14.7%** |
| terra | markov | 1.758 | 1.951 | **+10.9%** |
| terra | selective_r5 | 2.049 | 2.276 | **+11.1%** |
| terra | compress | 2.901 | 3.308 | **+14.0%** |
| terra | deletion | 2.479 | 2.877 | **+16.0%** |
| terra | l2 | 1.966 | 2.328 | **+18.4%** |

### 4.2 View C — the 4 collapse problems excluded (n=46)

**gpt-oss-120b**

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 | geo@30 (all 50) |
|---|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.804 | 0.457 | 0.065 | 1.014 | 0.870 | 0.609 | 0.109 | 1.219 | 1.297 |
| folding | 0.783 | 0.413 | 0.109 | 1.129 | 0.913 | 0.630 | 0.152 | 1.333 | 1.443 |
| markov | 0.804 | 0.500 | 0.022 | 0.945 | 0.957 | 0.674 | 0.043 | 1.064 | 1.189 |
| selective_r5 | 0.913 | 0.435 | 0.109 | 1.079 | 1.000 | 0.630 | 0.174 | 1.315 | 1.402 |
| compress | 0.891 | 0.609 | 0.065 | 1.086 | 1.000 | 0.761 | 0.152 | 1.302 | 1.336 |
| deletion | 0.891 | 0.457 | 0.022 | 0.981 | 0.978 | 0.652 | 0.130 | 1.312 | 1.379 |
| merge_sim08 | 0.804 | 0.435 | 0.043 | 0.936 | 0.957 | 0.652 | 0.130 | 1.304 | 1.309 |
| refinement | 0.826 | 0.370 | 0.109 | 1.151 | 0.978 | 0.652 | 0.174 | 1.332 | 1.525 |
| l2 | 0.913 | 0.435 | 0.043 | 0.910 | 0.978 | 0.717 | 0.196 | 1.393 | 1.471 |

**gpt-5.6-terra**

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 | geo@30 (all 50) |
|---|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 1.000 | 0.783 | 0.217 | 1.615 | 1.000 | 0.848 | 0.457 | 2.341 | 2.417 |
| markov | 0.978 | 0.717 | 0.152 | 1.378 | 0.978 | 0.826 | 0.261 | 1.609 | 1.758 |
| selective_r5 | 1.000 | 0.804 | 0.304 | 1.697 | 1.000 | 0.826 | 0.348 | 2.118 | 2.049 |
| compress | 1.000 | 0.848 | 0.457 | 2.156 | 1.000 | 0.891 | 0.630 | 2.864 | 2.901 |
| deletion | 1.000 | 0.848 | 0.304 | 1.871 | 1.000 | 0.978 | 0.478 | 2.555 | 2.479 |
| l2 | 1.000 | 0.717 | 0.261 | 1.520 | 1.000 | 0.783 | 0.391 | 2.056 | 1.966 |

Dropping the four problems lowers geomean on **12 of 15 arms** (all 9 gpt-oss, range -0.4% to -12.6%) and *raises* it on 3 terra arms (`deletion` +3.1%, `l2` +4.5%, `selective_r5` +3.4%) — those three had a *below-average* best on the collapse problems, so removing them helps. The terra-vs-oss gap survives intact, so the collapse class was inflating levels, not manufacturing the gap. Note the within-model ordering does shift: on gpt-oss, `refinement` (-12.6%) drops from best geomean to mid-pack, and on terra `deletion` overtakes `truncation`. Do not assume the exclusion is rank-preserving.

---

## 5. What this does and does not license

- **n=1 per cell.** Replicate log-SD is 0.147 (open item 10); a 95% band needs x1.50, Bonferroni across 8 contrasts needs x1.77. **Nothing in these tables separates any treatment from its control.** Report descriptively with n stated.
- **Do not read the terra-vs-oss gap as a model-quality result on L2P13/42/51.** It is mostly that terra found the algebraic shortcut more often.
- **Do not compare across the 10x/30x seam** without saying which view you used.

## 6. Follow-ups this surfaced

1. **Checker gap (highest value, independent of this wave).** Add FP8 patterns (`float8_e4m3fn`, `float8_e5m2`, `__nv_fp8`, `torch._scaled_mm`) to `FP32_TO_FP16_PATTERNS`, and match `__half*`/`half2` declarations, not only conversion idioms. Two real hacks are currently invisible.
2. **Recompute `is_hack` at analysis time** rather than trusting the stored label — land `--hack-threshold` in `generate_run_performance_stats.py` (line 276 currently trusts it).
3. **Decide the collapse-problem policy** and record it in `ANALYSIS_RULES.md`: exclude L2P13/42/51/56 from headline aggregates, or report them in a separate 'algebraic reduction' column.
4. **Regenerate** once the three contaminated-cell replays finish.

## 7. Reproduction

```bash
# 1. regenerate performance_stats.json for both roots
for m in gpt-oss-120b gpt-5.6-terra; do
  .venv/bin/python Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py \
    --all-runs --runs-root runs_evolving/$m/median --hardware NVIDIA_GH200x2_median
done

# 2. uniform-threshold re-score (view B)
.venv/bin/python scripts_integration/new_evolving_agent_analysis/rescore_hack_threshold.py \
  --threshold 30 --all-dirs \
  --runs-root runs_evolving/gpt-oss-120b/median --runs-root runs_evolving/gpt-5.6-terra/median \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_median
```

