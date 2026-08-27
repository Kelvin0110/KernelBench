# Kernel legitimacy audit — are the high-speedup samples real?

Generated 2026-08-27T10:43:49+00:00. Cohort filter: `_2026_08_22_`. Arms: 15.

Regenerate with `make_kernel_legitimacy_report.py` (see §6). Reads only `performance_stats.json`, the per-problem eval records, and `rescore_hack_threshold.json` — no scratch state.

**Headline tables are the uniform-30x view**, i.e. exactly what `generate_run_performance_stats.py` writes by default since submodule `fa133b1`. Metrics are best-so-far: `corr` = `fast_p_best@0.0`, `fast@1`/`fast@2` = `fast_p_best@{1.0,2.0}`, `geo` = geometric mean of best speedup over correct problems.

---

## 1. Headline (uniform 30x, n=50)

### gpt-oss-120b

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 |
|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.820 | 0.480 | 0.080 | 1.036 | 0.900 | 0.660 | 0.140 | 1.325 |
| folding | 0.800 | 0.420 | 0.100 | 1.106 | 0.920 | 0.660 | 0.180 | 1.443 |
| markov | 0.820 | 0.540 | 0.080 | 1.075 | 0.960 | 0.700 | 0.100 | 1.196 |
| selective_r5 | 0.920 | 0.460 | 0.140 | 1.160 | 1.000 | 0.660 | 0.200 | 1.402 |
| compress | 0.900 | 0.620 | 0.080 | 1.119 | 1.000 | 0.780 | 0.160 | 1.336 |
| deletion | 0.920 | 0.460 | 0.020 | 0.979 | 1.000 | 0.700 | 0.160 | 1.382 |
| merge_sim08 | 0.780 | 0.440 | 0.040 | 0.940 | 0.960 | 0.640 | 0.140 | 1.301 |
| refinement | 0.840 | 0.420 | 0.140 | 1.275 | 0.980 | 0.680 | 0.220 | 1.525 |
| l2 | 0.900 | 0.460 | 0.060 | 0.949 | 0.960 | 0.720 | 0.220 | 1.471 |

### gpt-5.6-terra

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 |
|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.980 | 0.780 | 0.240 | 1.764 | 1.000 | 0.860 | 0.500 | 2.773 |
| markov | 0.980 | 0.740 | 0.220 | 1.627 | 0.980 | 0.840 | 0.320 | 1.951 |
| selective_r5 | 0.980 | 0.800 | 0.320 | 1.871 | 1.000 | 0.840 | 0.360 | 2.276 |
| compress | 1.000 | 0.860 | 0.500 | 2.542 | 1.000 | 0.900 | 0.660 | 3.308 |
| deletion | 1.000 | 0.860 | 0.340 | 2.133 | 1.000 | 0.980 | 0.500 | 2.877 |
| l2 | 1.000 | 0.740 | 0.300 | 1.763 | 1.000 | 0.800 | 0.420 | 2.328 |

## 2. The same arms with the four collapse problems removed (n=46)

Rationale in §4. `geo@30 (all 50)` is repeated so the cost of the exclusion is visible.

**gpt-oss-120b**

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 | geo@30 (all 50) |
|---|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 0.804 | 0.457 | 0.065 | 1.014 | 0.891 | 0.630 | 0.109 | 1.215 | 1.325 |
| folding | 0.783 | 0.413 | 0.109 | 1.129 | 0.913 | 0.630 | 0.152 | 1.333 | 1.443 |
| markov | 0.804 | 0.500 | 0.022 | 0.945 | 0.957 | 0.674 | 0.043 | 1.064 | 1.196 |
| selective_r5 | 0.913 | 0.435 | 0.109 | 1.079 | 1.000 | 0.630 | 0.174 | 1.315 | 1.402 |
| compress | 0.891 | 0.609 | 0.065 | 1.086 | 1.000 | 0.761 | 0.152 | 1.302 | 1.336 |
| deletion | 0.913 | 0.457 | 0.022 | 0.978 | 1.000 | 0.674 | 0.130 | 1.316 | 1.382 |
| merge_sim08 | 0.804 | 0.435 | 0.043 | 0.936 | 0.978 | 0.652 | 0.130 | 1.296 | 1.301 |
| refinement | 0.826 | 0.370 | 0.109 | 1.151 | 0.978 | 0.652 | 0.174 | 1.332 | 1.525 |
| l2 | 0.913 | 0.435 | 0.043 | 0.910 | 0.978 | 0.717 | 0.196 | 1.393 | 1.471 |

**gpt-5.6-terra**

| arm | corr@10 | fast@1 itr10 | fast@2 itr10 | geo@10 | corr@30 | fast@1 itr30 | fast@2 itr30 | geo@30 | geo@30 (all 50) |
|---|---|---|---|---|---|---|---|---|---|
| truncation (ctrl) | 1.000 | 0.783 | 0.217 | 1.615 | 1.000 | 0.848 | 0.457 | 2.341 | 2.773 |
| markov | 0.978 | 0.717 | 0.152 | 1.378 | 0.978 | 0.826 | 0.261 | 1.609 | 1.951 |
| selective_r5 | 1.000 | 0.804 | 0.304 | 1.697 | 1.000 | 0.826 | 0.348 | 2.118 | 2.276 |
| compress | 1.000 | 0.848 | 0.457 | 2.156 | 1.000 | 0.891 | 0.630 | 2.864 | 3.308 |
| deletion | 1.000 | 0.848 | 0.304 | 1.871 | 1.000 | 0.978 | 0.478 | 2.555 | 2.877 |
| l2 | 1.000 | 0.717 | 0.261 | 1.520 | 1.000 | 0.783 | 0.391 | 2.056 | 2.328 |

Geomean falls on 15 of 15 arms and rises on 0 (none) — those had a below-average best on the collapse problems. **The exclusion is not rank-preserving**; do not carry a ranking across views.

## 3. The 10x -> 30x `is_hack` seam

`src/kernelbench/eval.py` changed `excessive_speedup_threshold` 10 -> 30 at **2026-08-24T15:11:45** (commit `588a6a5`). `eval.py` is re-imported by every eval spawn, so it reached live arms with no restart and each run's stored `is_hack` column is a mixture of two rules. Uniform re-scoring is what the tables above use.

| model | arm | geo@30 stored | geo@30 uniform-30 | change |
|---|---|---|---|---|
| gpt-oss-120b | truncation | 1.291 | 1.325 | **+2.7%** |
| gpt-oss-120b | folding | 1.443 | 1.443 | — |
| gpt-oss-120b | markov | 1.189 | 1.196 | **+0.6%** |
| gpt-oss-120b | selective_r5 | 1.402 | 1.402 | — |
| gpt-oss-120b | compress | 1.336 | 1.336 | — |
| gpt-oss-120b | deletion | 1.382 | 1.382 | — |
| gpt-oss-120b | merge_sim08 | 1.301 | 1.301 | — |
| gpt-oss-120b | refinement | 1.525 | 1.525 | — |
| gpt-oss-120b | l2 | 1.471 | 1.471 | — |
| terra | truncation | 2.417 | 2.773 | **+14.7%** |
| terra | markov | 1.758 | 1.951 | **+10.9%** |
| terra | selective_r5 | 2.049 | 2.276 | **+11.1%** |
| terra | compress | 2.901 | 3.308 | **+14.0%** |
| terra | deletion | 2.479 | 2.877 | **+16.0%** |
| terra | l2 | 1.966 | 2.328 | **+18.4%** |

## 4. Are the (10x, 30x] kernels legitimate?

**Yes — 264 of the 268 re-scored evals are exact, fp32, real custom CUDA.** Audited three ways:

- **Static.** All 268 run through `validate_kernel_static(backend='cuda', precision='fp32')`, including `global_module_patch` (the reference-corruption check added in `ede1898`, which did not exist when they were evaluated). **0 STRICT errors.**
- **Numerical.** Re-run independently with parameters synced from the reference at `atol=rtol=1e-4` (the tolerance eval uses): L2P13 correct, max|diff| 2.17e-05, 26.5x measured vs 30.0x recorded; L2P42 correct, max|diff| 2.29e-05, 22.7x vs 22.9x. Live reference timings (8.412 ms, 6.018 ms) match the fixed baseline (8.4, 6.02).
- **Code.** All 21 metric-moving samples contain real `__global__` CUDA via `load_inline`.

The band is concentrated in four problems:

| problem | reference model |
|---|---|
| L2P13 | ConvTranspose3d -> mean over depth |
| L2P42 | ConvTranspose2d -> global avg pool |
| L2P51 | Gemm 8192x8192 -> mean over features |
| L2P56 | Linear 32768 -> sigmoid -> sum (does NOT collapse) |

Each computes something enormous then discards most of it, so an exact algebraic shortcut is worth 10-30x. `logsumexp` over a size-1 dim (L2P51) is literally the identity.

**Why they are excluded anyway.** KernelBench's prompt (`src/kernelbench/prompts/prompts.toml:13`) explicitly permits *"algorithmic changes (such as online softmax)... only limited by your imagination"*, so these are legal. But `EVAL.md` warns *"a >2x speedup for anything is highly unlikely"*, and there is a real distinction: online softmax restructures **how** the same work is done, whereas mean-of-GEMM -> matvec proves most of the reference's work is never observed and **deletes** it. Only the first is kernel engineering. On these problems the speedup measures how wasteful the reference model is, and at 10-30x it dominates any geometric mean it enters.

Note this cut removes **problems**, not a speedup band. Cutting only >30x would be the same magnitude heuristic with a different constant: on L2P51 the ~150x kernels and the 22-30x kernels are the same trick, differing only in whether weight prep is cached. (The ~150x figure is also physically achievable — 134 MB of essential traffic in 36.8 us is ~3.6 TB/s, ordinary for GH200 HBM3e — so magnitude alone is not evidence of cheating.)

## 5. The exception: an FP8 hack the checker used to miss

Of the 268, four use reduced precision. One is decisive: a terra L2P56 kernel cast **both operands of a 32768-wide GEMM to `torch.float8_e4m3fn`** and ran it through `torch._scaled_mm` with fp16 accumulate. It passed the 1e-4 gate only because the following sigmoid **saturates**, which hides the FP8 error, and measured 15.6x.

`check_precision_downgrade` missed it: `FP32_TO_FP16_PATTERNS` matched conversion idioms (`__float2half(`, `.half()`) but not `__half*` declarations, and had **no FP8 pattern at all**. `torch._scaled_mm` was absent from `TORCH_COMPUTATION_OPS`.

**Fixed 2026-08-27** in `src/kernelbench/kernel_static_checker.py`:

1. New **STRICT** check `fp8_downgrade` (`check_fp8_downgrade`, `FP32_TO_FP8_PATTERNS`) matching `torch.float8_e[45]m[23]*`, `torch._scaled_mm`, `__nv_fp8*`, `__nv_cvt_*_to_fp8*`, for required precision FP32/FP16/BF16. FP8 sets `is_hack`. **FP16 deliberately stays a WARNING** — a 10-bit mantissa can legitimately meet a 1e-4 gate; a 3-bit one cannot.
2. FP16 **storage/consumption** patterns added: `__half*` declarations, `half2`/`__half2`, `__half2float(`, `at::Half`, `torch::kHalf`. A bare `torch.float16` token was deliberately NOT added — the CUDA source lives inside a Python string literal, so string contents cannot be stripped before matching and it false-positives on prose.
3. `torch._scaled_mm`, `torch._int_mm`, `torch.addmm`, `torch.addbmm` added to `TORCH_COMPUTATION_OPS`.
4. `_parse_python_module` retries `ast.parse` on a dedented copy — an indented snippet raised `SyntaxError` and silently disabled the STRICT `code_bypass` check (pre-existing failing test `test_strict_checks_are_errors`, now green).

**Validation.** 9 new unit tests; 76 pass across the three static-checker modules. On the wave corpus the 3 FP8 evals become STRICT errors, the 1 FP16 eval stays a warning, and all 264 legitimate kernels stay clean. Across **729 best-forming kernels**: **0 STRICT errors** — no false positives. 8 of the 729 carry FP16 warnings, and **7 of those 8 were already detectable before this change** — they were recorded as warnings and counted as bests anyway, because `precision_downgrade` does not set `is_hack`. Promoting FP16 too would change 8 problems' bests across 5 arms (largest: terra truncation geo 2.417 -> 2.270) with no correctness lost; that was considered and deliberately not done.

**No pipeline change was needed.** A STRICT failure short-circuits in `governor.py:549` with `compiled=False, correct=False` and no eval, so it never carries a speedup, and `_resolve_is_hack` returns the stored label when speedup is absent — static-check hacks survive uniform re-scoring intact.

**Seam.** The checker is parent-side (`governor.py` only) and bound at import in the long-lived parent, so it cannot reach a running arm. Arms launched from 2026-08-27 10:37 onward enforce `fp8_downgrade`; this cohort did not.

## 6. Reproduction

```bash
# stats (uniform-30 is the default; --use-stored-hack for the stored view)
for m in gpt-oss-120b gpt-5.6-terra; do
  .venv/bin/python Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py \
    --all-runs --runs-root runs_evolving/$m/median --hardware NVIDIA_GH200x2_median
done

# uniform-threshold re-score (--completed-only drops in-flight arms)
.venv/bin/python scripts_integration/new_evolving_agent_analysis/rescore_hack_threshold.py \
  --threshold 30 --all-dirs --completed-only \
  --runs-root runs_evolving/gpt-oss-120b/median --runs-root runs_evolving/gpt-5.6-terra/median \
  --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_median

# this report
.venv/bin/python scripts_integration/new_evolving_agent_analysis/make_kernel_legitimacy_report.py \
  --runs-root gpt-oss-120b=runs_evolving/gpt-oss-120b/median \
  --runs-root gpt-5.6-terra=runs_evolving/gpt-5.6-terra/median \
  --cohort _2026_08_22_ --output-dir scripts_integration/new_evolving_agent_analysis/output/GH200x2_median
```

## 7. Caveat

n=1 per cell against a replicate log-SD of 0.147 (open item 10): a 95% band needs x1.50, Bonferroni across 8 contrasts x1.77. **Nothing here separates any treatment from its control.** Read descriptively, with n stated.

