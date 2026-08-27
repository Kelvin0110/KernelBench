# L2 redesign wave — runbook

Launched 2026-08-27 on **GPU0**, worktree `.claude/worktrees/l2-redesign`,
branch `worktree-l2-redesign` (submodule branch `l2-redesign`).

```
results root : runs_evolving/gpt-oss-120b/l2redesign/
spec         : scripts_integration/new_evolving_agent/env/wave_l2_redesign.spec
manifest     : wave_gpu0_base_agent_gpt_oss_120b_Aug_27.manifest.tsv
eval config  : SLOTS=3  MEM_GATE=7  HOIST=1  SKIP_REF=1  UNLOCK_CORR=0
baseline     : NVIDIA_GH200x2_median
```

| arm | flags | offline prediction |
|---|---|---|
| `truncation` | — | no L2 at all; the base control |
| `l2` | `--enable-l2` | **0** rules (reproduces the on-disk null) |
| `l2_hit` | `+ --l2-use-hit-rate --l2-min-hit-rate 0.70` | **4** rules |
| `l2_redesign` | `+ hit 0.60, --l2-standing-cap 6, --l2-dedup-similarity 0.80` | **6** rules, 0 pairs ≥0.80 |

The predictions come from the validated offline replay of the *recorded* gpt-oss
L2 arm. This wave is a different run, so the LLM will mint different skills — treat
them as the expected shape (0 / a few / ~6 and bounded), not as exact targets.

## Why these four

`truncation` is launched fresh rather than reusing the median-wave one: that arm ran
in a 9-arm contention window, and unequal arms-per-GPU biases comparisons
one-directionally (CLAUDE.md §3.4). `l2` is the same-GPU L2 control §8.10 requires,
and it independently tests whether the 0-promotion result is reproducible — which,
given 0 / 4 / 9 across three arms, is genuinely open. `l2_hit` isolates the metric
fix from the cap and the dedup.

## While it runs

```bash
bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/launch_wave.sh 0 \
  scripts_integration/new_evolving_agent/env/wave_l2_redesign.spec status

# generic health (CLAUDE.md 3.5)
grep -c CUDA_HOME <log>                 # must stay 0
grep -E "^\[batch\]" <log> | tail -2

# L2-specific: a zero here is otherwise silent
for d in runs_evolving/gpt-oss-120b/l2redesign/*/; do
  echo "$d $(wc -l < $d/l2_promotions.jsonl 2>/dev/null || echo 0)"
done
```

**Confirm the mem gate is armed** — an all-zero `mem_need_gb` column means it is off,
which is exactly how `factor=2.5` went unnoticed for a day (CLAUDE.md §3.4):

```bash
grep -o '"mem_need_gb":[0-9.]*' *_phase.jsonl | sort -u | head
```

**Confirm the offer counter is live** on each L2 arm — without it `hit_rate` is 0 for
every skill and both hit-rate arms silently degrade to promoting nothing:

```bash
.venv/bin/python -c "
import json,sys
s=json.load(open(sys.argv[1]+'/l1_skill_usage.json'))
t=sum(v.get('total_offers',0) for v in s['skills'].values())
print('total_offers sum =', t, '=>', 'LIVE' if t else 'DEAD')" <run_dir>
```

## When it finishes

```bash
HW=NVIDIA_GH200x2_median
OUT=scripts_integration/new_evolving_agent_analysis/output/GH200x2_l2redesign

.venv/bin/python scripts_integration/new_evolving_agent_analysis/aggregate_runs.py \
  --hardware $HW --runs-root runs_evolving/gpt-oss-120b/l2redesign \
  --output-dir $OUT --regenerate-stats

.venv/bin/python scripts_integration/new_evolving_agent_analysis/compare_runs.py \
  --hardware $HW --runs-root runs_evolving/gpt-oss-120b/l2redesign \
  --output-dir $OUT --baseline-run <the truncation arm from THIS wave>
```

`--hardware` is mandatory (the default is a median-less baseline), and
`--regenerate-stats` is mandatory (cached stats across runs were written by
different code versions). The design column should now read
`truncation`, `truncation+l2`, `truncation+l2:hit0.7`,
`truncation+l2:hit0.6:cap6:dedup0.8` — if two arms share a label, the open-item-7
fix has regressed.

Then re-run the offline harness on the new arms to compare predicted vs actual:

```bash
python3 .../l2_redesign/build_visibility_cache.py out_l2 <arm>...
python3 .../l2_redesign/validate_replay.py <arm>...
.venv/bin/python .../l2_redesign/inspect_dupes.py <the l2_redesign arm>
```

## Reading the result honestly

- **n=1 per cell.** Replicate noise is log-SD 0.147, so a single arm-vs-arm contrast
  needs ≈×1.50 to clear 95% and ×1.77 under Bonferroni across the contrasts here
  (open item 10). **This wave cannot name a winner.** It answers: does the mechanism
  fire live, is the standing set bounded and deduplicated, and is the quality effect
  large enough to be worth replicating.
- Headline metric is `fast_p_best@1.0` on the full aligned denominator, with
  `best_geomean` secondary (`ANALYSIS_RULES.md:81-85`, `:158`).
- Filter hacks **per sample**, not per problem, and recompute `is_hack` at one
  threshold — this wave runs entirely after the 10×→30× cutover, so it is internally
  uniform, but it is *not* comparable to pre-2026-08-24 arms on that axis.
- A 0-promotion L2 arm is indistinguishable from a truncation arm in every metric.
  If `l2` again promotes nothing, that is the finding — not a tie.
