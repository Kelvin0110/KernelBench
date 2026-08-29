# L2 promotion — measured redesign

Companion to `CLAUDE.md` §8. Everything here is derived from artifacts of the two
completed L2 arms and is reproducible with the scripts in this directory.

## 0. What prompted this

`CLAUDE.md` §8 is written from a single arm,
`base_agent_gpt_oss_120b_l2_itr30_GH200_2026_08_22_20_34`, which promoted 9 rules.
**That run directory no longer exists on this host** — it survives only as rows in
`output/GH200x2_2nd_aug22_wave/`. The two L2 arms that *are* on disk behave very
differently from it and from each other:

| | gpt-oss-120b (`…_21_32`) | gpt-5.6-terra (`…_21_23`) |
|---|---|---|
| flags | identical | identical |
| problems | 50/50 | 50/50 |
| **rules promoted** | **0** | **4** |

Same gate, same defaults, same protocol — 0 vs 4. §8 treats "9 rules, then a null
result" as the thing to explain. The prior question is why the promotion count is
not reproducible at all.

## 1. The gate is exactly replayable offline

§8.8 says eligibility "is only replayable boundary by boundary". It is, and these
scripts do it. Every input is already in the artifacts:

- `workspaces/*/chat_history.jsonl`, `phase=="extractor"` — the prompt renders the
  candidate catalog as `- id=N | challenge=… | title=…`, and `assistant_text` is
  `{"selected_entry_ids": [...]}`. So both the **offered** set and the **selected**
  set are recoverable per iteration, with a wall clock.
- `l1_skill_usage.json` — `created_at_global_iter` per skill.

`global_iteration` advances once per governor iteration, so numbering iterations in
wall-clock order reconstructs it. **Both arms were resumed**, and the discarded
pre-resume iterations still bumped the real counter, so the reconstruction is offset
by `ledger_final_global_iteration − reconstructed_iterations` (26 for terra, 42 for
gpt-oss). Mixing the ledger's `created_at` with an un-offset scale corrupts
`opportunity` — that mistake produced 22 phantom promotions before it was caught.

**Validation.** With the offset applied the replay reproduces the recorded
promotions exactly — entry id, boundary `global_iter`, selections, distinct tasks,
opportunity and rate, on all four terra promotions, and 0-for-0 on gpt-oss:

```
python3 validate_replay.py <arm>        # exact-match check vs l2_promotions.jsonl
python3 calibrate_offset.py <arm>       # derives the offset; error 0 at 26
python3 regression_real_gate.py <arm>   # same, driven through the SHIPPED functions
```

## 2. Three defects, each measured

### 2.1 `selection_rate`'s denominator is not the support of its numerator

`rate = total_selections / (global_iter − created_at)`. A skill can only be selected
while it is inside the extractor's candidate set, and `read_l1_extractor_catalog`
returns `entries[-50:]` when governance is off (`memory_manager.py:798`). Once a
skill scrolls out of that tail its numerator freezes while its denominator keeps
ticking, so its rate **decays monotonically**. How fast it decays is set by how
quickly the arm mints new L1 skills — an arm-level property.

Measured with the exact per-iteration candidate sets:

| | gpt-oss | terra |
|---|---|---|
| L1 skills minted | 557 | 308 |
| catalog growth / iteration | 0.371 | 0.205 |
| median iterations a skill is *offered* | 118 | 231 |
| mean picks per extractor call | 6.81 | 3.85 |
| skills clearing tasks≥3 **and** selections≥50 | 49 | 35 |
| **best `selection_rate` ever reached** | **0.7059** | 1.0 |
| best `hit_rate` (= selections / offers) | 0.9667 | 1.0 |

The gpt-oss arm's best candidate **missed the 0.70 floor by 0.0115** while being
picked in 96.7% of the iterations it was actually offered. Individual skills show
the distortion plainly — id 11 has `hit_rate` 0.9412 and `selection_rate` 0.427.

### 2.2 `--l2-max-entries` does not cap the standing set

It caps **one pass**. The pass runs at every task boundary and the standing set
accumulates across them. Verified in `probe_cap_semantics.py`, and in replay:
`--l2-max-entries 4` admits **19** rules on the gpt-oss arm.

Worse, already-standing skills were still *counted against* the cap:
`compute_l2_candidates` reads `read_selectable_l1_jsonl`, which filters on
**status**, while `set_skill_tier` writes **tier** — `entry_tier` is imported in
`l2_promotion.py` and never used. So incumbents were ranked, consumed cap slots, and
only then skipped at `:417`.

`CLAUDE.md` §8.6 reads the cap as bounding the standing set ("4 → all three distinct
rules + one representative"), and §8.9's `l2_cap4` arm is designed on that reading.
Both need correcting.

### 2.3 No redundancy gate

§8.4's duplicate-family defect reproduces independently on a **different arm and a
different model**. Ranked pairwise cosine over the gpt-oss candidate set at
`hit≥0.60` (`inspect_dupes.py`, same embedding model as skill merging):

```
0.8733  398 TF32 matmul flag vs. cuDNN benchmark   || 449 Enable TF32 + cuDNN Autotuning
0.8655  403 Isolate Custom CUDA Ops & Enable TF32  || 410 Enable TF32 for FP32 workloads
0.8546  398 …                                      || 410 …
0.8530  410 …                                      || 449 …
0.8239  403 …                                      || 449 …
0.8119   56 Avoid Trivial Custom CUDA Kernels      || 493 Avoid Trivial No-Op CUDA Kernels
0.8099  394 Avoid Trivial Custom CUDA Kernels      || 401 In-place Elementwise Fusion
0.8044  372 Avoid Unnecessary Custom Elementwise   || 394 Avoid Trivial Custom CUDA Kernels
0.8038  139 In-place Elementwise Fusion            || 401 In-place Elementwise Fusion
------- 0.80 threshold -------
0.7988  398 …                                      || 403 …          (kept)
0.7967  394 …                                      || 450 Tiny Standalone CUDA Kernels  (kept)
```

τ=0.80 separates a four-member "enable TF32/cuDNN" family and an "avoid trivial
kernels" family from rules that merely share vocabulary. It was chosen by reading
the ranked list, not fitted.

## 3. The design comparison

`compare_designs.py`, both arms. `xctl` is the terminal coder-prompt size as a
multiple of the 4,190-char control; `pairs` counts promoted pairs at cosine ≥ 0.80.

```
gpt-oss                                    n   chars  xctl  dupmax  pairs>=0.8
  D0 SHIPPED  rate>=0.70                    0       0  1.00  0.0000    0/0
  D1 hit>=0.70                              4    7378  2.76  0.7119    0/6
  D3 hit>=0.60 + PER-PASS cap4 (shipped)   19   33226  8.93  0.8737    9/171
  D5 hit>=0.60 + STANDING cap6 + dedup      6   10186  3.43  0.7481    0/15

terra                                      n   chars  xctl  dupmax  pairs>=0.8
  D0 SHIPPED  rate>=0.70                    4    9446  3.25  0.7892    0/6
  D1 hit>=0.70                              6   14178  4.38  0.7922    0/15
  D3 hit>=0.60 + PER-PASS cap4 (shipped)    8   18862  5.50  0.8409    4/28
  D5 hit>=0.60 + STANDING cap6 + dedup      6   14178  4.38  0.7922    0/15
```

**D5 is the proposal.** Both arms land on exactly 6 rules with zero duplicate pairs,
against D0's 0-vs-4 and D3's 19-vs-8.

**An honest caveat.** `hit_rate` alone is *not* arm-invariant — its floor response
is a shifted version of `selection_rate`'s (gpt-oss 19→4→2→0 across floors
0.60→0.80; terra 8→6→6→4). It removes an unphysical decay term and it un-sticks the
zero-collapse, but the arm-invariance in D5 comes from the **standing cap**, which
delivers it by construction. Do not credit `hit_rate` with more than it earns.

## 4. What was changed

All four knobs default OFF; the shipped gate is untouched unless flagged. Proven by
`regression_real_gate.py`, which drives the boundary replay through the patched
`passes_floors` / `select_l2_promotions` and still reproduces both arms exactly.

| flag | effect |
|---|---|
| `--l2-use-hit-rate` / `--l2-min-hit-rate` | gate on `selections / times-offered`; **replaces** `--l2-min-rate` |
| `--l2-standing-cap N` | bound the accumulated standing set |
| `--l2-dedup-similarity τ` | reject a candidate restating a standing or higher-ranked rule |

Supporting changes:

- `total_offers` added to the usage ledger, incremented from the extractor's actual
  candidate list via a `offered_ids_out` out-parameter on `run_l1_skill_selection`
  (an out-parameter so the counter cannot drift from the catalog/L2/allow-list
  filtering). Missing in old ledgers → `hit_rate` 0 → fails the floor, never a
  spurious perfect score.
- Already-standing skills are excluded before ranking, so a cap ranks only what
  could actually be promoted.
- A `pass` census row is appended to `l2_promotions.jsonl` each boundary, recording
  `candidate_count`, `eligible_count`, `standing_after`, the full gate config and
  per-candidate drop reasons. §8.8 notes `eligible_count` was computed and thrown
  away, leaving capped arms uninterpretable; this fixes that.
- Dedup **fails open and loud** if embeddings are unavailable — it prints and keeps
  the candidates, rather than silently promoting nothing (the failure mode
  `--skill-merging` has).
- Open item 7 fixed: `aggregate_runs.py` extracts the L2 config and
  `compare_runs.py::design_variant_label` renders `truncation+l2:hit0.6:cap6:dedup0.8`
  instead of collapsing an L2 arm and its control to the same design string.
- MLE-side parity: the same knobs added to `GovernorConfig` / `_build_l2_config`,
  which `test_mle_and_kb_configs_expose_the_same_l2_knobs` requires.

## 4b. Threshold sharpness, measured against the live embedder

`live_promote_dedup.py` drives the real `run_l2_promotion_pass` over a synthetic
catalog (three restatements of one idea + two distinct rules) using the actual
`embed_texts_nvidia`, not a stub. Pairwise cosine over that corpus:

```
within the restatement family : 0.8302, 0.7595, 0.7419
family vs distinct rules      : 0.47 - 0.58
distinct vs distinct          : 0.4748
```

Separation between "same idea" and "different idea" is wide, which is what makes a
threshold viable at all. Two properties follow, both verified end to end:

- **Dedup is greedy-pairwise against already-kept rules, not transitive-closure
  clustering.** With family members at 0.83 / 0.76 / 0.74, keeping the top-ranked
  rule drops the 0.83 neighbour and keeps the 0.74 one. τ=0.70 collapses the whole
  family to one. This is intended: a promoted rule is permanent, so the gate should
  only refuse a candidate that restates something *specific* already standing.
- **τ=0.80 is calibrated on real L1 text**, where the observed families sit at
  0.80–0.87 (§2.3), not on hand-written prose. Re-derive it per model by reading
  `inspect_dupes.py` output rather than porting the number.

## 4c. The finding that reframes all of the above: selection has no outcome validity

Sections 2-4 treat the gate as a measurement problem -- wrong denominator, wrong
cap, no dedup. All three are real. But they assume the underlying signal means
something. It does not.

Every floor (`min_tasks`, `min_selections`, `min_rate`, and `hit_rate` too) counts
how often the extractor *chose* a skill. The one outcome floor, `min_new_bests`,
ships disabled. Comparing iterations where a skill was offered **and selected**
against those where it was offered and **not** selected -- so both sides condition
on the skill being available at that moment (`outcome_lift.py`):

| | gpt-oss | terra |
|---|---|---|
| P(new best \| selected) | 0.1423 | 0.2057 |
| P(new best \| offered, not selected) | 0.1440 | 0.2275 |
| **pooled lift** | **-0.0017** | **-0.0219** |
| per-skill median lift | -0.0125 | -0.0221 |
| fraction of skills with lift > 0 | 0.439 | 0.442 |

New bests concentrate early (base rate 0.218 in attempts 1-5 vs 0.152 in 6-15 on
gpt-oss), so position is a real confound -- but stratifying on it does not rescue
the signal: gpt-oss -0.0002 / -0.0055, terra +0.0281 / -0.0308. On terra the three
promoted skills scored **worse** than non-promoted (-0.0608 vs -0.0216).

And `min_new_bests` is not an independent second signal: `corr(selections,
new_bests)` is 0.867 / 0.680 and the ratio's median (0.129 / 0.203) matches the
pooled new-best base rate (0.142 / 0.206). It is `min_selections` rescaled
(`newbests_vs_selections.py`).

**So the ledger holds no validated outcome evidence, and no threshold on it can.**
This partly undercuts §2.1: the hit-rate change fixes the *precision* of a measure
that lacks *validity*. It is still correct on its own terms -- the old denominator
counts iterations in which selection was impossible -- but it should not be
expected to improve kernel quality by itself.

**Caveat.** This is observational. The extractor picks conditioned on the current
failure, so selection correlates with the state of the search. A large positive
lift would not have proven causation. A lift at zero does show the ranking signal
carries no outcome information, which is the decision-relevant direction.

### What follows from it

| knob | what it does instead |
|---|---|
| `--l2-judge` | an LLM reads the rule TEXT -- general, actionable, non-redundant, non-obvious -- rather than ranking counts. Floors drop to a weak admission bar. Fails CLOSED. |
| `--l2-preseed` | install a previous run's standing set at problem 1, so an arm asks "do standing rules help" WITHOUT also asking "can the gate find them" |
| `--l2-freeze` | promote nothing; with pre-seed the treatment is exactly the injected set |

## 5. Reproducing

```bash
V=.venv/bin/python          # repo venv; needs the embedding client for dedup
python3 build_visibility_cache.py out_l2 <arm> [<arm>…]   # slow, once per arm
python3 validate_replay.py <arm>
python3 sweep_gates.py <arm> [<arm>…]
$V compare_designs.py <arm> [<arm>…]
$V inspect_dupes.py <arm>
$V test_redesign.py            # 15 unit tests incl. defaults-are-inert
$V test_design_label.py        # L2 arm must not render as its own control
$V regression_real_gate.py <arm> [<arm>…]
$V live_promote_dedup.py       # promote+dedup+cap against the REAL embedder
$V probe_cap_semantics.py      # what --l2-max-entries actually caps
```

## 6. Limits

- n=2 arms, one of each model. Thresholds (`hit≥0.60`, cap 6, τ=0.80) are chosen
  from those two and should be re-derived per model, exactly as §8 warns.
- Replicate noise is log-SD 0.147 (open item 10), so a single arm-vs-arm contrast
  needs ≈×1.50 to clear 95%. Any L2 batch is a screen, not a winner-claim.
- These changes alter which rules get promoted. They do **not** show that L2 helps
  quality; §8's null result stands until a fresh arm says otherwise.

---

## Merged to `features/evolving-agent-final` (2026-08-29)

The redesign is now reachable behind one switch, with the **shipped gate still the
default and byte-unchanged**, so the two designs are a matched pair:

```bash
--enable-l2                  # shipped: selection_rate, no dedup, no cap
--enable-l2 --redesign-l2    # hit_rate 0.60 + dedup 0.80, no cap
```

Preset lives in `l2_promotion.L2_REDESIGN_PRESET` (shared by the KernelBench and
MLE entry points). Precedence is **explicit flag > preset > shipped default**.
`--redesign-l2` without `--enable-l2` is a hard error.

**`--l2-standing-cap` now defaults to `-1` = no cap** (any value `<= 0` means no
cap; `0` is a legacy alias). Rationale, measured: the two arms on the 2026-08-27
wave that ran with no cap **both ended at exactly 6 standing rules**, so at
ordinary run lengths the floors are what bound the set. A cap is still needed when
the floors are loosened (the judge arm hit 6 rules at problem 3 of 50) or for long
runs. Pair any cap with `--l2-min-new-bests 1` — see §8.7.

Launch both designs against a shared control with
`env/wave_l2_designs.spec`.

### Scripts added for the wave analysis

| script | what it answers |
|---|---|
| `paired_report.py` | paired per-problem log-ratio vs a control, hacks filtered per sample |
| `pair_contrast.py` | explicit A-vs-B contrasts incl. the identical-config null |
| `robust_contrast.py` | median + sign test + top-3 movers (geomean is outlier-driven here) |
| `collapse_check.py` | are the suspect problems bimodal? |
| `lottery_adjusted.py` | **the one to read** — re-runs contrasts with lottery problems removed by an arm-agnostic rule |
| `final_ranking.py` | one consolidated table across designs |
| `cap_binding.py` | did the standing cap actually bind? |
| `mechanism_report.py` | promotions, dedup drops, judge decisions, pre-seed survival |
| `standing_diversity.py` | pairwise cosine over each arm's final standing set |
| `show_configs.py` | per-arm gate config from `run_summary.json` (not from the spec) |
| `test_redesign_preset.py` | preset precedence + cap sentinel semantics |

**Read `lottery_adjusted.py` before quoting any arm-vs-arm number.** ~14 of the 50
subset problems are bimodal and set the geomean; on the 2026-08-27 wave the entire
apparent ranking was an artifact of them, and every adjusted CI contained 1.0 —
including the identical-configuration null. Full write-up in `RESULTS.md`.
