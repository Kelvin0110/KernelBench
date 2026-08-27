# Run KernelBench Evolving Agent with L0 Context Management (uv)

Companion to [RUN_WITH_UV.md](RUN_WITH_UV.md) for **L0 context-management** modes.
Use that guide for prerequisites, `uv` env setup, resume mechanics, and skill
refinement / deletion / merge commands.

This file covers `--context-management` on
`scripts_integration/new_evolving_agent/evolve_kb_batch.py`.

## Modes

| Mode | Default? | Behavior |
|------|----------|----------|
| `truncation` | yes | Keep only the latest N raw L0 rounds in prompts |
| `folding` | no | Archived L0 summaries + per-round summaries + unfold preflight; archived summary catalog is packed newest-first under ~90% of the model context window (disk L0 unchanged) |
| `markov_report` | no | Each iteration: **goal + evolving report + latest L0 only**; after eval, a dedicated rewriter LLM updates `evolving_report.md` |
| `selective_retention` | no | Each iteration: **goal + milestone memory (selected past rounds, full detail) + latest 5 full L0 rounds**; milestones are labeled per round (rules + additive LLM judge) and packed under 90% of the model context window |
| `compress_trigger` | no | Microcompact old L0 rounds every iteration, then run structured LLM compression when the prompt reaches a token threshold or an iteration interval |

Skill refinement and skill merging stay **off** by default (omit those flags).
Skill deletion defaults **on**, so mode-only examples below pass `--no-skill-deletion`.

## Unit tests (no GPU / no API key)

From repository root:

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_markov_report.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_markov_report_context_management -q
```

## Dry run (CLI plumbing only)

Validates that `--context-management markov_report` is accepted; dry-run skips GPU
eval and governor execution.

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name markov_report_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --context-management markov_report \
  --no-skill-deletion \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

After dry-run, check `runs_evolving/<run_name>_*/run_summary.json` for
`context_management: "markov_report"` and `skill_deletion: false`.

## Small real CUDA run (smoke)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_markov_report_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --context-management markov_report \
  >> base_agent_markov_report_smoke.log 2>&1 &
```

## Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_markov_report_itr30 \
  --max-iterations 30 \
  --context-management markov_report \
  --evolving-report-max-tokens 65536 \
  >> base_agent_markov_report_itr30_Jul_21.log 2>&1 &
```

Optional rewriter knobs (defaults are usually fine):

- `--evolving-report-max-tokens` (default `65536`)
- `--evolving-report-timeout-sec` (default `600`)

## After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `context_management`, `evolving_report_max_tokens`, `evolving_report_timeout_sec`
- `workspaces/level_*_problem_*/evolving_report.md` — current evolving report for that problem
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns with `phase=evolving_report`

## Resume

Resume the same way as [RUN_WITH_UV.md](RUN_WITH_UV.md) §6. Keep
`--context-management markov_report` and `--no-skill-deletion` if the original
run used them so prompt mode and L1 GC policy stay consistent.

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_markov_report_itr50_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 50 \
  --start-problem 11 \
  --context-management markov_report \
  --no-skill-deletion \
  >> base_agent_markov_report_itr50_resume.log 2>&1 &
```

---

## Selective retention mode

Design: [selective-retention spec](../../Self-Evolving-Agent/docs/superpowers/specs/2026-07-18-selective-retention-context-design.md).
Each iteration rebuilds the prompt as **original goal + milestone memory (full L0
detail) + latest 5 full L0 rounds** (`DEFAULT_SELECTIVE_RECENT_ROUNDS=5`). A round becomes a milestone via rules
(`propose_new`, new best metric, first compile success, first correct) plus an
additive, fail-soft LLM judge. Non-milestone rounds outside the recent window are
omitted from the prompt (never deleted from disk). When the previous coder turn used
≥90% of the model context window, milestones are packed under budget (recent kept,
new-best kept, oldest non-new-best dropped from the prompt).

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_selective_retention.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_selective_retention_context_management -q
```

### Dry run (CLI plumbing only)

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name selective_retention_recent5_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --context-management selective_retention \
  --no-skill-deletion \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

Check `runs_evolving/<run_name>_*/run_summary.json` for
`context_management: "selective_retention"` and `skill_deletion: false`.

### Small real CUDA run (smoke)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_selective_retention_recent5_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --context-management selective_retention \
  >> base_agent_selective_retention_recent5_smoke.log 2>&1 &
```

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_selective_retention_recent5_itr30 \
  --max-iterations 30 \
  --context-management selective_retention \
  >> base_agent_selective_retention_recent5_itr30.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_selective_retention_recent5_itr30_YYYY_MM_DD_HH_MM \
  --max-iterations 30 \
  --context-management selective_retention \
  --resume \
  --start-problem 34 \
  >> base_agent_selective_retention_recent5_itr30.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_selective_retention_recent5_itr30_YYYY_MM_DD_HH_MM \
  --max-iterations 30 \
  --context-management selective_retention \
  --resume \
  --start-problem 13 \
  --end-problem 26 \
  >> base_agent_selective_retention_recent5_itr30.log 2>&1 &
```

The 128K window for `gpt-oss-120b` is built in; override with a different model only
if you also update the context-window registry in
`Self-Evolving-Agent/evolving_common/context_management.py`.

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `context_management: "selective_retention"`
- `workspaces/level_*_problem_*/l0_milestones.json` — per-problem milestone index
  (`round_id`, `reasons`, `source`, `is_new_best`); one file per problem workspace
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns incl. `phase=milestone_judge`

Resume identically to markov: keep `--context-management selective_retention` (and
`--no-skill-deletion` if used) so the prompt mode and L1 GC policy stay consistent.

---

## Compress trigger mode

Design: [compress-trigger spec](../../Self-Evolving-Agent/docs/superpowers/specs/2026-07-27-compress-trigger-context-design.md).
Each iteration keeps the latest hot rounds in full detail and replaces older
post-compression rounds with one-line stubs. Once either trigger fires, a fail-soft
summarizer call combines the prior summary with new full rounds and advances the
compression boundary. Full L0 history remains on disk.

Command knobs used for fair comparison runs (selective recent=5; compress hot=15):

- `--compress-hot-rounds 15`
- `--compress-token-ratio 0.85`
- `--compress-every-n-iters 15`

(The governor default for hot rounds remains `3` if the flag is omitted.)
The structured summary output limit (`2048` tokens) and summarizer timeout (`90`
seconds) remain governor defaults rather than batch CLI options.

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_compress_trigger.py -q
uv run python -m pytest Self-Evolving-Agent/tests/test_context_management.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_compress_trigger_context_management -q
```

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_compress_trigger_hot15_itr30 \
  --max-problems 50 \
  --max-iterations 30 \
  --context-management compress_trigger \
  --compress-hot-rounds 15 \
  --compress-token-ratio 0.85 \
  --compress-every-n-iters 15 \
  --no-skill-deletion \
  >> base_agent_compress_trigger_hot15_itr30.log 2>&1 &
```

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `context_management` and the three `compress_*` CLI values
- `workspaces/level_*_problem_*/compression_summary.md` — current structured summary
- `workspaces/level_*_problem_*/compression_state.json` — compression boundary and count
- `workspaces/level_*_problem_*/compression_events.jsonl` — successful trigger events

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_compress_trigger_hot15_itr30_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem 11 \
  --context-management compress_trigger \
  --compress-hot-rounds 15 \
  --compress-token-ratio 0.85 \
  --compress-every-n-iters 15 \
  --no-skill-deletion \
  >> base_agent_compress_trigger_hot15_itr30_resume.log 2>&1 &
```

Keep the compression mode and tuning flags identical to the original run.

---

## Folding mode

Current folding choice: each iteration keeps **recent N full L0 rounds** (default
`N=15`: code, terminal, reasoning) plus an **archived summary catalog** for older
rounds. Before the coder, an **unfold preflight** may request full detail for a few
archived `round_id`s from that catalog (default: enabled; up to 2 attempts × 3
rounds). Archived summaries in the prompt are packed **newest-first** under ~90% of
the model context window (disk L0 unchanged; unfold may only select IDs still in the
packed catalog).

Batch CLI only needs `--context-management folding`; unfold / recent-N knobs use
governor defaults unless you run via `Self-Evolving-Agent/cli.py`.

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_l0_rounds.py -q
```

Packing / archive-catalog cases live in that file. For batch CLI acceptance, use the
dry-run below.
### Dry run (CLI plumbing only)

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name folding_dryrun \
  --subset-csv subset_selection/selected_problems_50.csv \
  --max-problems 2 \
  --max-iterations 2 \
  --context-management folding \
  --no-skill-deletion \
  --backend cuda \
  --precision fp32 \
  --dry-run
```

Check `runs_evolving/<run_name>_*/run_summary.json` for
`context_management: "folding"` and `skill_deletion: false`.

### Small real CUDA run (smoke)

```bash
CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_folding_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --context-management folding \
  >> base_agent_folding_smoke.log 2>&1 &
```

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_folding_itr30 \
  --max-iterations 30 \
  --context-management folding \
  >> base_agent_folding_itr30_Jul_28.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_folding_itr30_2026_07_28_01_09 \
  --max-iterations 30 \
  --context-management folding \
  --resume \
  --start-problem 4 \
  >> base_agent_folding_itr30_Jul_28.log 2>&1 &
```

The 128K window for `gpt-oss-120b` is built in (same registry as selective retention);
archived-summary packing reuses `CONTEXT_PACK_RATIO` in
`Self-Evolving-Agent/evolving_common/context_management.py`.

### After a real run

Under `runs_evolving/<run_name>_*/`:

- `run_summary.json` — `context_management: "folding"`
- `workspaces/level_*_problem_*/chat_history.jsonl` — LLM turns incl. `phase=preflight`
  (unfold) and `phase=l0_round_summary` (per-round archive summaries)
- Per-round `round_summary` lives on L0 in the problem workspace / snapshots (no
  separate milestones file)

### Resume

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_folding_itr30_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem 11 \
  --context-management folding \
  --no-skill-deletion \
  >> base_agent_folding_itr30_resume.log 2>&1 &
```

Keep `--context-management folding` (and `--no-skill-deletion` if the original run
used it). Optional range resume: add `--end-problem M` as in [RUN_WITH_UV.md](RUN_WITH_UV.md) §6.

---

For runs on the **inference-api.nvidia.com** endpoint (e.g. `gpt-5.6-terra`), see
[infer_api/RUN_WITH_UV_INFER.md](infer_api/RUN_WITH_UV_INFER.md).
