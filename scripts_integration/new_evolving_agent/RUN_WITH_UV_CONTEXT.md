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
| `folding` | no | Archived L0 summaries + per-round summaries + unfold preflight |
| `markov_report` | no | Each iteration: **goal + evolving report + latest L0 only**; after eval, a dedicated rewriter LLM updates `evolving_report.md` |
| `selective_retention` | no | Each iteration: **goal + milestone memory (selected past rounds, full detail) + latest L0 only**; milestones are labeled per round (rules + additive LLM judge) and packed under 90% of the model context window |

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
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_markov_report_itr10 \
  --max-iterations 10 \
  --context-management markov_report \
  --evolving-report-max-tokens 65536 \
  >> base_agent_markov_report_itr10_Jul_18.log 2>&1 &
```

Optional rewriter knobs (defaults are usually fine):

- `--evolving-report-max-tokens` (default `1536`)
- `--evolving-report-timeout-sec` (default `90`)

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
detail) + latest 1 L0 round**. A round becomes a milestone via rules (`propose_new`,
new best metric, first compile success, first correct) plus an additive, fail-soft LLM
judge. Non-milestone rounds are omitted from the prompt (never deleted from disk).
When the previous coder turn used ≥90% of the model context window, milestones are
packed under budget (latest kept, new-best kept, oldest non-new-best dropped from the
prompt).

### Unit tests (no GPU / no API key)

```bash
uv run python -m pytest Self-Evolving-Agent/tests/test_selective_retention.py -q
uv run python -m pytest scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py::test_main_dry_run_accepts_selective_retention_context_management -q
```

### Dry run (CLI plumbing only)

```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name selective_retention_dryrun \
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
  --run-name base_agent_selective_retention_smoke \
  --max-problems 5 \
  --max-iterations 20 \
  --context-management selective_retention \
  >> base_agent_selective_retention_smoke.log 2>&1 &
```

### Full 50 problems

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_selective_retention_itr10 \
  --max-iterations 10 \
  --context-management selective_retention \
  >> base_agent_selective_retention_itr10_Jul_18.log 2>&1 &
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
