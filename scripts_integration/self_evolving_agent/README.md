# Self-Evolving-Agent integration (KernelBench)

This folder contains thin wrappers to run the Self-Evolving-Agent (SEA) integration with KernelBench.

Files
- `run_batch.py`: Orchestrates batched evaluations across problem subsets and writes level-first JSON results.
- `kb_agent.py`: Legacy wrapper that imports `KernelBenchEvolvingAgent` from the SEA core.
- `kb_environment.py`: Legacy wrapper that imports `KernelBenchEnvironment` from the SEA core.

Quickstart

1. Ensure repository dependencies and environment variables are available. See `RUN_WITH_UV.md` for `uv sync`/`uv run` examples.

2. Example dry-run (no external LLM calls):

```bash
uv run python scripts_integration/self_evolving_agent/run_batch.py --subset-csv subset_selection/selected_problems_50.csv --output-path runs/sea_integration_run/eval_results.json --max-steps 2 --dry-run
```

3. Example real run (uses SEA provider registry):

```bash
# ensure .env exports SEA_PROVIDERS and SEA_ROLE_* as described in RUN_WITH_UV.md
uv run python scripts_integration/self_evolving_agent/run_batch.py --subset-csv subset_selection/selected_problems_50.csv --output-path runs/sea_integration_run/eval_results.json --max-steps 5
```

Environment variables
- `SEA_PROVIDERS`: JSON array of provider entries. Each entry must include `id`, `provider`, and `api_key_env`. Example in `RUN_WITH_UV.md`.
- `SEA_ROLE_<NAME>`: JSON object mapping a role name to a `provider_id` and `model`.

Troubleshooting
- If you see provider/API key errors, verify the API key env var referenced by `api_key_env` is present.
- Use `--dry-run` to validate orchestration without hitting LLMs.

Support
- These are wrapper scripts; behavioral changes to LLM interaction are implemented in the Self-Evolving-Agent package under `Self-Evolving-Agent/src/self_evolving_agent/`.

## Reminder:
This integration combine with the 2nd prototype branch of Self-Evolving-Agent