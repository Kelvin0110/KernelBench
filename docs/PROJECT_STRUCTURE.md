Project structure notes (automatically updated)

Summary of recent changes (features/evolving-agent branch):

- Consolidated SEA integration scripts into `scripts_integration/self_evolving_agent/run_batch.py`.
- Removed redundant wrapper files: `scripts_integration/self_evolving_agent/kb_agent.py` and `scripts_integration/self_evolving_agent/kb_environment.py`.
- `run_batch.py` now imports `KernelBenchEvolvingAgent` and `KernelBenchEnvironment` directly from the installed `self_evolving_agent` package when available, and falls back to minimal mocks only for `--dry-run` mode.

Notes for maintainers:

- If you run via `uv run`, ensure `uv sync --extra evolving-agent` has been executed so the editable `Self-Evolving-Agent` is available in the environment.
- For local development without installing `self-evolving-agent`, use `--dry-run` to exercise runner orchestration without external dependencies.
- The canonical SEA code resides in `Self-Evolving-Agent/src/self_evolving_agent/` and should be the source of truth for integration behavior.

Files changed in this update:

- Modified: `scripts_integration/self_evolving_agent/run_batch.py`
- Deleted: `scripts_integration/self_evolving_agent/kb_agent.py`
- Deleted: `scripts_integration/self_evolving_agent/kb_environment.py`
- Added: `scripts_integration/self_evolving_agent/README.md`
- Updated: `scripts_integration/self_evolving_agent/RUN_WITH_UV.md`

This file is intended to be a short, append-only note for reviewers and maintainers; keep it minimal and update when the integration layout changes.
