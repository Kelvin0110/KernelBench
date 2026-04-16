<!-- Canonical project structure. Update when files/folders change. -->
# Project Structure

This file documents the repository top-level layout and describes the purpose of key directories and files. Update this file in the same PR that modifies repository structure.

## How to update

- Run: `git diff --name-only origin/main...HEAD` to list changed files.
- For each affected top-level directory or notable file, add or edit a short section below (1 sentence + 1-3 example files).

---

## Top-level directories

- `src/` — Primary library code. Example files:
  - `src/kernelbench/compile.py` — compile-and-cache helpers.
  - `src/kernelbench/eval.py` — evaluation orchestration and helpers.

- `scripts/` — Utility scripts for running evaluations and helpers.

- `runs/` — Per-run outputs. Each run folder contains generated kernels and `eval_results.json`.

- `cache/` — Build cache for compiled CUDA extensions.

- `notebooks/` — Jupyter notebooks for inspection and analysis.

- `assets/` — Static assets such as figures and images.

- `docs/` — Design and planning artifacts for agent workflows and implementation specs.
  - `docs/superpowers/specs/` — design specifications (for example, L0/L1 memory-selection architecture docs).
  - `docs/superpowers/plans/` — executable implementation plans used by agentic workflows.

- `.github/skills/` — Skills and documentation templates used by agents and contributors.

Below are more detailed, repo-specific entries discovered in this repository (update when you add/remove subprojects):

- `KernelBench/` — Packaged dataset and canonical problem definitions (the KernelBench dataset tree used by the project). Example contents:
  - `KernelBench/level1/` — Problem definitions and reference implementations for Level 1 problems.
  - `KernelBench/level2/`, `KernelBench/level3/`, `KernelBench/level4/` — additional problem sets by difficulty.

- `aideml/` — A separate Python package included in the repo. Key files:
  - `aide_quick_test.py`, `README.md`, `TUTORIAL.md` — quickstart and tutorial materials.
  - `aide/` — package modules and agents (e.g., `agent.py`, `interpreter.py`, `run.py`).

- `scripts_integration/` — Integration helpers and test harnesses for external workflows:
  - `self_evolving_agent/` — integration harness and `run_batch.py` for the Self-Evolving Agent integration with KernelBench. Recent changes: the integration was consolidated into a single `run_batch.py` and legacy wrapper files were removed.
    - Example files: `scripts_integration/self_evolving_agent/run_batch.py`, `scripts_integration/self_evolving_agent/README.md`, `scripts_integration/self_evolving_agent/RUN_WITH_UV.md`
  - `new_evolving_agent/` — new KernelBench integration path using shared `evolving_common` governor helpers and recorder-backed metrics.
    - Example files: `scripts_integration/new_evolving_agent/kb_governor.py`, `scripts_integration/new_evolving_agent/evolve_kb_batch.py`, `scripts_integration/new_evolving_agent/RUN_WITH_UV.md`
  - `docker/` — Docker helpers and images for reproducible runs.
  - `evolving_agent/` — utilities and templates for evolving-agent experiments.

- `Self-Evolving-Agent/` — A separate subproject living inside the repo (agent prototype and tests). Notable contents:
  - `src/self_evolving_agent/` — agent core, integrations, memory backends.
  - `tests/` — shared unit tests for reusable memory/prompt infrastructure.
    - Example files: `Self-Evolving-Agent/tests/test_memory_manager.py`, `Self-Evolving-Agent/tests/test_prompt_context.py`
  - `visualizations/kernelbench/` — browser UI + FastAPI backend for per-problem inspection and run-level cached performance charts.
    - Example files: `Self-Evolving-Agent/visualizations/kernelbench/index.html`, `Self-Evolving-Agent/visualizations/kernelbench/server/app.py`, `Self-Evolving-Agent/visualizations/kernelbench/server/generate_run_performance_stats.py`
  - `documents/PROGRESS.md` — project progress ledger for that subproject.

- `analysis/` — Post-processing and benchmarking results, for example:
  - `analysis/SONG_CPU2_A6000x2/baseline_time_torch/` — baseline timing JSON snapshots used for reporting.

- `scripts/` — Utility scripts used by KernelBench evaluation harness (already listed above). Representative scripts:
  - `scripts/remove_oom_samples.py` — helper to remove OOM entries from `eval_results.json`.
  - `scripts/generate_samples.py`, `scripts/generate_baseline_time.py` — generation and baseline tooling.

## Recommended maintenance notes

- When adding or removing a top-level subproject (e.g., `aideml/`, `Self-Evolving-Agent/`, `KernelBench/`), add or remove the corresponding section above.
- Keep each directory description to one sentence plus 1–3 representative files.
- Prefer updating `documents/PROJECT_STRUCTURE.md` in the same PR that introduces structural changes.

## Important files

- `README.md` — Project overview and getting-started instructions.
- `requirements.txt` / `pyproject.toml` — Python dependencies and packaging metadata.
- `run_evals.sh` — Convenience script to start evaluations.

---

When adding new top-level directories or moving significant code, add a short entry above describing the directory and list representative files.
