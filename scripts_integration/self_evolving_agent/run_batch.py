"""Subset-driven KernelBench runner using Self-Evolving-Agent memory framework."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_ROOT = _REPO_ROOT / "Self-Evolving-Agent"
_SEA_SRC = _SEA_ROOT / "src"

if str(_SEA_SRC) not in sys.path:
    sys.path.insert(0, str(_SEA_SRC))
if str(_SEA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEA_ROOT))

# Import modular components from SEA core
from self_evolving_agent.integrations.kernelbench.reflection import (  # noqa: E402
    SimpleKernelBenchReflectionEngine,
)
from self_evolving_agent.integrations.kernelbench.batch_runner import (  # noqa: E402
    load_subset_csv,
    to_level_first_entry,
    _read_json,
    _write_json,
)

KernelBenchEnvironment = None
KernelBenchEvolvingAgent = None


def _ensure_runtime_dependencies() -> None:
    global KernelBenchEnvironment
    global KernelBenchEvolvingAgent

    if KernelBenchEnvironment is None:
        from scripts_integration.self_evolving_agent.kb_environment import (  # noqa: WPS433,E402
            KernelBenchEnvironment as _Env,
        )

        KernelBenchEnvironment = _Env
    if KernelBenchEvolvingAgent is None:
        from scripts_integration.self_evolving_agent.kb_agent import (  # noqa: WPS433,E402
            KernelBenchEvolvingAgent as _Agent,
        )

        KernelBenchEvolvingAgent = _Agent


def run_subset(
    *,
    agent,
    subset_rows: list[dict[str, int]],
    output_path: str | Path,
    max_steps: int,
    backend: str,
    precision: str,
) -> dict[str, dict[str, list[dict]]]:
    output = Path(output_path)
    existing = _read_json(output, default={})
    eval_doc: dict[str, dict[str, list[dict]]] = existing if isinstance(existing, dict) else {}

    for row in subset_rows:
        level = int(row["level"])
        problem_id = int(row["problem_id"])
        task_id = f"L{level}_P{problem_id}"
        try:
            result = agent.run_benchmark_task(
                task_id=task_id,
                challenge_data={
                    "level": level,
                    "problem_id": problem_id,
                    "max_steps": max_steps,
                    "backend": backend,
                    "precision": precision,
                },
            )
        except Exception as exc:
            result = {
                "sample_id": 0,
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "metadata": {
                    "backend": backend,
                    "precision": precision,
                    "iterations_run": 0,
                    "best_speedup": 0.0,
                    "error": (
                        f"task_failed: {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}"
                    ),
                },
            }

        if not isinstance(result, dict):
            result = {
                "sample_id": 0,
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "metadata": {
                    "backend": backend,
                    "precision": precision,
                    "iterations_run": 0,
                    "best_speedup": 0.0,
                    "error": f"unexpected_result_type: {type(result).__name__}",
                },
            }

        level_key = str(level)
        pid_key = str(problem_id)
        eval_doc.setdefault(level_key, {})
        eval_doc[level_key].setdefault(pid_key, [])
        eval_doc[level_key][pid_key].append(
            to_level_first_entry(result, level=level, problem_id=problem_id)
        )

        _write_json(output, eval_doc)

    return eval_doc


@dataclass
class _LocalTrialSummaryLite:
    id: str
    timestamp: float
    content: str
    metadata: dict
    task_id: str
    execution_step: int


@dataclass
class _GlobalStrategyMemoryLite:
    id: str
    timestamp: float
    content: str
    metadata: dict
    applicability_scope: list[str]
    utility_score: float
    invocation_count: int


try:
    from self_evolving_agent.memory.schemas import GlobalStrategyMemory as _GlobalStrategyMemory  # noqa: E402
    from self_evolving_agent.memory.schemas import LocalTrialSummary as _LocalTrialSummary  # noqa: E402
except Exception:
    _GlobalStrategyMemory = _GlobalStrategyMemoryLite
    _LocalTrialSummary = _LocalTrialSummaryLite


CODER_SYSTEM_PROMPT = """You are an expert GPU kernel engineer solving KernelBench optimization tasks.

Output rules:
1. Return exactly one fenced Python code block.
2. The code block must define ModelNew implementing behavior equivalent to the reference model.
3. Prioritize correctness first, then speed optimization.
4. Do not include explanations outside the code block.
"""


class InMemoryLocalMemory:
    """Simple task-scoped local memory implementation for integration runner."""

    def __init__(self) -> None:
        self._entries: list[_LocalTrialSummary] = []
        self._raw_logs: list[str] = []

    def add_entry(self, entry: _LocalTrialSummary) -> None:
        self._entries.append(entry)

    def get_recent(self, limit: int = 10, task_id: Optional[str] = None) -> list[_LocalTrialSummary]:
        entries = self._entries
        if task_id is not None:
            entries = [e for e in entries if e.task_id == task_id]
        if limit <= 0:
            return list(entries)
        return entries[-limit:]

    def flush(self, task_id: str) -> None:
        self._entries = [e for e in self._entries if e.task_id != task_id]

    def add_raw_log(self, text: str) -> None:
        self._raw_logs.append(text)

    def flush_and_summarize(self) -> None:
        if not self._raw_logs:
            return
        summary = _LocalTrialSummary(
            id=f"local-summary-{uuid4().hex}",
            timestamp=time.time(),
            content="\n".join(self._raw_logs)[-2000:],
            metadata={"source": "raw_log_buffer"},
            task_id="raw_log",
            execution_step=0,
        )
        self.add_entry(summary)
        self._raw_logs.clear()

    def finalize_task(self, score: float, success: bool) -> None:
        summary = _LocalTrialSummary(
            id=f"final-{uuid4().hex}",
            timestamp=time.time(),
            content=f"finalize score={score:.6f} success={success}",
            metadata={"final": True},
            task_id="final",
            execution_step=0,
        )
        self.add_entry(summary)


class InMemoryGlobalMemory:
    """Simple global strategy store for script runner."""

    def __init__(self) -> None:
        self._items: list[_GlobalStrategyMemory] = []

    def add_wisdom(self, entry: _GlobalStrategyMemory) -> None:
        self._items.append(entry)

    def retrieve_relevant(self, query: str, top_k: int = 5, min_utility: float = 0.0) -> list[_GlobalStrategyMemory]:
        _ = query
        candidates = [x for x in self._items if x.utility_score >= min_utility]
        candidates.sort(key=lambda x: x.utility_score, reverse=True)
        return candidates[:top_k]

    def update_utility(self, memory_id: str, reward_signal: float) -> None:
        for item in self._items:
            if item.id != memory_id:
                continue
            item.utility_score = item.utility_score + 0.1 * (reward_signal - item.utility_score)
            item.invocation_count += 1
            return

    def find_similar(self, query: str, threshold_distance: float = 0.15) -> Optional[_GlobalStrategyMemory]:
        _ = threshold_distance
        query_lower = query.lower()
        for item in self._items:
            if query_lower in item.content.lower() or item.content.lower() in query_lower:
                return item
        return None

    def update_wisdom(self, entry: _GlobalStrategyMemory) -> None:
        for idx, item in enumerate(self._items):
            if item.id == entry.id:
                self._items[idx] = entry
                return


def _build_coder_fn(*, dry_run: bool, dry_run_template_path: str | None):
    if dry_run:
        template = None
        if dry_run_template_path:
            template = Path(dry_run_template_path).read_text(encoding="utf-8")

        def _dry_run(prompt: str) -> str:
            _ = prompt
            if template:
                return template
            return """```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *inputs):
        raise RuntimeError(\"dry-run placeholder ModelNew\")
```
"""

        return _dry_run

    def _coder(prompt: str) -> str:
        from llm_client import call_coder

        raw, _ = call_coder(
            [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return raw or ""

    return _coder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Self-Evolving-Agent on a KernelBench subset")
    parser.add_argument(
        "--subset-csv",
        type=str,
        default="subset_selection/selected_problems_50.csv",
        help="CSV containing level/problem_id columns",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="runs/sea_integration_run/eval_results.json",
        help="Output JSON path for eval results",
    )
    parser.add_argument("--backend", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--dataset-source", type=str, default="local")
    parser.add_argument("--prompt-option", type=str, default="one_shot")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--global-top-k", type=int, default=3)
    parser.add_argument("--min-utility", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and use a placeholder generator")
    parser.add_argument(
        "--dry-run-template-path",
        type=str,
        default=None,
        help="Optional path to python code used as dry-run candidate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    subset_rows = load_subset_csv(args.subset_csv)

    local_memory = InMemoryLocalMemory()
    global_memory = InMemoryGlobalMemory()
    try:
        _ensure_runtime_dependencies()
    except Exception:
        if not args.dry_run:
            raise

    reflection_engine = SimpleKernelBenchReflectionEngine(
        local_memory=local_memory,
        global_memory=global_memory,
    )

    if KernelBenchEnvironment is None or KernelBenchEvolvingAgent is None:
        raise RuntimeError(
            "KernelBench integration dependencies are unavailable. "
            "Install Self-Evolving-Agent dependencies or run with --dry-run and monkeypatched dependencies."
        )

    environment = KernelBenchEnvironment(
        dataset_source=args.dataset_source,
        backend=args.backend,
        precision=args.precision,
        prompt_option=args.prompt_option,
    )

    coder_fn = _build_coder_fn(dry_run=args.dry_run, dry_run_template_path=args.dry_run_template_path)

    agent = KernelBenchEvolvingAgent(
        local_memory=local_memory,
        global_memory=global_memory,
        reflection_engine=reflection_engine,
        environment=environment,
        generate_code=coder_fn,
        max_steps=args.max_steps,
        global_top_k=args.global_top_k,
        min_utility=args.min_utility,
        stop_on_first_correct=True,
    )

    payload = run_subset(
        agent=agent,
        subset_rows=subset_rows,
        output_path=args.output_path,
        max_steps=args.max_steps,
        backend=args.backend,
        precision=args.precision,
    )

    print(json.dumps({"output_path": args.output_path, "levels": sorted(payload.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
