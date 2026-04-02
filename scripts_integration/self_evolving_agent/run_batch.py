"""Subset-driven KernelBench runner using Self-Evolving-Agent memory framework."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from uuid import uuid4

try:
    from dotenv import load_dotenv
    # Search for .env in KernelBench and Self-Evolving-Agent dirs
    load_dotenv() # current dir
    load_dotenv(Path(__file__).parents[2] / ".env") # KernelBench root
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from self_evolving_agent.integrations.kernelbench.reflection import (
    SimpleKernelBenchReflectionEngine,
)
from self_evolving_agent.integrations.kernelbench.batch_runner import (
    load_subset_csv,
    to_level_first_entry,
    _read_json,
    _write_json,
)
from self_evolving_agent.integrations.kernelbench.environment import KernelBenchEnvironment
from self_evolving_agent.integrations.kernelbench.agent import KernelBenchEvolvingAgent
from self_evolving_agent.memory.schemas import (
    GlobalStrategyMemory as _GlobalStrategyMemory,
    LocalTrialSummary as _LocalTrialSummary,
)

# Local implementations for basic orchestration
class InMemoryLocalMemory:
    def __init__(self) -> None:
        self._entries: list[Any] = []
    def add_entry(self, entry: Any) -> None:
        self._entries.append(entry)
    def get_recent(self, limit: int = 10, task_id: Optional[str] = None) -> list[Any]:
        return [e for e in self._entries if (task_id is None or e.task_id == task_id)][-limit:]
    def flush(self, task_id: str) -> None: pass
    def add_raw_log(self, text: str) -> None: pass
    def finalize_task(self, score: float, success: bool) -> None: pass

class InMemoryGlobalMemory:
    def __init__(self) -> None:
        self._items: list[Any] = []
    def add_wisdom(self, entry: Any) -> None:
        self._items.append(entry)
    def retrieve_relevant(self, query: str, top_k: int = 5, min_utility: float = 0.0) -> list[Any]:
        return self._items[:top_k]
    def update_utility(self, memory_id: str, reward_signal: float) -> None: pass
    def find_similar(self, query: str, threshold_distance: float = 0.15) -> Optional[Any]:
        return None  # Dummy for dry-run compatibility

def run_subset(
    *, agent: Any, subset_rows: list[dict], output_path: str | Path, max_steps: int, backend: str, precision: str
) -> dict:
    output = Path(output_path)
    eval_doc = _read_json(output, default={})
    
    for row in subset_rows:
        level, pid = int(row["level"]), int(row["problem_id"])
        logger.info(f"Running L{level} P{pid}")
        try:
            result = agent.run_benchmark_task(
                f"L{level}_P{pid}", 
                {"level": level, "problem_id": pid, "max_steps": max_steps, "backend": backend, "precision": precision}
            )
        except Exception:
            logger.error(f"Failed L{level} P{pid}: {traceback.format_exc()}")
            result = {"compiled": False, "correctness": False, "error": traceback.format_exc()}

        l_key, p_key = str(level), str(pid)
        eval_doc.setdefault(l_key, {}).setdefault(p_key, []).append(
            to_level_first_entry(result, level=level, problem_id=pid)
        )
        _write_json(output, eval_doc)
    return eval_doc

def _build_coder_fn():
    from self_evolving_agent.llm.unified_client import UnifiedLLMClient
    client = UnifiedLLMClient(role="coder")

    def _coder(prompt: str) -> str:
        # UnifiedLLMClient.generate returns a string by default
        return client.generate(prompt)

    return _coder

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="subset_selection/selected_problems_50.csv")
    parser.add_argument("--output-path", default="runs_evolving/evolving_memory_integration_run/eval_results.json")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    # Require SEA modules to be importable; failures should be visible and not masked.

    subset_rows = load_subset_csv(args.subset_csv)
    local_mem, global_mem = InMemoryLocalMemory(), InMemoryGlobalMemory()

    env = KernelBenchEnvironment(backend=args.backend, precision=args.precision)
    agent_cls = KernelBenchEvolvingAgent
    reflex = SimpleKernelBenchReflectionEngine(local_memory=local_mem, global_memory=global_mem)

    agent = agent_cls(
        local_memory=local_mem,
        global_memory=global_mem,
        reflection_engine=reflex,
        environment=env,
        generate_code=_build_coder_fn(),
        max_steps=args.max_steps,
    )

    payload = run_subset(
        agent=agent, subset_rows=subset_rows, output_path=args.output_path,
        max_steps=args.max_steps, backend=args.backend, precision=args.precision
    )
    logger.info(f"Done. Results in {args.output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
