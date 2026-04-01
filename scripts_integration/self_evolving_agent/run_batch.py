"""Subset-driven KernelBench runner using Self-Evolving-Agent memory framework."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Try to import modular components from SEA core
try:
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
    HAS_SEA = True
except ImportError:
    HAS_SEA = False

    # Minimal fallbacks for dry-run mode
    def load_subset_csv(path: str | Path) -> list[dict[str, int]]:
        rows: list[dict[str, int]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    rows.append({"level": int(row["level"]), "problem_id": int(row["problem_id"])})
        return rows

    def _read_json(path: Path, default: Any):
        if not path.is_file(): return default
        with path.open("r", encoding="utf-8") as f: return json.load(f)

    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

    def to_level_first_entry(run_result: dict, *, level: int, problem_id: int) -> dict:
        metadata = run_result.get("metadata", {})
        return {
            "sample_id": int(run_result.get("sample_id", 0)),
            "compiled": bool(run_result.get("compiled", False)),
            "correctness": bool(run_result.get("correctness", False)),
            "metadata": {**metadata, "level": level, "problem_id": problem_id},
            "runtime": float(run_result.get("runtime", -1.0)),
            "runtime_stats": run_result.get("runtime_stats", {}),
        }

    class SimpleKernelBenchReflectionEngine:
        def __init__(self, **kwargs): pass
        def reflect_now(self, task_id: str): pass

    @dataclass
    class _GlobalStrategyMemory:
        id: str; timestamp: float; content: str; metadata: dict; utility_score: float

    @dataclass
    class _LocalTrialSummary:
        id: str; timestamp: float; content: str; metadata: dict; task_id: str; execution_step: int

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

def _build_coder_fn(dry_run: bool, dry_run_template_path: Optional[str]):
    if dry_run:
        code = Path(dry_run_template_path).read_text() if dry_run_template_path else "import torch\nclass ModelNew(torch.nn.Module): pass"
        return lambda prompt: f"```python\n{code}\n```"
    
    def _coder(prompt: str) -> str:
        from llm_client import call_coder
        raw, _ = call_coder([{"role": "user", "content": prompt}])
        return raw or ""
    return _coder

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="subset_selection/selected_problems_50.csv")
    parser.add_argument("--output-path", default="runs/sea_integration_run/eval_results.json")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-template-path", default=None)
    args = parser.parse_args()

    if not HAS_SEA and not args.dry_run:
        logger.error("SEA dependencies missing. Run with --dry-run or install via 'uv sync --extra evolving-agent'")
        return 1

    subset_rows = load_subset_csv(args.subset_csv)
    local_mem, global_mem = InMemoryLocalMemory(), InMemoryGlobalMemory()
    
    if HAS_SEA:
        env = KernelBenchEnvironment(backend=args.backend, precision=args.precision)
        agent_cls = KernelBenchEvolvingAgent
        reflex = SimpleKernelBenchReflectionEngine(local_memory=local_mem, global_memory=global_mem)
    else:
        # Minimal mock for dry-run
        class MockEnv:
            def __init__(self, **kwargs): pass
            def build_prompt(self, l, p): return "mock"
            def evaluate_candidate(self, **kwargs): return {"correctness": False}
        class MockAgent:
            def __init__(self, **kwargs): self.env = kwargs["environment"]; self.gen = kwargs["generate_code"]
            def run_benchmark_task(self, tid, data):
                self.gen(self.env.build_prompt(data["level"], data["problem_id"]))
                return {"compiled": True, "correctness": False, "runtime": 0.1}
        env, agent_cls, reflex = MockEnv(), MockAgent, SimpleKernelBenchReflectionEngine()

    agent = agent_cls(
        local_memory=local_mem, global_memory=global_mem, reflection_engine=reflex,
        environment=env, generate_code=_build_coder_fn(args.dry_run, args.dry_run_template_path),
        max_steps=args.max_steps
    )

    payload = run_subset(
        agent=agent, subset_rows=subset_rows, output_path=args.output_path,
        max_steps=args.max_steps, backend=args.backend, precision=args.precision
    )
    logger.info(f"Done. Results in {args.output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
