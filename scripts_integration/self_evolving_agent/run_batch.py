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

from self_evolving_agent.memory.local_file_memory import JSONLinesLocalMemory
from self_evolving_agent.memory.chroma_backend import ChromaGlobalMemory
from self_evolving_agent.llm.unified_client import UnifiedLLMClient
from self_evolving_agent.integrations.kernelbench.reflection import (
    SimpleKernelBenchReflectionEngine,
)
from self_evolving_agent.integrations.kernelbench.batch_runner import (
    load_subset_csv,
    run_subset,
)
from self_evolving_agent.integrations.kernelbench.environment import KernelBenchEnvironment
from self_evolving_agent.integrations.kernelbench.agent import KernelBenchEvolvingAgent

def _build_coder_fn():
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
    parser.add_argument("--persist-directory", default="./chroma_db")
    parser.add_argument("--local-memory-dir", default=".agent_memory/local_runs")
    args = parser.parse_args()

    subset_rows = load_subset_csv(args.subset_csv)
    
    # Initialize Persistent Global Memory
    global_mem = ChromaGlobalMemory(persist_directory=args.persist_directory)
    
    # Initialize LLM Client for reflection and local memory summarization
    coder_client = UnifiedLLMClient(role="coder")
    
    env = KernelBenchEnvironment(backend=args.backend, precision=args.precision)

    from self_evolving_agent.integrations.kernelbench.batch_runner import _read_json, _write_json, to_level_first_entry

    output = Path(args.output_path)
    eval_doc = _read_json(output, default={})
    
    for row in subset_rows:
        level, pid = int(row["level"]), int(row["problem_id"])
        task_id = f"L{level}_P{pid}"
        logger.info(f"Running {task_id}")
        
        # Fresh Local Memory for each task
        local_mem = JSONLinesLocalMemory(
            task_id=task_id, 
            llm_client=coder_client,
            base_dir=args.local_memory_dir
        )
        
        reflex = SimpleKernelBenchReflectionEngine(local_memory=local_mem, global_memory=global_mem)
        
        agent = KernelBenchEvolvingAgent(
            local_memory=local_mem,
            global_memory=global_mem,
            reflection_engine=reflex,
            environment=env,
            generate_code=_build_coder_fn(),
            max_steps=args.max_steps,
        )
        
        try:
            result = agent.run_benchmark_task(
                task_id, 
                {"level": level, "problem_id": pid, "max_steps": args.max_steps, "backend": args.backend, "precision": args.precision}
            )
        except Exception:
            logger.error(f"Failed {task_id}: {traceback.format_exc()}")
            result = {"compiled": False, "correctness": False, "error": traceback.format_exc()}

        l_key, p_key = str(level), str(pid)
        eval_doc.setdefault(l_key, {}).setdefault(p_key, []).append(
            to_level_first_entry(result, level=level, problem_id=pid)
        )
        _write_json(output, eval_doc)

    logger.info(f"Done. Results in {args.output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
