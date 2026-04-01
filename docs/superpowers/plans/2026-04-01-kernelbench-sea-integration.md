# KernelBench Self-Evolving-Agent (SEA) Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the SEA dual-memory system into KernelBench as a standalone evaluation track that handles batch tasks, leverages existing `KernelBench` evaluation harnesses, and produces standardized JSON benchmark metrics.

**Architecture:** We use a "Harness Wrapper" approach. The `KernelBenchEnvironment` bridges existing `eval.py` scripts to standard environment payload conventions. The `KernelBenchEvolvingAgent` sub-classes the `SelfEvolvingAgent` core to run iterative memory-augmented generation loops. Finally, `run_batch.py` dispatches multiple kernel problems to this system and consolidates metrics into `eval_results.json`.

**Tech Stack:** Python, PyTorch, Triton/CUDA, ChromaDB (via SEA memory module).

---

### Task 1: Create the KernelBench Environment Adapter

**Files:**
- Create: `scripts_integration/self_evolving_agent/kb_environment.py`

- [ ] **Step 1: Write the minimal environment class**

```python
import traceback
from typing import Any, Dict, Tuple
from kernelbench.dataset import get_problem
from kernelbench.prompts.prompt_constructor_toml import generate_prompt
from kernelbench.eval import run_and_check_code

class KernelBenchEnvironment:
    def __init__(self, backend: str = "cuda", precision: str = "fp32", device: str = "cuda:0"):
        self.backend = backend
        self.precision = precision
        self.device = device
    
    def get_prompt_for_problem(self, level: int, problem_id: int) -> str:
        # We rely on existing kernelbench dataset and prompt generation logic
        problem = get_problem("local", level, problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} at level {level} not found")
        prompt = generate_prompt(problem, None, self.backend, self.precision)
        return prompt
        
    def evaluate_kernel(self, level: int, problem_id: int, generated_code: str) -> Tuple[bool, float, str, Dict[str, Any]]:
        # Evaluate using standard internal tool
        try:
            result = run_and_check_code(
                eval_mode="local",
                ref_origin="local",
                level=level,
                problem_id=problem_id,
                backend=self.backend,
                precision=self.precision,
                candidate_source=generated_code,
                device=self.device
            )
            
            # Unpack results standard from run_and_check_code
            correctness = result.get("correctness", False)
            speedup = result.get("best_speedup", 0.0)
            
            # Format error feedback if failed
            feedback = ""
            if not correctness:
                metadata = result.get("metadata", {})
                if "runtime_error" in metadata:
                    feedback = metadata["runtime_error"]
                elif "correctness_trials" in metadata:
                    feedback = f"Failed correctness tests. Trials: {metadata['correctness_trials']}"
                else:
                    feedback = "Unknown validation failure."
            
            return correctness, speedup, feedback, result
        except Exception as e:
            return False, 0.0, f"Critical harness failure:\n{traceback.format_exc()}", {}
```

- [ ] **Step 2: Commit**

```bash
git add scripts_integration/self_evolving_agent/kb_environment.py
git commit -m "feat(sea-integration): Add KernelBenchEnvironment adapter"
```

### Task 2: Create the KernelBench Evolving Agent Subclass

**Files:**
- Create: `scripts_integration/self_evolving_agent/kb_agent.py`

- [ ] **Step 1: Write the SEA subclass implementation**

```python
from typing import Any, Dict
import litellm

from self_evolving_agent.agent.core import SelfEvolvingAgent
from self_evolving_agent.agent.iteration import IterationController
from scripts_integration.self_evolving_agent.kb_environment import KernelBenchEnvironment

class KernelBenchEvolvingAgent(SelfEvolvingAgent):
    def __init__(self, local_memory, global_memory, reflection_engine, environment, llm_client=None):
        super().__init__(local_memory, global_memory, reflection_engine)
        self.env = environment
        self.llm_client = llm_client

    def run_benchmark_task(self, task_id: str, challenge_data: Dict[str, Any]) -> Dict[str, Any]:
        level = challenge_data["level"]
        problem_id = challenge_data["problem_id"]
        max_steps = challenge_data.get("max_steps", 5)
        
        # 1. Fetch Problem Prompt
        base_prompt = self.env.get_prompt_for_problem(level, problem_id)
        
        # 2. Get Global Memory Strategies
        task_context = {"level": level, "problem_id": problem_id, "prompt": base_prompt}
        strategy_contexts = self.get_global_prompt_context(
            environment_state=f"KernelBench Level {level}, problem {problem_id}",
            top_k=3
        )
        strategies_str = "\n".join(strategy_contexts)
        
        # Initialize loop variables
        best_result = None
        best_speedup = 0.0
        success = False
        
        for step in range(max_steps):
            # 3. Format Iteration Prompt with Local Memory (Trace)
            local_context = self.get_local_prompt_context(task_id=task_id)
            
            llm_prompt = f"{base_prompt}\n\n"
            if strategies_str:
                llm_prompt += f"--- GLOBAL STRATEGIES ---\n{strategies_str}\n\n"
            if local_context:
                llm_prompt += f"--- PREVIOUS ATTEMPTS & ITERATIONS ---\n{local_context}\n\n"
                llm_prompt += "Fix the errors and optimize the kernel:\n"
            
            # Generate Code (Using LiteLLM directly or through a provided client block)
            # Assuming self.llm_client provides a .generate(prompt) method or using a fallback
            try:
                if self.llm_client:
                    response = self.llm_client.generate(llm_prompt)
                else:
                    response = litellm.completion(
                        model="openai/gpt-4o",  # Fallback model
                        messages=[{"role": "user", "content": llm_prompt}]
                    ).choices[0].message.content
            except Exception as e:
                response = f"LLM Error: {str(e)}"
            
            # Simple code extraction block
            generated_code = response
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            
            # 4. Evaluate Code
            is_correct, speedup, feedback, raw_eval_metadata = self.env.evaluate_kernel(level, problem_id, generated_code)
            
            # Log local
            trace_content = f"Attempt {step}\nFeedback: {feedback}\nSpeedup: {speedup}\n"
            self.local_memory.add_trace(task_id=task_id, execution_step=step, content=trace_content)
            
            if is_correct and speedup > best_speedup:
                best_speedup = speedup
                best_result = raw_eval_metadata
                best_result["metadata"]["iterations_run"] = step + 1
            
            if is_correct:
                success = True
                break  # Stop early if correct (or we can keep iterating for better speedup based on logic)
                
        # 5. Extract global insight via Reflection after loop finishes
        # (Assuming local traces are captured for the task)
        self.reflection_engine.reflect_and_distil(task_id=task_id)
        
        # If no result passed throughout iterations, return the last raw_eval_metadata
        if best_result is None:
            best_result = raw_eval_metadata
            best_result["metadata"]["iterations_run"] = max_steps
            
        return best_result
```

- [ ] **Step 2: Commit**

```bash
git add scripts_integration/self_evolving_agent/kb_agent.py
git commit -m "feat(sea-integration): Add KernelBench evolving agent core logic"
```

### Task 3: Create the Batch Runner Script

**Files:**
- Create: `scripts_integration/self_evolving_agent/run_batch.py`

- [ ] **Step 1: Write the batch dispatcher script**

```python
import os
import json
import argparse
from typing import Dict, Any

from self_evolving_agent.memory.local_file_memory import FileBasedLocalMemory
from self_evolving_agent.memory.chroma_backend import ChromaGlobalMemory
from self_evolving_agent.memory.reflection import DummyReflectionEngine  # or specific implementation
from scripts_integration.self_evolving_agent.kb_environment import KernelBenchEnvironment
from scripts_integration.self_evolving_agent.kb_agent import KernelBenchEvolvingAgent

def load_subset(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return [(row['level'], row['problem_id']) for _, row in df.iterrows()]

def main():
    parser = argparse.add_argument_group("Self Evolving Agent Runner")
    parser.add_argument("--subset_csv", type=str, default="subset_selection/selected_problems_50.csv")
    parser.add_argument("--output_path", type=str, default="runs/sea_integration_run/eval_results.json")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--backend", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Initialize SEA Components
    local_mem = FileBasedLocalMemory(log_dir=os.path.join(os.path.dirname(args.output_path), "local_logs"))
    global_mem = ChromaGlobalMemory(persist_directory=os.path.join(os.path.dirname(args.output_path), "chroma_db"))
    reflection = DummyReflectionEngine(local_memory=local_mem, global_memory=global_mem)
    
    env = KernelBenchEnvironment(backend=args.backend)
    agent = KernelBenchEvolvingAgent(
        local_memory=local_mem,
        global_memory=global_mem,
        reflection_engine=reflection,
        environment=env
    )

    tasks = load_subset(args.subset_csv)
    
    # Group results heavily matched to eval.py structures
    eval_results: Dict[str, Dict[str, list]] = {}
    
    for level, problem_id in tasks:
        task_id = f"L{level}_P{problem_id}"
        print(f"Starting {task_id}")
        
        result_metadata = agent.run_benchmark_task(
            task_id=task_id, 
            challenge_data={"level": level, "problem_id": problem_id, "max_steps": args.max_steps}
        )
        
        # Inject custom SEA metadata
        result_metadata["metadata"]["source"] = "sea_prototype"
        
        lvl_str = str(level)
        prob_str = str(problem_id)
        
        if lvl_str not in eval_results:
            eval_results[lvl_str] = {}
        if prob_str not in eval_results[lvl_str]:
            eval_results[lvl_str][prob_str] = []
            
        eval_results[lvl_str][prob_str].append(result_metadata)
        
        # Continuously save logic
        with open(args.output_path, "w") as f:
            json.dump(eval_results, f, indent=4)
            
    print(f"Run completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git chmod +x scripts_integration/self_evolving_agent/run_batch.py
git add scripts_integration/self_evolving_agent/run_batch.py
git commit -m "feat(sea-integration): Add batch runner for KernelBench evaluating subset"
```
