# Evolving Agent Save Logic and Iteration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the KernelBench evolving agent to evaluate all defined steps (disabling early stopping) and structured logging output matching the required directory hierarchy.

**Architecture:** We will change the default value of `stop_on_first_correct` in `KernelBenchEvolvingAgent` to `False`. To support structured saving, we will modify `KernelBenchEvolvingAgent.run_benchmark_task` to return the `candidate_code` and full `prompts` / `feedback` trace for the best execution. Then, `run_subset` in `batch_runner.py` will be updated to write these artifacts into the `eval_results.json`, `config.yaml`, `kernels/`, and `logs/` directory hierarchy.

**Tech Stack:** Python, pathlib, json, yaml

---

### Task 1: Disable `stop_on_first_correct` by Default

**Files:**
- Modify: `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py`

- [ ] **Step 1: Set `stop_on_first_correct` to False**

```python
# In KernelBenchEvolvingAgent.__init__

    def __init__(
        self,
        *,
        local_memory,
        global_memory,
        reflection_engine,
        environment: KernelBenchEnvironment,
        generate_code: Callable[[str], str],
        max_steps: int = 5,
        global_top_k: int = 3,
        min_utility: float = 0.5,
        stop_on_first_correct: bool = False, # Change this from True to False
    ) -> None:
        super().__init__(local_memory, global_memory, reflection_engine)
```

- [ ] **Step 2: Commit**

```bash
git add Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py
git commit -m "fix: set stop_on_first_correct default to False"
```

### Task 2: Capture Source Code and Iteration Logs in Agent Payload

**Files:**
- Modify: `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py`

We need to return the generated code and the prompt traces so the batch runner can save them.

- [ ] **Step 1: Add execution traces to the return payload**

```python
# In KernelBenchEvolvingAgent.run_benchmark_task

        best_payload: dict[str, Any] | None = None
        best_speedup = -1.0
        last_feedback = ""
        iteration_logs: list[dict[str, str]] = [] # Initialize this above the for loop

        for step_idx in range(max_steps):
            # ... existing prompt composition ...
            prompt = self._compose_prompt(
                base_prompt=base_prompt,
                strategies=strategies,
                local_context=local_context,
                last_feedback=last_feedback,
            )

            raw_output = self.generate_code(prompt)
            candidate_code = self._extract_python_code(raw_output)
            
            # Log the round
            iteration_logs.append({
                "step": step_num,
                "prompt": prompt,
                "raw_response": raw_output,
            })
            
            # ... existing evaluate_candidate ...
            
            # ... existing internal tracking ...
            
            candidate_payload = {
                "sample_id": 0,
                "compiled": outcome.compiled,
                "correctness": outcome.correctness,
                "runtime": outcome.runtime,
                "runtime_stats": outcome.runtime_stats,
                "source_code": candidate_code, # Add source code
                "metadata": {
                    # ... existing metadata updates ...
                }
            }
            iteration_logs[-1]["feedback"] = outcome.feedback # Append feedback after evaluation

            # Update best_payload condition to keep iteration trace and code
            if outcome.correctness and outcome.speedup >= best_speedup:
                best_speedup = outcome.speedup
                best_payload = candidate_payload
                
            # ... existing break and default set ...

        if best_payload is None:
            # ... existing default payload ...
            best_payload["source_code"] = ""
            
        best_payload["iteration_logs"] = iteration_logs # Add logs to final return

        # ... existing reflection calls ...
        return best_payload
```

- [ ] **Step 2: Commit**

```bash
git add Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/agent.py
git commit -m "feat: include source code and iteration logs in agent return payload"
```

### Task 3: Update Save Logic in Batch Runner

**Files:**
- Modify: `Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/batch_runner.py`

Update `run_subset` to save the structured folder hierarchy.

- [ ] **Step 1: Import yaml and construct directory logic**

```python
# Ensure yaml is imported at the top of batch_runner.py
import yaml
from pathlib import Path
# ... existing imports ...
```

- [ ] **Step 2: Intercept the result and write files**

```python
# In batch_runner.py inside run_subset

    output = Path(output_path)
    output_dir = output.parent
    kernels_dir = output_dir / "kernels"
    logs_dir = output_dir / "logs"
    
    kernels_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    existing = _read_json(output, default={})
    eval_doc: dict[str, dict[str, list[dict[str, Any]]]] = existing if isinstance(existing, dict) else {}

    for row in subset_rows:
        # ... existing run_benchmark_task call wrapped in try/except ...
        
        # New disk saving logic for the structured output
        source_code = result.get("source_code", "")
        iteration_logs = result.get("iteration_logs", [])
        
        # 1. Save Best Kernel
        kernel_filename = f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
        kernel_file = kernels_dir / kernel_filename
        kernel_file.write_text(source_code, encoding="utf-8")
        
        # 2. Create Log Directory
        task_log_dir = logs_dir / f"level_{level}_problem_{problem_id}"
        task_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Save config.yaml
        config_data = {
            "level": level,
            "problem_id": problem_id,
            "max_steps": max_steps,
            "backend": backend,
            "precision": precision,
            "best_speedup": result.get("metadata", {}).get("best_speedup", 0.0),
            "correctness": result.get("correctness", False)
        }
        with open(task_log_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, indent=2)
            
        # 4. Save best solution in logs
        (task_log_dir / "best_solution.py").write_text(source_code, encoding="utf-8")
        
        # 5. Save iteration logs
        with open(task_log_dir / "iteration_logs.json", "w", encoding="utf-8") as f:
            json.dump(iteration_logs, f, indent=2)

        # Remove heavy blobs before writing eval_results.json
        if "source_code" in result:
            del result["source_code"]
        if "iteration_logs" in result:
            del result["iteration_logs"]

        # ... existing eval_doc append and _write_json ...
```

- [ ] **Step 3: Commit**

```bash
git add Self-Evolving-Agent/src/self_evolving_agent/integrations/kernelbench/batch_runner.py
git commit -m "feat: output best kernel, iteration logs, and config.yaml in batch runner"
```
