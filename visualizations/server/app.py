from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os

app = FastAPI(title="KernelBench Evolution Visualizer")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory for evolving logs - ADJUST THIS IF NEEDED
RESULTS_ROOT = Path("/home/kwtamai/KernelBench/results")
RUNS_EVOLVING_ROOT = Path("/home/kwtamai/KernelBench/runs_evolving")

def get_valid_run_dirs():
    """Find directories containing valid evolving agent data."""
    valid_dirs = []
    # Check both potential root locations
    roots = [RESULTS_ROOT, RUNS_EVOLVING_ROOT]
    for root in roots:
        if not root.exists():
            continue
        # Scan subdirectories of each root
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            
            # Check if this folder itself has a workspaces subdir
            # OR if it's a category folder (like evolving_logs used to be)
            # We'll check for 'workspaces' in all first-level subdirs.
            workspaces_path = entry / "workspaces"
            if workspaces_path.exists() and workspaces_path.is_dir():
                valid_dirs.append({
                    "name": entry.name,
                    "path": str(entry.absolute()),
                    "root": str(root.name)
                })
    return valid_dirs

def is_valid_problem_dir(p_dir: Path) -> bool:
    """Check if a problem directory has all 4 required files."""
    required = [
        "chat_history.jsonl",
        "terminal_output.jsonl",
        "snapshots.jsonl",
        "metrics_by_iteration.jsonl"
    ]
    return all((p_dir / f).exists() for f in required)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

@app.get("/api/runs")
async def list_runs():
    return get_valid_run_dirs()

@app.get("/api/runs/{run_name}/problems")
async def list_problems(run_name: str, root_name: str = "results"):
    # Determine the actual path
    base_root = RESULTS_ROOT if root_name == "evolving_logs" else RUNS_EVOLVING_ROOT
    run_dir = base_root / run_name
    workspaces_path = run_dir / "workspaces"
    
    if not workspaces_path.exists():
        raise HTTPException(status_code=404, detail="Workspaces not found")
    
    problems = []
    for p_dir in workspaces_path.iterdir():
        if p_dir.is_dir() and is_valid_problem_dir(p_dir):
            problems.append({
                "id": p_dir.name,
                "name": p_dir.name.replace("_", " ").title()
            })
    return sorted(problems, key=lambda x: x["id"])

@app.get("/api/runs/{run_name}/problems/{problem_id}/iterations")
async def get_iterations(run_name: str, problem_id: str, root_name: str = "results"):
    base_root = RESULTS_ROOT if root_name == "evolving_logs" else RUNS_EVOLVING_ROOT
    p_dir = base_root / run_name / "workspaces" / problem_id
    
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Problem directory not found")
        
    chat = read_jsonl(p_dir / "chat_history.jsonl")
    terminal = read_jsonl(p_dir / "terminal_output.jsonl")
    snapshots = read_jsonl(p_dir / "snapshots.jsonl")
    metrics = read_jsonl(p_dir / "metrics_by_iteration.jsonl")
    
    # Simple join by iteration index if available, or just index them 
    iterations = []
    max_iters = len(metrics) # Primary driver
    
    for i in range(max_iters):
        # Find matching entries. This logic might need refinement based on exact JSONL structures.
        m = metrics[i] if i < len(metrics) else {}
        s = snapshots[i] if i < len(snapshots) else {}
        
        # Chat history often has multiple turns per iteration (coder, terminal, summarizer)
        # We'll group them for this display
        c_turns = [t for t in chat if t.get("iteration") == i]
        t_turns = [t for t in terminal if t.get("iteration") == i]
        
        iterations.append({
            "index": i,
            "metrics": m,
            "snapshot": s,
            "chat": c_turns,
            "terminal": t_turns
        })
        
    return iterations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
