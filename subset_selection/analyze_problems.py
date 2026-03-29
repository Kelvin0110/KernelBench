#!/usr/bin/env python3
"""
Script to analyze and classify KernelBench problems by difficulty.

This script identifies learnable problems (correctness improvements), runtime improvers,
and classifies problems into categories (A: Learnable, B: Runtime Improver, C: Moderate, D: Difficult).
"""

import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Use relative paths - assumes script is in KernelBench/subset_selection/
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent  # KernelBench directory
RUN_INTEGRATION_PATH = BASE_PATH / "run_integration"


def load_eval_results(level: int, step: int) -> Optional[Dict]:
    """Load eval_results.json for a given level and step."""
    folder_name = f"docker_level_{level}_inte_gpt_oss_120b_step{step}"
    file_path = RUN_INTEGRATION_PATH / folder_name / "eval_results.json"

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_problem_data(eval_results: Dict, problem_id: str) -> Tuple[Optional[bool], Optional[float]]:
    """
    Extract correctness and runtime from eval_results.

    Returns: (correctness: bool, runtime: float or None)
    """
    if problem_id not in eval_results:
        return None, None

    samples = eval_results[problem_id]
    if not samples or len(samples) == 0:
        return None, None

    sample = samples[0]  # First (and typically only) sample

    correctness = sample.get('correctness', None)
    runtime = sample.get('runtime', None)

    # Handle invalid runtime values
    if runtime is not None and runtime < 0:
        runtime = None

    return correctness, runtime


def classify_problem(
    step5_correct: Optional[bool],
    step20_correct: Optional[bool],
    step5_runtime: Optional[float],
    step20_runtime: Optional[float]
) -> Tuple[str, float, float]:
    """
    Classify a problem and assign it a difficulty score.

    Returns: (problem_type: str, score: float, runtime_improvement_pct: float)
    - Type A: Learnable (false→true)
    - Type B: Runtime Improver (both correct with ≥20% improvement)
    - Type C: Moderate (other successes or partial improvements)
    - Type D: Difficult (both false or insufficient improvement)
    """

    # Process None values
    if step5_correct is None or step20_correct is None:
        # If we have no data, classify as C (moderate - unknown)
        return 'C', 0.5, 0.0

    runtime_improvement_pct = 0.0
    if step5_runtime and step20_runtime and step5_runtime > 0:
        runtime_improvement_pct = ((step5_runtime - step20_runtime) / step5_runtime) * 100

    # Type A: Learnable (improved from wrong to correct)
    if not step5_correct and step20_correct:
        score = 0.95  # Highest priority - shows learning potential
        return 'A', score, runtime_improvement_pct

    # Type B: Runtime Improver (both correct, significant runtime improvement)
    if step5_correct and step20_correct:
        if runtime_improvement_pct >= 20:
            score = 0.7  # Good priority - shows optimization potential
            return 'B', score, runtime_improvement_pct
        else:
            # Both correct but low runtime improvement - still moderate
            score = 0.4
            return 'C', score, runtime_improvement_pct

    # Type D: Difficult (both incorrect or minimal improvement)
    if not step5_correct and not step20_correct:
        score = 0.2  # Lowest priority - consistently failing
        return 'D', score, runtime_improvement_pct

    # Type C: Moderate (some improvement but not learnable, or regressed)
    score = 0.5
    return 'C', score, runtime_improvement_pct


def analyze_level(level: int) -> Dict[str, Dict]:
    """Analyze all problems in a given level."""
    print(f"\nAnalyzing Level {level}...")

    step5_results = load_eval_results(level, 5)
    step20_results = load_eval_results(level, 20)

    if not step5_results or not step20_results:
        print(f"Skipping level {level}: Missing eval_results.json")
        return {}

    # Get all problem IDs across both steps
    all_problem_ids = set(step5_results.keys()) | set(step20_results.keys())
    print(f"  Total problems in level {level}: {len(all_problem_ids)}")

    problem_data = {}

    for problem_id in sorted(all_problem_ids):
        step5_correct, step5_runtime = extract_problem_data(step5_results, problem_id)
        step20_correct, step20_runtime = extract_problem_data(step20_results, problem_id)

        prob_type, score, runtime_improvement = classify_problem(
            step5_correct, step20_correct, step5_runtime, step20_runtime
        )

        # Create unique key combining level and problem_id to avoid overwriting
        unique_key = f"L{level}_P{problem_id}"
        problem_data[unique_key] = {
            'level': level,
            'problem_id': problem_id,
            'type': prob_type,
            'score': score,
            'step5_correct': step5_correct,
            'step20_correct': step20_correct,
            'step5_runtime': step5_runtime,
            'step20_runtime': step20_runtime,
            'runtime_improvement_pct': runtime_improvement
        }

    # Print summary statistics
    type_counts = {}
    for data in problem_data.values():
        t = data['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"  Type distribution: A={type_counts.get('A', 0)}, B={type_counts.get('B', 0)}, " +
          f"C={type_counts.get('C', 0)}, D={type_counts.get('D', 0)}")

    return problem_data


def main():
    parser = argparse.ArgumentParser(description="Analyze KernelBench problems by difficulty")
    parser.add_argument(
        '--output',
        type=str,
        default=str(SCRIPT_DIR / "problem_analysis.csv"),
        help="Output CSV file path"
    )
    args = parser.parse_args()

    all_problems = {}

    # Analyze all levels
    for level in [1, 2, 3]:
        level_problems = analyze_level(level)
        all_problems.update(level_problems)

    if not all_problems:
        print("No problems found to analyze!")
        return

    # Write to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'problem_id', 'level', 'type', 'score',
            'step5_correct', 'step20_correct',
            'step5_runtime', 'step20_runtime', 'runtime_improvement_pct'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by level and then by score (descending)
        sorted_problems = sorted(
            all_problems.values(),
            key=lambda x: (x['level'], -x['score'])
        )

        for problem in sorted_problems:
            writer.writerow(problem)

    print(f"\n✓ Analysis complete. Results written to: {output_path}")
    print(f"  Total problems analyzed: {len(all_problems)}")

    # Print summary by level
    print("\nSummary by level:")
    for level in [1, 2, 3]:
        level_problems = [p for p in all_problems.values() if p['level'] == level]
        if level_problems:
            type_counts = {}
            for p in level_problems:
                t = p['type']
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  Level {level}: {len(level_problems)} problems " +
                  f"(A={type_counts.get('A', 0)}, B={type_counts.get('B', 0)}, " +
                  f"C={type_counts.get('C', 0)}, D={type_counts.get('D', 0)})")


if __name__ == "__main__":
    main()
