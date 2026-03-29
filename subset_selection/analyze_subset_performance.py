#!/usr/bin/env python3
"""
Script to analyze performance of the selected 50-problem subset.

This script calculates performance metrics for the selected problems across
different levels and training steps (step5 and step20), producing both
per-level analysis and aggregate results.

KEY IMPROVEMENT: Uses actual baseline comparison for accurate speedup metrics
- Loads baseline_time_torch.json which contains baseline runtimes for each problem
- Maps problem IDs to problem names to find baseline data
- Calculates TRUE speedup: baseline_runtime / actual_runtime
- Computes fast_p and geo_mean_speedup using real speedup values

Usage:
```bash
python3 analyze_subset_performance.py
```

Output:
- analysis_results.json: Contains per-level metrics and aggregate statistics
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict

# Use relative paths - assumes script is in KernelBench/subset_selection/
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent  # KernelBench directory
RUN_INTEGRATION_PATH = BASE_PATH / "run_integration"
BASELINE_FILE = BASE_PATH / "results" / "timing" / "SONG_CPU2_A6000x2" / "baseline_time_torch.json"


def load_selected_problems() -> Dict[int, set]:
    """
    Load selected problem IDs organized by level.

    Returns: {level: set of problem_ids}
    """
    problems_by_level = {1: set(), 2: set(), 3: set()}

    csv_path = SCRIPT_DIR / "selected_problems_50.csv"
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                level = int(row['level'])
                problem_id = row['problem_id']
                problems_by_level[level].add(problem_id)
    except Exception as e:
        print(f"Error loading selected problems: {e}")
        return {}

    print(f"Loaded {sum(len(p) for p in problems_by_level.values())} selected problems")
    for level in [1, 2, 3]:
        print(f"  Level {level}: {len(problems_by_level[level])} problems")

    return problems_by_level


def load_baseline_results() -> Dict[int, Dict[str, float]]:
    """
    Load baseline runtimes from baseline_time_torch.json.

    Creates a mapping: {level: {problem_id: baseline_runtime}}
    For example: {1: {"1": 5.85, "2": 5.87, ...}, 2: {...}, 3: {...}}

    Returns: {level: {problem_id: mean_runtime}}
    """
    if not BASELINE_FILE.exists():
        print(f"Warning: Baseline file not found: {BASELINE_FILE}")
        return {}

    try:
        with open(BASELINE_FILE, 'r') as f:
            baseline_data = json.load(f)
    except Exception as e:
        print(f"Error loading baseline file: {e}")
        return {}

    # Map problem IDs to baseline runtimes
    baseline_by_level = {}
    for level in [1, 2, 3]:
        level_key = f"level{level}"
        baseline_by_level[level] = {}

        if level_key not in baseline_data:
            print(f"Warning: {level_key} not found in baseline data")
            continue

        # Extract baseline runtimes keyed by problem ID
        for problem_name, metrics in baseline_data[level_key].items():
            # Problem names start with ID: "1_...", "2_...", etc.
            # Extract the ID part
            try:
                problem_id = problem_name.split('_')[0]
                baseline_runtime = metrics.get('mean', None)
                if baseline_runtime is not None:
                    baseline_by_level[level][problem_id] = baseline_runtime
            except Exception as e:
                print(f"Warning: Could not parse problem name {problem_name}: {e}")
                continue

        print(f"  Level {level}: Loaded {len(baseline_by_level[level])} baseline entries")

    return baseline_by_level


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
    Extract correctness and runtime from eval_results for a problem.

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


def geometric_mean(values: List[float]) -> float:
    """Calculate geometric mean of values."""
    if not values or any(v <= 0 for v in values):
        return 0.0

    import math
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def calculate_speedup_ratio(baseline_time: float, actual_time: float) -> float:
    """Calculate speedup ratio: baseline_time / actual_time."""
    if actual_time is None or actual_time <= 0:
        return 0.0
    return baseline_time / actual_time


def calculate_fast_p(is_correct_list: List[bool],
                    baseline_speeds: List[float],
                    actual_speeds: List[float],
                    p_threshold: float) -> float:
    """
    Calculate fast_p score for a given speedup threshold p.

    fast_p is the percentage of correct samples that meet the speedup threshold.
    """
    if not is_correct_list or len(is_correct_list) == 0:
        return 0.0

    count_meeting_threshold = 0
    count_correct = 0

    for is_correct, baseline, actual in zip(is_correct_list, baseline_speeds, actual_speeds):
        if is_correct is None:
            continue
        count_correct += 1

        if is_correct and baseline is not None and actual is not None and actual > 0:
            speedup = baseline / actual
            if speedup >= p_threshold:
                count_meeting_threshold += 1

    if count_correct == 0:
        return 0.0

    return count_meeting_threshold / count_correct


def analyze_level_step(selected_problem_ids: set, level: int, step: int,
                      baseline_by_level: Dict[int, Dict[str, float]]) -> Dict:
    """
    Analyze performance for a specific level and step using baseline comparison.

    Returns dictionary with metrics including TRUE speedup calculations.
    """
    eval_results = load_eval_results(level, step)
    if not eval_results:
        return None

    # Get baseline for this level
    baseline_for_level = baseline_by_level.get(level, {})

    # Initialize counters
    total_count = len(selected_problem_ids)
    compiled_count = 0
    correct_count = 0

    # Tracking for speedup calculations (now with TRUE speedup)
    is_correct_list = []
    speedup_values = []  # TRUE speedup: baseline / actual

    # Process each selected problem
    for problem_id in selected_problem_ids:
        correctness, actual_runtime = extract_problem_data(eval_results, problem_id)

        if correctness is None:
            continue

        # Count compiled (we consider it compiled if we have correctness data)
        compiled_count += 1

        if correctness:
            correct_count += 1

        # Get baseline runtime for this problem
        baseline_runtime = baseline_for_level.get(problem_id, None)

        # Calculate TRUE speedup: baseline_runtime / actual_runtime
        is_correct_list.append(correctness)

        if correctness and actual_runtime is not None and actual_runtime > 0 and baseline_runtime is not None:
            # Only count speedup for correct samples with valid baseline
            speedup = baseline_runtime / actual_runtime
            speedup_values.append(speedup)

    # Calculate rates
    compilation_rate = compiled_count / total_count if total_count > 0 else 0.0
    correctness_rate = correct_count / total_count if total_count > 0 else 0.0

    # Calculate geometric mean speedup (TRUE speedup for correct samples)
    geo_mean_speedup = geometric_mean(speedup_values) if speedup_values else 0.0

    # Calculate fast_p for different thresholds (using TRUE speedup)
    p_thresholds = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    fast_p_results = {}

    # For each threshold, calculate percentage of correct samples that meet it
    for p in p_thresholds:
        count_meeting = 0
        count_correct = sum(1 for c in is_correct_list if c)

        if count_correct > 0:
            # Re-count for this threshold using TRUE speedup
            speedup_idx = 0
            correct_idx = 0
            for problem_id in selected_problem_ids:
                correctness, actual_runtime = extract_problem_data(eval_results, problem_id)
                if correctness is None:
                    continue
                correct_idx += 1

                if correctness:
                    baseline_runtime = baseline_for_level.get(problem_id, None)
                    if actual_runtime is not None and actual_runtime > 0 and baseline_runtime is not None:
                        speedup = baseline_runtime / actual_runtime
                        if speedup >= p:
                            count_meeting += 1

            fast_p_results[str(p)] = count_meeting / count_correct
        else:
            fast_p_results[str(p)] = 0.0

    return {
        "level": level,
        "step": step,
        "total_count": total_count,
        "compiled_count": compiled_count,
        "correct_count": correct_count,
        "compilation_rate": round(compilation_rate, 4),
        "correctness_rate": round(correctness_rate, 4),
        "geo_mean_speedup": round(geo_mean_speedup, 4),
        "fast_p": {str(k): round(v, 4) for k, v in fast_p_results.items()},
    }


def aggregate_results(results_by_level_step: Dict) -> Dict:
    """
    Aggregate metrics separately for step5 and step20.

    Returns dictionary with both step5_aggregate and step20_aggregate.
    """
    # Separate results by step
    step5_results = [r for r in results_by_level_step.values() if r and r.get("step") == 5]
    step20_results = [r for r in results_by_level_step.values() if r and r.get("step") == 20]

    def calculate_aggregate_for_step(step_results):
        """Calculate aggregate metrics for a given step's results."""
        if not step_results:
            return {
                "total_count": 0,
                "compiled_count": 0,
                "correct_count": 0,
                "compilation_rate": 0.0,
                "correctness_rate": 0.0,
                "avg_geo_mean_speedup": 0.0,
                "avg_fast_p": {str(p): 0.0 for p in [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]},
                "breakdown": {"level_1": 0, "level_2": 0, "level_3": 0}
            }

        # Sum metrics
        total_count = sum(r["total_count"] for r in step_results)
        compiled_count = sum(r["compiled_count"] for r in step_results)
        correct_count = sum(r["correct_count"] for r in step_results)

        # Calculate rates
        compilation_rate = compiled_count / total_count if total_count > 0 else 0.0
        correctness_rate = correct_count / total_count if total_count > 0 else 0.0

        # Average fast_p across all levels
        avg_fast_p = {}
        for p_key in ["0.0", "0.5", "0.8", "1.0", "1.5", "2.0"]:
            values = [r["fast_p"].get(p_key, 0.0) for r in step_results]
            avg_fast_p[p_key] = round(sum(values) / len(values), 4) if values else 0.0

        # Average geo_mean_speedup
        geo_mean_speedups = [r["geo_mean_speedup"] for r in step_results
                             if r.get("geo_mean_speedup", 0) > 0]
        avg_geo_mean = round(sum(geo_mean_speedups) / len(geo_mean_speedups), 4) if geo_mean_speedups else 0.0

        # Breakdown by level
        breakdown = {
            "level_1": sum(1 for r in step_results if r.get("level") == 1),
            "level_2": sum(1 for r in step_results if r.get("level") == 2),
            "level_3": sum(1 for r in step_results if r.get("level") == 3),
        }

        return {
            "total_count": total_count,
            "compiled_count": compiled_count,
            "correct_count": correct_count,
            "compilation_rate": round(compilation_rate, 4),
            "correctness_rate": round(correctness_rate, 4),
            "avg_geo_mean_speedup": avg_geo_mean,
            "avg_fast_p": avg_fast_p,
            "breakdown": breakdown
        }

    return {
        "step5_aggregate": calculate_aggregate_for_step(step5_results),
        "step20_aggregate": calculate_aggregate_for_step(step20_results)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze subset problem performance")
    parser.add_argument(
        '--output',
        type=str,
        default=str(SCRIPT_DIR / "analysis_results.json"),
        help="Output JSON file path"
    )
    args = parser.parse_args()

    print("Loading selected problems...")
    selected_problems = load_selected_problems()

    if not selected_problems or all(not v for v in selected_problems.values()):
        print("Error: No selected problems found!")
        return

    print("\nLoading baseline results...")
    baseline_by_level = load_baseline_results()

    print("\nAnalyzing performance for each level and step...")
    results_by_level_step = {}

    # Analyze each level and step combination
    for level in [1, 2, 3]:
        if not selected_problems[level]:
            continue

        for step in [5, 20]:
            key = f"level_{level}_step_{step}"
            print(f"  Analyzing {key}...")

            result = analyze_level_step(
                selected_problems[level],
                level,
                step,
                baseline_by_level  # Pass baseline results for speedup calculation
            )

            if result:
                results_by_level_step[key] = result
                print(f"    Correctness: {result['correctness_rate']:.1%}, " +
                     f"Compilation: {result['compilation_rate']:.1%}, " +
                     f"Speedup: {result['geo_mean_speedup']:.2f}x")

    # Calculate aggregate results
    print("\nCalculating aggregate metrics...")
    aggregates = aggregate_results(results_by_level_step)

    # Build final output
    output_data = {
        "metadata": {
            "subset_size": sum(len(problems) for problems in selected_problems.values()),
            "levels": 3,
            "problems_per_level": {
                str(level): len(selected_problems[level])
                for level in [1, 2, 3]
            },
            "steps_analyzed": [5, 20]
        },
        **results_by_level_step,
        "step5_aggregate": aggregates["step5_aggregate"],
        "step20_aggregate": aggregates["step20_aggregate"]
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Analysis complete. Results written to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUBSET PERFORMANCE SUMMARY")
    print("="*60)

    step5_agg = aggregates["step5_aggregate"]
    step20_agg = aggregates["step20_aggregate"]

    print("\nStep 5 Results:")
    print(f"  Total problems: {step5_agg['total_count']}")
    print(f"  Compilation rate: {step5_agg['compilation_rate']:.1%}")
    print(f"  Correctness rate: {step5_agg['correctness_rate']:.1%}")
    print(f"  Avg geo_mean_speedup: {step5_agg['avg_geo_mean_speedup']:.4f}x")
    print("  Fast-p scores:")
    for p_key in ["0.0", "0.5", "0.8", "1.0", "1.5", "2.0"]:
        print(f"    fast_p[{p_key}]: {step5_agg['avg_fast_p'][p_key]:.4f}")

    print("\nStep 20 Results:")
    print(f"  Total problems: {step20_agg['total_count']}")
    print(f"  Compilation rate: {step20_agg['compilation_rate']:.1%}")
    print(f"  Correctness rate: {step20_agg['correctness_rate']:.1%}")
    print(f"  Avg geo_mean_speedup: {step20_agg['avg_geo_mean_speedup']:.4f}x")
    print("  Fast-p scores:")
    for p_key in ["0.0", "0.5", "0.8", "1.0", "1.5", "2.0"]:
        print(f"    fast_p[{p_key}]: {step20_agg['avg_fast_p'][p_key]:.4f}")

    print("\nPer-level breakdown:")
    for key in sorted(results_by_level_step.keys()):
        result = results_by_level_step[key]
        print(f"  {key}: Correct={result['correct_count']}/{result['total_count']}, " +
             f"Rate={result['correctness_rate']:.1%}")


if __name__ == "__main__":
    main()
