#!/usr/bin/env python3
"""
Script to select ~50 balanced problems from analyzed KernelBench problems.

PROBLEM SELECTION STRATEGY:
===========================
This script selects 50 problems (10 from level 1, 15 from level 2, 25 from level 3)
using a prioritized learning-focused approach:

1. PRIORITY WEIGHTING BY PROBLEM TYPE:
   - Type A (Learnable, score=0.95): Problems where model improved from wrong→correct
     * Highest value: demonstrates model capability growth
     * Selected FIRST because these show direct learning potential

   - Type B (Runtime Improver, score=0.7): Problems correct at both steps but with ≥20% speedup
     * High value: shows optimization learning even if correctness didn't change
     * Selected SECOND as these demonstrate efficiency improvements

   - Type C (Moderate, score=0.4-0.5): Mixed results or partial improvements
     * Selected THIRD as filler to reach target count

   - Type D (Difficult, score=0.2): Problems failing at both steps
     * Selected LAST (2-4 per level only) for post-training validation

2. LEVEL ALLOCATION STRATEGY:
   - Level 1: 10 problems (20%) - Easier level with faster model learning
   - Level 2: 15 problems (30%) - Medium difficulty
   - Level 3: 25 problems (50%) - Hardest level, most learning opportunity

   This distribution prioritizes harder problems (less than 10% improvement at step20)
   where the model has more opportunity to learn.

3. WITHIN EACH LEVEL, SELECTION ORDER:
   - First: All available Type A problems (learnable)
   - Then: Type B problems (runtime improvers)
   - Then: Type C problems (moderate difficulty)
   - Finally: Up to 4 Type D problems (difficult) for validation

   Within each type, problems are ranked by score (highest first).
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Use relative paths - assumes script is in KernelBench/subset_selection/
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent  # KernelBench directory
ANALYSIS_PATH = BASE_PATH / "analysis" / "SONG_CPU2_A6000x2" / "baseline_time_torch"


def load_baseline_metrics(level: int) -> Tuple[float, float]:
    """
    Load baseline metrics for a level to assess difficulty.

    Returns: (correctness_change: float, fast_1_value: float)
    """
    # Load step5 and step20 files
    step5_file = ANALYSIS_PATH / f"docker_level_{level}_inte_gpt_oss_120b_step5.json"
    step20_file = ANALYSIS_PATH / f"docker_level_{level}_inte_gpt_oss_120b_step20.json"

    try:
        with open(step5_file, 'r') as f:
            step5_data = json.load(f)
        with open(step20_file, 'r') as f:
            step20_data = json.load(f)

        correctness_change = step20_data['correctness_rate'] - step5_data['correctness_rate']
        fast_1_value = step20_data.get('fast_p', {}).get('1.0', 0.5)

        return correctness_change, fast_1_value
    except Exception as e:
        print(f"Warning: Could not load baseline metrics for level {level}: {e}")
        return 0.0, 0.5


def calculate_level_difficulty(correctness_change: float) -> float:
    """
    Calculate difficulty score for a level.

    Higher correctness_change = easier = lower difficulty score.
    Lower correctness_change = harder = higher difficulty score.
    """
    # Use inverse calculation: harder levels (less improvement) get higher difficulty
    # Add epsilon to avoid division by zero
    epsilon = 0.05
    difficulty = 1.0 / (correctness_change + epsilon)
    return difficulty


def allocate_problems_per_level(
    problem_counts: Dict[int, int],
    target_total: int = 50
) -> Dict[int, int]:
    """
    Allocate problems across levels using hard-coded distribution.

    Hard-coded allocation (optimized for self-evolving memory training):
    - Level 1: 10 problems (20%) - Easier, faster initial learning
    - Level 2: 15 problems (30%) - Medium difficulty
    - Level 3: 25 problems (50%) - Hardest level, maximum learning opportunity

    Rationale: Model learning rate is inverse to level difficulty.
    Level 1 improves by ~30% (easier), Level 3 by only ~6% (harder).
    By allocating more from harder levels, we prioritize problems where
    the model has the most opportunity to learn and improve over training cycles.
    """
    # Hard-coded allocation optimized for learning potential
    allocation = {
        1: 10,  # Level 1: 20% (easier, faster learning)
        2: 15,  # Level 2: 30% (medium difficulty)
        3: 25,  # Level 3: 50% (hardest, maximum learning opportunity)
    }

    # Print level analysis for reference
    for level in [1, 2, 3]:
        correctness_change, _ = load_baseline_metrics(level)
        difficulty = calculate_level_difficulty(correctness_change)
        print(f"  Level {level}: correctness_change={correctness_change:.3f}, difficulty={difficulty:.3f}, " +
              f"allocated={allocation[level]} problems")

    return allocation


def load_analyzed_problems(csv_path: str) -> Dict[int, List[Dict]]:
    """Load analyzed problems from CSV and organize by level."""
    problems_by_level = {1: [], 2: [], 3: []}

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                level = int(row['level'])
                # Convert string booleans
                row['step5_correct'] = row['step5_correct'].lower() == 'true'
                row['step20_correct'] = row['step20_correct'].lower() == 'true'
                row['score'] = float(row['score'])
                row['runtime_improvement_pct'] = float(row['runtime_improvement_pct'])
                problems_by_level[level].append(row)
    except Exception as e:
        print(f"Error loading analyzed problems: {e}")
        return {}

    return problems_by_level


def select_problems_from_level(
    problems: List[Dict],
    target_count: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select problems from a level, prioritizing types and marking difficult ones.

    Returns: (selected_problems, difficult_problems_marked)
    """
    # Separate by type
    type_a = [p for p in problems if p['type'] == 'A']  # Learnable
    type_b = [p for p in problems if p['type'] == 'B']  # Runtime improver
    type_c = [p for p in problems if p['type'] == 'C']  # Moderate
    type_d = [p for p in problems if p['type'] == 'D']  # Difficult

    # Sort within each type by score (descending)
    type_a.sort(key=lambda p: -p['score'])
    type_b.sort(key=lambda p: -p['score'])
    type_c.sort(key=lambda p: -p['score'])
    type_d.sort(key=lambda p: -p['score'])

    selected = []
    difficult_marked = []

    # Priority 1: Type A (Learnable) - take all or as many as we can
    selected.extend(type_a[:target_count])
    remaining = target_count - len(selected)

    # Priority 2: Type B (Runtime Improver)
    selected.extend(type_b[:remaining])
    remaining = target_count - len(selected)

    # Priority 3: Type C (Moderate)
    selected.extend(type_c[:remaining])
    remaining = target_count - len(selected)

    # Priority 4: Type D (Difficult) - limit to 2-4 per level for validation
    max_difficult = min(4, remaining, len(type_d))
    for i in range(max_difficult):
        p = type_d[i].copy()
        p['difficulty_flag'] = True
        selected.append(p)
        difficult_marked.append(p)
        remaining -= 1

    # If still need more, add remaining Type C
    if remaining > 0:
        start_idx = min(remaining, len(type_c))
        selected.extend(type_c[-(remaining):])

    # Add difficulty_flag to non-difficult problems
    for p in selected:
        if 'difficulty_flag' not in p:
            p['difficulty_flag'] = False

    return selected[:target_count], difficult_marked


def main():
    parser = argparse.ArgumentParser(description="Select ~50 balanced KernelBench problems")
    parser.add_argument(
        '--analysis-csv',
        type=str,
        default=str(SCRIPT_DIR / "problem_analysis.csv"),
        help="Input CSV from analyze_problems.py"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(SCRIPT_DIR / "selected_problems_50.csv"),
        help="Output CSV with selected problems"
    )
    parser.add_argument(
        '--target',
        type=int,
        default=50,
        help="Target number of problems to select"
    )
    args = parser.parse_args()

    analysis_csv = Path(args.analysis_csv)
    if not analysis_csv.exists():
        print(f"Error: Analysis CSV not found: {analysis_csv}")
        print("Please run analyze_problems.py first")
        return

    print("Loading analyzed problems...")
    problems_by_level = load_analyzed_problems(str(analysis_csv))

    print("\nAnalyzing level difficulties...")
    allocation = allocate_problems_per_level(
        {level: len(problems_by_level[level]) for level in [1, 2, 3]},
        args.target
    )

    print(f"\nProblem allocation (total=50):")
    for level in [1, 2, 3]:
        count = allocation[level]
        total = len(problems_by_level[level])
        print(f"  Level {level}: {count} / {total} problems")

    # Select problems per level
    all_selected = []
    all_difficult = []

    for level in [1, 2, 3]:
        print(f"\nSelecting problems for level {level}...")
        selected, difficult = select_problems_from_level(
            problems_by_level[level],
            allocation[level]
        )
        all_selected.extend(selected)
        all_difficult.extend(difficult)

        if difficult:
            print(f"  Marked {len(difficult)} difficult problems for validation:")
            for p in difficult:
                print(f"    - Problem {p['problem_id']} (type={p['type']}, score={p['score']:.2f})")

    # Sort final selection by level and score
    all_selected.sort(key=lambda p: (int(p['level']), -p['score']))

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'problem_id', 'level', 'type', 'score',
            'step5_correct', 'step20_correct',
            'step5_runtime', 'step20_runtime', 'runtime_improvement_pct',
            'difficulty_flag'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for problem in all_selected:
            writer.writerow(problem)

    print(f"\n✓ Selection complete. Results written to: {output_path}")
    print(f"  Total problems selected: {len(all_selected)}")
    print(f"  Total difficult problems: {len(all_difficult)}")

    # Print summary by level
    print("\nFinal selection by level:")
    for level in [1, 2, 3]:
        level_problems = [p for p in all_selected if int(p['level']) == level]
        if level_problems:
            type_counts = {}
            difficult_count = sum(1 for p in level_problems if p['difficulty_flag'])
            for p in level_problems:
                t = p['type']
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  Level {level}: {len(level_problems)} problems " +
                  f"(A={type_counts.get('A', 0)}, B={type_counts.get('B', 0)}, " +
                  f"C={type_counts.get('C', 0)}, D={type_counts.get('D', 0)}) " +
                  f"[{difficult_count} difficult]")


if __name__ == "__main__":
    main()
