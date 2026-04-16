from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.score import fastp

DEFAULT_FAST_P_THRESHOLDS = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isfinite(out):
        return out
    return None


def speedup(baseline_runtime: float, runtime: float | None) -> float:
    if runtime is None or runtime <= 0:
        return 0.0
    return float(baseline_runtime / runtime)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def parse_fastp_values(raw: str | None) -> list[float]:
    if not raw:
        return list(DEFAULT_FAST_P_THRESHOLDS)

    parsed: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(float(token))
        except Exception:
            continue

    if not parsed:
        return list(DEFAULT_FAST_P_THRESHOLDS)

    return sorted(set(parsed))


def parse_workspace_name(name: str) -> tuple[int, int] | None:
    match = re.match(r"^level_(\d+)_problem_(\d+)$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_checkpoint_node_name(name: str) -> int | None:
    match = re.match(r"^node_(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", model).strip("_")


def pick_sample_zero(entries: Any) -> dict[str, Any] | None:
    if isinstance(entries, dict):
        return entries
    if not isinstance(entries, list):
        return None

    sample_zero = [e for e in entries if isinstance(e, dict) and e.get("sample_id") == 0]
    if sample_zero:
        return sample_zero[-1]

    for entry in entries:
        if isinstance(entry, dict):
            return entry
    return None


def build_baseline_lookup(baseline_results: dict[str, Any], level: int) -> dict[int, float]:
    level_key = f"level{level}"
    level_baseline = baseline_results.get(level_key)
    if not isinstance(level_baseline, dict):
        return {}

    dataset = construct_kernelbench_dataset(level=level, source="local")
    lookup: dict[int, float] = {}
    for pid in dataset.get_problem_ids():
        problem = dataset.get_problem_by_id(pid)
        baseline_entry = level_baseline.get(problem.name)
        if not isinstance(baseline_entry, dict):
            continue
        baseline_mean = safe_float(baseline_entry.get("mean"))
        if baseline_mean is None:
            continue
        lookup[int(pid)] = baseline_mean
    return lookup


def load_subset_problem_ids_by_level(csv_path: Path) -> dict[int, set[int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Subset CSV not found: {csv_path}")

    by_level: dict[int, set[int]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            try:
                level = int(row.get("level", ""))
                problem_id = int(row.get("problem_id", ""))
            except Exception:
                continue
            by_level.setdefault(level, set()).add(problem_id)

    return by_level


def compute_fastp_from_records(
    records: Iterable[dict[str, Any]],
    thresholds: list[float],
    *,
    runtime_field: str,
    correct_field: str,
) -> tuple[dict[str, float], int]:
    correctness: list[bool] = []
    baseline: list[float] = []
    runtime: list[float] = []

    for record in records:
        baseline_runtime = safe_float(record.get("baseline_runtime"))
        if baseline_runtime is None:
            continue

        correctness.append(bool(record.get(correct_field, False)))
        baseline.append(float(baseline_runtime))

        runtime_value = safe_float(record.get(runtime_field))
        if runtime_value is not None and runtime_value > 0:
            runtime.append(float(runtime_value))
        else:
            runtime.append(-1.0)

    aligned_count = len(baseline)
    if aligned_count == 0:
        return {str(p): 0.0 for p in thresholds}, 0

    correct_np = np.array(correctness)
    baseline_np = np.array(baseline, dtype=float)
    runtime_np = np.array(runtime, dtype=float)

    scores = {
        str(p): float(fastp(correct_np, baseline_np, runtime_np, aligned_count, p))
        for p in thresholds
    }
    return scores, aligned_count


def build_fastp_series(
    iterations: list[dict[str, Any]],
    *,
    fast_p_key: str,
    thresholds: list[float],
) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {str(p): [] for p in thresholds}

    for row in iterations:
        iteration = int(row.get("iteration", 0))
        fastp_map = row.get(fast_p_key)
        if not isinstance(fastp_map, dict):
            fastp_map = {}
        for p in thresholds:
            key = str(p)
            out[key].append(
                {
                    "iteration": iteration,
                    "value": float(safe_float(fastp_map.get(key)) or 0.0),
                }
            )

    return out


def resolve_threshold_key(series: dict[str, Any], threshold: float) -> str | None:
    if not series:
        return None
    threshold_float = float(threshold)
    candidates: list[tuple[float, str]] = []
    for key in series.keys():
        try:
            candidates.append((float(key), key))
        except Exception:
            continue
    if not candidates:
        return None

    candidates.sort(key=lambda x: abs(x[0] - threshold_float))
    return candidates[0][1]


def align_series_for_comparison(
    *,
    evolving: list[dict[str, Any]],
    aide: list[dict[str, Any]],
) -> list[dict[str, float]]:
    e_map: dict[int, float] = {}
    a_map: dict[int, float] = {}

    for point in evolving:
        try:
            i = int(point.get("iteration"))
        except Exception:
            continue
        value = safe_float(point.get("value"))
        if value is None:
            continue
        e_map[i] = float(value)

    for point in aide:
        try:
            i = int(point.get("iteration"))
        except Exception:
            continue
        value = safe_float(point.get("value"))
        if value is None:
            continue
        a_map[i] = float(value)

    if not e_map or not a_map:
        return []

    max_iter = min(max(e_map.keys()), max(a_map.keys()))
    common_iterations = sorted(i for i in set(e_map.keys()).intersection(a_map.keys()) if i <= max_iter)

    return [
        {
            "iteration": i,
            "evolving": e_map[i],
            "aide": a_map[i],
        }
        for i in common_iterations
    ]


def auto_axis_bounds(
    values: Iterable[float],
    *,
    log_scale: bool,
    pad_ratio: float = 0.08,
    min_span: float = 1e-6,
    min_positive: float = 1e-8,
) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return (min_positive, 1.0) if log_scale else (0.0, 1.0)

    if log_scale:
        positive = [v for v in finite if v > 0]
        if not positive:
            return (min_positive, 1.0)
        vmin = min(positive)
        vmax = max(positive)
        if vmin == vmax:
            return (max(min_positive, vmin * 0.5), vmax * 2.0)
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)
        log_span = max(min_span, float(log_max - log_min))
        low = 10 ** (log_min - log_span * pad_ratio)
        high = 10 ** (log_max + log_span * pad_ratio)
        return max(min_positive, low), high

    vmin = min(finite)
    vmax = max(finite)
    if vmin == vmax:
        delta = max(min_span, abs(vmin) * pad_ratio)
        return vmin - delta, vmax + delta

    span = max(min_span, vmax - vmin)
    pad = span * pad_ratio
    return vmin - pad, vmax + pad
