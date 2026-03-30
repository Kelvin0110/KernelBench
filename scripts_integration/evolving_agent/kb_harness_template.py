"""Template helper for benchmark execution in evolving-agent integration."""

from __future__ import annotations

from kernelbench import eval as kb_eval
from kernelbench.dataset import construct_kernelbench_dataset


def run_benchmark(
    kernel_source_code: str,
    level: int,
    problem_id: int,
    backend: str = "cuda",
    precision: str = "fp32",
) -> float:
    """Evaluate generated kernel source and return speedup (0.0 on failure)."""
    try:
        dataset = construct_kernelbench_dataset(level=level, source="local")
        problem = dataset.get_problem_by_id(problem_id)

        result = kb_eval.eval_kernel_against_ref(
            problem.code,
            kernel_source_code,
            backend=backend,
            precision=kb_eval.get_torch_dtype_from_string(precision),
            measure_performance=True,
        )

        print(f"KERNEL_BENCH_CORRECT: {result.correctness}")
        if result.correctness and result.runtime > 0:
            speedup = result.ref_runtime / result.runtime
            print(f"KERNEL_BENCH_SPEEDUP: {speedup:.4f}")
            return float(speedup)

        print("KERNEL_BENCH_SPEEDUP: 0.0")
        error_info = (
            result.metadata.get("compilation_error")
            or result.metadata.get("runtime_error")
            or "Unknown error"
        )
        print(f"KERNEL_BENCH_ERROR: {error_info}")
        return 0.0
    except Exception as exc:
        print(f"Harness Error: {exc}")
        return 0.0
