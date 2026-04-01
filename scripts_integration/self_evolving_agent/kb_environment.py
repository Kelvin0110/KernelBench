"""KernelBench environment adapter used by scripts_integration/self_evolving_agent."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import traceback
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_SRC = _REPO_ROOT / "Self-Evolving-Agent" / "src"
if str(_SEA_SRC) not in sys.path:
    sys.path.insert(0, str(_SEA_SRC))


@dataclass
class KernelBenchEvalOutcome:
    compiled: bool
    correctness: bool
    speedup: float
    runtime: float
    runtime_stats: dict[str, Any]
    metadata: dict[str, Any]
    feedback: str


class KernelBenchEnvironment:
    def __init__(
        self,
        *,
        dataset_source: str = "local",
        backend: str = "cuda",
        precision: str = "fp32",
        prompt_option: str = "one_shot",
        include_hardware: bool = False,
        gpu_name: str | None = None,
    ) -> None:
        self.dataset_source = dataset_source
        self.backend = backend
        self.precision = precision
        self.prompt_option = prompt_option
        self.include_hardware = include_hardware
        self.gpu_name = gpu_name

    def _load_dataset(self, level: int):
        from kernelbench.dataset import construct_kernelbench_dataset

        return construct_kernelbench_dataset(level=level, source=self.dataset_source)

    def fetch_problem(self, level: int, problem_id: int):
        dataset = self._load_dataset(level)
        return dataset.get_problem_by_id(problem_id)

    def build_prompt(self, level: int, problem_id: int) -> str:
        from kernelbench.prompt_constructor_toml import get_prompt_for_backend

        problem = self.fetch_problem(level, problem_id)
        return get_prompt_for_backend(
            ref_arch_src=problem.code,
            backend=self.backend,
            option=self.prompt_option,
            precision=self.precision,
            include_hardware=self.include_hardware,
            gpu_name=self.gpu_name,
        )

    def evaluate_candidate(
        self,
        *,
        level: int,
        problem_id: int,
        candidate_code: str,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        measure_performance: bool = True,
        device: Any = None,
    ) -> KernelBenchEvalOutcome:
        try:
            from kernelbench import eval as kb_eval

            problem = self.fetch_problem(level, problem_id)
            precision_dtype = kb_eval.get_torch_dtype_from_string(self.precision)
            result = kb_eval.eval_kernel_against_ref(
                original_model_src=problem.code,
                custom_model_src=candidate_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                measure_performance=measure_performance,
                backend=self.backend,
                precision=precision_dtype,
                device=device,
            )

            if result is None:
                return KernelBenchEvalOutcome(
                    compiled=False,
                    correctness=False,
                    speedup=0.0,
                    runtime=-1.0,
                    runtime_stats={},
                    metadata={},
                    feedback="evaluation returned None",
                )

            metadata = dict(result.metadata or {})
            runtime_stats = dict(result.runtime_stats or {})
            runtime = float(result.runtime) if result.runtime is not None else -1.0

            speedup = 0.0
            if (
                bool(result.correctness)
                and result.ref_runtime is not None
                and result.runtime is not None
                and result.ref_runtime > 0
                and result.runtime > 0
            ):
                speedup = float(result.ref_runtime / result.runtime)

            feedback = ""
            if not bool(result.correctness):
                feedback = str(
                    metadata.get("runtime_error")
                    or metadata.get("compilation_error")
                    or metadata.get("correctness_issue")
                    or "kernel evaluation failed"
                )

            return KernelBenchEvalOutcome(
                compiled=bool(result.compiled),
                correctness=bool(result.correctness),
                speedup=speedup,
                runtime=runtime,
                runtime_stats=runtime_stats,
                metadata=metadata,
                feedback=feedback,
            )
        except Exception:
            return KernelBenchEvalOutcome(
                compiled=False,
                correctness=False,
                speedup=0.0,
                runtime=-1.0,
                runtime_stats={},
                metadata={},
                feedback=traceback.format_exc(),
            )

__all__ = ["KernelBenchEnvironment", "KernelBenchEvalOutcome"]
