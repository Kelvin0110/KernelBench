from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_SRC = _REPO_ROOT / "Self-Evolving-Agent" / "src"
if str(_SEA_SRC) not in sys.path:
    sys.path.insert(0, str(_SEA_SRC))

from self_evolving_agent.integrations.kernelbench.agent import (  # noqa: E402
    KernelBenchEvolvingAgent as BaseKernelBenchEvolvingAgent,
)


class KernelBenchEvolvingAgent(BaseKernelBenchEvolvingAgent):
    """Legacy wrapper for KernelBenchEvolvingAgent, now using the modular core."""

    pass

        problem_id = int(challenge_data["problem_id"])
        max_steps = int(challenge_data.get("max_steps", self.max_steps))
        num_correct_trials = int(challenge_data.get("num_correct_trials", 5))
        num_perf_trials = int(challenge_data.get("num_perf_trials", 100))

        base_prompt = self.environment.build_prompt(level, problem_id)
        strategies = self.get_global_prompt_context(
            environment_state=(
                f"kernelbench level={level} problem={problem_id} "
                f"backend={self.environment.backend} precision={self.environment.precision}"
            ),
            top_k=self.global_top_k,
            min_utility=self.min_utility,
        )

        best_payload: dict[str, Any] | None = None
        best_speedup = -1.0
        last_feedback = ""

        for step_idx in range(max_steps):
            step_num = step_idx + 1
            local_context = self.get_local_prompt_context(task_id=task_id, limit=10)
            prompt = self._compose_prompt(
                base_prompt=base_prompt,
                strategies=strategies,
                local_context=local_context,
                last_feedback=last_feedback,
            )

            raw_output = self.generate_code(prompt)
            candidate_code = self._extract_python_code(raw_output)
            outcome = self.environment.evaluate_candidate(
                level=level,
                problem_id=problem_id,
                candidate_code=candidate_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
            )
            last_feedback = outcome.feedback

            self._record_local_step(
                task_id=task_id,
                step=step_num,
                content=(
                    f"compiled={outcome.compiled} correctness={outcome.correctness} "
                    f"speedup={outcome.speedup:.6f} feedback={outcome.feedback[:400]}"
                ),
                metadata={
                    "level": level,
                    "problem_id": problem_id,
                    "runtime": outcome.runtime,
                },
            )

            candidate_payload = {
                "sample_id": 0,
                "compiled": outcome.compiled,
                "correctness": outcome.correctness,
                "runtime": outcome.runtime,
                "runtime_stats": outcome.runtime_stats,
                "metadata": {
                    **outcome.metadata,
                    "level": level,
                    "problem_id": problem_id,
                    "best_speedup": outcome.speedup,
                    "backend": self.environment.backend,
                    "precision": self.environment.precision,
                    "iterations_run": step_num,
                    "source": "self_evolving_agent",
                    "error": outcome.feedback if not outcome.correctness else None,
                },
            }

            if outcome.correctness and outcome.speedup >= best_speedup:
                best_speedup = outcome.speedup
                best_payload = candidate_payload

            if outcome.correctness and self.stop_on_first_correct:
                break

            if best_payload is None:
                best_payload = candidate_payload

        if best_payload is None:
            best_payload = {
                "sample_id": 0,
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "runtime_stats": {},
                "metadata": {
                    "level": level,
                    "problem_id": problem_id,
                    "best_speedup": 0.0,
                    "backend": self.environment.backend,
                    "precision": self.environment.precision,
                    "iterations_run": 0,
                    "source": "self_evolving_agent",
                    "error": "no iterations executed",
                },
            }

        reflect_now = getattr(self.reflection_engine, "reflect_now", None)
        if callable(reflect_now):
            reflect_now(task_id)
        else:
            trigger_reflection = getattr(self.reflection_engine, "trigger_reflection", None)
            if callable(trigger_reflection):
                trigger_reflection(task_id)

        return best_payload

__all__ = ["KernelBenchEvolvingAgent"]
