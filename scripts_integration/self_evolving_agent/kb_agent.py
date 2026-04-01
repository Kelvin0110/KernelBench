"""KernelBench-specific evolving agent used by scripts_integration/self_evolving_agent."""

from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_SRC = _REPO_ROOT / "Self-Evolving-Agent" / "src"
if str(_SEA_SRC) not in sys.path:
    sys.path.insert(0, str(_SEA_SRC))

from self_evolving_agent.agent.core import SelfEvolvingAgent  # noqa: E402
from self_evolving_agent.memory.schemas import LocalTrialSummary  # noqa: E402

from scripts_integration.self_evolving_agent.kb_environment import KernelBenchEnvironment  # noqa: E402


class KernelBenchEvolvingAgent(SelfEvolvingAgent):
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
        stop_on_first_correct: bool = True,
    ) -> None:
        super().__init__(local_memory, global_memory, reflection_engine)
        self.environment = environment
        self.generate_code = generate_code
        self.max_steps = max_steps
        self.global_top_k = global_top_k
        self.min_utility = min_utility
        self.stop_on_first_correct = stop_on_first_correct

    @staticmethod
    def _extract_python_code(text: str) -> str:
        start = text.find("```")
        if start == -1:
            return text.strip()

        end = text.find("```", start + 3)
        if end == -1:
            return text.strip()

        block = text[start + 3 : end]
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _compose_prompt(
        self,
        *,
        base_prompt: str,
        strategies: list[str],
        local_context: str,
        last_feedback: str,
    ) -> str:
        sections = [base_prompt.strip()]
        if strategies:
            sections.append("Global strategies:\n" + "\n".join(f"- {item}" for item in strategies))
        if local_context:
            sections.append("Recent local trace:\n" + local_context)
        if last_feedback:
            sections.append("Most recent evaluation feedback:\n" + last_feedback)

        sections.append(
            "Return exactly one Python code block that defines ModelNew and fixes correctness/performance issues."
        )
        return "\n\n".join(sections)

    def _record_local_step(
        self,
        *,
        task_id: str,
        step: int,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        entry = LocalTrialSummary(
            id=f"kb-step-{uuid4().hex}",
            timestamp=time.time(),
            content=content,
            metadata=metadata,
            task_id=task_id,
            execution_step=step,
        )
        self.local_memory.add_entry(entry)

    def run_benchmark_task(self, task_id: str, challenge_data: dict[str, Any]) -> Any:
        level = int(challenge_data["level"])
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
