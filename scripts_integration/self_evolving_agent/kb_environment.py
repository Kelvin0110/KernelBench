from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_SRC = _REPO_ROOT / "Self-Evolving-Agent" / "src"
if str(_SEA_SRC) not in sys.path:
    sys.path.insert(0, str(_SEA_SRC))

from self_evolving_agent.integrations.kernelbench.environment import (  # noqa: E402
    KernelBenchEnvironment as BaseKernelBenchEnvironment,
    KernelBenchEvalOutcome,
)


class KernelBenchEnvironment(BaseKernelBenchEnvironment):
    """Legacy wrapper for KernelBenchEnvironment, now using the modular core."""

    pass

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
