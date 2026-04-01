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


__all__ = ["KernelBenchEvolvingAgent"]
