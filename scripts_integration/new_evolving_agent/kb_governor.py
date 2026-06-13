"""Deprecated import path — use ``kernelbench_integration`` instead."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEA_ROOT = _REPO_ROOT / "Self-Evolving-Agent"
if str(_SEA_ROOT) not in sys.path:
    sys.path.append(str(_SEA_ROOT))

warnings.warn(
    "scripts_integration.new_evolving_agent.kb_governor is deprecated; "
    "import from kernelbench_integration instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kernelbench_integration import (  # noqa: E402,F401
    CODER_SYSTEM_PROMPT,
    KBEvalResult,
    KBGovernor,
    KBGovernorConfig,
    KBGovernorResult,
    KBIterationRecord,
    LEGACY_CODER_SYSTEM_PROMPT,
    cleanup_problem_build_artifacts,
    governor_result_to_dict,
    safe_run_kb_governor,
)
from kernelbench_integration.eval_runner import (  # noqa: E402
    kernelbench_eval_worker,
    run_kernelbench_eval,
)
from evolving_common.governor.l0_round_summary import maybe_summarize_l0_round  # noqa: E402
from evolving_common.llm_client import call_coder_with_meta  # noqa: E402
from evolving_common.governor import maybe_promote_l0_to_l1  # noqa: E402
from importlib import import_module  # noqa: E402

__all__ = [
    "CODER_SYSTEM_PROMPT",
    "KBEvalResult",
    "KBGovernor",
    "KBGovernorConfig",
    "KBGovernorResult",
    "KBIterationRecord",
    "LEGACY_CODER_SYSTEM_PROMPT",
    "call_coder_with_meta",
    "cleanup_problem_build_artifacts",
    "governor_result_to_dict",
    "import_module",
    "kernelbench_eval_worker",
    "maybe_promote_l0_to_l1",
    "maybe_summarize_l0_round",
    "run_kernelbench_eval",
    "safe_run_kb_governor",
]
