import importlib.util
import sys
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SEA_ROOT = REPO_ROOT / "Self-Evolving-Agent"
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SEA_ROOT), str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

if 'torch' not in sys.modules:
    _torch = ModuleType('torch')
    _torch.cuda = MagicMock()
    _torch.cuda.is_available = lambda: False
    sys.modules['torch'] = _torch

_kb_pkg = ModuleType('kernelbench')
_kb_dataset = ModuleType('kernelbench.dataset')
_kb_dataset.construct_kernelbench_dataset = MagicMock()
_kb_prompt = ModuleType('kernelbench.prompt_constructor_toml')
_kb_prompt.get_prompt_for_backend = MagicMock()
sys.modules.setdefault('kernelbench', _kb_pkg)
sys.modules.setdefault('kernelbench.dataset', _kb_dataset)
sys.modules.setdefault('kernelbench.prompt_constructor_toml', _kb_prompt)

if "kernelbench.performance_stats" not in sys.modules:
    score_path = SRC_ROOT / "kernelbench" / "score.py"
    score_spec = importlib.util.spec_from_file_location("kernelbench.score", score_path)
    if score_spec and score_spec.loader:
        score_mod = importlib.util.module_from_spec(score_spec)
        sys.modules["kernelbench.score"] = score_mod
        score_spec.loader.exec_module(score_mod)

    perf_path = SRC_ROOT / "kernelbench" / "performance_stats.py"
    spec = importlib.util.spec_from_file_location("kernelbench.performance_stats", perf_path)
    if spec and spec.loader:
        perf_mod = importlib.util.module_from_spec(spec)
        sys.modules["kernelbench.performance_stats"] = perf_mod
        spec.loader.exec_module(perf_mod)

_kb_governor = ModuleType('scripts_integration.new_evolving_agent.kb_governor')
_kb_governor.KBGovernorConfig = MagicMock
_kb_governor.governor_result_to_dict = lambda result: dict(result)
_kb_governor.safe_run_kb_governor = MagicMock()
sys.modules.setdefault('scripts_integration.new_evolving_agent.kb_governor', _kb_governor)
