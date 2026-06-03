import sys
import os
from types import ModuleType
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Self-Evolving-Agent')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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

_kb_governor = ModuleType('scripts_integration.new_evolving_agent.kb_governor')
_kb_governor.KBGovernorConfig = MagicMock
_kb_governor.governor_result_to_dict = lambda result: dict(result)
_kb_governor.safe_run_kb_governor = MagicMock()
sys.modules.setdefault('scripts_integration.new_evolving_agent.kb_governor', _kb_governor)
