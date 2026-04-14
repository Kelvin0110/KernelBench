from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_integration.new_evolving_agent import kb_governor as governor


class _FakeEvalResult:
    compiled = True
    correctness = True
    runtime = 2.0
    ref_runtime = 4.0
    runtime_stats = {"mean": 2.0, "std": 0.1}
    metadata = {"hardware": "fake-gpu", "device": "cuda:0"}


class _FakeEvalModule:
    @staticmethod
    def get_torch_dtype_from_string(_precision: str):
        return "float32"

    @staticmethod
    def eval_kernel_against_ref(*_args, **_kwargs):
        return _FakeEvalResult()


def test_config_supports_batch_integration_fields(tmp_path: Path) -> None:
    cfg = governor.KBGovernorConfig(
        problem_id="100",
        reference_code="print('ref')",
        level=1,
        run_name="new-agent-test",
        results_root=tmp_path,
    )

    assert cfg.level == 1
    assert cfg.run_name == "new-agent-test"
    assert cfg.results_root == tmp_path


def test_governor_run_returns_best_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(governor, "import_module", lambda _name: _FakeEvalModule())
    monkeypatch.setattr(
        governor,
        "call_coder_with_meta",
        lambda *_a, **_k: ("```python\nprint('candidate')\n```", 32, {"model_id": "fake-coder"}),
    )

    def _fake_promote(*_args, **_kwargs):
        # no-op promotion for unit test
        return None

    monkeypatch.setattr(governor, "maybe_promote_l0_to_l1", _fake_promote)

    cfg = governor.KBGovernorConfig(
        problem_id="100",
        reference_code="print('ref')",
        level=1,
        run_name="new-agent-test",
        results_root=tmp_path,
        max_iterations=1,
        promote_entry_threshold=99,
    )

    kb_gov = governor.KBGovernor(cfg)
    result = kb_gov.run(task_prompt="Optimize this model")

    assert result.best_correct is True
    assert result.best_compiled is True
    assert result.best_speedup == 2.0
    assert result.iterations_run == 1
    assert len(result.records) == 1
