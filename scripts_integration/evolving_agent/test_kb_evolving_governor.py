from __future__ import annotations

import time
from pathlib import Path

from scripts_integration.evolving_agent import kb_evolving_governor as governor


class _DummyProblem:
    code = "class Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"


class _DummyDataset:
    def get_problem_by_id(self, _problem_id: int):
        return _DummyProblem()


def test_run_kb_governor_continues_after_fatal_cuda_error(tmp_path: Path, monkeypatch) -> None:
    run_name = "fatal_cuda_case"
    run_root = tmp_path / "results"
    shared_l1 = tmp_path / "shared_l1.txt"
    shared_l1.write_text("", encoding="utf-8")

    calls = {"count": 0}

    monkeypatch.setattr(governor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(governor, "construct_kernelbench_dataset", lambda **_k: _DummyDataset())
    monkeypatch.setattr(governor, "get_prompt_for_backend", lambda **_k: "prompt")
    monkeypatch.setattr(governor, "call_coder", lambda *_a, **_k: ("```python\nprint('x')\n```", 1))

    def _fake_eval(**_k):
        calls["count"] += 1
        if calls["count"] == 1:
            return 0.0, False, False, "CUDA error: an illegal memory access was encountered"
        return 2.0, True, True, None

    monkeypatch.setattr(governor, "_evaluate_candidate_isolated", _fake_eval)

    cfg = governor.KBGovernorConfig(
        run_name=run_name,
        level=1,
        problem_id=100,
        max_iterations=2,
        shared_l1_path=shared_l1,
        results_root=run_root,
        verbose=False,
        promote_entry_threshold=99,
    )

    result = governor.run_kb_governor(cfg)

    assert result.iterations_run == 2
    assert result.best_correct is True
    assert result.best_compiled is True
    assert result.best_speedup == 2.0
    assert result.error is not None
    assert "continued" in result.error.lower()


def test_evaluate_candidate_isolated_timeout(monkeypatch) -> None:
    def _slow_eval(**_k):
        time.sleep(1.0)
        return 1.0, True, True, None

    monkeypatch.setattr(governor, "_evaluate_candidate", _slow_eval)

    speedup, correctness, compiled, err = governor._evaluate_candidate_isolated(
        reference_code="x",
        candidate_code="y",
        backend="cuda",
        precision="fp32",
        timeout_sec=0.1,
        start_method="fork",
    )

    assert speedup == 0.0
    assert correctness is False
    assert compiled is False
    assert err is not None
    assert "evaluation_timeout" in err
