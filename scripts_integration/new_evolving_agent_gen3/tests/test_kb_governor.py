from __future__ import annotations

import json
from pathlib import Path

from scripts_integration.new_evolving_agent_gen3 import kb_governor as governor


class _FakeEvalResult:
    compiled = True
    correctness = True
    runtime = 2.0
    ref_runtime = 4.0
    runtime_stats = {'mean': 2.0, 'std': 0.1}
    metadata = {'hardware': 'fake-gpu', 'device': 'cuda:0'}


class _FakeEvalModule:
    @staticmethod
    def get_torch_dtype_from_string(_precision: str):
        return 'float32'

    @staticmethod
    def eval_kernel_against_ref(*_args, **_kwargs):
        print('FAKE_EVAL_STDOUT')
        return _FakeEvalResult()


def test_config_supports_batch_integration_fields(tmp_path: Path) -> None:
    cfg = governor.KBGovernorConfig(
        problem_id='100',
        reference_code='print("ref")',
        level=1,
        run_name='new-agent-test',
        results_root=tmp_path,
        isolate_evaluation_process=False,
    )

    assert cfg.level == 1
    assert cfg.run_name == 'new-agent-test'
    assert cfg.results_root == tmp_path


def test_governor_run_returns_best_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(governor, 'import_module', lambda _name: _FakeEvalModule())
    
    code = '```python\nprint("candidate")\n```'
    monkeypatch.setattr(
        governor,
        'call_coder_with_meta',
        lambda *_a, **_k: (code, 32, {'model_id': 'fake-coder'}),
    )

    def _fake_promote(*_args, **_kwargs):
        return None

    monkeypatch.setattr(governor, 'maybe_promote_l0_to_l1', _fake_promote)

    def _fake_round_summarize(rnd, **_kwargs):
        from evolving_common.l0_context import format_round_summary_fallback
        from evolving_common.memory_manager import set_l0_round_summary

        text = format_round_summary_fallback(rnd)
        set_l0_round_summary(rnd, text)
        return text

    monkeypatch.setattr(governor, 'maybe_summarize_l0_round', _fake_round_summarize)

    cfg = governor.KBGovernorConfig(
        problem_id='100',
        reference_code='print("ref")',
        level=1,
        run_name='new-agent-test',
        results_root=tmp_path,
        max_iterations=1,
        isolate_evaluation_process=False,
        enable_action_selector=False,
        enable_l0_unfold=False,
    )

    kb_gov = governor.KBGovernor(cfg)
    result = kb_gov.run(task_prompt='Optimize this model')

    assert result.best_correct is True
    assert result.best_compiled is True
    assert result.best_speedup == 2.0
    assert result.iterations_run == 1
    assert len(result.records) == 1
    assert 'FAKE_EVAL_STDOUT' in (result.records[0].evaluation.terminal_output or '')

    eval_terminal_log = (
        tmp_path
        / 'new-agent-test'
        / 'workspaces'
        / 'level_1_problem_100'
        / 'evaluation_terminal_output.jsonl'
    )
    assert eval_terminal_log.is_file()
    rows = [
        json.loads(line)
        for line in eval_terminal_log.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert rows
    assert 'FAKE_EVAL_STDOUT' in rows[0]['terminal_output']
    assert rows[0]['extra']['runtime'] == 2.0


def test_kb_governor_reports_missing_kernelbench_runtime(monkeypatch) -> None:
    from importlib import import_module as real_import
    def _mock_import(name):
        if 'kernelbench.eval' in name:
            raise ModuleNotFoundError('kernelbench not installed')
        return real_import(name)

    monkeypatch.setattr(governor, 'import_module', _mock_import)

    cfg = governor.KBGovernorConfig(
        problem_id='demo',
        reference_code='class Model: pass',
        isolate_evaluation_process=False,
    )
    kb_gov = governor.KBGovernor(cfg)
    result = kb_gov._evaluate_candidate('class ModelNew: pass', attempt=1)

    assert result.compiled is False
    assert result.correct is False
    assert result.error_message == 'kernelbench.eval unavailable in this environment'


def test_kb_governor_reports_missing_reference_code(monkeypatch) -> None:
    monkeypatch.setattr(governor, 'import_module', lambda _name: object())

    cfg = governor.KBGovernorConfig(problem_id='demo')
    kb_gov = governor.KBGovernor(cfg)
    result = kb_gov._evaluate_candidate('class ModelNew: pass', attempt=1)

    assert result.compiled is False
    assert result.correct is False
    assert result.error_message == 'missing reference_code for KernelBench evaluation'


def test_kb_governor_calls_kernelbench_eval(monkeypatch) -> None:
    class _FakeResult:
        compiled = True
        correctness = True
        runtime = 2.0
        ref_runtime = 5.0
        metadata = {}

    class _FakeEval:
        @staticmethod
        def get_torch_dtype_from_string(precision: str) -> str:
            return f'dtype:{precision}'

        @staticmethod
        def eval_kernel_against_ref(
            reference_code: str,
            candidate_code: str,
            *,
            backend: str,
            precision: str,
            measure_performance: bool,
            build_dir: str | None = None,
        ) -> _FakeResult:
            assert reference_code == 'class Model: pass'
            assert candidate_code == 'class ModelNew: pass'
            assert backend == 'cuda'
            assert precision == 'dtype:fp32'
            assert measure_performance is True
            assert build_dir is not None
            return _FakeResult()

    monkeypatch.setattr(governor, 'import_module', lambda _name: _FakeEval)

    cfg = governor.KBGovernorConfig(
        problem_id='demo',
        reference_code='class Model: pass',
        isolate_evaluation_process=False,
    )
    kb_gov = governor.KBGovernor(cfg)
    result = kb_gov._evaluate_candidate('class ModelNew: pass', attempt=1)

    assert result.compiled is True
    assert result.correct is True
    assert result.speedup == 2.5
    assert result.error_message is None


def test_governor_uses_extractor_selected_l1_in_coder_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(governor, "import_module", lambda _name: _FakeEvalModule())

    l1_txt = tmp_path / "shared_l1.txt"
    l1_txt.write_text("# shared l1\n", encoding="utf-8")
    l1_jsonl = l1_txt.with_suffix(".jsonl")
    selected_entry = {
        "entry_id": "selected-memory-1",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "description": "Use vectorized loads",
        "content": "Prefer contiguous access in the inner loop.",
        "source": "summarizer",
    }
    non_selected_entry = {
        "entry_id": "skip-memory-2",
        "timestamp": "2026-04-15T00:01:00+00:00",
        "description": "DO_NOT_INCLUDE_DESC",
        "content": "DO_NOT_INCLUDE_CONTENT",
        "source": "summarizer",
    }
    l1_jsonl.write_text(
        json.dumps(selected_entry, ensure_ascii=False)
        + "\n"
        + json.dumps(non_selected_entry, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    captured_messages: list[list[dict[str, str]]] = []

    def _fake_coder(messages, **_kwargs):
        captured_messages.append(messages)
        return ('```python\nprint("candidate")\n```', 16, {"model_id": "fake-coder"})

    monkeypatch.setattr(governor, "call_coder_with_meta", _fake_coder)
    monkeypatch.setattr(
        governor,
        "call_extractor_with_meta",
        lambda *_a, **_k: (
            '{"selected_entry_ids": ["selected-memory-1"]}',
            8,
            {"model_id": "fake-extractor"},
        ),
    )
    monkeypatch.setattr(governor, "maybe_promote_l0_to_l1", lambda *_a, **_k: None)

    cfg = governor.KBGovernorConfig(
        problem_id="100",
        reference_code='print("ref")',
        level=1,
        run_name="extractor-selection-test",
        results_root=tmp_path,
        shared_l1_path=l1_txt,
        max_iterations=1,
        isolate_evaluation_process=False,
        enable_l1_extractor=True,
        extractor_max_memories=1,
    )

    result = governor.KBGovernor(cfg).run(task_prompt="Optimize this model")

    assert result.iterations_run == 1
    assert captured_messages
    user_prompt = captured_messages[0][1]["content"]
    assert "## Selected L1 memory" in user_prompt
    assert "selected-memory-1" in user_prompt
    assert "Use vectorized loads" in user_prompt
    assert "Prefer contiguous access in the inner loop." in user_prompt
    assert "DO_NOT_INCLUDE_DESC" not in user_prompt
    assert "DO_NOT_INCLUDE_CONTENT" not in user_prompt
    assert "best_speedup_so_far" in user_prompt


def test_governor_promotion_keeps_l0_after_promotion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(governor, "import_module", lambda _name: _FakeEvalModule())
    monkeypatch.setattr(
        governor,
        "call_coder_with_meta",
        lambda *_a, **_k: ('```python\nprint("candidate")\n```', 32, {"model_id": "fake-coder"}),
    )

    captured_kwargs: dict[str, object] = {}

    def _capture_promote(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(governor, "maybe_promote_l0_to_l1", _capture_promote)

    cfg = governor.KBGovernorConfig(
        problem_id="100",
        reference_code='print("ref")',
        level=1,
        run_name="promotion-keep-l0-test",
        results_root=tmp_path,
        max_iterations=1,
        isolate_evaluation_process=False,
    )

    result = governor.KBGovernor(cfg).run(task_prompt="Optimize this model")

    assert result.iterations_run == 1
    assert "clear_l0_after_promotion" in captured_kwargs
    assert captured_kwargs["clear_l0_after_promotion"] is False


def test_governor_allows_extractor_model_none_uses_env_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(governor, "import_module", lambda _name: _FakeEvalModule())

    l1_txt = tmp_path / "shared_l1.txt"
    l1_txt.write_text("# shared l1\n", encoding="utf-8")
    l1_jsonl = l1_txt.with_suffix(".jsonl")
    l1_jsonl.write_text(
        json.dumps(
            {
                "entry_id": "m1",
                "timestamp": "2026-04-15T00:00:00+00:00",
                "description": "d1",
                "content": "c1",
                "source": "summarizer",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    captured_extractor_model_ids: list[object] = []

    def _fake_extractor(_messages, **kwargs):
        captured_extractor_model_ids.append(kwargs.get("model_id"))
        return ('{"selected_entry_ids": ["m1"]}', 8, {"model_id": "fake-extractor"})

    monkeypatch.setattr(governor, "call_extractor_with_meta", _fake_extractor)
    monkeypatch.setattr(
        governor,
        "call_coder_with_meta",
        lambda *_a, **_k: ('```python\nprint("candidate")\n```', 32, {"model_id": "fake-coder"}),
    )
    monkeypatch.setattr(governor, "maybe_promote_l0_to_l1", lambda *_a, **_k: None)

    cfg = governor.KBGovernorConfig(
        problem_id="100",
        reference_code='print("ref")',
        level=1,
        run_name="extractor-model-none-test",
        results_root=tmp_path,
        shared_l1_path=l1_txt,
        max_iterations=1,
        isolate_evaluation_process=False,
        enable_l1_extractor=True,
        extractor_model=None,
    )

    governor.KBGovernor(cfg).run(task_prompt="Optimize this model")

    assert captured_extractor_model_ids
    assert captured_extractor_model_ids[0] is None


def test_summarizer_prompt_mentions_description_max_length() -> None:
    assert "Description" in governor.SUMMARIZER_SYSTEM_PROMPT
    assert "max" in governor.SUMMARIZER_SYSTEM_PROMPT.lower()
