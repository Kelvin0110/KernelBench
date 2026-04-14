from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_integration.new_evolving_agent import kb_governor as governor
from kernelbench.config import KBGovernorConfig
from kernelbench.schemas import KBEvalResult


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

    cfg = governor.KBGovernorConfig(
        problem_id='100',
        reference_code='print("ref")',
        level=1,
        run_name='new-agent-test',
        results_root=tmp_path,
        max_iterations=1,
        promote_entry_threshold=99,
        isolate_evaluation_process=False,
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

    cfg = governor.KBGovernorConfig(problem_id='demo', reference_code='class Model: pass')
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
