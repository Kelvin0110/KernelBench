"""Tests for KernelBench integrity prompts."""

from __future__ import annotations

from kernelbench_integration.prompts import (
    CODER_SYSTEM_PROMPT,
    KERNELBENCH_INTEGRITY_RULES,
    LEGACY_CODER_SYSTEM_PROMPT,
)


def test_hybrid_guidance_present() -> None:
    for prompt in (KERNELBENCH_INTEGRITY_RULES, CODER_SYSTEM_PROMPT, LEGACY_CODER_SYSTEM_PROMPT):
        assert "Mixing custom kernels with native PyTorch" in prompt
        assert "expected and encouraged" in prompt
        assert "__global__" in prompt
        assert "load_inline" in prompt


def test_old_native_pytorch_ban_absent() -> None:
    for prompt in (KERNELBENCH_INTEGRITY_RULES, CODER_SYSTEM_PROMPT, LEGACY_CODER_SYSTEM_PROMPT):
        assert "Do not use native PyTorch operations" not in prompt
        assert "Always use custom kernels" not in prompt


def test_integrity_guidance_mentions_key_patterns() -> None:
    rules = KERNELBENCH_INTEGRITY_RULES
    assert "try/except" in rules
    assert "__global__" in rules
    assert "load_inline" in rules
    assert "Event.elapsed_time" in rules
    assert "pass" in rules


def test_dummy_noop_cuda_discouraged() -> None:
    rules = KERNELBENCH_INTEGRITY_RULES
    assert "placeholder or identity-copy kernel" in rules
    assert "meaningful work" in rules


def test_instructional_not_checker_catalog() -> None:
    rules = KERNELBENCH_INTEGRITY_RULES
    assert "Static check" not in rules
    assert "STRICT" not in rules
    assert "How to structure your submission" in rules
    assert "Required behavior on every forward pass" in rules


def test_hybrid_allowed_in_feedback_guidance() -> None:
    rules = KERNELBENCH_INTEGRITY_RULES
    assert "Using PyTorch alongside a real custom kernel is fine" in rules
