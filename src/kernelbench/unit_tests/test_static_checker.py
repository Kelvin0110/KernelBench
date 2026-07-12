"""
Tests for kernel_static_checker.py

Validates that the static checker correctly identifies:
- Valid DSL kernels (no false positives)  
- Known adversarial/hack patterns (no false negatives)

We welcome contributions to improve the static checker and providing adversarial kernels.


Run with: uv run pytest src/kernelbench/unit_tests/test_static_checker.py -v
"""

import pytest
from pathlib import Path
from kernelbench.kernel_static_checker import (
    check_code_bypass,
    check_workload_shrink,
    validate_kernel_static,
)


# =============================================================================
# Fixtures - Common paths and helpers
# =============================================================================

@pytest.fixture
def prompts_dir():
    """Path to DSL example kernels."""
    return Path(__file__).parent.parent / "prompts"

@pytest.fixture
def test_kernels_dir():
    """Path to adversarial test kernels."""
    return Path(__file__).parent / "test_kernels"


def read_kernel(path: Path) -> str:
    """Read kernel code from file, skip if not found."""
    if not path.exists():
        pytest.skip(f"Kernel file not found: {path}")
    return path.read_text()


# =============================================================================
# Valid DSL Kernels - Should Pass (No False Positives)
# These are real, correct kernels from src/kernelbench/prompts/
# =============================================================================

def test_cuda_example_valid(prompts_dir):
    """Real CUDA kernel example should pass with default settings."""
    code = read_kernel(prompts_dir / "model_new_ex_add.py")
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    # May have warnings (F import), but should be valid
    assert valid or "import F" in str(warnings), f"CUDA example should pass: {errors}"


def test_triton_example_valid(prompts_dir):
    """Real Triton kernel example should pass."""
    code = read_kernel(prompts_dir / "model_new_ex_add_triton.py")
    valid, errors, warnings = validate_kernel_static(code, backend="triton")
    assert valid or len(warnings) > 0, f"Triton example should pass: {errors}"


def test_cute_example_valid(prompts_dir):
    """Real CuTe/CUTLASS kernel example should pass."""
    code = read_kernel(prompts_dir / "model_new_ex_add_cute.py")
    valid, errors, warnings = validate_kernel_static(code, backend="cute")
    assert valid or len(warnings) > 0, f"CuTe example should pass: {errors}"


def test_tilelang_example_valid(prompts_dir):
    """Real TileLang kernel example should pass."""
    code = read_kernel(prompts_dir / "model_new_ex_add_tilelang.py")
    valid, errors, warnings = validate_kernel_static(code, backend="tilelang")
    assert valid or len(warnings) > 0, f"TileLang example should pass: {errors}"


# =============================================================================
# Adversarial Kernels - Should Detect Issues  
# These are known hack patterns from test_kernels/
# =============================================================================

def test_stream_kernel_flagged(test_kernels_dir):
    """Non-default stream kernel should trigger stream_injection warning."""
    code = read_kernel(test_kernels_dir / "non_default_stream_kernel.py")
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    # Stream injection is in warnings by default
    all_messages = errors + warnings
    has_stream_warning = any("stream" in msg.lower() for msg in all_messages)
    # Note: The CUDA code is in a string literal, so static checker may not catch it
    # This test documents the limitation


def test_result_reuse_kernel_flagged(test_kernels_dir):
    """Result reuse (empty tensor) kernel - static checker can't catch this."""
    code = read_kernel(test_kernels_dir / "result_reuse_kernel.py")
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    # This is a runtime check, static checker won't catch it
    # Just verify it doesn't crash


def test_zero_out_kernel_flagged(test_kernels_dir):
    """Zero-out kernel - static checker can't catch this."""
    code = read_kernel(test_kernels_dir / "zero_out_kernel.py")
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    # This is a correctness issue, not detectable statically


# =============================================================================
# Hack Patterns - Synthetic Examples
# =============================================================================

def test_bypass_try_except():
    """Try-except fallback should be flagged as error."""
    code = """
try:
    result = custom_kernel(x)
except:
    result = torch.matmul(x, w)  # Fallback to torch
"""
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Try-except should be flagged"
    assert any("try-except" in e.lower() for e in errors)


def test_bypass_pass_statement():
    """Pass statement (inheritance bypass) should be flagged."""
    code = """
class ModelNew(Model):
    def forward(self, x):
        pass  # Does nothing, inherits parent
"""
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Pass statement should be flagged"
    assert any("pass" in e.lower() for e in errors)


def test_bypass_pass_prose_in_docstring_not_flagged():
    """Prose like 'forward pass' in docstrings must not trigger pass bypass."""
    code = '''
class ModelNew(nn.Module):
    """The heavy part of the forward pass is compiled with torch.compile."""
    def forward(self, x):
        return x
'''
    flagged, msg = check_code_bypass(code)
    assert not flagged, msg


def test_bypass_pass_prose_in_string_not_flagged():
    """The word pass inside string literals must not trigger bypass."""
    code = '''
note = "must pass validation before launch"
'''
    flagged, msg = check_code_bypass(code)
    assert not flagged, msg


def test_bypass_try_in_string_not_flagged():
    """try: inside string literals must not trigger bypass."""
    code = '''
doc = """
try:
    this is documentation only
"""
'''
    flagged, msg = check_code_bypass(code)
    assert not flagged, msg


def test_bypass_forward_pass_with_cuda_kernel_not_blocked_by_pass():
    """Identity CUDA kernel + forward pass docstring should pass code_bypass."""
    code = '''
import torch
from torch.utils.cpp_extension import load_inline

_identity_cuda_src = r"""
#include <torch/extension.h>
__global__ void identity_kernel(float* data, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { data[idx] = data[idx]; }
}
torch::Tensor identity_cuda(torch::Tensor inp) { return inp; }
"""
_ext = load_inline(
    name="identity_ext",
    cpp_sources="torch::Tensor identity_cuda(torch::Tensor inp);",
    cuda_sources=_identity_cuda_src,
    functions=["identity_cuda"],
    verbose=False,
)

class ModelNew(torch.nn.Module):
    """The heavy part of the forward pass is compiled with torch.compile."""
    def forward(self, x):
        return _ext.identity_cuda(x)
'''
    flagged, msg = check_code_bypass(code)
    assert not flagged, msg
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not any("pass" in e.lower() for e in errors), errors


def test_lazy_eval_make_subclass():
    """_make_subclass (lazy tensor hack) should be flagged."""
    code = """
fake_tensor = torch.Tensor._make_subclass(FakeTensor, real_tensor)
"""
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "_make_subclass should be flagged"


def test_timing_monkey_patch():
    """Monkey patching timing functions should be flagged."""
    code = """
# Override timing to fake benchmarks
torch.cuda.Event.elapsed_time = lambda self, end: 0.001
"""
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Timing monkey patch should be flagged"


def test_thread_injection():
    """Threading in kernel code should be flagged."""
    code = """
import threading
t = threading.Thread(target=background_work)
t.start()
"""
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Threading should be flagged"


def test_workload_shrink_warning():
    """Redefining shape globals or get_inputs should be detected."""
    code = """
batch_size = 1
dim1 = 1
dim2 = 2

def get_inputs():
    return [torch.rand(batch_size, 64, dim1, dim2)]
"""
    has_issue, msg = check_workload_shrink(code)
    assert has_issue
    assert "workload" in msg.lower() or "get_inputs" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
