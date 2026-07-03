"""Tests for eval namespace isolation (workload global pollution)."""

from kernelbench.eval import _bind_function_to_context, load_original_model_and_inputs


REF_SRC = """
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

class Model:
    def __init__(self, num_features: int):
        self.num_features = num_features

    def forward(self, x):
        return x

def get_inputs():
    return [(batch_size, features, dim1, dim2)]

def get_init_inputs():
    return [features]
"""

CUSTOM_SRC = """
batch_size = 1
features = 64
dim1 = 1
dim2 = 2

class ModelNew:
    def __init__(self, num_features: int):
        self.num_features = num_features

    def forward(self, x):
        return x
"""


def test_get_inputs_uses_original_globals_after_custom_exec():
    original_context: dict = {}
    custom_context: dict = {}

    _model, _get_init, get_inputs = load_original_model_and_inputs(REF_SRC, original_context)
    assert get_inputs is not None
    assert get_inputs() == [(112, 64, 512, 512)]

    compile(CUSTOM_SRC, "<string>", "exec")
    exec(CUSTOM_SRC, custom_context)

    # Original-bound get_inputs must ignore custom_context pollution.
    assert get_inputs() == [(112, 64, 512, 512)]
    assert custom_context["batch_size"] == 1


def test_bind_function_to_context_pins_globals():
    ctx_a = {"value": 1}
    ctx_b = {"value": 99}
    exec("def reader():\n    return value", ctx_a)
    fn_a = _bind_function_to_context(ctx_a["reader"], ctx_a)
    assert fn_a() == 1

    exec("def reader():\n    return value", ctx_b)
    fn_b = _bind_function_to_context(ctx_b["reader"], ctx_b)
    assert fn_b() == 99
