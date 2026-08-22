"""Tests for classifying build-lock contention vs a real compile failure.

A failed ninja build writes no shared object, so the load that follows raises
``ImportError: <name>.so: cannot open shared object file: No such file or
directory``. The classifier used to match on "No such file or directory" (and on
the bare substring "lock", which also occurs in "blockIdx"), so real compile
failures were reported as retryable and their diagnostics were discarded.
"""

from kernelbench.eval import is_transient_lock_error


def test_missing_shared_object_is_a_compile_failure_not_a_lock_error():
    """The regression: a failed build surfaces as a missing .so."""
    exc = ImportError(
        "/builds/l1_p100_iter1/hinge_loss_mod/hinge_loss_mod.so: "
        "cannot open shared object file: No such file or directory"
    )
    assert is_transient_lock_error(exc) is False


def test_nvcc_diagnostics_mentioning_blockidx_are_not_lock_errors():
    """"blockIdx" contains the substring "lock"; it must not trigger a retry."""
    exc = RuntimeError(
        "Error building extension 'custom_tanh': [1/3] c++ ... \n"
        "cuda.cu(12): error: identifier 'blockIdx' is undefined"
    )
    assert is_transient_lock_error(exc) is False


def test_deadlock_wording_is_not_a_lock_error():
    exc = RuntimeError("kernel launch failed: potential deadlock detected")
    assert is_transient_lock_error(exc) is False


def test_redefinition_error_is_a_compile_failure():
    exc = RuntimeError(
        "Error building extension 'mod': main.cpp:4: error: redefinition of "
        "'PyObject* PyInit_mod()'"
    )
    assert is_transient_lock_error(exc) is False


def test_genuine_lock_contention_is_still_retryable():
    for message in (
        "The file lock '/builds/mod/lock' could not be acquired",
        "Waiting for lock on /builds/mod",
        "failed to open .lock file",
        "flock: Resource temporarily unavailable",
    ):
        assert is_transient_lock_error(RuntimeError(message)) is True, message
