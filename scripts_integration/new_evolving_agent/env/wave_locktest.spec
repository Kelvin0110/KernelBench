# Contention validation for the 2026-08-22 eval-deadline fix (submodule 7ac0e87).
# NOT a data run -- its speedups are deliberately contended and must be discarded.
#
# The mechanism is already unit-covered by
# Self-Evolving-Agent/tests/test_eval_timeout_excludes_lock_wait.py (4 passing:
# lock wait excluded from the budget, a real hang still times out, legacy
# two-arg workers still work, gpu_lock publishes its wait). What that cannot
# cover is two real arms contending on a real GPU, which is what CLAUDE.md 3.4
# asks for before committing a full wave.
#
#   KB_GPU_EVAL_LOCK_TIMEOUT_SEC=6 LAG_SEC=61 MAX_PROBLEMS=2 MAX_ITERATIONS=5 \
#     bash scripts_integration/new_evolving_agent/env/launch_wave.sh 1 \
#          scripts_integration/new_evolving_agent/env/wave_locktest.spec
#
# Timeout 6s is chosen against two constants, not picked round: gpu_lock only
# logs "waiting"/"acquired after" once a wait reaches _SLOW_WAIT_LOG_SEC = 5.0s,
# and a lock hold is ~4.9s (max 9.3s). At 6s a contended wait therefore logs
# "waiting" at 5s and then either acquires (logged) or times out into
# "proceeding UNLOCKED" at 6s -- every outcome observable. A timeout below 5s
# would time out before anything logged, making the audit vacuously clean.
#
# Pass criteria:
#   1. Both arms run to completion.
#   2. Orphaned waits == 0: every "waiting" pairs with an "acquired after" or an
#      UNLOCKED. A non-zero count is an eval killed mid-wait -- the exact pre-fix
#      corruption, which surfaces as compiled=False on a kernel that was fine.
#   3. "proceeding UNLOCKED" is reachable. Pre-fix the 600s eval deadline always
#      killed the waiter first, so it could never fire. Seeing it here proves the
#      child now outlives its wait. Zero occurrences means the arms never
#      actually contended -- lower the timeout, do not call that a pass.
#
# Distinct tags so the two arms cannot collide on a UTC-minute run-name.
#
# tag          | context-mode | extra flags
locktest_a     | truncation   |
locktest_b     | truncation   |
