#!/usr/bin/env python3
"""
Test script to verify CheckpointFileLock synchronization works correctly
with concurrent access to checkpoint files.

Usage:
    python test_checkpoint_lock.py
"""

import os
import sys
import json
import time
import tempfile
import threading
from pathlib import Path

# Add the scripts_integration/docker directory to path so we can import the lock class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts_integration', 'docker'))

from docker_single_run import CheckpointFileLock


def test_lock_acquisition_and_release():
    """Test 1: Basic lock acquisition and release."""
    print("Test 1: Basic lock acquisition and release...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")

        try:
            with CheckpointFileLock(lock_file, timeout_secs=5):
                assert os.path.exists(lock_file), "Lock file should exist during lock acquisition"
                print("  ✓ Lock acquired successfully")

            assert not os.path.exists(lock_file), "Lock file should be cleaned up after exit"
            print("  ✓ Lock file cleaned up successfully")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False


def test_concurrent_lock_contention():
    """Test 2: Concurrent lock contention - ensure serialization."""
    print("\nTest 2: Concurrent lock contention (2 threads)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")
        test_file = os.path.join(tmpdir, "test_data.json")

        results = {"thread_1": [], "thread_2": []}
        lock_times = {"thread_1": [], "thread_2": []}

        def write_with_lock(thread_id, iterations=3):
            """Simulate writing to protected file."""
            for i in range(iterations):
                start_time = time.time()
                try:
                    with CheckpointFileLock(lock_file, timeout_secs=10):
                        acquire_time = time.time() - start_time
                        lock_times[thread_id].append(acquire_time)

                        # Read-modify-write
                        if os.path.exists(test_file):
                            with open(test_file) as f:
                                data = json.load(f)
                        else:
                            data = {"writes": []}

                        data["writes"].append({
                            "thread": thread_id,
                            "iteration": i,
                            "timestamp": time.time()
                        })

                        with open(test_file, "w") as f:
                            json.dump(data, f)

                        results[thread_id].append(i)
                        time.sleep(0.1)  # Simulate work

                except Exception as e:
                    print(f"    ✗ Thread {thread_id} failed: {e}")
                    return False
            return True

        # Create two threads that compete for the lock
        t1 = threading.Thread(target=write_with_lock, args=("thread_1", 3), daemon=False)
        t2 = threading.Thread(target=write_with_lock, args=("thread_2", 3), daemon=False)

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # Verify all iterations completed
        if len(results["thread_1"]) == 3 and len(results["thread_2"]) == 3:
            print(f"  ✓ Both threads completed all iterations")
            print(f"    Thread 1 average lock acquire time: {sum(lock_times['thread_1'])/len(lock_times['thread_1']):.4f}s")
            print(f"    Thread 2 average lock acquire time: {sum(lock_times['thread_2'])/len(lock_times['thread_2']):.4f}s")

            # Verify final file has all writes
            with open(test_file) as f:
                final_data = json.load(f)

            if len(final_data["writes"]) == 6:  # 3 + 3
                print(f"  ✓ All 6 writes preserved (no data loss)")
                return True
            else:
                print(f"  ✗ Expected 6 writes, got {len(final_data['writes'])}")
                return False
        else:
            print(f"  ✗ Thread 1 completed {len(results['thread_1'])}/3 iterations")
            print(f"  ✗ Thread 2 completed {len(results['thread_2'])}/3 iterations")
            return False


def test_lock_timeout():
    """Test 3: Lock timeout behavior."""
    print("\nTest 3: Lock timeout behavior...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")

        # Simulate stuck lock by creating pre-existing lock file
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)

        # Create a lock file that's fresh (not stale)
        with open(lock_file, "w") as f:
            f.write("")

        # Try to acquire with short timeout - should fail quickly
        start_time = time.time()
        try:
            with CheckpointFileLock(lock_file, timeout_secs=2):
                pass
            print(f"  ✗ Should have timed out")
            return False
        except RuntimeError as e:
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                print(f"  ✓ Timeout occurred after {elapsed:.1f}s as expected")
                print(f"    Error message: {str(e)[:60]}...")
                return True
            else:
                print(f"  ✗ Timeout too fast: {elapsed:.1f}s")
                return False
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return False


def test_stale_lock_detection():
    """Test 4: Stale lock detection and cleanup."""
    print("\nTest 4: Stale lock detection and cleanup...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")

        # Create a stale lock file (timestamp > 10 minutes old)
        with open(lock_file, "w") as f:
            f.write("")

        # Set modification time to 15 minutes ago
        old_mtime = time.time() - (15 * 60)
        os.utime(lock_file, (old_mtime, old_mtime))

        try:
            with CheckpointFileLock(lock_file, timeout_secs=5):
                print("  ✓ Stale lock was removed and new lock acquired")
                return True
        except Exception as e:
            print(f"  ✗ Failed to handle stale lock: {e}")
            return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing CheckpointFileLock Synchronization")
    print("=" * 70)

    tests = [
        test_lock_acquisition_and_release,
        test_concurrent_lock_contention,
        test_lock_timeout,
        test_stale_lock_detection,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
