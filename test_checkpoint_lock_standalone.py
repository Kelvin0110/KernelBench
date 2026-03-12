#!/usr/bin/env python3
"""
Simple unit test for CheckpointFileLock that doesn't require external dependencies.
"""

import os
import sys
import json
import time
import tempfile
import threading
import fcntl


class CheckpointFileLock:
    """Context manager for fcntl-based file locking of checkpoint files.

    Provides reliable cross-container synchronization for checkpoint-level file updates
    on local filesystems. Handles stale lock detection and cleanup to prevent deadlock
    from crashed containers.
    """

    def __init__(self, lock_file_path, timeout_secs=30):
        self.lock_file_path = lock_file_path
        self.timeout_secs = timeout_secs
        self.lock_fh = None

    def __enter__(self):
        """Acquire exclusive lock with timeout and stale lock detection."""
        os.makedirs(os.path.dirname(self.lock_file_path), exist_ok=True)

        # Stale lock detection
        if os.path.exists(self.lock_file_path):
            try:
                lock_age_secs = time.time() - os.path.getmtime(self.lock_file_path)
                if lock_age_secs > 600:
                    print(f"WARNING: Stale lock file {self.lock_file_path} (age: {lock_age_secs}s), removing.")
                    try:
                        os.remove(self.lock_file_path)
                    except OSError:
                        pass
            except Exception:
                pass

        # Retry loop
        start_time = time.time()
        while True:
            try:
                self.lock_fh = open(self.lock_file_path, "w")
                fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self

            except (IOError, BlockingIOError):
                if self.lock_fh:
                    self.lock_fh.close()
                    self.lock_fh = None

                elapsed = time.time() - start_time
                if elapsed > self.timeout_secs:
                    raise RuntimeError(
                        f"Could not acquire lock {self.lock_file_path} after {self.timeout_secs}s."
                    )
                time.sleep(0.5)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock and cleanup."""
        if self.lock_fh:
            try:
                fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                try:
                    self.lock_fh.close()
                except Exception:
                    pass

        try:
            os.remove(self.lock_file_path)
        except OSError:
            pass


def test_lock_acquisition_and_release():
    """Test 1: Basic lock acquisition and release."""
    print("Test 1: Basic lock acquisition and release...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")

        try:
            with CheckpointFileLock(lock_file, timeout_secs=5):
                assert os.path.exists(lock_file), "Lock file should exist"
                print("  ✓ Lock acquired")

            assert not os.path.exists(lock_file), "Lock file should be cleaned up"
            print("  ✓ Lock file cleaned up")
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False


def test_concurrent_lock_contention():
    """Test 2: Concurrent lock serialization."""
    print("\nTest 2: Concurrent lock contention (2 threads)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")
        test_file = os.path.join(tmpdir, "data.json")

        writes_completed = []
        errors = []

        def write_with_lock(thread_id, iterations=3):
            try:
                for i in range(iterations):
                    with CheckpointFileLock(lock_file, timeout_secs=10):
                        # Read-modify-write
                        if os.path.exists(test_file):
                            with open(test_file) as f:
                                data = json.load(f)
                        else:
                            data = {"writes": []}

                        data["writes"].append({"thread": thread_id, "iter": i})

                        with open(test_file, "w") as f:
                            json.dump(data, f)

                        writes_completed.append((thread_id, i))
                        time.sleep(0.05)  # Simulate work
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=write_with_lock, args=("thread_1", 3), daemon=False)
        t2 = threading.Thread(target=write_with_lock, args=("thread_2", 3), daemon=False)

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        if errors:
            print(f"  ✗ Errors occurred: {errors}")
            return False

        if len(writes_completed) == 6:
            print(f"  ✓ Both threads completed all iterations ({len(writes_completed)} writes)")

            with open(test_file) as f:
                final = json.load(f)

            if len(final["writes"]) == 6:
                print(f"  ✓ All 6 writes preserved (no data loss)")
                return True
            else:
                print(f"  ✗ Expected 6 writes, got {len(final['writes'])}")
                return False
        else:
            print(f"  ✗ Expected 6 writes, got {len(writes_completed)}")
            return False


def test_lock_timeout():
    """Test 3: Lock timeout."""
    print("\nTest 3: Lock timeout behavior...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        with open(lock_file, "w") as f:
            f.write("")

        start = time.time()
        try:
            with CheckpointFileLock(lock_file, timeout_secs=2):
                pass
            print(f"  ✗ Should have timed out")
            return False
        except RuntimeError:
            elapsed = time.time() - start
            if 2.0 <= elapsed <= 3.0:
                print(f"  ✓ Timeout after {elapsed:.1f}s (expected ~2s)")
                return True
            else:
                print(f"  ✗ Timeout too different: {elapsed:.1f}s")
                return False


def test_stale_lock_detection():
    """Test 4: Stale lock removal."""
    print("\nTest 4: Stale lock detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = os.path.join(tmpdir, ".test.lock")
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)

        with open(lock_file, "w") as f:
            f.write("")

        # Make it 15 minutes old
        old_mtime = time.time() - (15 * 60)
        os.utime(lock_file, (old_mtime, old_mtime))

        try:
            with CheckpointFileLock(lock_file, timeout_secs=5):
                print("  ✓ Stale lock was removed and new lock acquired")
                return True
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False


def main():
    print("=" * 70)
    print("Testing CheckpointFileLock Implementation")
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
