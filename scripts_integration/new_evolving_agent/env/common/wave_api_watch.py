#!/usr/bin/env python3
"""Detached LLM/API-side health watchdog for a running wave.

wave_watch.sh + wave_collect.py already cover the EVAL side (OOM, eval timeout,
lock starvation, missing nvcc). Neither looks at the LLM side, which is where a
wave on a shared inference endpoint actually dies:

  * 429 rate limiting -- the endpoint is shared with other tenants, so saturation
    is not under our control. llm_client retries with backoff, so a few are
    harmless; a sustained rate means every arm is sleeping instead of working.
  * upstream 500s / request timeouts -- retried too, but a burst means the model
    is unhealthy.
  * EMPTY `content` -- for a reasoning model, max_tokens bounds reasoning+answer
    together. When reasoning exhausts it, `content` is empty and
    _assistant_visible_text SILENTLY substitutes truncated chain-of-thought. No
    error is raised anywhere; the agent just receives garbage. This is the single
    most important thing to watch on a qwen wave and nothing else checks it.

Run detached -- it must outlive the ssh session:

    cd /localhome/local-tianzheng/KernelBench
    setsid nohup ./.venv/bin/python \
      scripts_integration/new_evolving_agent/env/common/wave_api_watch.py \
      --results-root runs_evolving/qwen3.6-27b/ \
      --run-prefix base_agent_qwen3_6_27b --expect-arms 9 \
      >/dev/null 2>&1 < /dev/null &

    tail -f wave_api_watch.log
    grep -E "ALARM|WARN" wave_api_watch.log

Two traps this deliberately avoids, both documented in CLAUDE.md:
  * it filters on the LIVE PROCESS LIST, never on a directory glob -- a killed
    arm's directory keeps its old records and silently pollutes any glob;
  * it never concludes "all arms died" from its own empty read; a zero is
    cross-checked against ps before it is believed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

# substring -> counter. Matched case-insensitively against arm-log lines.
LOG_PATTERNS = {
    "ratelimiterror": "rate_limit",
    "error code: 429": "rate_limit",
    "internalservererror": "api_5xx",
    "error code: 500": "api_5xx",
    "error code: 502": "api_5xx",
    "error code: 503": "api_5xx",
    "apitimeouterror": "api_timeout",
    "coder_call_error": "coder_call_error",
    "extract_error": "extract_error",
    "budget": "budget",
}
# Rates above these (per arm, per poll) raise an ALARM rather than a WARN.
ALARM_RATE = {"rate_limit": 20, "api_5xx": 10, "api_timeout": 10, "coder_call_error": 5}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def live_arms(run_prefix: str) -> dict[str, str]:
    """run_name -> CUDA_VISIBLE_DEVICES, for arms actually running right now."""
    out: dict[str, str] = {}
    pids = sh(["pgrep", "-f", "evolve_kb_batch"]).split()
    for pid in pids:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0")
            if "--run-name" not in cmd:
                continue
            name = cmd[cmd.index("--run-name") + 1]
            if run_prefix and not name.startswith(run_prefix):
                continue
            env = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
            gpu = "?"
            for kv in env.split("\0"):
                if kv.startswith("CUDA_VISIBLE_DEVICES="):
                    gpu = kv.split("=", 1)[1] or "?"
            out[name] = gpu
        except (OSError, ValueError, IndexError):
            continue
    return out


def tail_new(path: str, offsets: dict[str, int]) -> list[str]:
    """Lines appended since the last poll. Handles truncation/rotation."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    prev = offsets.get(path, 0)
    if size < prev:          # rotated or truncated -> start over
        prev = 0
    if size == prev:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(prev)
            data = fh.read()
    except OSError:
        return []
    offsets[path] = size
    return data.decode("utf-8", errors="replace").splitlines()


def scan_logs(run_names: list[str], offsets: dict[str, int]) -> dict[str, dict[str, int]]:
    per_arm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in run_names:
        for log in [f for f in os.listdir(".") if f.startswith(name) and f.endswith(".log")]:
            for line in tail_new(log, offsets):
                low = line.lower()
                for pat, counter in LOG_PATTERNS.items():
                    if pat in low:
                        per_arm[name][counter] += 1
    return per_arm


def scan_content_field(results_root: str, run_names: list[str],
                       offsets: dict[str, int]) -> dict[str, tuple[int, int]]:
    """run_name -> (new llm turns, of which EMPTY content i.e. CoT substituted)."""
    out: dict[str, tuple[int, int]] = {}
    for name in run_names:
        total = empty = 0
        try:
            dirs = [d for d in os.listdir(results_root) if d.startswith(name)]
        except OSError:
            continue
        for d in dirs:
            ws = os.path.join(results_root, d, "workspaces")
            if not os.path.isdir(ws):
                continue
            for prob in os.listdir(ws):
                ch = os.path.join(ws, prob, "chat_history.jsonl")
                if not os.path.isfile(ch):
                    continue
                for line in tail_new(ch, offsets):
                    if '"llm_turn"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    extra = rec.get("extra") or {}
                    if not isinstance(extra, dict):
                        continue
                    field = extra.get("assistant_content_field")
                    if field is None:
                        continue
                    total += 1
                    if field in ("reasoning_content", "reasoning", "none"):
                        empty += 1
        if total:
            out[name] = (total, empty)
    return out


def newest_activity_min(results_root: str, run_names: list[str]) -> dict[str, float]:
    """Minutes since each arm last wrote an LLM turn -- stall detection."""
    now = time.time()
    out: dict[str, float] = {}
    for name in run_names:
        newest = 0.0
        try:
            dirs = [d for d in os.listdir(results_root) if d.startswith(name)]
        except OSError:
            continue
        for d in dirs:
            ws = os.path.join(results_root, d, "workspaces")
            if not os.path.isdir(ws):
                continue
            for prob in os.listdir(ws):
                ch = os.path.join(ws, prob, "chat_history.jsonl")
                try:
                    newest = max(newest, os.path.getmtime(ch))
                except OSError:
                    continue
        if newest:
            out[name] = (now - newest) / 60.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--run-prefix", default="")
    ap.add_argument("--expect-arms", type=int, default=0)
    ap.add_argument("--interval-sec", type=int, default=600)
    ap.add_argument("--stall-min", type=float, default=45.0,
                    help="ALARM if an arm has written no LLM turn for this long")
    ap.add_argument("--empty-pct-alarm", type=float, default=5.0,
                    help="ALARM if this %% of new LLM turns came back with empty content")
    ap.add_argument("--log", default="wave_api_watch.log")
    a = ap.parse_args()

    offsets: dict[str, int] = {}
    seen_arms: set[str] = set()
    cum: dict[str, int] = defaultdict(int)

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        with open(a.log, "a") as fh:
            fh.write(line + "\n")
            fh.flush()

    say(f"=== wave_api_watch start: root={a.results_root} prefix={a.run_prefix or '(any)'} "
        f"expect={a.expect_arms} interval={a.interval_sec}s ===")

    while True:
        try:
            arms = live_arms(a.run_prefix)
            names = sorted(arms)
            seen_arms.update(names)

            alarms: list[str] = []
            warns: list[str] = []

            # --- liveness. A zero is only believed after a second look. -------
            if not names:
                time.sleep(5)
                arms = live_arms(a.run_prefix)
                names = sorted(arms)
            if seen_arms and not names:
                finished = 0
                for n in seen_arms:
                    try:
                        finished += any(
                            os.path.isfile(os.path.join(a.results_root, d, "run_summary.json"))
                            for d in os.listdir(a.results_root) if d.startswith(n))
                    except OSError:
                        pass
                if finished == len(seen_arms):
                    say(f"ALL {finished} ARMS FINISHED (run_summary.json present). watcher exiting.")
                    return 0
                alarms.append(f"0 arms live but only {finished}/{len(seen_arms)} have run_summary.json "
                              f"-- arms DIED. Inspect logs, then resume_wave.sh.")
            missing = sorted(seen_arms - set(names))
            if missing:
                still = []
                for n in missing:
                    try:
                        done = any(os.path.isfile(os.path.join(a.results_root, d, "run_summary.json"))
                                   for d in os.listdir(a.results_root) if d.startswith(n))
                    except OSError:
                        done = False
                    if not done:
                        still.append(n)
                if still:
                    alarms.append(f"{len(still)} arm(s) vanished WITHOUT run_summary.json: "
                                  f"{', '.join(still)} -- died, needs resume")
            if a.expect_arms and len(names) < a.expect_arms and not missing:
                warns.append(f"only {len(names)}/{a.expect_arms} arms live (still launching?)")

            # --- LLM/API incidents, delta since last poll ---------------------
            per_arm = scan_logs(names, offsets)
            totals: dict[str, int] = defaultdict(int)
            for n, c in per_arm.items():
                for k, v in c.items():
                    totals[k] += v
                    cum[k] += v
            for k, v in sorted(totals.items()):
                worst = max((c.get(k, 0) for c in per_arm.values()), default=0)
                thr = ALARM_RATE.get(k)
                if thr and worst >= thr:
                    alarms.append(f"{k}: {v} this poll (worst single arm {worst} >= {thr})")
                elif v:
                    warns.append(f"{k}: {v} this poll (cum {cum[k]})")

            # --- empty content == silent CoT substitution ---------------------
            cf = scan_content_field(a.results_root, names, offsets)
            for n, (tot, empt) in sorted(cf.items()):
                if tot >= 20 and empt:
                    pct = 100.0 * empt / tot
                    msg = f"{n}: {empt}/{tot} new LLM turns returned EMPTY content ({pct:.1f}%)"
                    (alarms if pct >= a.empty_pct_alarm else warns).append(msg)

            # --- stalls -------------------------------------------------------
            for n, mins in sorted(newest_activity_min(a.results_root, names).items()):
                if mins >= a.stall_min:
                    alarms.append(f"{n}: no LLM turn for {mins:.0f} min (stalled)")

            gpus = defaultdict(int)
            for g in arms.values():
                gpus[g] += 1
            gpu_s = " ".join(f"gpu{g}={n}" for g, n in sorted(gpus.items()))
            say(f"arms={len(names)} [{gpu_s}] "
                f"rate_limit={totals.get('rate_limit',0)} api_5xx={totals.get('api_5xx',0)} "
                f"api_timeout={totals.get('api_timeout',0)} "
                f"empty_content={sum(e for _, e in cf.values())} "
                f"| cum rate_limit={cum['rate_limit']} api_5xx={cum['api_5xx']}")
            for w in warns:
                say(f"  WARN  {w}")
            for al in alarms:
                say(f"  ALARM {al}")
            if alarms:
                say("  ACTION: rate_limit/api_5xx bursts usually self-heal via backoff -- only stop if "
                    "sustained across polls. EMPTY content and stalls do NOT self-heal: stop the wave.")
        except Exception as exc:                      # never let the watcher die
            say(f"  WARN  watcher error (continuing): {type(exc).__name__}: {exc}")
        time.sleep(a.interval_sec)


if __name__ == "__main__":
    sys.exit(main())
