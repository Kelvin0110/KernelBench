#!/usr/bin/env python3
"""Detached LLM/API-side health watchdog for a running wave.

wave_watch.sh + wave_collect.py already cover the EVAL side (OOM, eval timeout,
lock starvation, missing nvcc). Neither looks at the LLM side, which is where a
wave on a shared inference endpoint actually dies.

WHAT KILLS A WAVE, in increasing order of how quietly it does it
----------------------------------------------------------------
  * 429 rate limiting -- the endpoint is shared, so saturation is not under our
    control. llm_client retries with backoff, so a few are harmless.
  * upstream 500s / connection drops / request timeouts -- retried too, but a
    burst means the model group is unhealthy.
  * EMPTY `content` -- for a reasoning model, max_tokens bounds reasoning+answer
    together. When reasoning exhausts it, `content` is empty and
    _assistant_visible_text SILENTLY substitutes truncated chain-of-thought.
  * ABANDONED PROBLEMS -- the one that actually destroyed a wave, and the one
    nothing used to watch. A coder exception that is not a timeout increments a
    per-problem `fatal_error_count`; at `max_fatal_errors` (config.py, default
    3, NEVER reset within a problem) the governor gives up on the problem and
    moves on. The batch then completes normally, writes run_summary.json, and
    reports 50/50 problems -- while most of those problems were thrown away at
    iteration 3 of 30.

    On the 2026-08-27 qwen3.6-27b wave this cost 399 of 450 problem-runs (89%)
    and 72% of all iterations, on all 9 arms roughly equally, and the previous
    version of THIS script watched it happen for 46 h without one alarm.

WHY THE ARM LOG IS NOT ENOUGH (the blind spot that was fixed 2026-08-29)
-----------------------------------------------------------------------
The arm log only ever carries EXTRACTOR and ACTION-SELECTOR failures --
non-fatal auxiliary roles. CODER failures, the only ones that can abandon a
problem, are never printed to it; they exist solely in the per-problem
artifacts. Tailing the log therefore measures endpoint weather, not damage:
on that wave the log showed 229 5xx while the artifacts showed 1,248 lost
coder iterations. Always read scan_iteration_errors() / scan_aborts() -- the
`coder_lost=` and `aborted=` fields -- not the log-pattern counters.

Run detached -- it must outlive the ssh session:

    cd /localhome/local-tianzheng/KernelBench
    setsid nohup ./.venv/bin/python \
      scripts_integration/new_evolving_agent/env/common/wave_api_watch.py \
      --results-root runs_evolving/qwen3.6-27b/ \
      --run-prefix base_agent_qwen3_6_27b --expect-arms 9 \
      >/dev/null 2>&1 < /dev/null &

    tail -f wave_api_watch.log
    grep -E "ALARM" wave_api_watch.log

One-shot triage of a wave that already finished (skips the live-process filter
and the stall check, exits 2 if anything ALARMs):

    ./.venv/bin/python .../wave_api_watch.py --once \
      --results-root runs_evolving/qwen3.6-27b/ --run-prefix base_agent_qwen3_6_27b

Calibration, measured on two real waves 2026-08-29:
    healthy (gpt-5.6-terra): coder_lost 0%,  aborts 2-4%   -> exit 0, no alarms
    ruined  (qwen3.6-27b):   coder_lost 33%, aborts 84-94% -> exit 2, 10 alarms

Two traps this deliberately avoids, both documented in CLAUDE.md:
  * it filters on the LIVE PROCESS LIST, never on a directory glob -- a killed
    arm's directory keeps its old records and silently pollutes any glob.
    (--once is the sole exception, and is safe precisely because nothing is
    live when you use it.)
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
#
# NOTE (2026-08-29): the arm log only ever carries EXTRACTOR and ACTION-SELECTOR
# failures. Coder failures -- the only ones that can abandon a problem -- are
# never printed to it; they exist solely in the per-problem artifacts. So these
# patterns are a leading indicator of endpoint health, NOT a damage measure.
# scan_iteration_errors() + scan_aborts() below are the authoritative signals.
LOG_PATTERNS = {
    "ratelimiterror": "rate_limit",
    "error code: 429": "rate_limit",
    "internalservererror": "api_5xx",
    "error code: 500": "api_5xx",
    "error code: 502": "api_5xx",
    "error code: 503": "api_5xx",
    "apiconnectionerror": "api_conn",
    "connection error": "api_conn",
    "server disconnected": "api_conn",
    "apitimeouterror": "api_timeout",
    "endpoint outage": "outage_wait",
    "coder_call_error": "coder_call_error",
    "extract_error": "extract_error",
    "budget": "budget",
}
# Rates above these (per arm, per poll) raise an ALARM rather than a WARN.
ALARM_RATE = {"rate_limit": 20, "api_5xx": 10, "api_conn": 10,
              "api_timeout": 10, "coder_call_error": 5}
# `outage_wait` is deliberately absent: it means the run is CORRECTLY waiting out
# an endpoint outage instead of burning problems, which is the desired behaviour.
# It suppresses the stall alarm (see below) rather than raising one.

# Classification of the per-iteration `error` string recorded in
# metrics_by_iteration.jsonl. Order matters -- first match wins.
# NOTE the order: litellm wraps upstream 500s in the text "Server disconnected",
# so a conn-first table silently reclassifies every 5xx as a connection error.
ITER_ERR_PATTERNS = [
    # The governor records an exhausted outage budget under its own prefix, so it
    # never looks like an ordinary coder fault (it does not burn a fatal life).
    ("coder/outage",  re.compile(r"endpoint_outage", re.I | re.S)),
    ("coder/5xx",     re.compile(r"coder_call_error.*(internalservererror|code: 5\d\d)", re.I | re.S)),
    ("coder/429",     re.compile(r"coder_call_error.*(ratelimit|code: 429)", re.I | re.S)),
    ("coder/conn",    re.compile(r"coder_call_error.*(apiconnectionerror|connection error|server disconnected)", re.I | re.S)),
    ("coder/timeout", re.compile(r"coder_call_error.*(timeout|timed out)", re.I | re.S)),
    ("coder/other",   re.compile(r"coder_call_error", re.I | re.S)),
]

# governor abandons a problem after this many NON-timeout coder exceptions.
# kernelbench_integration/config.py: max_fatal_errors, default 3. The counter is
# per problem and is never reset, so three scattered blips across 30 iterations
# are enough. Timeouts are exempt (is_timeout_error).
MAX_FATAL_DEFAULT = 3


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


def _arm_dirs(results_root: str, name: str) -> list[str]:
    try:
        return [os.path.join(results_root, d) for d in os.listdir(results_root)
                if d.startswith(name)]
    except OSError:
        return []


def scan_iteration_errors(results_root: str, run_names: list[str],
                          offsets: dict[str, int]) -> dict[str, dict[str, int]]:
    """run_name -> {class: count} over iterations recorded since the last poll.

    This is the CODER-side signal. It is the one that matters: a coder exception
    burns one of the problem's `max_fatal_errors` lives, and the arm log never
    shows it. Also counts `ok` (iterations that produced a real evaluation) so a
    rate can be reported rather than a bare count.
    """
    per_arm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in run_names:
        for d in _arm_dirs(results_root, name):
            ws = os.path.join(d, "workspaces")
            if not os.path.isdir(ws):
                continue
            for prob in os.listdir(ws):
                mi = os.path.join(ws, prob, "metrics_by_iteration.jsonl")
                if not os.path.isfile(mi):
                    continue
                for line in tail_new(mi, offsets):
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("record_type") != "metrics_by_iteration":
                        continue
                    err = (rec.get("metrics_iteration") or {}).get("error") or ""
                    if not err:
                        per_arm[name]["ok"] += 1
                        continue
                    for label, pat in ITER_ERR_PATTERNS:
                        if pat.search(err):
                            per_arm[name][label] += 1
                            break
                    else:
                        per_arm[name]["ok"] += 1   # an ordinary kernel/eval error
    return per_arm


def scan_aborts(results_root: str, run_names: list[str],
                known: set[str]) -> tuple[list[tuple[str, str, int, str]], dict[str, int]]:
    """Problems ABANDONED mid-run (run_finished.json carries metadata.error).

    This is the fatal outcome -- the wave can look perfectly healthy by every
    other measure while most of its problems are being thrown away at iteration
    3 of 30. Returns (new_aborts, per_arm_total).

    new_aborts: (run_name, problem, iterations_done, error_head)
    """
    new: list[tuple[str, str, int, str]] = []
    totals: dict[str, int] = defaultdict(int)
    for name in run_names:
        for d in _arm_dirs(results_root, name):
            ws = os.path.join(d, "workspaces")
            if not os.path.isdir(ws):
                continue
            for prob in os.listdir(ws):
                rf = os.path.join(ws, prob, "run_finished.json")
                if not os.path.isfile(rf):
                    continue
                try:
                    md = (json.load(open(rf)).get("metadata") or {})
                except (ValueError, OSError):
                    continue
                if not md.get("error"):
                    continue
                totals[name] += 1
                key = f"{name}/{prob}"
                if key in known:
                    continue
                known.add(key)
                mi = os.path.join(ws, prob, "metrics_by_iteration.jsonl")
                try:
                    n = sum(1 for _ in open(mi, errors="replace"))
                except OSError:
                    n = 0
                new.append((name, prob, n, str(md["error"])[:110]))
    return new, totals


def scan_fatal_pressure(results_root: str, run_names: list[str],
                        max_fatal: int) -> dict[str, tuple[str, int]]:
    """Problems currently IN PROGRESS that have already burned lives.

    Early warning: an arm at 2 of 3 on its live problem is one blip from losing
    it. Only considers problems without run_finished.json (i.e. still running).
    """
    out: dict[str, tuple[str, int]] = {}
    for name in run_names:
        worst = ("", 0)
        for d in _arm_dirs(results_root, name):
            ws = os.path.join(d, "workspaces")
            if not os.path.isdir(ws):
                continue
            for prob in os.listdir(ws):
                pdir = os.path.join(ws, prob)
                if os.path.isfile(os.path.join(pdir, "run_finished.json")):
                    continue
                mi = os.path.join(pdir, "metrics_by_iteration.jsonl")
                if not os.path.isfile(mi):
                    continue
                burned = 0
                try:
                    for line in open(mi, errors="replace"):
                        if "coder_call_error" not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        err = (rec.get("metrics_iteration") or {}).get("error") or ""
                        # timeouts are exempt from the fatal counter
                        if re.search(r"timeout|timed out", err, re.I):
                            continue
                        burned += 1
                except OSError:
                    continue
                if burned > worst[1]:
                    worst = (prob, burned)
        if worst[1]:
            out[name] = worst
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
    ap.add_argument("--once", action="store_true",
                    help="single pass then exit; also audits FINISHED runs "
                         "(skips the live-process filter). Use for post-hoc triage.")
    ap.add_argument("--max-fatal", type=int, default=MAX_FATAL_DEFAULT,
                    help="governor's max_fatal_errors; problems at max-1 raise an ALARM")
    ap.add_argument("--abort-pct-alarm", type=float, default=10.0,
                    help="ALARM if this %% of an arm's finished problems were abandoned")
    a = ap.parse_args()

    offsets: dict[str, int] = {}
    seen_arms: set[str] = set()
    cum: dict[str, int] = defaultdict(int)
    known_aborts: set[str] = set()

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        with open(a.log, "a") as fh:
            fh.write(line + "\n")
            fh.flush()

    say(f"=== wave_api_watch start: root={a.results_root} prefix={a.run_prefix or '(any)'} "
        f"expect={a.expect_arms} interval={a.interval_sec}s ===")

    while True:
        try:
            if a.once:
                # Post-hoc: the arms are gone from ps, so read the results root.
                # Safe here precisely BECAUSE nothing is live -- the killed-arm
                # glob trap in CLAUDE.md only bites while a wave is running.
                try:
                    arms = {re.sub(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}$", "", d): "?"
                            for d in os.listdir(a.results_root)
                            if d.startswith(a.run_prefix) and
                            os.path.isdir(os.path.join(a.results_root, d))}
                except OSError:
                    arms = {}
            else:
                arms = live_arms(a.run_prefix)
            names = sorted(arms)
            seen_arms.update(names)

            alarms: list[str] = []
            warns: list[str] = []

            # --- liveness. A zero is only believed after a second look. -------
            if a.once and not names:
                say(f"  ALARM no run dirs under {a.results_root} matching "
                    f"prefix {a.run_prefix!r}")
                return 1
            if not names and not a.once:
                time.sleep(5)
                arms = live_arms(a.run_prefix)
                names = sorted(arms)
            if seen_arms and not names and not a.once:
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
            missing = [] if a.once else sorted(seen_arms - set(names))
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
                thr = None if a.once else ALARM_RATE.get(k)
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

            # --- CODER-side errors, from the artifacts (arm log cannot show these)
            ie = scan_iteration_errors(a.results_root, names, offsets)
            iter_tot: dict[str, int] = defaultdict(int)
            for n, c in ie.items():
                for k, v in c.items():
                    iter_tot[k] += v
                    if k != "ok":
                        cum[k] += v
            burned_now = sum(v for k, v in iter_tot.items() if k != "ok")
            graded = burned_now + iter_tot.get("ok", 0)
            if graded >= 20:
                pct = 100.0 * burned_now / graded
                msg = (f"{burned_now}/{graded} new iterations ({pct:.0f}%) lost to CODER "
                       f"API errors -- " +
                       ", ".join(f"{k}={v}" for k, v in sorted(iter_tot.items()) if k != "ok"))
                if pct >= 25.0:
                    alarms.append(msg)
                elif burned_now:
                    warns.append(msg)

            # --- problems ABANDONED (the fatal outcome) -----------------------
            new_ab, ab_tot = scan_aborts(a.results_root, names, known_aborts)
            for n, prob, iters, err in new_ab[:6]:
                warns.append(f"{n}: problem {prob} ABANDONED after {iters}/30 iters -- {err}")
            if len(new_ab) > 6:
                warns.append(f"... and {len(new_ab)-6} more problems abandoned this poll")
            for n in names:
                tot = ab_tot.get(n, 0)
                if not tot:
                    continue
                fin = 0
                for d in _arm_dirs(a.results_root, n):
                    ws = os.path.join(d, "workspaces")
                    if os.path.isdir(ws):
                        fin += sum(os.path.isfile(os.path.join(ws, p, "run_finished.json"))
                                   for p in os.listdir(ws))
                if fin >= 5:
                    pct = 100.0 * tot / fin
                    m = f"{n}: {tot}/{fin} finished problems ABANDONED ({pct:.0f}%)"
                    (alarms if pct >= a.abort_pct_alarm else warns).append(m)

            # --- early warning: live problems that have burned fatal lives ----
            for n, (prob, burned) in sorted(scan_fatal_pressure(
                    a.results_root, names, a.max_fatal).items()):
                if burned >= a.max_fatal - 1:
                    alarms.append(f"{n}: problem {prob} has burned {burned}/{a.max_fatal} "
                                  f"fatal lives -- one more error abandons it")

            # --- stalls (meaningless once the run is over) --------------------
            # An arm waiting out an endpoint outage writes no LLM turns by
            # design -- the outage budget is hours, far past --stall-min. Report
            # it as a wait, not as a hang, or every outage looks like a crash.
            waiting_out = {n for n, c in per_arm.items() if c.get("outage_wait")}
            for n, mins in sorted(newest_activity_min(a.results_root, names).items()):
                if a.once:
                    break
                if n in waiting_out:
                    warns.append(f"{n}: no LLM turn for {mins:.0f} min -- waiting out an "
                                 f"endpoint outage (expected; it will stop the arm if the "
                                 f"outage outlives its budget)")
                elif mins >= a.stall_min:
                    alarms.append(f"{n}: no LLM turn for {mins:.0f} min (stalled)")

            gpus = defaultdict(int)
            for g in arms.values():
                gpus[g] += 1
            gpu_s = " ".join(f"gpu{g}={n}" for g, n in sorted(gpus.items()))
            say(f"arms={len(names)} [{gpu_s}] "
                f"rate_limit={totals.get('rate_limit',0)} api_5xx={totals.get('api_5xx',0)} "
                f"api_conn={totals.get('api_conn',0)} "
                f"api_timeout={totals.get('api_timeout',0)} "
                f"outage_wait={totals.get('outage_wait',0)} "
                f"empty_content={sum(e for _, e in cf.values())} "
                f"| coder_lost={burned_now}/{graded} "
                f"aborted={sum(ab_tot.values())} "
                f"| cum rate_limit={cum['rate_limit']} api_5xx={cum['api_5xx']} "
                f"api_conn={cum['api_conn']}")
            for w in warns:
                say(f"  WARN  {w}")
            for al in alarms:
                say(f"  ALARM {al}")
            if alarms:
                say("  ACTION: rate_limit/api_5xx/api_conn bursts in the ARM LOG are extractor and "
                    "action-selector failures -- degraded but survivable, and they self-heal via "
                    "backoff. The ones that do NOT self-heal, and that cost real data: "
                    "coder_lost (burns max_fatal lives) and aborted (problem thrown away at "
                    "whatever iteration it reached). A sustained coder_lost above ~25%, or any "
                    "abort rate above ~10%, means the wave is producing unusable arms -- stop it "
                    "and requeue rather than let it run to a clean-looking completion.")
            if a.once:
                say(f"=== one-shot audit done: {len(names)} run dir(s), "
                    f"{sum(ab_tot.values())} abandoned problems ===")
                return 2 if alarms else 0
        except Exception as exc:                      # never let the watcher die
            say(f"  WARN  watcher error (continuing): {type(exc).__name__}: {exc}")
        time.sleep(a.interval_sec)


if __name__ == "__main__":
    sys.exit(main())
