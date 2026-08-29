"""Tests for --redesign-l2 and the no-cap default.

Run:  .venv/bin/python .../l2_redesign/test_redesign_preset.py

Standalone rather than pytest, matching the other test_*.py in this directory
(pytest tries to collect them as modules and trips on their sys.exit).

The load-bearing guarantee is the FIRST test: without --redesign-l2 the shipped
gate must be byte-identical to what it was before the redesign existed. Everything
else in this project's L2 history is comparisons across arms, and a silently
changed default would invalidate every one of them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Self-Evolving-Agent"))

from evolving_common.governor.l2_promotion import (  # noqa: E402
    DEFAULT_L2_DEDUP_SIMILARITY,
    DEFAULT_L2_MIN_HIT_RATE,
    DEFAULT_L2_STANDING_CAP,
    L2_REDESIGN_PRESET,
    L2Candidate,
    L2PromotionConfig,
    select_l2_promotions,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def parse(argv: list[str]) -> dict:
    """Run the real CLI parser in --dry-run and return the recorded config.

    Goes through the actual argparse + _resolve_l2_preset path rather than
    reimplementing precedence, because precedence is the thing under test.
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable,
            str(REPO / "scripts_integration/new_evolving_agent/evolve_kb_batch.py"),
            "--run-name", "preset_probe",
            "--max-problems", "1", "--max-iterations", "1",
            "--model", "gpt-oss-120b",
            "--results-root", td,
            "--dry-run",
            *argv,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if p.returncode != 0:
            return {"__error__": (p.stderr or p.stdout)[-400:]}
        hits = sorted(Path(td).glob("**/run_summary.json"))
        if not hits:
            return {"__error__": "no run_summary.json written"}
        return json.loads(hits[0].read_text())


def cand(eid: str, score: float = 1.0) -> L2Candidate:
    return L2Candidate(
        entry_id=eid,
        entry={"entry_id": eid, "title": f"t{eid}", "content": f"body {eid}"},
        total_selections=99, tasks=5, opportunity=100, rate=0.99,
        new_bests=5, total_offers=100, hit_rate=0.99, score=score,
    )


print("=== defaults: the shipped gate must be untouched ===")
s = parse([])
check("no --enable-l2 -> l2 off", s.get("enable_l2") is False, str(s)[:200])
check("use_hit_rate defaults False", s.get("l2_use_hit_rate") is False)
check("dedup defaults 0", float(s.get("l2_dedup_similarity", -1)) == DEFAULT_L2_DEDUP_SIMILARITY)
check("standing_cap defaults -1 (no cap)", int(s.get("l2_standing_cap", 0)) == -1)
check("redesign_l2 recorded False", s.get("redesign_l2") is False)

s = parse(["--enable-l2"])
check("--enable-l2 alone keeps shipped metric",
      s.get("l2_use_hit_rate") is False and float(s.get("l2_dedup_similarity")) == 0.0,
      str(s)[:200])
check("--enable-l2 alone has no cap", int(s.get("l2_standing_cap")) == -1)

print("\n=== --redesign-l2 applies the preset ===")
s = parse(["--enable-l2", "--redesign-l2"])
check("redesign_l2 recorded True", s.get("redesign_l2") is True, str(s)[:200])
check("use_hit_rate on", s.get("l2_use_hit_rate") is True)
check("min_hit_rate 0.60", abs(float(s.get("l2_min_hit_rate")) - 0.60) < 1e-9)
check("dedup 0.80", abs(float(s.get("l2_dedup_similarity")) - 0.80) < 1e-9)
check("NO cap in the preset", int(s.get("l2_standing_cap")) == -1,
      f"got {s.get('l2_standing_cap')}")
check("preset table agrees with the constant",
      L2_REDESIGN_PRESET["standing_cap"] == DEFAULT_L2_STANDING_CAP == -1)

print("\n=== explicit flags beat the preset ===")
s = parse(["--enable-l2", "--redesign-l2", "--l2-min-hit-rate", "0.70",
           "--l2-standing-cap", "6", "--l2-dedup-similarity", "0.9"])
check("explicit hit-rate wins", abs(float(s.get("l2_min_hit_rate")) - 0.70) < 1e-9,
      str(s.get("l2_min_hit_rate")))
check("explicit cap wins", int(s.get("l2_standing_cap")) == 6)
check("explicit dedup wins", abs(float(s.get("l2_dedup_similarity")) - 0.9) < 1e-9)

print("\n=== --redesign-l2 without --enable-l2 is an error ===")
s = parse(["--redesign-l2"])
check("errors out", "__error__" in s and "requires --enable-l2" in s["__error__"],
      str(s)[:200])

print("\n=== cap sentinel semantics at the gate ===")
base = dict(enabled=True, min_tasks=0, min_selections=0, min_rate=0.0, min_new_bests=0)
cands = [cand(x) for x in "abcdef"]
for cap, want, label in ((-1, 6, "-1 = no cap"), (0, 6, "0 = legacy alias for no cap"),
                         (3, 3, "3 = cap at 3")):
    got = len(select_l2_promotions([cand(x) for x in "abcdef"],
                                   L2PromotionConfig(**base, standing_cap=cap)))
    check(f"standing_cap {label}", got == want, f"got {got} of 6")

print("\n=== a cap bounds the STANDING set across boundaries, not one pass ===")
cfg = L2PromotionConfig(**base, standing_cap=4)
standing = [{"entry_id": "a"}, {"entry_id": "b"}, {"entry_id": "c"}]
got = select_l2_promotions([cand(x) for x in "defgh"], cfg, standing=standing)
check("3 standing + cap 4 -> room for 1", len(got) == 1, f"got {len(got)}")

print("\n=== a cap refusal is recorded (CLAUDE.md 8.8) ===")
pool = [cand(x) for x in "defgh"]
select_l2_promotions(pool, L2PromotionConfig(**base, standing_cap=4), standing=standing)
refused = [c for c in pool if any(str(r).startswith("standing cap:") for r in c.reasons)]
check("refused candidates carry a reason", len(refused) == 4, f"got {len(refused)}")

print("\n=== no cap -> no refusal reasons ===")
pool = [cand(x) for x in "defgh"]
select_l2_promotions(pool, L2PromotionConfig(**base, standing_cap=-1), standing=standing)
check("nothing refused when uncapped",
      not any(c.reasons for c in pool), str([c.reasons for c in pool])[:160])

print("\n" + "=" * 60)
print("FAILED:", ", ".join(FAILED) if FAILED else "none")
sys.exit(1 if FAILED else 0)
