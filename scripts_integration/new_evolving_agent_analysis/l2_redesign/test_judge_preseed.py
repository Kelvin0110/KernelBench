"""Tests for the LLM judge, freeze, and pre-seed paths."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Self-Evolving-Agent"))

from evolving_common.governor.l2_promotion import (  # noqa: E402
    L2Candidate,
    L2PromotionConfig,
    load_l2_standing,
    preseed_l2_standing,
    resolve_l2_standing_path,
    score_candidate,
    select_l2_promotions,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def mk(eid: str, title: str = "t") -> L2Candidate:
    c = L2Candidate(
        entry_id=eid,
        entry={"entry_id": eid, "title": title, "description": "d", "trigger": "g"},
        total_selections=80, tasks=5, opportunity=100, rate=0.8, new_bests=10,
        total_offers=100, hit_rate=0.8,
    )
    c.score = score_candidate(0.8, 5, 10)
    return c


BASE = dict(enabled=True, judge=True, min_tasks=1, min_selections=1, min_rate=0.0)

print("1) JUDGE PICKS A SUBSET")
def good_call(messages, **kw):
    return ('{"promote": [{"entry_id": "2", "reason": "general and actionable"}]}', 0, {})
sel = select_l2_promotions([mk("1"), mk("2"), mk("3")], L2PromotionConfig(**BASE),
                           standing=[], call_fn=good_call)
check("judge subset honoured", [c.entry_id for c in sel] == ["2"],
      str([c.entry_id for c in sel]))
check("reason recorded", any("judge:" in r for r in sel[0].reasons), str(sel[0].reasons))

print()
print("2) JUDGE FAILS CLOSED (promotion is permanent, so never fail open)")
def boom(messages, **kw):
    raise RuntimeError("endpoint down")
sel = select_l2_promotions([mk("1"), mk("2")], L2PromotionConfig(**BASE),
                           standing=[], call_fn=boom)
check("LLM error -> promote nothing", sel == [], str([c.entry_id for c in sel]))

def junk(messages, **kw):
    return ("not json at all", 0, {})
sel = select_l2_promotions([mk("1")], L2PromotionConfig(**BASE), standing=[], call_fn=junk)
check("unparseable reply -> promote nothing", sel == [])

sel = select_l2_promotions([mk("1")], L2PromotionConfig(**BASE), standing=[], call_fn=None)
check("no call_fn -> promote nothing", sel == [])

print()
print("3) JUDGE CANNOT INVENT IDS")
def hallucinate(messages, **kw):
    return ('{"promote": [{"entry_id": "999"}, {"entry_id": "1"}]}', 0, {})
sel = select_l2_promotions([mk("1"), mk("2")], L2PromotionConfig(**BASE),
                           standing=[], call_fn=hallucinate)
check("unknown id dropped, real id kept", [c.entry_id for c in sel] == ["1"],
      str([c.entry_id for c in sel]))

print()
print("4) JUDGE RESPECTS THE STANDING CAP")
st3 = [{"entry_id": f"s{i}", "entry": {"title": f"s{i}"}} for i in range(3)]
def greedy(messages, **kw):
    return ('{"promote": [{"entry_id": "1"}, {"entry_id": "2"}, {"entry_id": "3"}]}', 0, {})
sel = select_l2_promotions([mk("1"), mk("2"), mk("3")],
                           L2PromotionConfig(**BASE, standing_cap=4),
                           standing=st3, call_fn=greedy)
check("cap 4 with 3 standing -> at most 1", len(sel) == 1, f"got {len(sel)}")
sel = select_l2_promotions([mk("1")], L2PromotionConfig(**BASE, standing_cap=3),
                           standing=st3, call_fn=greedy)
check("cap already full -> judge not even consulted", sel == [])

print()
print("5) FREEZE PROMOTES NOTHING")
sel = select_l2_promotions([mk("1"), mk("2")],
                           L2PromotionConfig(enabled=True, freeze=True,
                                             min_tasks=1, min_selections=1, min_rate=0.0),
                           standing=[])
check("freeze -> empty", sel == [])

print()
print("6) PRE-SEED INSTALLS AND SURVIVES")
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    l1 = tmp / "shared_l1.jsonl"
    l1.write_text("")
    src = tmp / "src_standing.jsonl"
    src.write_text("\n".join(json.dumps({
        "entry_id": str(i), "title": f"rule {i}", "render": "verbatim",
        "text": f"RULE {i}: do the thing.", "l2_meta": {},
    }) for i in (1, 2, 3)) + "\n")

    n = preseed_l2_standing(l1, src)
    check("all rows installed", n == 3, f"got {n}")
    rows = load_l2_standing(l1)
    check("standing set readable", len(rows) == 3, f"got {len(rows)}")
    check("rendered text preserved verbatim",
          rows[0]["text"] == "RULE 1: do the thing.", rows[0].get("text", ""))
    check("provenance stamped", all(r.get("preseeded_from") for r in rows))

    try:
        preseed_l2_standing(l1, src)
        check("refuses to overwrite a non-empty standing set", False)
    except FileExistsError:
        check("refuses to overwrite a non-empty standing set", True)

    audit = (resolve_l2_standing_path(l1).parent / "l2_promotions.jsonl")
    events = [json.loads(x)["event"] for x in audit.read_text().splitlines() if x.strip()]
    check("preseed audited", events.count("preseed") == 3, str(events))

    # A row with no text is not a usable standing rule.
    bad = tmp / "bad.jsonl"
    bad.write_text(json.dumps({"entry_id": "9", "title": "x"}) + "\n")
    l1b = tmp / "other" / "shared_l1.jsonl"
    l1b.parent.mkdir()
    l1b.write_text("")
    try:
        preseed_l2_standing(l1b, bad)
        check("rejects rows with no rendered text", False)
    except ValueError:
        check("rejects rows with no rendered text", True)

print()
print("=" * 60)
print("FAILED:", FAILED if FAILED else "none")
sys.exit(1 if FAILED else 0)
