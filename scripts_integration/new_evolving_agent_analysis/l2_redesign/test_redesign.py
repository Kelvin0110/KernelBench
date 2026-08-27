"""Unit tests for the L2 redesign.

The first test is the important one: with every new knob at its default, the
gate must behave EXACTLY as it did before, so an unflagged arm is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Self-Evolving-Agent"))

from evolving_common.governor.l2_promotion import (  # noqa: E402
    L2Candidate,
    L2PromotionConfig,
    passes_floors,
    score_candidate,
    select_l2_promotions,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def mk(eid, sel=80, tasks=5, rate=0.9, nb=10, offers=100, hits=None, title=None, desc=None):
    hit = (hits / offers) if hits is not None and offers else 0.0
    c = L2Candidate(
        entry_id=eid,
        entry={
            "entry_id": eid,
            "title": title or f"title {eid}",
            "description": desc or f"description {eid}",
            "trigger": "t",
        },
        total_selections=sel,
        tasks=tasks,
        opportunity=max(1, int(sel / rate)),
        rate=rate,
        new_bests=nb,
        total_offers=offers,
        hit_rate=hit,
    )
    c.score = score_candidate(rate, tasks, nb)
    return c


def fake_embed(dim_map):
    """Deterministic embeddings: identical text -> identical vector."""
    def _embed(texts):
        out = []
        for t in texts:
            out.append(dim_map.get(t.split("\n")[0], [1.0, 0.0, 0.0]))
        return out
    return _embed


print("1) DEFAULTS ARE INERT")
cfg_default = L2PromotionConfig(enabled=True)
check("use_hit_rate defaults off", cfg_default.use_hit_rate is False)
check("standing_cap defaults 0", cfg_default.standing_cap == 0)
check("dedup_similarity defaults 0", cfg_default.dedup_similarity == 0.0)
c_lowhit = mk("1", rate=0.9, offers=100, hits=1)  # hit_rate 0.01, rate 0.9
check("default gate ignores hit_rate", passes_floors(c_lowhit, cfg_default) is True,
      "rate 0.9 passes despite hit_rate 0.01")
sel = select_l2_promotions([mk(str(i)) for i in range(1, 6)], cfg_default)
check("default gate promotes all eligible (no cap)", len(sel) == 5)

print()
print("2) HIT-RATE GATE REPLACES, NOT ADDS")
cfg_hit = L2PromotionConfig(enabled=True, use_hit_rate=True, min_hit_rate=0.60)
check("high rate + low hit_rate now REJECTED", passes_floors(c_lowhit, cfg_hit) is False)
c_lowrate_highhit = mk("2", rate=0.10, offers=100, hits=90)
check("low rate + high hit_rate now ACCEPTED",
      passes_floors(c_lowrate_highhit, cfg_hit) is True,
      "this is the gpt-oss case the shipped gate rejected")

print()
print("3) MISSING OFFER EVIDENCE FAILS CLOSED")
c_nooffers = mk("3", offers=0, hits=None)
check("offers=0 -> hit_rate 0 -> rejected", passes_floors(c_nooffers, cfg_hit) is False,
      "must never score an unmeasured skill as a perfect hit")

print()
print("4) STANDING CAP BOUNDS THE ACCUMULATED SET")
cfg_cap = L2PromotionConfig(enabled=True, standing_cap=4)
standing3 = [{"entry_id": f"s{i}", "entry": {"title": f"s{i}", "description": "d"}} for i in range(3)]
sel = select_l2_promotions([mk(str(i)) for i in range(10, 20)], cfg_cap, standing=standing3)
check("3 standing + cap 4 -> room for exactly 1", len(sel) == 1, f"got {len(sel)}")
standing4 = standing3 + [{"entry_id": "s3", "entry": {"title": "s3", "description": "d"}}]
sel = select_l2_promotions([mk(str(i)) for i in range(10, 20)], cfg_cap, standing=standing4)
check("cap already full -> room for 0", len(sel) == 0, f"got {len(sel)}")

print()
print("5) PER-PASS CAP IS STILL PER-PASS (unchanged semantics)")
cfg_pp = L2PromotionConfig(enabled=True, max_entries=4)
sel = select_l2_promotions([mk(str(i)) for i in range(10, 20)], cfg_pp, standing=standing3)
check("max_entries=4 with 3 standing still returns 4", len(sel) == 4, f"got {len(sel)}")

print()
print("6) ALREADY-STANDING CANDIDATES NO LONGER CONSUME CAP SLOTS")
cands = [mk("s0"), mk("s1"), mk("s2"), mk("new1"), mk("new2")]
sel = select_l2_promotions(cands, L2PromotionConfig(enabled=True, max_entries=2),
                           standing=standing3)
check("standing ids excluded before ranking",
      all(c.entry_id.startswith("new") for c in sel), f"got {[c.entry_id for c in sel]}")

print()
print("7) DEDUP DROPS RESTATEMENTS, KEEPS DISTINCT RULES")
V = {"dup A": [1.0, 0.0], "dup B": [0.999, 0.0447], "distinct": [0.0, 1.0]}
cfg_dd = L2PromotionConfig(enabled=True, dedup_similarity=0.80)
cands = [mk("a", title="dup A"), mk("b", title="dup B"), mk("c", title="distinct")]
sel = select_l2_promotions(cands, cfg_dd, standing=[], embed_fn=fake_embed(V))
check("near-duplicate dropped, distinct kept",
      [c.entry_id for c in sel] == ["a", "c"], f"got {[c.entry_id for c in sel]}")
check("dropped candidate records a reason",
      any("deduped" in r for r in cands[1].reasons), str(cands[1].reasons))

st = [{"entry_id": "x", "entry": {"title": "dup A", "description": "", "trigger": ""}}]
sel = select_l2_promotions([mk("b", title="dup B")], cfg_dd, standing=st,
                           embed_fn=fake_embed(V))
check("candidate duplicating a STANDING rule is dropped", len(sel) == 0, f"got {len(sel)}")

print()
print("8) DEDUP FAILS OPEN AND LOUD")
def boom(texts):
    raise RuntimeError("embedding endpoint down")
sel = select_l2_promotions([mk("a"), mk("b")], cfg_dd, standing=[], embed_fn=boom)
check("embedding failure -> candidates preserved, not silently zeroed", len(sel) == 2)

print()
print("=" * 60)
print("FAILED:", FAILED if FAILED else "none")
sys.exit(1 if FAILED else 0)
