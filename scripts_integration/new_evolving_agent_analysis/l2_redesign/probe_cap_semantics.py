"""Probe what --l2-max-entries actually caps.

CLAUDE.md 8.6 reads the cap as bounding the STANDING SET ("4 -> all three
distinct rules + one representative"). This checks that against the code.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Self-Evolving-Agent"))

from evolving_common.governor import l2_promotion as L  # noqa: E402
from evolving_common.governor.l2_promotion import (  # noqa: E402
    L2Candidate,
    L2PromotionConfig,
    score_candidate,
    select_l2_promotions,
)


def mk(eid: str, sel: int, tasks: int, rate: float, nb: int) -> L2Candidate:
    c = L2Candidate(
        entry_id=eid,
        entry={"entry_id": eid, "title": f"t{eid}", "description": "d", "trigger": "g"},
        total_selections=sel,
        tasks=tasks,
        opportunity=int(sel / rate),
        rate=rate,
        new_bests=nb,
    )
    c.score = score_candidate(rate, tasks, nb)
    return c


def main() -> None:
    cands = [mk(str(i), 80, 5, 0.9, 10) for i in range(1, 8)]
    cfg = L2PromotionConfig(enabled=True, max_entries=4)
    sel = select_l2_promotions(cands, cfg)
    print("A) cap=4 applied to 7 eligible ->", [c.entry_id for c in sel])
    print("   select_l2_promotions is called ONCE PER TASK BOUNDARY, and the")
    print("   standing set is accumulated across boundaries, so this is a")
    print("   PER-PASS cap, not a standing-set cap.")
    print()

    src_pass = inspect.getsource(L.run_l2_promotion_pass)
    src_cands = inspect.getsource(L.compute_l2_candidates)
    src_mod = inspect.getsource(L)

    print("B) compute_l2_candidates filters by tier?      ", "tier" in src_cands)
    print("   ...it reads:", [l.strip() for l in src_cands.splitlines()
                              if "read_selectable" in l])
    print("C) entry_tier occurrences in the whole module: ", src_mod.count("entry_tier"),
          "(1 == import only, never used)")
    print("D) the standing-skip happens AFTER the cap?    ",
          src_pass.index("select_l2_promotions") < src_pass.index("in standing_by_id"))
    print()
    print("=> Already-standing skills are still returned by compute_l2_candidates")
    print("   (read_selectable_l1_jsonl filters on STATUS, set_skill_tier writes TIER),")
    print("   are ranked, consume cap slots, and only then are skipped at :417.")
    print("   So --l2-max-entries N throttles new promotions once N incumbents")
    print("   outrank every newcomer -- it does not bound the standing set.")


if __name__ == "__main__":
    main()
