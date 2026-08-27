"""Show which pairs a dedup threshold would actually collapse.

A similarity gate is only defensible if the pairs it removes are the same idea
restated, not distinct rules that happen to share vocabulary. This prints the
ranked pairs so the threshold can be chosen by reading them.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "Self-Evolving-Agent"))

from compare_designs import cosine, embed_all  # noqa: E402
from sweep_gates import load_arm, make_gate, run_gate  # noqa: E402


def main(arm: str, min_hit: float = 0.60) -> None:
    rd = Path(arm)
    rows, ca, off, nb, ent = load_arm(rd)
    promo = run_gate(rows, ca, off, make_gate(min_hit=min_hit, nb=nb), ent)
    entries = [ent.get(p["entry_id"], {}) for p in promo]
    ids = [p["entry_id"] for p in promo]
    vecs = embed_all(entries)
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pairs.append((cosine(vecs[i], vecs[j]), i, j))
    pairs.sort(reverse=True)
    print(f"{rd.name}  (hit>={min_hit}, {len(promo)} candidates)")
    print(f"{'cos':>7}  {'id':>4} {'title':50s} || {'id':>4} title")
    for s, i, j in pairs[:14]:
        ti = str(entries[i].get("title", ""))[:48]
        tj = str(entries[j].get("title", ""))[:48]
        mark = "DROP" if s >= 0.80 else "keep"
        print(f"{s:7.4f} {mark} {ids[i]:>4} {ti:50s} || {ids[j]:>4} {tj}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
        print()
