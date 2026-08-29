"""Did dedup actually produce a semantically diverse standing set?

CLAUDE.md 8.4: the one historical L2 arm promoted 9 rules of which 6 said the same
thing ("don't add a trivial kernel, fuse instead"), 69% of the standing text. That
is the defect --l2-dedup-similarity exists to fix, and 'dedup fired 13 times' is
only evidence of activity, not of the outcome.

This measures the outcome directly: embed each arm's FINAL standing rules and
report the pairwise cosine distribution. A deduped arm should have no pair at or
above its own threshold; an un-deduped arm is expected to.

Uses the same embedding path as skill merging (evolving_common.llm_client), so a
similarity here is comparable to the merge threshold.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Self-Evolving-Agent").resolve()))

ROOTS = [Path("runs_evolving/gpt-oss-120b/l2redesign"),
         Path("runs_evolving/gpt-oss-120b/l2quick")]


def cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def main() -> None:
    from evolving_common.llm_client import embed_texts_nvidia

    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            f = d / "l2_standing.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if len(rows) < 2:
                continue
            name = d.name.replace("base_agent_gpt_oss_120b_", "")
            i = name.find("_itr30_GH200")
            name = name[:i] if i >= 0 else name

            texts = [str(r.get("text") or "")[:2000] for r in rows]
            titles = [str(r.get("title") or r.get("entry_id") or "?") for r in rows]
            try:
                vecs = embed_texts_nvidia(texts)
            except Exception as exc:
                print(f"{name}: embedding failed ({type(exc).__name__}: {exc})")
                continue

            sims = [(cos(vecs[a], vecs[b]), titles[a], titles[b])
                    for a, b in itertools.combinations(range(len(rows)), 2)]
            sims.sort(reverse=True)
            n80 = sum(1 for s, _, _ in sims if s >= 0.80)
            chars = sum(len(t) for t in texts)
            print(f"\n{name}: {len(rows)} rules, {chars} chars, "
                  f"{len(sims)} pairs, {n80} at cosine >= 0.80  "
                  f"(max {sims[0][0]:.3f})")
            for s, t1, t2 in sims[:3]:
                print(f"    {s:.3f}  {t1[:36]:<36} | {t2[:36]}")


if __name__ == "__main__":
    main()
