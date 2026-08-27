"""Measure semantic duplication among a set of L1/L2 entries.

Uses the same embedding path the skill-merge machinery uses
(``evolving_common.llm_client.embed_texts_nvidia``), so a similarity threshold
chosen here transfers directly to a promotion-time dedup gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/localhome/local-tianzheng/KernelBench")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Self-Evolving-Agent"))


def _load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def entry_text(entry: dict) -> str:
    """Same fields the merge path embeds: title + description + trigger."""
    parts = [
        str(entry.get("title") or ""),
        str(entry.get("description") or entry.get("content") or ""),
        str(entry.get("trigger") or ""),
    ]
    return "\n".join(p for p in parts if p).strip()


def embed(texts: list[str]) -> list[list[float]]:
    _load_env()
    from evolving_common.llm_client import embed_texts_nvidia

    return embed_texts_nvidia(texts)


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def pairwise(entries: list[dict]) -> dict:
    texts = [entry_text(e) for e in entries]
    vecs = embed(texts)
    n = len(vecs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((round(cosine(vecs[i], vecs[j]), 4), i, j))
    pairs.sort(reverse=True)
    return {"n": n, "pairs": pairs, "vecs": vecs}


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    entries = data if isinstance(data, list) else data["entries"]
    out = pairwise(entries)
    print(f"n={out['n']}  top pairs by cosine:")
    for sim, i, j in out["pairs"][:20]:
        ti = entries[i].get("title", "")[:44]
        tj = entries[j].get("title", "")[:44]
        print(f"  {sim:.4f}  {ti:46s} || {tj}")
