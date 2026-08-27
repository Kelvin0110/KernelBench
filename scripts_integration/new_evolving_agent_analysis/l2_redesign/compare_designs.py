"""Definitive offline comparison of L2 promotion designs.

For each design, on each completed L2 arm, reports:
  n promoted, standing chars (real render_l2_entry), terminal prompt ratio,
  and semantic duplication (max/mean pairwise cosine, pairs above tau)
  measured with the same embedding model the skill-merge path uses.

Run with the repo venv (needs the embedding client):
  .venv/bin/python compare_designs.py <arm> [<arm> ...]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path("/localhome/local-tianzheng/KernelBench")
WORKTREE = HERE.parents[2]
sys.path.insert(0, str(WORKTREE / "Self-Evolving-Agent"))

from sweep_gates import load_arm, make_gate, run_gate  # noqa: E402

CONTROL_CODER_PROMPT_CHARS = 4190  # measured, CLAUDE.md 8.5


def _load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_VEC_CACHE: dict[str, list[float]] = {}


def entry_text(entry: dict) -> str:
    parts = [
        str(entry.get("title") or ""),
        str(entry.get("description") or entry.get("content") or ""),
        str(entry.get("trigger") or ""),
    ]
    return "\n".join(p for p in parts if p).strip()


def embed_all(entries: list[dict]) -> list[list[float]]:
    _load_env()
    from evolving_common.llm_client import embed_texts_nvidia

    texts = [entry_text(e) for e in entries]
    missing = [t for t in texts if t and t not in _VEC_CACHE]
    if missing:
        uniq = list(dict.fromkeys(missing))
        for t, v in zip(uniq, embed_texts_nvidia(uniq)):
            _VEC_CACHE[t] = v
    return [_VEC_CACHE.get(t, []) for t in texts]


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    n = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return n / (na * nb) if na and nb else 0.0


def dup_stats(entries: list[dict], tau: float) -> dict:
    if len(entries) < 2:
        return {"max": 0.0, "mean": 0.0, "pairs_over_tau": 0, "n_pairs": 0}
    vecs = embed_all(entries)
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine(vecs[i], vecs[j]))
    return {
        "max": round(max(sims), 4),
        "mean": round(sum(sims) / len(sims), 4),
        "pairs_over_tau": sum(1 for s in sims if s >= tau),
        "n_pairs": len(sims),
    }


def render_chars(entries: list[dict], mode: str) -> int:
    from evolving_common.governor.l2_promotion import L2PromotionConfig, render_l2_entry

    cfg = L2PromotionConfig(enabled=True, render=mode)
    total = 0
    for e in entries:
        try:
            total += len(render_l2_entry(e, cfg=cfg))
        except Exception:
            total += len(json.dumps(e))
    return total


def embed_dedup(tau: float):
    def sim(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        va, vb = embed_all([a, b])
        return cosine(va, vb)

    return sim


DESIGNS = [
    ("D0 SHIPPED  rate>=0.70                ", dict(min_rate=0.70)),
    ("D1 hit>=0.70                          ", dict(min_hit=0.70)),
    ("D2 hit>=0.70 + dedup.80               ", dict(min_hit=0.70, dedup_tau=0.80)),
    ("D3 hit>=0.60 + PER-PASS cap4 (shipped)", dict(min_hit=0.60, max_entries=4)),
    ("D4 hit>=0.60 + STANDING cap6          ", dict(min_hit=0.60, standing_cap=6)),
    ("D5 hit>=0.60 + STANDING cap6 + dedup  ", dict(min_hit=0.60, standing_cap=6, dedup_tau=0.80)),
    ("D6 hit>=0.60 + STANDING cap4 + dedup  ", dict(min_hit=0.60, standing_cap=4, dedup_tau=0.80)),
    ("D7 PROPOSED hit>=.60 cap6 dedup .78   ", dict(min_hit=0.60, standing_cap=6, dedup_tau=0.78)),
]


def main(arm_dirs: list[str], tau_report: float = 0.80, render: str = "verbatim") -> None:
    for a in arm_dirs:
        rd = Path(a)
        rows, ca, off, nb, ent = load_arm(rd)
        print("=" * 96)
        print(rd.name)
        print("=" * 96)
        print(f"  {'design':38s} {'n':>3s} {'chars':>7s} {'xctl':>6s} "
              f"{'dupmax':>7s} {'dupavg':>7s} {'pairs>=' + str(tau_report):>10s}  ids")
        for label, kw in DESIGNS:
            kw = dict(kw)
            tau = kw.pop("dedup_tau", None)
            gate = make_gate(nb=nb, dedup=embed_dedup(tau) if tau else None,
                             dedup_tau=tau, **kw)
            promo = run_gate(rows, ca, off, gate, ent)
            entries = [ent.get(p["entry_id"], {}) for p in promo]
            chars = render_chars(entries, render)
            ratio = (CONTROL_CODER_PROMPT_CHARS + chars) / CONTROL_CODER_PROMPT_CHARS
            d = dup_stats(entries, tau_report)
            ids = ",".join(p["entry_id"] for p in promo)
            print(f"  {label:38s} {len(promo):3d} {chars:7d} {ratio:6.2f} "
                  f"{d['max']:7.4f} {d['mean']:7.4f} {d['pairs_over_tau']:>4d}/{d['n_pairs']:<5d}  {ids}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
