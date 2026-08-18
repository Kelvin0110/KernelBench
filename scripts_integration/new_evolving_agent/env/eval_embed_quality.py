#!/usr/bin/env python
"""Choose the skill-merge embedding model on INTRINSIC quality, not on agreement
with the (currently dead) nv-embedcode-7b-v1.

Labels come from a merge run that really executed: each proposed cluster was
merged by an LLM and then unit-tested, giving 72 accepted / 134 rejected
clusters. A good embedding model should pull the skills of an *accepted* cluster
tighter together than those of a *rejected* one -- i.e. its geometry should
predict which merges actually work.

Metrics per candidate:
  AUC acc/rej   P(mean intra-cluster sim of an accepted cluster > a rejected one).
                0.5 = no signal, 1.0 = perfect. THE headline number.
  sep           mean intra-sim of accepted clusters minus that of rejected ones.
  AUC vs rand   accepted clusters vs random skill pairs. Sanity floor: any usable
                model should be well above 0.5 here.
  thresh        similarity cutoff matching the reference eligibility rate, to
                pass as --skill-merge-similarity.

Caveat: every cluster here was originally *proposed* by nv-embedcode, so this
re-ranks within its candidate set; a model that would discover entirely
different clusters gets no credit. It is still the only outcome-labelled signal
available, and it is unbiased across the candidates being compared.

    uv run python scripts_integration/new_evolving_agent/env/eval_embed_quality.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "Self-Evolving-Agent")

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from evolving_common.governor.skill_embedding import build_skill_embed_text  # noqa: E402

load_dotenv(".env")

REF_RUN = Path(
    "runs_evolving/inference_oss_120b/"
    "base_agent_oss120b_merge_only_sim_07_itr30_2026_08_05_15_49"
)
MAX_CLUSTERS_PER_CLASS = int(os.getenv("MAX_CLUSTERS", "30"))

INTEGRATE = ("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
INFERENCE = ("https://inference-api.nvidia.com/v1", os.getenv("NVIDIA_INF_API_KEY"))

# (label, endpoint, model_id, send_input_type)
CANDIDATES = [
    ("qwen3-embedding-0.6b", INFERENCE, "nvidia/qwen/qwen3-embedding-0.6b", False),
    ("llama-embed-nemotron-8b", INFERENCE, "nvidia/nvidia/llama-embed-nemotron-8b", False),
    ("gemini-embedding-001", INFERENCE, "gcp/google/gemini-embedding-001", False),
    ("gemini-embedding-2", INFERENCE, "gcp/google/gemini-embedding-2", False),
    ("text-embedding-3-large", INFERENCE, "openai/openai/text-embedding-3-large", False),
    ("nemotron-3-embed-1b", INFERENCE, "nvidia/nvidia/nemotron-3-embed-1b", False),
    ("llama-3.2-nv-embedqa-1b-v2", INFERENCE, "nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2", True),
    ("nv-embed-v1", INTEGRATE, "nvidia/nv-embed-v1", True),
]


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg), ties counted as 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def embed(endpoint, model, texts, input_type):
    base, key = endpoint
    client = OpenAI(base_url=base, api_key=key, max_retries=1, timeout=240)
    out = []
    for s in range(0, len(texts), 16):
        kw = {"extra_body": {"input_type": "passage"}} if input_type else {}
        r = client.embeddings.create(model=model, input=texts[s : s + 16], **kw)
        out.extend(
            list(d.embedding)
            for d in sorted(r.data, key=lambda d: int(getattr(d, "index", 0)))
        )
    return out


def main() -> None:
    skills = {}
    for line in (REF_RUN / "shared_l1.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            skills[str(e["entry_id"])] = e

    merges = [
        json.loads(line)
        for line in (REF_RUN / "l1_skill_merges.jsonl").read_text().splitlines()
        if line.strip()
    ]

    def clusters(status):
        out = []
        for m in merges:
            if m.get("status") != status:
                continue
            ids = [str(i) for i in m.get("source_entry_ids", []) if str(i) in skills]
            if len(ids) >= 2:
                out.append(ids)
        return out

    random.seed(0)
    acc = clusters("accepted")
    rej = clusters("rejected")
    random.shuffle(acc)
    random.shuffle(rej)
    acc = acc[:MAX_CLUSTERS_PER_CLASS]
    rej = rej[:MAX_CLUSTERS_PER_CLASS]

    ids = sorted({i for c in acc + rej for i in c})
    texts = [build_skill_embed_text(skills[i]) for i in ids]
    idx = {i: n for n, i in enumerate(ids)}

    print(f"reference run : {REF_RUN.name}")
    print(f"clusters      : {len(acc)} accepted / {len(rej)} rejected")
    print(f"skills to embed: {len(ids)}   max_chars={max(len(t) for t in texts)}\n")

    rand_pairs = [tuple(random.sample(range(len(ids)), 2)) for _ in range(400)]

    hdr = (
        f"{'candidate':<27} {'dim':>5} {'AUC acc/rej':>12} {'sep':>7} "
        f"{'AUC vs rand':>12} {'thresh':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, ep, model, itype in CANDIDATES:
        try:
            V = [l2(v) for v in embed(ep, model, texts, itype)]

            def intra(c):
                vs = [V[idx[i]] for i in c]
                ps = [
                    cos(vs[a], vs[b])
                    for a in range(len(vs))
                    for b in range(a + 1, len(vs))
                ]
                return sum(ps) / len(ps)

            pa = [intra(c) for c in acc]
            pr = [intra(c) for c in rej]
            rnd = [cos(V[a], V[b]) for a, b in rand_pairs]

            a1 = auc(pa, pr)
            a2 = auc(pa, rnd)
            sep = sum(pa) / len(pa) - sum(pr) / len(pr)
            # threshold reproducing nv-embedcode's 1.2% eligibility on random pairs
            thr = sorted(rnd, reverse=True)[max(0, int(0.012 * len(rnd)) - 1)]
            results.append((label, a1, a2, sep, thr))
            print(
                f"{label:<27} {len(V[0]):>5} {a1:>12.3f} {sep:>+7.3f} "
                f"{a2:>12.3f} {thr:>7.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<27} {'--':>5} {'FAIL':>12}  {type(exc).__name__}: {str(exc)[:40]}")

    if results:
        results.sort(key=lambda r: -r[1])
        b = results[0]
        print(f"\nbest discriminator: {b[0]}")
        print(f"  AUC accepted-vs-rejected = {b[1]:.3f}   sep = {b[3]:+.3f}")
        print(f"  suggested --skill-merge-similarity {b[4]:.2f}")


if __name__ == "__main__":
    main()
