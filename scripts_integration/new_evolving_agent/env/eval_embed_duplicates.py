#!/usr/bin/env python
"""Rank skill-merge embedding candidates by NEAR-DUPLICATE RETRIEVAL, which is
what the merge pass actually does.

Ground truth without circularity: skill refinement rewrites a skill in place and
records `parent_id` / `refinement_round`. A refined child and its parent are the
SAME skill in different words -- exactly the relation merge must detect. Those
pairs are true positives, and nothing about them came from an embedding model,
so no candidate is favoured.

Task: embed a corpus of skills; for each refined child, rank every other skill by
cosine and ask where its true parent lands.

  recall@1   fraction of children whose parent is the single nearest skill
             (the metric that matters -- clustering merges nearest neighbours)
  recall@5   parent within the top 5
  MRR        mean reciprocal rank of the true parent
  AUC        P(parent pair scores above a random pair)
  thresh     cutoff admitting ~1.2% of random pairs, matching the eligibility
             rate nv-embedcode had at 0.7 -> pass as --skill-merge-similarity

    uv run python scripts_integration/new_evolving_agent/env/eval_embed_duplicates.py
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
    "base_agent_oss120b_deletion_merge_refine_sim_07_itr30_2026_08_09_13_48"
)
N_DISTRACTORS = int(os.getenv("N_DISTRACTORS", "180"))

INTEGRATE = ("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
INFERENCE = ("https://inference-api.nvidia.com/v1", os.getenv("NVIDIA_INF_API_KEY"))

CANDIDATES = [
    ("qwen3-embedding-0.6b", INFERENCE, "nvidia/qwen/qwen3-embedding-0.6b", False),
    ("llama-embed-nemotron-8b", INFERENCE, "nvidia/nvidia/llama-embed-nemotron-8b", False),
    ("gemini-embedding-001", INFERENCE, "gcp/google/gemini-embedding-001", False),
    ("text-embedding-3-large", INFERENCE, "openai/openai/text-embedding-3-large", False),
    ("text-embedding-3-small", INFERENCE, "openai/openai/text-embedding-3-small", False),
    ("nemotron-3-embed-1b", INFERENCE, "nvidia/nvidia/nemotron-3-embed-1b", False),
    ("llama-3.2-nv-embedqa-1b-v2", INFERENCE, "nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2", True),
    ("nv-embed-v1", INTEGRATE, "nvidia/nv-embed-v1", True),
]


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def embed(endpoint, model, texts, input_type):
    base, key = endpoint
    client = OpenAI(base_url=base, api_key=key, max_retries=2, timeout=240)
    out = []
    for s in range(0, len(texts), 16):
        kw = {"extra_body": {"input_type": "passage"}} if input_type else {}
        r = client.embeddings.create(model=model, input=texts[s : s + 16], **kw)
        batch = sorted(r.data, key=lambda d: int(getattr(d, "index", 0)))
        if len(batch) != len(texts[s : s + 16]):
            raise RuntimeError(f"returned {len(batch)} of {len(texts[s:s+16])}")
        out.extend(list(d.embedding) for d in batch)
    return out


def main() -> None:
    rows = [
        json.loads(line)
        for line in (REF_RUN / "shared_l1.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_id = {str(r["entry_id"]): r for r in rows}

    pairs = [
        (str(r["entry_id"]), str(r["parent_id"]))
        for r in rows
        if r.get("parent_id") and str(r["parent_id"]) in by_id
    ]
    # dedupe children, keep pairs whose text actually differs
    seen, gold = set(), []
    for child, parent in pairs:
        if child in seen:
            continue
        if build_skill_embed_text(by_id[child]) == build_skill_embed_text(by_id[parent]):
            continue
        seen.add(child)
        gold.append((child, parent))

    involved = {i for p in gold for i in p}
    others = [i for i in by_id if i not in involved]
    random.seed(0)
    distract = random.sample(others, min(N_DISTRACTORS, len(others)))

    corpus = sorted(involved | set(distract))
    idx = {i: n for n, i in enumerate(corpus)}
    texts = [build_skill_embed_text(by_id[i]) for i in corpus]

    print(f"reference run : {REF_RUN.name}")
    print(f"gold pairs    : {len(gold)} (refined child -> parent)")
    print(f"corpus        : {len(corpus)} skills  max_chars={max(len(t) for t in texts)}")
    print(f"chance recall@1 ~ {1.0/(len(corpus)-1):.3%}\n")

    rand_pairs = [tuple(random.sample(range(len(corpus)), 2)) for _ in range(600)]

    hdr = (
        f"{'candidate':<27} {'dim':>5} {'recall@1':>9} {'recall@5':>9} "
        f"{'MRR':>6} {'AUC':>6} {'thresh':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, ep, model, itype in CANDIDATES:
        try:
            V = [l2(v) for v in embed(ep, model, texts, itype)]
            hits1 = hits5 = 0
            rr = 0.0
            pos_sims = []
            for child, parent in gold:
                ci, pi = idx[child], idx[parent]
                sims = [
                    (cos(V[ci], V[j]), j) for j in range(len(corpus)) if j != ci
                ]
                sims.sort(key=lambda t: -t[0])
                rank = next(k for k, (_, j) in enumerate(sims, 1) if j == pi)
                hits1 += rank == 1
                hits5 += rank <= 5
                rr += 1.0 / rank
                pos_sims.append(cos(V[ci], V[pi]))
            n = len(gold)
            rnd = [cos(V[a], V[b]) for a, b in rand_pairs]
            wins = sum((p > r) + 0.5 * (p == r) for p in pos_sims for r in rnd)
            a = wins / (len(pos_sims) * len(rnd))
            thr = sorted(rnd, reverse=True)[max(0, int(0.012 * len(rnd)) - 1)]
            results.append((label, hits1 / n, hits5 / n, rr / n, a, thr, len(V[0])))
            print(
                f"{label:<27} {len(V[0]):>5} {hits1/n:>8.1%} {hits5/n:>9.1%} "
                f"{rr/n:>6.3f} {a:>6.3f} {thr:>7.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<27} {'--':>5} {'FAIL':>9}  {type(exc).__name__}: {str(exc)[:38]}")

    if results:
        results.sort(key=lambda r: (-r[1], -r[3]))
        b = results[0]
        print(f"\nbest near-duplicate retriever: {b[0]}  (dim={b[6]})")
        print(f"  recall@1={b[1]:.1%}  recall@5={b[2]:.1%}  MRR={b[3]:.3f}  AUC={b[4]:.3f}")
        print(f"  suggested --skill-merge-similarity {b[5]:.2f}")


if __name__ == "__main__":
    main()
