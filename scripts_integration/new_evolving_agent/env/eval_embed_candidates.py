#!/usr/bin/env python
"""Pick a replacement for the (currently 500ing) nv-embedcode-7b-v1 skill-merge
embedding model, by measuring which candidate best reproduces its behaviour.

Ground truth: the cached nv-embedcode vectors from a merge run that actually
worked (206 merges). A good substitute must rank skill pairs the same way, so
the merge clusters stay comparable to the earlier runs.

Reported per candidate:
  spearman   rank correlation of all pairwise similarities vs nv-embedcode
  top-overlap  agreement on which pairs land in nv-embedcode's merge-eligible
               top band (the pairs that actually drive clustering)
  thresh@X%  the similarity cutoff reproducing nv-embedcode's eligibility rate,
             i.e. what to pass to --skill-merge-similarity for that model

    uv run python scripts_integration/new_evolving_agent/env/eval_embed_candidates.py
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
N_SKILLS = int(os.getenv("N_SKILLS", "40"))
REF_THRESHOLD = 0.70  # the value used in the successful nv-embedcode runs

INTEGRATE = ("https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY"))
INFERENCE = ("https://inference-api.nvidia.com/v1", os.getenv("NVIDIA_INF_API_KEY"))

# (label, endpoint, model_id, send_input_type)
CANDIDATES = [
    ("qwen3-embed-0.6b", INFERENCE, "nvidia/qwen/qwen3-embedding-0.6b", False),
    ("llama-embed-nemotron-8b", INFERENCE, "nvidia/nvidia/llama-embed-nemotron-8b", False),
    ("nemotron-3-embed-1b", INFERENCE, "nvidia/nvidia/nemotron-3-embed-1b", False),
    ("text-embedding-3-large", INFERENCE, "openai/openai/text-embedding-3-large", False),
    ("text-embedding-3-small", INFERENCE, "openai/openai/text-embedding-3-small", False),
    ("nv-embedqa-e5-v5", INFERENCE, "nvidia/nvidia/nv-embedqa-e5-v5", True),
    ("llama-3.2-nv-embedqa-1b-v2", INFERENCE, "nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2", True),
    ("nv-embed-v1 (integrate)", INTEGRATE, "nvidia/nv-embed-v1", True),
]


def l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def pairs(vs: list[list[float]]) -> list[float]:
    out = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            out.append(sum(a * b for a, b in zip(vs[i], vs[j])))
    return out


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def embed(endpoint, model: str, texts: list[str], input_type: bool) -> list[list[float]]:
    base, key = endpoint
    client = OpenAI(base_url=base, api_key=key, max_retries=1, timeout=180)
    out: list[list[float]] = []
    for s in range(0, len(texts), 16):
        chunk = texts[s : s + 16]
        kw = {"extra_body": {"input_type": "passage"}} if input_type else {}
        r = client.embeddings.create(model=model, input=chunk, **kw)
        out.extend(
            list(d.embedding)
            for d in sorted(r.data, key=lambda d: int(getattr(d, "index", 0)))
        )
    return out


def main() -> None:
    cache = json.loads((REF_RUN / "l1_skill_embeddings.json").read_text())
    skills = {
        json.loads(line)["entry_id"]: json.loads(line)
        for line in (REF_RUN / "shared_l1.jsonl").read_text().splitlines()
        if line.strip()
    }
    cached = cache["skills"]
    usable = [e for e in cached if e in skills and cached[e].get("vector")]
    random.seed(0)
    sample = sorted(random.sample(usable, min(N_SKILLS, len(usable))))

    ref = [l2(cached[e]["vector"]) for e in sample]
    ref_sims = pairs(ref)
    texts = [build_skill_embed_text(skills[e]) for e in sample]

    n = len(ref_sims)
    ref_elig = [s >= REF_THRESHOLD for s in ref_sims]
    k = sum(ref_elig)
    rate = k / n
    print(f"reference : nv-embedcode-7b-v1  (cache model={cache['model']})")
    print(f"skills={len(sample)}  pairs={n}  eligible@{REF_THRESHOLD}={k} ({rate:.1%})")
    print(f"max_chars={max(len(t) for t in texts)}\n")

    ref_top = {i for i, e in enumerate(ref_elig) if e}
    hdr = f"{'candidate':<26} {'dim':>5} {'spearman':>9} {'top-overlap':>12} {'thresh':>7}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, ep, model, itype in CANDIDATES:
        try:
            vs = [l2(v) for v in embed(ep, model, texts, itype)]
            sims = pairs(vs)
            rho = spearman(ref_sims, sims)
            # threshold reproducing the reference eligibility rate
            thr = sorted(sims, reverse=True)[k - 1] if k else float("nan")
            cand_top = {i for i, s in enumerate(sims) if s >= thr}
            overlap = len(ref_top & cand_top) / len(ref_top) if ref_top else float("nan")
            results.append((label, rho, overlap, thr))
            print(f"{label:<26} {len(vs[0]):>5} {rho:>9.3f} {overlap:>11.0%} {thr:>7.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<26} {'--':>5} {'FAIL':>9}  {type(exc).__name__}: {str(exc)[:44]}")

    if results:
        results.sort(key=lambda r: -(r[1] + r[2]))
        best = results[0]
        print(
            f"\nbest fidelity: {best[0]}  "
            f"(spearman={best[1]:.3f}, top-overlap={best[2]:.0%})"
        )
        print(f"  --skill-merge-similarity {best[3]:.2f}")


if __name__ == "__main__":
    main()
