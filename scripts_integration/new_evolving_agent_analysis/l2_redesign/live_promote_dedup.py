"""Exercise run_l2_promotion_pass end to end with the REAL embedding client.

test_redesign.py stubs the embedder, and a 3x3 smoke run is too short to reach
the promote branch. This drives the actual promotion pass over a synthetic L1
catalog seeded with a known duplicate family, using the same
embed_texts_nvidia the skill-merge path uses -- so it proves the promote +
dedup + standing-cap path works against the live endpoint, not a fake.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path("/localhome/local-tianzheng/KernelBench")
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "Self-Evolving-Agent"))

for line in (REPO / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from evolving_common.governor.l2_promotion import (  # noqa: E402
    L2PromotionConfig,
    load_l2_standing,
    run_l2_promotion_pass,
)
from evolving_common.governor.skill_usage_tracker import (  # noqa: E402
    resolve_skill_usage_path,
    SkillUsageRecord,
    SkillUsageState,
    save_usage_state,
)

# Three restatements of one idea + two genuinely distinct rules.
SKILLS = [
    ("1", "Avoid Trivial Custom CUDA Kernels",
     "Replacing a cheap elementwise op with a standalone custom CUDA kernel adds launch overhead and loses fusion; keep the native path or fuse.",
     "When considering a custom kernel for a simple elementwise op."),
    ("2", "Avoid Trivial No-Op CUDA Kernels for Marginal Speedups",
     "A standalone kernel that only copies or scales data costs a launch and a round trip to memory; fuse it into an adjacent kernel instead.",
     "When a custom kernel performs a trivial elementwise transform."),
    ("3", "Fuse Compute Instead of Adding Trivial Kernels",
     "Adding a small kernel next to an existing one pays launch overhead twice; fuse the computation into the neighbouring kernel.",
     "When adding a kernel adjacent to an existing elementwise kernel."),
    ("4", "Read-Only Cache (__ldg) Boost for Vectorized CUDA Kernels",
     "Loading through __ldg routes reads via the read-only data cache, improving bandwidth for vectorized float4 loads on modern GPUs.",
     "When a bandwidth-bound kernel performs repeated global reads."),
    ("5", "Naive Direct Conv2D Kernel Indexing Pitfalls",
     "Direct conv2d kernels mis-index padded halos and stride offsets; tile over output and precompute input bases to keep addressing correct.",
     "When writing a direct convolution kernel by hand."),
]


def seed(tmp: Path) -> Path:
    l1 = tmp / "shared_l1.jsonl"
    with l1.open("w") as fh:
        for eid, title, desc, trig in SKILLS:
            fh.write(json.dumps({
                "entry_id": eid, "status": "active", "tier": "L1",
                "title": title, "description": desc, "trigger": trig,
                "content": desc, "source": "Level 1 problem 1",
            }) + "\n")
    state = SkillUsageState(global_iteration=120)
    for eid, *_ in SKILLS:
        state.skills[eid] = SkillUsageRecord(
            entry_id=eid, created_at_global_iter=0,
            total_selections=90, tasks_used=["L1P1", "L1P2", "L1P3", "L1P4"],
            new_best_attributions=9, total_offers=100,
        )
    save_usage_state(l1, state)
    return l1


def run(label: str, cfg: L2PromotionConfig, l1: Path) -> list[str]:
    for f in ("l2_standing.jsonl", "l2_promotions.jsonl"):
        p = l1.parent / f
        if p.exists():
            p.unlink()
    # reset tiers
    rows = [json.loads(x) for x in l1.read_text().splitlines() if x.strip()]
    for r in rows:
        r["tier"] = "L1"
        r.pop("l2_meta", None)
    l1.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary = run_l2_promotion_pass(l1, cfg)
    standing = load_l2_standing(l1)
    ids = [str(r["entry_id"]) for r in standing]
    titles = {str(r["entry_id"]): r.get("title") for r in standing}
    print(f"\n{label}")
    print(f"  promoted={summary['promoted']}  standing={len(standing)}  eligible={summary.get('eligible_count')}")
    for i in ids:
        print(f"    id={i}  {titles[i]}")
    return ids


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        l1 = seed(tmp)

        base = dict(enabled=True, use_hit_rate=True, min_hit_rate=0.60,
                    min_tasks=3, min_selections=50)

        # Measured pairwise cosine over this corpus, real embedder:
        #   1-2 0.8302   1-3 0.7419   2-3 0.7595   (the restatement family)
        #   family vs 4/5: 0.47-0.58   4 vs 5: 0.4748
        # Separation is clean, but the family straddles 0.80, so tau=0.80 must
        # collapse ONLY the 1-2 pair. Dedup is greedy-pairwise against already
        # kept rules, not transitive-closure clustering: keeping 1 drops 2
        # (0.83) but not 3 (0.74). tau=0.70 collapses the whole family.
        a = run("A) no dedup, no cap  -> all 5 promoted, duplicates included",
                L2PromotionConfig(**base), l1)
        b = run("B) dedup 0.80        -> drops only the >=0.80 pair (id 2)",
                L2PromotionConfig(**base, dedup_similarity=0.80), l1)
        d = run("D) dedup 0.70        -> whole restatement family collapses to 1",
                L2PromotionConfig(**base, dedup_similarity=0.70), l1)
        c = run("C) dedup 0.80 + cap 2 -> cap bounds the standing set",
                L2PromotionConfig(**base, dedup_similarity=0.80, standing_cap=2), l1)

        ok = True
        if len(a) != 5:
            print(f"\nFAIL: expected all 5 without dedup, got {len(a)}"); ok = False
        if set(b) != {"1", "3", "4", "5"}:
            print(f"\nFAIL: tau=0.80 should drop exactly id 2; got {b}"); ok = False
        if set(d) != {"1", "4", "5"}:
            print(f"\nFAIL: tau=0.70 should leave one of the family; got {d}"); ok = False
        if not ({"4", "5"} <= set(b) and {"4", "5"} <= set(d)):
            print("\nFAIL: dedup dropped a genuinely distinct rule"); ok = False
        if len(c) != 2:
            print(f"\nFAIL: standing cap 2 gave {len(c)}"); ok = False

        print("\n" + "=" * 62)
        print("VERDICT:", "PASS -- live embedder separates the family from distinct "
              "rules; threshold behaves as measured" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
