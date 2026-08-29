"""L2 arms must not render as their own control (CLAUDE.md open item 7)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_runs import design_variant_label as label  # noqa: E402

CASES = [
    ({"context_management": "truncation"}, "truncation"),
    ({"context_management": "truncation", "enable_l2": True}, "truncation+l2"),
    (
        {
            "context_management": "truncation",
            "enable_l2": True,
            "l2_use_hit_rate": True,
            "l2_min_hit_rate": 0.6,
            "l2_standing_cap": 6,
            "l2_dedup_similarity": 0.8,
        },
        "truncation+l2/hit0.6/standcap6/dedup0.8",
    ),
    (
        {"context_management": "truncation", "enable_l2": True, "l2_render": "extract"},
        "truncation+l2@extract",
    ),
    (
        {
            "context_management": "truncation",
            "skill_merging": True,
            "skill_merge_similarity": 0.8,
            "enable_l2": True,
        },
        "truncation+merge@0.8+l2",
    ),
]

bad = 0
for cfg, want in CASES:
    got = label({"config": cfg})
    ok = got == want
    bad += not ok
    print(("PASS  " if ok else "FAIL  ") + repr(got) + ("" if ok else f"   want {want!r}"))

ctl = label({"config": {"context_management": "truncation"}})
arm = label({"config": {"context_management": "truncation", "enable_l2": True}})
distinct = ctl != arm
print(("PASS  " if distinct else "FAIL  ") + f"L2 arm distinguishable from control: {ctl!r} != {arm!r}")
bad += not distinct

print("FAILURES:", bad)
sys.exit(1 if bad else 0)
