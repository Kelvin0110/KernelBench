"""End-to-end check that the analysis scripts now see L2 on real run summaries.

Open item 7 / CLAUDE.md 8.10: an L2 arm and its plain control used to render as
the same design string, so every delta table compared an arm with itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aggregate_runs import _as_bool, _as_int, _as_str, safe_float  # noqa: E402,F401
from compare_runs import design_variant_label  # noqa: E402

ARMS = sys.argv[1:]
if not ARMS:
    print("usage: check_analysis_l2.py <run_dir> [<run_dir>...]")
    raise SystemExit(2)

bad = 0
labels = {}
for a in ARMS:
    rd = Path(a)
    sp = rd / "run_summary.json"
    if not sp.exists():
        print(f"  SKIP {rd.name}: no run_summary.json")
        continue
    s = json.loads(sp.read_text())
    # Mirror aggregate_runs.py's extraction for the L2 block.
    cfg = {
        "context_management": _as_str(s.get("context_management")),
        "enable_l2": _as_bool(s.get("enable_l2")),
        "l2_render": _as_str(s.get("l2_render")),
        "l2_use_hit_rate": _as_bool(s.get("l2_use_hit_rate")),
        "l2_min_hit_rate": safe_float(s.get("l2_min_hit_rate")),
        "l2_standing_cap": _as_int(s.get("l2_standing_cap"))
        if s.get("l2_standing_cap") is not None
        else None,
        "l2_dedup_similarity": safe_float(s.get("l2_dedup_similarity")),
        "skill_merging": _as_bool(s.get("skill_merging")),
        "skill_merge_similarity": safe_float(s.get("skill_merge_similarity")),
        "skill_deletion": _as_bool(s.get("skill_deletion")),
        "enable_skill_refinement": _as_bool(s.get("enable_skill_refinement")),
    }
    label = design_variant_label({"config": cfg})
    labels[rd.name] = label
    standing = s.get("l2_standing_count")
    print(f"  {rd.name[:52]:54s} -> {label}   (standing_count={standing})")
    if cfg["enable_l2"] and "l2" not in label:
        print("      FAIL: L2 arm rendered without an l2 marker")
        bad += 1

# The whole point: an L2 arm must not share a label with a non-L2 arm.
l2_labels = {n: l for n, l in labels.items() if "l2" in l}
plain = {n: l for n, l in labels.items() if "l2" not in l}
clash = set(l2_labels.values()) & set(plain.values())
if clash:
    print(f"  FAIL: L2 and non-L2 arms share design label(s): {clash}")
    bad += 1

print()
print("VERDICT:", "PASS -- L2 is visible to the analysis" if not bad else "FAIL")
raise SystemExit(1 if bad else 0)
