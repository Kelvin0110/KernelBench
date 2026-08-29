"""Print each arm's L2 gate configuration from its own run_summary.json.

Read the config from the ARTIFACT, never from the spec file: the spec records what
was requested, run_summary.json records what the runner actually bound.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOTS = [Path("runs_evolving/gpt-oss-120b/l2redesign"),
         Path("runs_evolving/gpt-oss-120b/l2quick")]
KEYS = ["enable_l2", "l2_render", "l2_min_tasks", "l2_min_selections", "l2_min_rate",
        "l2_use_hit_rate", "l2_min_hit_rate", "l2_min_new_bests", "l2_max_entries",
        "l2_standing_cap", "l2_dedup_similarity", "l2_judge", "l2_freeze", "l2_preseed"]


def main() -> None:
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            f = d / "run_summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            name = d.name.replace("base_agent_gpt_oss_120b_", "")
            i = name.find("_itr30_GH200")
            name = name[:i] if i >= 0 else name
            vals = {k: s[k] for k in KEYS if k in s and s[k] not in (None, False, 0, 0.0, "")}
            print(f"{name:<13} {vals}")


if __name__ == "__main__":
    main()
