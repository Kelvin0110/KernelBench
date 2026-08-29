"""What the gate variants actually DID -- the mechanism result, independent of quality.

Quality on this wave is a null (see lottery_adjusted.py), so the defensible claims
are mechanical: how many rules each gate promoted, whether the count is
reproducible, whether dedup removed the duplicate families documented in CLAUDE.md
8.4, and whether the judge and the pre-seed freeze behaved as specified.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("runs_evolving/gpt-oss-120b/l2redesign")
QUICK = Path("runs_evolving/gpt-oss-120b/l2quick")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def arm_dirs() -> dict[str, Path]:
    out = {}
    for root in (ROOT, QUICK):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            s = d.name.replace("base_agent_gpt_oss_120b_", "")
            i = s.find("_itr30_GH200")
            out[s[:i] if i >= 0 else s] = d
    return out


def main() -> None:
    dirs = arm_dirs()

    print("=== Promotion / census summary ===")
    print(f"{'arm':<13}{'standing':>9}{'promote':>9}{'preseed':>9}{'passes':>8}"
          f"{'cands':>7}{'elig':>6}{'judged':>7}")
    for name, d in dirs.items():
        pr = rows(d / "l2_promotions.jsonl")
        if not pr and not (d / "l2_standing.jsonl").exists():
            continue
        std = len(rows(d / "l2_standing.jsonl"))
        npro = sum(1 for r in pr if r.get("event") == "promote")
        npre = sum(1 for r in pr if r.get("event") == "preseed")
        census = [r for r in pr if r.get("event") == "pass"]
        cand = sum(int(r.get("candidate_count") or 0) for r in census)
        elig = sum(int(r.get("eligible_count") or 0) for r in census)
        judged = sum(1 for r in census if r.get("judge"))
        print(f"{name:<13}{std:>9}{npro:>9}{npre:>9}{len(census):>8}"
              f"{cand:>7}{elig:>6}{judged:>7}")

    print("\n=== Reproducibility of the SHIPPED gate ===")
    print("CLAUDE.md 8.11 records 9 / 4 / 0 rules from three identical-flag runs.")
    d = dirs.get("l2")
    if d:
        pr = rows(d / "l2_promotions.jsonl")
        n = sum(1 for r in pr if r.get("event") == "promote")
        print(f"This wave's shipped-gate arm promoted: {n}")
        print("-> a FOURTH distinct value; the shipped gate's count remains unstable.")

    print("\n=== Per-candidate reasons (dedup / cap / judge) ===")
    print("NOTE: the census key is named 'dropped' but emits EVERY candidate")
    print("carrying a reason, including PROMOTED ones -- the judge writes its")
    print("acceptance rationale into the same field. Cross-reference the promote")
    print("events to separate them; do not read 'dropped' as 'rejected'.\n")
    for name in ("l2_redesign", "l2_hit", "l2", "l2_judge", "l2_extract"):
        d = dirs.get(name)
        if not d:
            continue
        pr = rows(d / "l2_promotions.jsonl")
        promoted = {str(r.get("entry_id")) for r in pr if r.get("event") == "promote"}
        kinds: dict[str, int] = {}
        for r in pr:
            if r.get("event") != "pass":
                continue
            for c in r.get("dropped") or []:
                if str(c.get("entry_id")) in promoted:
                    continue  # accepted, not dropped
                for why in c.get("reasons") or []:
                    key = str(why).split(":")[0].strip()
                    kinds[key] = kinds.get(key, 0) + 1
        std = len(rows(d / "l2_standing.jsonl"))
        print(f"{name:<13} standing={std}  genuine drops: "
              f"{dict(sorted(kinds.items(), key=lambda kv: -kv[1])) or 'none'}")

    print("\n=== Judge decisions (accepted) ===")
    d = dirs.get("l2_judge")
    if d:
        pr = rows(d / "l2_promotions.jsonl")
        promoted = {str(r.get("entry_id")) for r in pr if r.get("event") == "promote"}
        seen: set[str] = set()
        for r in pr:
            if r.get("event") != "pass":
                continue
            for c in r.get("dropped") or []:
                eid = str(c.get("entry_id"))
                if eid not in promoted or eid in seen:
                    continue
                seen.add(eid)
                why = " ".join(str(x) for x in (c.get("reasons") or []))
                why = why.replace("judge:", "").strip().replace("\n", " ")
                print(f"  + id={eid:<4} {why[:96]}")

    print("\n=== Pre-seed freeze ===")
    for name in ("l2_preseed", "q15_pre_r1", "q15_pre_r2"):
        d = dirs.get(name)
        if not d:
            continue
        pr = rows(d / "l2_promotions.jsonl")
        npre = sum(1 for r in pr if r.get("event") == "preseed")
        npro = sum(1 for r in pr if r.get("event") == "promote")
        ndem = sum(1 for r in pr if r.get("event") == "demote")
        std = len(rows(d / "l2_standing.jsonl"))
        ok = "OK" if (npre == std and npro == 0 and ndem == 0) else "UNEXPECTED"
        print(f"  {name:<12} preseeded={npre} standing={std} promoted={npro} "
              f"demoted={ndem}  [{ok}]")


if __name__ == "__main__":
    main()
