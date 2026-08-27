#!/usr/bin/env bash
d=$(ls -d runs_evolving/gpt-oss-120b/l2redesign/*_l2_hit_itr30_*/ | head -1)
.venv/bin/python - "$d" <<'PY'
import json, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "l2_promotions.jsonl")]
print("promotions (l2_hit):")
for r in rows:
    if r["event"] != "promote":
        continue
    print(f"  gi={r['promoted_at_global_iter']:5d}  hit={r.get('hit_rate')}  "
          f"offers={r.get('total_offers')}  sel={r.get('total_selections')}  "
          f"tasks={r.get('distinct_tasks')}  rate={r.get('selection_rate')}  "
          f"| {str(r.get('title'))[:52]}")
print()
print("pass census (last 5):")
for r in [x for x in rows if x["event"] == "pass"][-5:]:
    print(f"  gi={r['global_iteration']:5d} cand={r['candidate_count']:4d} "
          f"elig={r['eligible_count']} standing_after={r['standing_after']}")
PY
