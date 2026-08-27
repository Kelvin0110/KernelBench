#!/usr/bin/env bash
# Per-arm L2 state for the live GPU0 wave (batch 1 + batch 2).
R=runs_evolving/gpt-oss-120b/l2redesign
printf "%-13s %-9s %-6s %-9s %-9s %s\n" ARM PROBLEMS L1 PROMOTED STANDING NOTE
for t in truncation l2 l2_hit l2_redesign l2_judge l2_preseed l2_extract; do
  d=$(ls -d $R/*_${t}_itr30_*/ 2>/dev/null | head -1)
  if [ -z "$d" ]; then
    printf "%-13s %-9s\n" "$t" "(not up)"
    continue
  fi
  p=$(wc -l < "$d/batch_timing.jsonl" 2>/dev/null || echo 0)
  sk=$(wc -l < "$d/shared_l1.jsonl" 2>/dev/null || echo 0)
  pr=$(grep -c '"event": "promote"' "$d/l2_promotions.jsonl" 2>/dev/null | head -1 || echo 0)
  ps=$(grep -c '"event": "preseed"' "$d/l2_promotions.jsonl" 2>/dev/null | head -1 || echo 0)
  st=$(wc -l < "$d/l2_standing.jsonl" 2>/dev/null || echo 0)
  note=""
  [ "$ps" != "0" ] && note="preseeded=$ps"
  printf "%-13s %-9s %-6s %-9s %-9s %s\n" "$t" "$p/50" "$sk" "$pr" "$st" "$note"
done
