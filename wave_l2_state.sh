#!/usr/bin/env bash
# Per-arm L2 state for the live wave.
for t in truncation l2 l2_hit l2_redesign; do
  d=$(ls -d runs_evolving/gpt-oss-120b/l2redesign/*_${t}_itr30_*/ 2>/dev/null | head -1)
  [ -z "$d" ] && continue
  p=$(wc -l < "$d/batch_timing.jsonl" 2>/dev/null || echo 0)
  pr=$(grep -c '"event": "promote"' "$d/l2_promotions.jsonl" 2>/dev/null || echo 0)
  st=$(wc -l < "$d/l2_standing.jsonl" 2>/dev/null || echo 0)
  sk=$(wc -l < "$d/shared_l1.jsonl" 2>/dev/null || echo 0)
  printf "%-13s problems=%-4s L1=%-5s promoted=%-4s standing=%s\n" "$t" "$p" "$sk" "$pr" "$st"
done
