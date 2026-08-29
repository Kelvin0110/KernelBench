#!/usr/bin/env bash
# Final per-arm state for the L2 wave (batch 1, batch 2, quick replicates).
# run_summary.json exists only after the LAST problem, so it is the completion flag.
cd "$(dirname "$0")"
printf "%-58s %-7s %-9s %-9s %-9s %s\n" ARM PROBS COMPLETE STANDING PROMOTED PRESEED
for d in runs_evolving/gpt-oss-120b/l2redesign/*/ runs_evolving/gpt-oss-120b/l2quick/*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  short=${n#base_agent_gpt_oss_120b_}; short=${short%_itr30_GH200_*}
  p=0;  [ -f "$d/batch_timing.jsonl" ] && p=$(wc -l < "$d/batch_timing.jsonl")
  sm=no; [ -f "$d/run_summary.json" ] && sm=YES
  st=0; [ -f "$d/l2_standing.jsonl" ] && st=$(wc -l < "$d/l2_standing.jsonl")
  pr=0; ps=0
  if [ -f "$d/l2_promotions.jsonl" ]; then
    pr=$(grep -c '"event": "promote"' "$d/l2_promotions.jsonl")
    ps=$(grep -c '"event": "preseed"' "$d/l2_promotions.jsonl")
  fi
  printf "%-58s %-7s %-9s %-9s %-9s %s\n" "$short" "$p" "$sm" "$st" "$pr" "$ps"
done
