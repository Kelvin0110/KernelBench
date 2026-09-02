#!/usr/bin/env bash
cd /localhome/local-tianzheng/KernelBench
while true; do
  ./.venv/bin/python scripts_integration/new_evolving_agent/env/common/wave_timeout_supervisor.py \
    --apply --threshold-pct 8 --window 80 --min-evals 30 \
    --max-restarts 2 --cooldown-h 8 --max-per-pass 1 --new-timeout 3600 \
    >> wave_timeout_supervisor.log 2>&1
  sleep 1800
done
