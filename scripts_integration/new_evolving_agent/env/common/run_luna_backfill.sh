#!/usr/bin/env bash
# Drain the gpt-5.6-luna backfill queue: keep each GPU topped up to TARGET_PER_GPU
# live arms (9 => total 18), launching ONE queued cell at a time as arms finish.
#
# HW is passed EXPLICITLY. wave_backfill.sh defaults it to env/NVIDIA_GH200x2_2nd,
# which on this host is the OTHER server's launcher folder -- and $HARDWARE is derived
# from the calling script's directory name, so the default would resolve the wrong
# baseline. CLAUDE.local.md keeps that folder pristine on purpose.
cd /localhome/local-tianzheng/KernelBench
REPO=/localhome/local-tianzheng/KernelBench \
HW=/localhome/local-tianzheng/KernelBench/scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2 \
QUEUE=scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2/luna_backfill.queue \
MODEL=gpt-5.6-luna \
RESULTS_ROOT=runs_evolving/gpt-5.6-luna/median/ \
RUN_PREFIX=base_agent_gpt_5_6_luna \
TARGET_PER_GPU=9 MAX_PER_GPU=12 INTERVAL=300 \
  exec bash scripts_integration/new_evolving_agent/env/common/wave_backfill.sh
