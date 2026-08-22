#!/usr/bin/env bash
# Repair a GH200 host whose GPUs are invisible because the PROPRIETARY NVIDIA
# driver is installed instead of the open one, and add the movable-memory
# onlining unit that ships in no package.
#
# Symptom this fixes:
#   nvidia-smi -> "couldn't communicate with the NVIDIA driver"
#   lsmod       -> no nvidia/nvidia_uvm (only nvidia_cspmu)
#   modinfo nvidia | grep license -> "NVIDIA"   (proprietary; the open one is Dual MIT/GPL)
#
# GH200 is GH100/Hopper, which requires the OPEN kernel module. The proprietary
# module cannot drive these GPUs no matter how cleanly DKMS builds it.
#
# NOTE: the module firmware list is NOT a diagnostic. Both the proprietary and the
# open 580.173.02 modules advertise only gsp_tu10x/gsp_ga10x and neither mentions
# GH100, so `grep -c gh100` returns 0 on a working host too. Use the licence.
#
# No reboot: safe because the nvidia modules are not currently loaded, so nothing
# has to be unloaded and no HBM has been onlined yet under the wrong policy.
#
#   sudo bash fix_closed_to_open_driver.sh
#
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "FATAL: run me with sudo"; exit 1; }

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "1/6  GH200 movable-memory onlining unit"
tee /etc/systemd/system/gh200-memory-online.service >/dev/null <<'UNIT'
[Unit]
Description=Configure movable memory onlining for NVIDIA GH200
DefaultDependencies=no
Before=systemd-modules-load.service
Before=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo online_movable > /sys/devices/system/memory/auto_online_blocks'
RemainAfterExit=yes

[Install]
WantedBy=sysinit.target
UNIT
systemctl daemon-reload
systemctl enable --now gh200-memory-online.service
echo "auto_online_blocks = $(cat /sys/devices/system/memory/auto_online_blocks)   # want: online_movable"

step "2/6  What the package swap will do (preview)"
apt-get update -qq || echo "(apt-get update failed; continuing with cached lists)"
apt-get install -y --dry-run nvidia-driver-580-open 2>&1 | grep -E '^(Inst|Remv)' || true

step "3/6  Installing nvidia-driver-580-open (removes the proprietary driver)"
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-580-open

step "4/6  Confirming the OPEN module is now in place"
depmod -a || true
lic=$(modinfo nvidia 2>/dev/null | awk '/^license:/{$1="";print $0}' | xargs)
echo "  module license : ${lic:-<none>}    # want: Dual MIT/GPL"
case "$lic" in
  *MIT*|*GPL*) echo "  -> open module OK" ;;
  *) echo "  -> STILL PROPRIETARY. Stop here; the swap did not take."; exit 1 ;;
esac

step "5/6  Loading modules + persistence daemon"
for m in nvidia nvidia_uvm; do
  if modprobe "$m"; then
    echo "  modprobe $m: ok"
  else
    echo "  modprobe $m: FAILED"
    echo "  --- dmesg (NVRM) ---"
    dmesg 2>/dev/null | grep -iE 'nvrm|nvidia' | tail -20 || echo "  (no dmesg output)"
    echo
    echo "  A reboot is the usual remedy at this point."
    exit 1
  fi
done
systemctl enable --now nvidia-persistenced || true
sleep 2

step "6/6  Verification"
echo "--- lsmod ---";      lsmod | grep -E '^nvidia' || echo "  NO NVIDIA MODULES LOADED"
echo "--- /dev nodes ---"; ls /dev/nvidia* 2>/dev/null || echo "  none"
echo "--- nvidia-smi ---"
if nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv; then
  echo "--- NUMA (GPU HBM should appear as nodes 2 and 10) ---"
  if command -v numactl >/dev/null 2>&1; then
    numactl -H | grep -E '^node (2|10) size' \
      || echo "  nodes 2/10 not online (see: cat /sys/devices/system/node/online). A reboot"
    echo "     usually brings them up now that auto_online_blocks=online_movable is set."
  else
    echo "  numactl not installed (apt install numactl) -- check /sys/devices/system/node/online"
  fi
  echo "--- dkms ---"; dkms status | head
  printf '\n\033[1;32mPASS\033[0m  GPUs are visible. Re-run the acceptance test:\n'
  echo '  cd /localhome/local-tianzheng/KernelBench'
  echo '  export CUDA_HOME=$HOME/opt/cuda-12.8'
  echo '  export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH'
  echo '  bash scripts_integration/new_evolving_agent/env/gh200_setup/acceptance_test.sh'
else
  printf '\n\033[1;31mFAIL\033[0m  modules loaded but nvidia-smi still fails.\n'
  echo "Next: check 'dmesg | grep -i nvrm | tail -30'. A reboot is the usual remedy"
  echo "(the memory-onlining policy only applies to memory brought online after it is set)."
  exit 1
fi
