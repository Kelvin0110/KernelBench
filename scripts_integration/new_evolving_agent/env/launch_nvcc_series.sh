#!/usr/bin/env bash
# Launch the post-nvcc-fix L0 context-management series: truncation + markov_report,
# 50 problems x 30 iterations, gpt-oss-120b on the inference endpoint.
#
#   bash scripts_integration/new_evolving_agent/env/launch_nvcc_series.sh
#
# Prerequisite: the CUDA 12.8 user prefix must already be installed --
#   PREFIX=$HOME/opt/cuda-12.8 VENV=$PWD/.venv \
#     bash scripts_integration/new_evolving_agent/env/install_cuda128_local.sh
# See ./README.md for why this is required.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Hardware/baseline is a parameter so this script works on another server:
#   HARDWARE=<folder under results/timing> bash <this script> ...
# shellcheck source=./hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/hardware_env.sh"
kb_resolve_hardware


RESULTS_ROOT="runs_evolving/gpt-oss-120b/"
DATE_TAG="$(date -u +%b_%-d)"

# ---------------------------------------------------------------------------
# Preflight. Each of these silently corrupts a 40+ hour run rather than failing
# it, so they are hard errors.
# ---------------------------------------------------------------------------
command -v nvcc >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }
kb_require_hardware "$REPO_ROOT"
grep -q "NVIDIA_INF_API_KEY" .env 2>/dev/null || { echo "FATAL: NVIDIA_INF_API_KEY not in .env"; exit 1; }

echo ">> nvcc:  $(nvcc --version | tail -1)"
echo ">> ninja: $(command -v ninja)"

# Refuse to start if either GPU is already busy -- these runs each pin one GPU
# for ~40-50h and silently sharing a device wrecks the timing measurements.
for gpu in 0 1; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if [ "$used" -gt 1000 ]; then
    echo "FATAL: GPU $gpu already has ${used} MiB in use. Refusing to launch."
    exit 1
  fi
done

# Verify a real CUDA extension actually builds before committing 80+ GPU-hours.
echo ">> probing load_inline(cuda_sources=...) ..."
uv run --no-sync python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="preflight_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                cuda_sources=src, functions=["f"], verbose=False)
ok = m.f(torch.zeros(8, device="cuda")).sum().item() == 8.0
print("   CUDA extension build+run:", "OK" if ok else "WRONG RESULT")
sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------------------
# Launch. Run names intentionally match the archived (pre-fix) runs; the new
# results live under $RESULTS_ROOT and the old ones under
# runs_evolving/archived/with_NVCC_bug/, so there is no collision.
# ---------------------------------------------------------------------------
launch() {
  local gpu="$1" run_name="$2" ctx="$3"
  local log="${run_name}_${DATE_TAG}.log"
  echo ">> GPU $gpu -> $run_name (context-management=$ctx), log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run --no-sync python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "$run_name" \
    --results-root "$RESULTS_ROOT" \
    --max-problems 50 \
    --max-iterations 30 \
    --hardware "$HARDWARE" \
    --nvidia-endpoint inference \
    --model gpt-oss-120b \
    --context-management "$ctx" \
    >> "$log" 2>&1 &
  echo "   pid=$!"
}

launch 0 base_agent_gpt_oss_120b_itr30_GH200        truncation
sleep 5
launch 1 base_agent_gpt_oss_120b_markov_itr30_GH200 markov_report

sleep 40
echo
echo "=== running processes ==="
pgrep -af evolve_kb_batch | cut -c1-140 || echo "NONE -- check the logs"
echo
echo "=== run dirs ==="
ls -1 "$RESULTS_ROOT" 2>/dev/null || echo "(not created yet)"
cat <<EOF

Launched. Expect ~40-50h each (~2 min/iteration with real CUDA compiles).

Early health check (gating signal is cuda_home_err, which must be 0):

  uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto

Watch for trouble:

  grep -c CUDA_HOME base_agent_gpt_oss_120b_*_${DATE_TAG}.log    # must stay 0
  grep -cE "404|APIConnectionError|ReadTimeout" base_agent_gpt_oss_120b_*_${DATE_TAG}.log
EOF
