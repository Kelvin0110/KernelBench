#!/usr/bin/env bash
# Launch ONE evolving-agent arm on a chosen GPU, on the repaired CUDA toolchain.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_run.sh <gpu> <run_name> <ctx_mode> [extra flags...]
#
# Examples:
#   # compress_trigger arm
#   ... launch_run.sh 0 base_agent_gpt_oss_120b_compress_itr30_GH200 compress_trigger \
#         --compress-hot-rounds 3 --compress-token-ratio 0.85 --compress-every-n-iters 15
#
#   # a skill-governance arm (context mode held at truncation)
#   ... launch_run.sh 1 base_agent_gpt_oss_120b_deletion_itr30_GH200 truncation --skill-deletion
#
# Prerequisite: bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/install_cuda128_local.sh
# See ./README.md for why CUDA_HOME and .venv/bin on PATH are both mandatory.

set -euo pipefail

GPU="${1:?usage: launch_run.sh <gpu> <run_name> <ctx_mode> [extra flags...]}"
RUN_NAME="${2:?missing run_name}"
CTX="${3:?missing context-management mode}"
shift 3
EXTRA=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Hardware/baseline is a parameter so this script works on another server:
#   HARDWARE=<folder under results/timing> bash <this script> ...
# shellcheck source=./hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
kb_resolve_hardware


RESULTS_ROOT="runs_evolving/gpt-oss-120b/"
LOG="${RUN_NAME}_$(date -u +%b_%-d).log"

# --- preflight: each of these silently corrupts a ~70h run rather than failing it
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }
kb_require_hardware "$REPO_ROOT"
grep -q "NVIDIA_INF_API_KEY" .env 2>/dev/null || { echo "FATAL: NVIDIA_INF_API_KEY not in .env"; exit 1; }

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
[ "$used" -gt 1000 ] && { echo "FATAL: GPU $GPU busy (${used} MiB). Refusing."; exit 1; }

echo ">> nvcc: $(nvcc --version | tail -1)"
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

echo ">> GPU $GPU -> $RUN_NAME (context-management=$CTX) ${EXTRA[*]:-}"
echo ">> log: $LOG"

CUDA_VISIBLE_DEVICES="$GPU" nohup uv run --no-sync python \
  scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name "$RUN_NAME" \
  --results-root "$RESULTS_ROOT" \
  --max-problems 50 \
  --max-iterations 30 \
  --hardware "$HARDWARE" \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --context-management "$CTX" \
  --coder-timeout-sec 600 \
  "${EXTRA[@]}" \
  >> "$LOG" 2>&1 &

echo "   pid=$!"
sleep 40
echo
echo "=== process ==="
pgrep -af "evolve_kb_batch.*$RUN_NAME" | cut -c1-140 || echo "NOT RUNNING -- check $LOG"
echo "=== first activity ==="
grep -E "\[batch\]|kb-governor|Error|Traceback" "$LOG" | head -4 || echo "(none yet)"
cat <<EOF

Watch:  tail -f $LOG
Health: uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
        grep -c CUDA_HOME $LOG   # must stay 0
EOF
