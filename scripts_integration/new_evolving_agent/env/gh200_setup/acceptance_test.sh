#!/usr/bin/env bash
# The gauntlet from 10-acceptance-test.md, as one runnable script.
# Every check here corresponds to a failure that once silently corrupted a ~70h run.
#
#   cd <repo> && bash scripts_integration/new_evolving_agent/env/gh200_setup/acceptance_test.sh
#
# Exits non-zero on the first hard failure. Run this before launching anything.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

pass=0; fail=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
info() { printf '        %s\n' "$*"; }

echo "repo: $REPO_ROOT"
echo "CUDA_HOME: $CUDA_HOME"
echo
echo "-- toolchain --"
v=$(nvcc --version 2>/dev/null | grep -o 'V[0-9][0-9.]*' | head -1)
[ "$v" = "V12.8.93" ] && ok "nvcc $v" || bad "nvcc: got '${v:-missing}', want V12.8.93"

n=$(command -v ninja || true)
case "$n" in "$REPO_ROOT/.venv/bin/ninja") ok "ninja from venv";; "") bad "ninja not on PATH (.venv/bin missing from PATH)";; *) bad "ninja resolves to $n, not the venv copy";; esac

g=$(gcc --version 2>/dev/null | head -1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
[ -n "$g" ] && ok "gcc $g" || bad "gcc missing"

p=$(.venv/bin/python -V 2>&1 | awk '{print $2}')
case "$p" in 3.10.*) ok "python $p";; *) bad "python: got '${p:-missing}', want 3.10.x";; esac

echo
echo "-- submodules --"
[ -f Self-Evolving-Agent/evolving_common/llm_client.py ] && ok "Self-Evolving-Agent checked out" || bad "Self-Evolving-Agent EMPTY -- run: git submodule update --init --recursive"
sea=$(git -C Self-Evolving-Agent rev-parse --short=12 HEAD 2>/dev/null); info "SEA at $sea"
pin=$(git rev-parse --short=12 HEAD:Self-Evolving-Agent 2>/dev/null)
[ "$sea" = "$pin" ] && ok "SEA matches superproject pin" || bad "SEA at $sea but superproject pins $pin"

echo
echo "-- config --"
grep -q NVIDIA_INF_API_KEY .env 2>/dev/null && ok "NVIDIA_INF_API_KEY present in .env" || bad "NVIDIA_INF_API_KEY missing from .env (launch_run.sh greps for it)"
[ -f results/timing/NVIDIA_GH200x2/baseline_time_torch.json ] && ok "GH200x2 timing baseline present" || bad "results/timing/NVIDIA_GH200x2/ missing"

echo
echo "-- driver --"
if nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader 2>/dev/null; then
  ok "nvidia-smi responds"
  lic=$(modinfo nvidia 2>/dev/null | awk '/^license:/{$1="";print}' | xargs)
  case "$lic" in *MIT*|*GPL*) ok "open kernel module ($lic)";; *) bad "PROPRIETARY module ($lic) -- GH200 needs nvidia-driver-580-open; see fix_closed_to_open_driver.sh";; esac
  numactl -H 2>/dev/null | grep -qE '^node (2|10) size: [1-9]' && ok "GPU HBM online as NUMA nodes 2/10" || info "NUMA nodes 2/10 empty -- check gh200-memory-online.service"
else
  bad "nvidia-smi cannot talk to the driver -- see fix_closed_to_open_driver.sh"
fi

echo
echo "-- python stack --"
.venv/bin/python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath("Self-Evolving-Agent"))
def ok(m):  print(f"  \033[32mPASS\033[0m  {m}")
def bad(m): print(f"  \033[31mFAIL\033[0m  {m}"); globals().__setitem__("BAD", True)
BAD = False
import torch
(ok if torch.__version__.startswith("2.11.0+cu128") else bad)(f"torch {torch.__version__} (cuda {torch.version.cuda})")
try:
    import kernelbench; ok("kernelbench importable (editable)")
except Exception as e: bad(f"kernelbench import: {e}")
try:
    import evolving_common; ok("evolving_common importable via sys.path")
except Exception as e: bad(f"evolving_common import: {e}")
try:
    import sklearn; ok(f"scikit-learn {sklearn.__version__} (needed by --skill-merging)")
except Exception as e: bad(f"scikit-learn missing -- --skill-merging will die with coder_call_error: {e}")
if torch.cuda.is_available():
    ok(f"torch sees {torch.cuda.device_count()} GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    _arch = torch.cuda.get_arch_list()
    (ok if "sm_90" in _arch else bad)(f"arch list {_arch}" if "sm_90" in _arch else f"sm_90 absent from {_arch}")
else:
    bad("torch.cuda.is_available() is False -- driver not usable")
sys.exit(1 if BAD else 0)
PY
[ $? -eq 0 ] && pass=$((pass+1)) || fail=$((fail+1))

echo
echo "-- the real test: end-to-end CUDA extension build + launch --"
.venv/bin/python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
try:
    m = load_inline(name="host_acceptance_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                    cuda_sources=src, functions=["f"], verbose=False)
except Exception as e:
    hint = " (no driver: torch cannot detect a GPU arch)" if not torch.cuda.is_available() else ""
    print(f"  \033[31mFAIL\033[0m  load_inline build{hint}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)
if not torch.cuda.is_available():
    print("  \033[31mFAIL\033[0m  built OK but cannot launch: no usable driver")
    sys.exit(1)
ok = m.f(torch.zeros(8, device="cuda")).sum().item() == 8.0
print(f"  \033[32mPASS\033[0m  CUDA extension build+run" if ok else "  \033[31mFAIL\033[0m  kernel ran but gave the WRONG RESULT")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] && pass=$((pass+1)) || fail=$((fail+1))

echo
if [ "$fail" -eq 0 ]; then
  printf '\033[1;32mOK\033[0m  %d checks passed. Safe to launch.\n' "$pass"; exit 0
else
  printf '\033[1;31mNOT READY\033[0m  %d passed, %d failed. Do NOT launch a run.\n' "$pass" "$fail"; exit 1
fi
