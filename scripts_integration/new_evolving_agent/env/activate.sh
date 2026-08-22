# Source me -- do not execute.
#   source scripts_integration/new_evolving_agent/env/activate.sh
#
# Sets the two exports that every hand-rolled evolve_kb_batch.py invocation needs.
# launch_run.sh sets them itself, so this is for interactive work only.
#
# Deliberately NOT added to ~/.bashrc: putting .venv/bin on PATH globally shadows
# the system python 3.12 with the project's 3.10 in every shell you open.

_kb_root="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )/../../.." && pwd )"

export CUDA_HOME="$HOME/opt/cuda-12.8"
export PATH="$CUDA_HOME/bin:$_kb_root/.venv/bin:$PATH"

# Uncomment when sharing one GPU across several arms (see 12-multi-arm-settings.md):
# export KB_GPU_RESERVE_GB=0

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "WARNING: no nvcc at $CUDA_HOME/bin/nvcc -- run install_cuda128_local.sh" >&2
fi
if [ ! -x "$_kb_root/.venv/bin/ninja" ]; then
  echo "WARNING: no ninja in the venv -- CUDA builds will fail. Run: uv sync --extra dev" >&2
fi

echo "KernelBench env ready: CUDA_HOME=$CUDA_HOME, python=$(command -v python)"
unset _kb_root
