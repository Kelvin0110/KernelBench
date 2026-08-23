#!/usr/bin/env bash
# Install a userspace CUDA 12.8 toolchain (nvcc) on aarch64/sbsa Ubuntu 24.04.
# NO SUDO. Nothing is written outside $PREFIX. The project .venv is only read.
#
#   PREFIX=$HOME/opt/cuda-12.8 VENV=/localhome/local-tianzheng/KernelBench/.venv ./install_cuda128_local.sh
#
set -euo pipefail

PREFIX="${PREFIX:-$HOME/opt/cuda-12.8}"
VENV="${VENV:-/localhome/local-tianzheng/KernelBench/.venv}"
WORK="${WORK:-$(mktemp -d)}"
REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa"
SP="$VENV/lib/python3.10/site-packages/nvidia"

# CUDA 12.8.x component debs. nvcc/crt/nvvm are .93; cudart/cccl are .90 (they
# are versioned per-component upstream; this is the real 12.8.1 combination).
DEBS=(
  cuda-nvcc-12-8_12.8.93-1_arm64.deb          # bin/nvcc, cudafe++, ptxas, nvlink, fatbinary
  cuda-crt-12-8_12.8.93-1_arm64.deb           # include/crt/*, bin/crt/link.stub
  cuda-nvvm-12-8_12.8.93-1_arm64.deb          # nvvm/bin/cicc, libnvvm, libdevice.10.bc
  cuda-cudart-12-8_12.8.90-1_arm64.deb        # libcudart.so.12
  cuda-cudart-dev-12-8_12.8.90-1_arm64.deb    # cuda_runtime.h, libcudart.so, stubs
  cuda-cccl-12-8_12.8.90-1_arm64.deb          # thrust/, cub/, cuda/std (libcu++)
  cuda-driver-dev-12-8_12.8.90-1_arm64.deb    # cuda.h, stubs/libcuda.so
  cuda-profiler-api-12-8_12.8.90-1_arm64.deb  # cuda_profiler_api.h
  cuda-nvtx-12-8_12.8.90-1_arm64.deb          # nvToolsExt
  cuda-nvrtc-dev-12-8_12.8.93-1_arm64.deb     # nvrtc.h + libnvrtc.so devlink
)

mkdir -p "$WORK/debs" "$WORK/root"
for d in "${DEBS[@]}"; do
  echo ">> fetching $d"
  curl -fsSL --retry 3 -o "$WORK/debs/$d" "$REPO/$d"
  dpkg -x "$WORK/debs/$d" "$WORK/root"          # dpkg -x needs no root
done

rm -rf "$PREFIX"
mkdir -p "$(dirname "$PREFIX")"
cp -a "$WORK/root/usr/local/cuda-12.8/." "$PREFIX/"

INC="$PREFIX/targets/sbsa-linux/include"
LIB="$PREFIX/targets/sbsa-linux/lib"      # $PREFIX/include and $PREFIX/lib64 symlink here

# Merge in the headers/libs that only exist as pip wheels (cuBLAS, cuRAND,
# cuSOLVER, cuSPARSE, cuFFT, cuDNN, CUPTI, NCCL...). Symlinks, not copies, so
# the extension links against exactly the .so torch already dlopens.
if [ -d "$SP" ]; then
  for d in cublas curand cusolver cusparse cufft cuda_cupti nvtx cudnn \
           cusparselt nvjitlink cuda_nvrtc cufile nccl; do
    [ -d "$SP/$d/include" ] && for f in "$SP/$d/include"/*; do
      b=$(basename "$f"); [ "$b" = "__init__.py" ] && continue
      [ -e "$INC/$b" ] || ln -s "$f" "$INC/$b"
    done
    [ -d "$SP/$d/lib" ] && for f in "$SP/$d/lib"/*.so*; do
      b=$(basename "$f"); [ -e "$LIB/$b" ] || ln -s "$f" "$LIB/$b"
    done
  done
fi

# pip wheels ship only versioned sonames; -lcublas needs the unversioned devlink.
for f in "$LIB"/lib*.so.*; do
  b=$(basename "$f"); dev="${b%%.so.*}.so"
  [ -e "$LIB/$dev" ] || ln -s "$b" "$LIB/$dev"
done

"$PREFIX/bin/nvcc" --version
cat <<EOF

DONE. Add to your shell / .env / run scripts:

  export CUDA_HOME=$PREFIX
  export PATH=\$CUDA_HOME/bin:\$PATH
  # optional, only if a kernel dlopens a CUDA lib itself:
  # export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

Scratch: $WORK  (safe to delete)
EOF
