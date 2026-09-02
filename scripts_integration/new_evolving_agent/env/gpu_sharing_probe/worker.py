"""One concurrency-probe worker.

Times two small custom CUDA kernels with the SAME method KernelBench uses to
score kernels (`time_execution_with_cuda_event`: CUDA-event window, L2 thrash
outside the window, sync before each trial). Workers rendezvous at a file
barrier before every timed block so that `degree` processes genuinely overlap.
"""
import argparse, glob, json, os, statistics as st, sys, time

MISH_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float sp = log1pf(expf(v));
        y[i] = v * tanhf(sp);
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

MM_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TS 32
__global__ void mm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        int ak = t * TS + threadIdx.x;
        int bk = t * TS + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TS; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor mm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (M + TS - 1) / TS);
    mm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                   C.data_ptr<float>(), M, N, K);
    return C;
}
"""


def build():
    from torch.utils.cpp_extension import load_inline
    mish = load_inline(name="conc_mish", cpp_sources="torch::Tensor mish_cuda(torch::Tensor x);",
                       cuda_sources=MISH_SRC, functions=["mish_cuda"], verbose=False)
    mm = load_inline(name="conc_mm", cpp_sources="torch::Tensor mm_cuda(torch::Tensor A, torch::Tensor B);",
                     cuda_sources=MM_SRC, functions=["mm_cuda"], verbose=False)
    return mish, mm


def barrier(bdir, tag, slot, degree, timeout=600.0):
    """Rendezvous: publish own token, wait for `degree` tokens of this round."""
    open(os.path.join(bdir, "b%s_%d" % (tag, slot)), "w").close()
    t0 = time.time()
    pat = os.path.join(bdir, "b%s_*" % tag)
    while time.time() - t0 < timeout:
        if len(glob.glob(pat)) >= degree:
            return True
        time.sleep(0.005)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bdir", required=True)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, "/localhome/local-tianzheng/KernelBench/src")
    import torch
    from kernelbench.timing import time_execution_with_cuda_event

    mish, mm = build()
    if a.build_only:
        print("build ok")
        return

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    # deliberately small footprint: 128 MB in / 128 MB out, and a 12 MB matmul
    n = 32 * 1024 * 1024
    x = torch.randn(n, device=dev, dtype=torch.float32)
    A = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    B = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    torch.cuda.synchronize(dev)

    A2 = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    B2 = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    torch.cuda.synchronize(dev)

    # (name, fn, args, trials). matmul_2048 is the SATURATING case: ~2 ms/launch, so a
    # single process already occupies the GPU almost continuously.
    cases = [("mish_32M", mish.mish_cuda, [x], a.trials),
             ("matmul_1024", mm.mm_cuda, [A, B], a.trials),
             ("matmul_2048", mm.mm_cuda, [A2, B2], max(20, a.trials // 5))]
    out = {"degree": a.degree, "slot": a.slot, "trials": a.trials, "blocks": a.blocks,
           "pid": os.getpid(), "cases": {}}

    devnull = open(os.devnull, "w")
    for cname, fn, args, ctrials in cases:
        blocks = []
        for b in range(a.blocks):
            if not barrier(a.bdir, "%s_%d" % (cname, b), a.slot, a.degree):
                out.setdefault("errors", []).append("barrier timeout %s %d" % (cname, b))
                break
            w0 = time.time()
            so = sys.stdout
            sys.stdout = devnull                       # timing fn prints a [Profiling] line
            try:
                el = time_execution_with_cuda_event(fn, args, num_warmup=5,
                                                    num_trials=ctrials, discard_first=1,
                                                    verbose=False, device=dev)
            finally:
                sys.stdout = so
            wall = time.time() - w0
            el_s = sorted(el)
            blocks.append({
                "block": b, "wall_sec": round(wall, 4), "trials": ctrials,
                "sum_kernel_ms": sum(el), "duty_pct": 100.0 * sum(el) / (wall * 1000.0),
                "median_ms": st.median(el), "mean_ms": st.fmean(el),
                "min_ms": el_s[0], "max_ms": el_s[-1],
                "p90_ms": el_s[int(0.9 * (len(el_s) - 1))],
                "cv_pct": 100.0 * st.pstdev(el) / st.fmean(el),
            })
        meds = [x_["median_ms"] for x_ in blocks]
        out["cases"][cname] = {
            "blocks": blocks,
            "median_of_block_medians_ms": st.median(meds) if meds else None,
            "min_block_median_ms": min(meds) if meds else None,
            "max_block_median_ms": max(meds) if meds else None,
        }

    with open(os.path.join(a.outdir, "w%02d.json" % a.slot), "w") as f:
        json.dump(out, f, indent=1)
    print("slot %d done" % a.slot)


if __name__ == "__main__":
    main()
