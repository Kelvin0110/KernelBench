
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor fused_bias_tanh_scale(torch::Tensor inp,
                                    torch::Tensor bias,
                                    float scaling)
{
    // Expect inp shape: (B, C, 1, H, W)
    inp = inp.contiguous();
    bias = bias.contiguous();

    const int B = inp.size(0);
    const int C = inp.size(1);
    const int H = inp.size(3);
    const int W = inp.size(4);
    const int total = B * C * H * W;

    auto out = torch::empty_like(inp);

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    // launch kernel
    fused_bias_tanh_scale_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        bias.view({C}).data_ptr<float>(),
        out.data_ptr<float>(),
        scaling,
        B, C, H, W
    );

    // check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_tanh_scale", &fused_bias_tanh_scale,
          "Fuse bias addition, tanh activation and scaling (CUDA)");
}
