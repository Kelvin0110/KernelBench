
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv3d_naive_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C_in, int D, int H, int W,
    int C_out, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D * H * W;
    if (idx >= total) return;

    // Decode linear index.
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int d = tmp % D;
    tmp = tmp / D;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    scalar_t sum = bias ? bias[oc] : static_cast<scalar_t>(0);

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int id = d + kd;   // stride=1, padding=0
                    int ih = h + kh;
                    int iw = w + kw;
                    int in_idx = (((n * C_in + ic) * D + id) * H + ih) * W + iw;
                    int w_idx = ((((oc * C_in) + ic) * K + kd) * K + kh) * K + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    int out_idx = (((n * C_out + oc) * D + d) * H + h) * W + w;
    output[out_idx] = sum;
}

// Wrapper called from Python.
torch::Tensor conv3d_naive_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    const auto C_out = weight.size(0);
    const auto K = weight.size(2); // kernel depth (assumed cubic)

    auto output = torch::zeros({N, C_out, D, H, W}, input.options());

    const int threads = 256;
    const int total = N * C_out * D * H * W;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_naive_kernel", ([&] {
        conv3d_naive_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, C_in, D, H, W,
            C_out, K);
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_naive_forward, "Naive Conv3d forward (CUDA)");
}
