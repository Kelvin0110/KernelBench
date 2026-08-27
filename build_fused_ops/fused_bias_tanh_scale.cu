
extern "C" __global__
void fused_bias_tanh_scale_kernel(const float* __restrict__ inp,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  const float scaling,
                                  const int B,
                                  const int C,
                                  const int H,
                                  const int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    // compute channel index to fetch bias
    int c = (idx / (H * W)) % C;
    float bias_val = bias[c];

    float val = inp[idx] + bias_val;
    val = tanhf(val);          // tanh activation
    out[idx] = val * scaling; // scaling
}
