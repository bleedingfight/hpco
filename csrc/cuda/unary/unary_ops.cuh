
#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#define ALPHA 1.f
namespace hpco::unary_ops::cuda {
__device__ __forceinline__ float elu_fp32(float x) {
    return x > 0 ? x : ALPHA * (__expf(x) - 1);
}
__global__ void elu_kernel_fp32(float *d_out, const float *d_in, const int N) {
    extern __shared__ float smem[];
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto idx = threadIdx.x;
    smem[idx] = block_data[idx];
    tb.sync();
    smem[threadIdx.x] = elu_fp32(smem[threadIdx.x]);
    tb.sync();
    d_out[blockIdx.x * blockDim.x + threadIdx.x] = smem[threadIdx.x];
}

} // namespace hpco::unary_ops::cuda
