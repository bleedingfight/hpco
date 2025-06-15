#include "unary_operators.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#define ALPHA 1.f
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
void elu_fp32_cuda(float *h_out, const float *h_in, const int N) {
    auto nbytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    dim3 block = {512, 1, 1};
    dim3 grid = {(N + block.x - 1) / block.x, 1, 1};
    elu_kernel_fp32<<<grid, block, block.x * sizeof(float)>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
