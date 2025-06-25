
#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#define ALPHA 1.f
#define UNARY_OP_REGISTER_TYPE(op_name, T)                                     \
    template void op_name##_cuda(T *h_dst, const T *h_src, const int);

#define UNARY_OP_REGISTER(op_name, BLOCK_SIZE, dtype)                          \
    template <typename T>                                                      \
    void op_name##_cuda(T *h_out, const T *h_in, const int N) {                \
        auto nbytes = N * sizeof(T);                                           \
        T *d_in, *d_out;                                                       \
        cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);                  \
        cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);                 \
        cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);                \
        dim3 block = {BLOCK_SIZE, 1, 1};                                       \
        dim3 grid = {(N + block.x - 1) / block.x, 1, 1};                       \
        op_name##_kernel<<<grid, block, block.x * sizeof(T)>>>(d_out, d_in,    \
                                                               N);             \
        cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);              \
        cudaFree(d_in);                                                        \
        cudaFree(d_out);                                                       \
    }                                                                          \
    UNARY_OP_REGISTER_TYPE(op_name, dtype)

namespace hpco::unary_ops::cuda {

__device__ __forceinline__ float elu(float x) {
    return x > 0 ? x : ALPHA * (__expf(x) - 1);
}
__device__ __forceinline__ float silu(float x) { return x / (1 + __expf(-x)); }
__global__ void elu_kernel(float *__restrict__ d_out,
                           const float *__restrict__ d_in, const int N) {
    extern __shared__ float smem[];
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto idx = threadIdx.x;
    smem[idx] = block_data[idx];
    tb.sync();
    smem[threadIdx.x] = elu(smem[threadIdx.x]);
    tb.sync();
    d_out[blockIdx.x * blockDim.x + threadIdx.x] = smem[threadIdx.x];
}

__global__ void silu_kernel(float *__restrict__ d_out,
                            const float *__restrict__ d_in, const int N) {
    extern __shared__ float smem[];
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto idx = threadIdx.x;
    smem[idx] = block_data[idx];
    tb.sync();
    smem[threadIdx.x] = silu(smem[threadIdx.x]);
    tb.sync();
    d_out[blockIdx.x * blockDim.x + threadIdx.x] = smem[threadIdx.x];
}
} // namespace hpco::unary_ops::cuda
