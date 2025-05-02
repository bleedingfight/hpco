#include "common.h"
#include <algorithm>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda_runtime.h>
#define WARP_SIZE 32
template <typename T>
__global__ void reduce_max_kernel(T *d_out, const T *d_in, const int N) {
    __shared__ T smem[1024];
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto block = cooperative_groups::this_thread_block();
    auto idx =
        block.dim_threads().x * block.group_index().x + block.thread_rank();
    if (idx >= N)
        return;
    smem[threadIdx.x] = block_data[threadIdx.x];

    if (blockDim.x >= 1024 && threadIdx.x < 512) {
        smem[threadIdx.x] = max(smem[threadIdx.x + 512], smem[threadIdx.x]);
    }
    block.sync();
    if (blockDim.x >= 512 && threadIdx.x < 256) {
        smem[threadIdx.x] = max(smem[threadIdx.x + 256], smem[threadIdx.x]);
    }
    block.sync();
    if (blockDim.x >= 256 && threadIdx.x < 128) {
        smem[threadIdx.x] = max(smem[threadIdx.x + 128], smem[threadIdx.x]);
    }
    block.sync();
    if (blockDim.x >= 128 && threadIdx.x < 64) {
        smem[threadIdx.x] = max(smem[threadIdx.x + 64], smem[threadIdx.x]);
    }
    block.sync();

    if (threadIdx.x < 32) {
        smem[threadIdx.x] =
            max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], 16),
                smem[threadIdx.x]);
        smem[threadIdx.x] =
            max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], 8),
                smem[threadIdx.x]);
        smem[threadIdx.x] =
            max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], 4),
                smem[threadIdx.x]);
        smem[threadIdx.x] =
            max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], 2),
                smem[threadIdx.x]);
        smem[threadIdx.x] =
            max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], 1),
                smem[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

template <typename T>
__global__ void reduce_max_kernel_opt(T *d_out, const T *d_in, const int N) {
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto tb = this_thread_block();
    auto grid = this_grid();
    __shared__ T smem[1024];

    auto idx = tb.thread_rank() + grid.block_rank() * grid.num_blocks();
    thread_block_tile<WARP_SIZE> tile32 = tiled_partition<WARP_SIZE>(tb);
    if (threadIdx.x + blockIdx.x * blockDim.x >= N)
        return;

    smem[threadIdx.x] = block_data[blockDim.x * blockIdx.x + threadIdx.x];

    T warp_max = cooperative_groups::reduce(tile32, smem[tb.thread_rank()],
                                            cooperative_groups::greater<T>());
    if (tile32.thread_rank() == 0) {
        smem[tb.thread_rank() / WARP_SIZE] = warp_max;
    }
    tb.sync();

    if (threadIdx.x < WARP_SIZE) {
#pragma unroll
        for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
            smem[threadIdx.x] =
                max(__shfl_xor_sync(__activemask(), smem[threadIdx.x], i),
                    smem[threadIdx.x]);
        }
    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}
namespace cuda {
template <typename T> T reduce_max_with_cuda(T *h_in, const int N) {
    auto nbytes = N * sizeof(T);
    T *d_out, *d_in;
    const size_t block_dim = 1024;
    const int grids = (N + block_dim - 1) / block_dim;
    T *h_out = new T[grids];
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(T) * grids));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice));
    reduce_max_kernel_opt<<<grids, block_dim>>>(d_out, d_in, N);
    CUDA_CHECK(
        cudaMemcpy(h_out, d_out, sizeof(T) * grids, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    T res = *std::max_element(h_out, h_out + grids);
    delete[] h_out;
    return res;
}

template int reduce_max_with_cuda(int *h_in, const int N);
template float reduce_max_with_cuda(float *h_in, const int N);
}; // namespace cuda
