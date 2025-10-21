#include "hpco/csrc/cuda/common.h"
#include "reduce_kernel.cuh"
#include <cassert>
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

template <typename T, uint32_t WARP_SIZE = 32>
__global__ void reduce_max_kernel_opt(T *d_out, const T *d_in, const int N) {
    auto block_data = d_in + blockDim.x * blockIdx.x;
    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    __shared__ T smem[1024];

    auto idx = tb.thread_rank() + grid.block_rank() * grid.num_blocks();
    cooperative_groups::thread_block_tile<WARP_SIZE> tile32 =
        cooperative_groups::tiled_partition<WARP_SIZE>(tb);
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

template <typename T, uint32_t BLOCK_SIZE, uint32_t WARP_SIZE>
__device__ T reduce_sum_block(const T *d_block) {
    constexpr uint32_t WA = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T smem[WA];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);
    auto thread_val = d_block[block.thread_rank()];
    T tile_sum = cooperative_groups::reduce(tile, thread_val,
                                            cooperative_groups::plus<T>());
    if (tile.thread_rank() == 0)
        smem[tile.meta_group_rank()] = tile_sum;
    block.sync();
    T block_sum = T();
    if (tile.meta_group_rank() == 0) {
        T tile_value = tile.thread_rank() < WA ? smem[tile.thread_rank()] : T();
        T tile_sum0 = cooperative_groups::reduce(tile, tile_value,
                                                 cooperative_groups::plus<T>());
        if (tile.thread_rank() == 0)
            smem[0] = tile_sum0;
    }
    block.sync();
    block_sum = smem[0];
    return block_sum;
}

template <typename T, uint32_t BLOCK_SIZE = 1024, uint32_t WARP_SIZE = 32>
__global__ void reduce_sum_kernel(T *d_out, const T *d_in,
                                  const uint32_t batch_size,
                                  const uint32_t num_tokens) {
    auto block = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    __shared__ T smem[BLOCK_SIZE];
    // assert(num_tokens / BLOCK_SIZE <= BLOCK_SIZE);

    cooperative_groups::thread_block_tile<WARP_SIZE> tile32 =
        cooperative_groups::tiled_partition<WARP_SIZE>(block);

    auto block_data = d_in + grid.block_index().x * num_tokens;

    auto repeat = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < repeat; i++) {
        T block_sum = reduce_sum_block<T, BLOCK_SIZE, WARP_SIZE>(
            block_data + i * BLOCK_SIZE);
        if (block.thread_rank() == 0)
            smem[i] = block_sum;
    }
    block.sync();
    T all_sum = reduce_sum_block<T, BLOCK_SIZE, WARP_SIZE>(smem);
    if (block.thread_rank() == 0) {
        d_out[grid.block_index().x] = all_sum;
    }
}

template <typename T>
void reduce_sum_with_cuda(T *h_out, const T *h_in, const uint32_t batch_size,
                          const uint32_t num_tokens) {
    T *d_out, *d_in;
    auto in_nbytes = batch_size * num_tokens * sizeof(T);
    auto out_nbytes = batch_size * sizeof(T);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), in_nbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), out_nbytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_nbytes, cudaMemcpyHostToDevice));
    dim3 block = {32, 32};
    dim3 grid = {batch_size, 1};
    reduce_sum_kernel<T><<<grid, block>>>(d_out, d_in, batch_size, num_tokens);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
};

template void reduce_sum_with_cuda(int *h_out, const int *h_in,
                                   const uint32_t batch_size,
                                   const uint32_t num_tokens);

template int reduce_max_with_cuda(int *h_in, const int N);
template float reduce_max_with_cuda(float *h_in, const int N);
}; // namespace cuda
