#include "hpco/csrc/op_kernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h>
namespace hpco::cuda {
// block级的运算
__device__ float2
reduce_max_and_exp_sum(const cooperative_groups::thread_block_tile<32> &tile,
                       const float *d_in, size_t N) {
    float thread_max = -FLT_MAX;

    // Step 1: Find per-thread max
    for (int i = tile.thread_rank(); i < N; i += tile.size()) {
        thread_max = max(thread_max, d_in[i]);
    }

    // Step 2: Reduce to find global max within tile
    float max_val = cooperative_groups::reduce(
        tile, thread_max, cooperative_groups::greater<float>());

    // Step 3: Broadcast max_val to all threads in tile
    max_val = tile.shfl(max_val, 0);

    // Step 4: Each thread computes exp(x - max)
    float thread_exp_sum = 0.f;
    for (int i = tile.thread_rank(); i < N; i += tile.size()) {
        thread_exp_sum += expf(d_in[i] - max_val);
    }

    // Step 5: Reduce exp sums
    float exp_sum = cooperative_groups::reduce(
        tile, thread_exp_sum, cooperative_groups::plus<float>());

    // Step 6: Return result
    // printf("max_val = %f exp_sum = %f\n", max_val, exp_sum);
    return make_float2(max_val, exp_sum);
}

__device__ float2 tile_max_and_exp_sum(
    const cooperative_groups::thread_block_tile<32> &tile,
    const float *d_in, // 输入数组
    size_t offset,     // 当前 tile 处理的起始下标
    size_t count       // 当前 tile 要处理的元素个数
) {
    float local_max = -FLT_MAX;

    // 每个线程找自己处理的数据中的最大值
    for (int i = tile.thread_rank(); i < count; i += tile.size()) {
        local_max = max(local_max, d_in[offset + i]);
    }

    // tile 内 reduce 最大值
    float tile_max = cooperative_groups::reduce(
        tile, local_max, cooperative_groups::greater<float>());
    tile_max = tile.shfl(tile_max, 0); // 广播

    // 每个线程计算 exp(x_i - tile_max)
    float local_sum = 0.0f;
    for (int i = tile.thread_rank(); i < count; i += tile.size()) {
        local_sum += expf(d_in[offset + i] - tile_max);
    }

    // tile 内 reduce 和
    float tile_sum = cooperative_groups::reduce(
        tile, local_sum, cooperative_groups::plus<float>());
    tile_sum = tile.shfl(tile_sum, 0); // 广播

    // 返回 tile 局部结果
    return make_float2(tile_max, tile_sum);
}
__device__ float2 reduce_max_and_exp_sum_tile(
    const cooperative_groups::thread_block_tile<32> &tile,
    const float *d_in_block, size_t N) {
    float m0 = -FLT_MAX;
    float den = 0;
    float m1 = cooperative_groups::reduce(tile, d_in_block[threadIdx.x],
                                          cooperative_groups::greater<float>());
    den = den * expf(m0 - m1) + expf(d_in_block[threadIdx.x] - m1);
    m0 = max(m0, m1);
    return make_float2(m0, den);
}
// 只能受限于最大blockDim.x列
__global__ void safe_softmax(float *d_out, const float *d_in, const size_t rows,
                             const size_t cols) {
    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto row_in = d_in + grid.thread_index().x / blockDim.x * cols;
    auto row_out = d_out + grid.thread_index().x / blockDim.x * cols;
    auto r = reduce_max_and_exp_sum(tile, row_in, cols);
    row_out[grid.thread_index().x % blockDim.x] =
        expf(row_in[grid.thread_index().x % blockDim.x] - r.x) / r.y;
}

__global__ void online_softmax(float *d_out, const float *d_in,
                               const size_t rows, const size_t cols) {
    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto row_in = d_in + grid.thread_index().x / blockDim.x * cols;
    auto row_out = d_out + grid.thread_index().x / blockDim.x * cols;
    auto r = reduce_max_and_exp_sum_tile(tile, row_in, cols);
    row_out[grid.thread_index().x % blockDim.x] =
        expf(row_in[grid.thread_index().x % blockDim.x] - r.x) / r.y;
}
void safesoftmax_cuda(float *h_out, const float *h_in, const size_t rows,
                      const size_t cols) {
    float *d_out, *d_in;
    const size_t nbytes = rows * cols * sizeof(float);
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    dim3 block = 512;
    dim3 grid = (rows * cols + block.x - 1) / block.x;
    safe_softmax<<<grid, block>>>(d_out, d_in, rows, cols);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

void onlinesoftmax_cuda(float *h_out, const float *h_in, const size_t rows,
                        const size_t cols) {
    float *d_out, *d_in;
    const size_t nbytes = rows * cols * sizeof(float);
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    dim3 block = 512;
    dim3 grid = (rows * cols + block.x - 1) / block.x;
    online_softmax<<<grid, block>>>(d_out, d_in, rows, cols);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

} // namespace hpco::cuda
