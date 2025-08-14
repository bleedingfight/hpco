#include "hpco/csrc/op_kernels.h"
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
namespace hpco::cuda {
template <typename T>
void generateRandom(T *h_data, const int N, int minVal, int maxVal,
                    const int seed = 42) {
    static std::mt19937 generator(seed);

    // 创建一个在 [minVal, maxVal] 范围内的均匀整数分布
    auto min = std::min(minVal, maxVal);
    auto max = std::max(minVal, maxVal);
    std::uniform_int_distribution<int> distribution(min, max);
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<T>(distribution(generator));
    }
}
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
// template <typename T>
// __global__ void softmax_kernel(T *out, const T *d_in,
//                                const int stride1 /*cols*/, const int stride2)
//                                {
//     auto tb = cooperative_groups::this_thread_block();
//     auto grid = cooperative_groups::this_grid();
//     auto tile = cooperative_groups::thread_block_tile<32>(tb);
//     T thead_value = -FLT_MAX;
//     // 缓存tile的计算结果
//     extern __shared__ T shared_data[];
//     for (auto idx = tile.thread_rank(); idx < stride1; idx += tile.size()) {
//         thead_value = d_in[grid.thread_index().x * stride1 + idx];
//         T max_value = cooperative_groups::reduce(
//             tile, thead_value, cooperative_groups::greater<T>());
//         max_value = fmax(max_value, thead_value);

//         // max_value = tile.shfl(max_value, 0); // 广播
//         T exp_sum = T();
//         for (auto i = tile.thread_rank(); i < stride2; i += tile.size()) {
//             exp_sum += expf(d_in[idx * stride2 + i] - max_value);
//         }
//         exp_sum = cooperative_groups::reduce(tile, exp_sum,
//                                              cooperative_groups::plus<T>());
//         shared_data[tile.meta_group_index()] = make_float2(max_value,
//         exp_sum);
//     }

//     // out[idx * stride2 + grid.thread_index().y] =
//     //     expf(thead_value - max_value) / exp_sum;
// }

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

template <typename T>
void safesoftmax(T *h_out, const T *h_in, const size_t rows,
                 const size_t cols) {
    for (int r = 0; r < rows; r++) {
        T max_value = std::reduce(h_in + r * cols, h_in + r * cols + cols,
                                  std::numeric_limits<T>::min(),
                                  [](T a, T b) { return std::max(a, b); });
        std::transform(h_in + r * cols, h_in + r * cols + cols,
                       h_out + r * cols,
                       [&](T x) { return std::exp(x - max_value); });
        T den = std::reduce(h_out + r * cols, h_out + r * cols + cols, T(),
                            [](T a, T b) { return a + b; });
        std::transform(h_out + r * cols, h_out + r * cols + cols,
                       h_out + r * cols, [&](T x) { return x / den; });
    }
}
struct __align__(8) MD {
    float m; // max
    float d; // sum of exp
};
template <const int kWarpSize = 32>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;

        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return value;
}

template <const int NUM_THREADS = 256, int WARP_SIZE = 32>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x,
                                                         float *y, int N) {
    // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer
    // calculation for softmax)
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
    const int WARP_NUM = NUM_THREADS / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    MD val;
    val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
    val.d = global_tid < N ? 1.0f : 0.0f;

    __shared__ MD shared[WARP_NUM];
    MD res = warp_reduce_md_op<WARP_SIZE>(val);

    if (lane_id == 0)
        shared[warp_id] = res;
    __syncthreads();

    if (local_tid < WARP_SIZE) {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WARP_NUM>(block_res);
        if (local_tid == 0) {
            shared[0] = block_res;
        }
    }
    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (global_tid < N) {
        y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
    }
}
template <typename T>
__device__ __inline__ MD tile_max_and_exp_sum_kernel(
    const cooperative_groups::thread_block_tile<32> &tile, MD init, T *sdata,
    size_t N) {
    T thread_value = sdata[tile.thread_rank()];
    T tile_max = cooperative_groups::reduce(tile, thread_value,
                                            cooperative_groups::greater<T>());
    T den = cooperative_groups::reduce(tile, expf(thread_value - tile_max),
                                       cooperative_groups::plus<T>());
    den = static_cast<T>(expf(static_cast<float>(max(init.m, tile_max)))) *
              init.d +
          den;
    return {tile_max, den};
}
template <const int BLOCK_SIZE = 512, int WARP_SIZE = 32>
__global__ void online_softmax_kernel(float *d_out, const float *d_in,
                                      const int rows, const int cols) {
    auto grid = cooperative_groups::this_grid();
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(tb);
    MD init;
    init.m = blockIdx.x < cols ? d_in[tb.thread_index()] : -FLT_MAX;
    // init.d = global_tid < N ? 1.0f : 0.0f;
    extern __shared__ float smem[];
    smem[tb.thread_rank()] =
        (grid.thread_rank() < rows * cols)
            ? d_in[grid.block_rank() * cols + tb.thread_rank()]
            : -FLT_MAX; // 仅处理有效数据
    tb.sync();
    MD value = tile_max_and_exp_sum_kernel(tile, init, smem, cols);
}
template <typename T, int BLOCK_SIZE>
void online_softmax_interface(T *h_out, const T *h_in, const int rows,
                              const int cols) {
    T *d_in, *d_out;
    const size_t nbytes = rows * cols * sizeof(T);
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE);
    dim3 grid((rows * cols + block.x - 1) / block.x);
    online_safe_softmax_f32_per_token_kernel<<<grid, block>>>(d_in, d_out,
                                                              rows * cols);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
template void online_softmax_interface<float>(float *h_out, const float *h_in,
                                              const int rows, const int cols);
} // namespace hpco::cuda
