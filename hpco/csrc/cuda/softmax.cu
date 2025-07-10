#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <limits>
#include <numeric>
// __device__ float2
// reduce_max_and_exp_sum(const cooperative_groups::thread_block_tile<32> &tile,
//                        const float *d_in, const size_t N) {
//     float2 p;
//     float m_0 = -100.f;
//     float den = 0;
//     for (int i = tile.thread_rank(); i < N; i += tile.num_threads()) {
//         float m_1 = max(m_0, d_in[i]);
//         den = den * expf(m_0 - m_1) + expf(d_in[i] - m_1);
//         m_0 = max(m_0, d_in[i]);
//         // printf("i = %d max = %f den = %f part before %f part after %f\n",
//         i,
//         //        m_0, den, den * expf(m_0 - max(m_0, d_in[i])),
//         //        expf(d_in[i] - max(m_0, d_in[i])));
//         printf("i = %d m_0 = %f %d m_1 = %f m_0-m_1 = %f\n", i, m_0, m_1,
//                m_0 - m_1);
//     }
//     if (tile.thread_rank() == 0) {
//         p.x = m_0;
//         p.y = den;
//     }
//     return p;
// }
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
    printf("max_val = %f exp_sum = %f\n", max_val, exp_sum);
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

__global__ void online_softmax(float *d_out, const float *d_in,
                               const size_t rows, const size_t cols) {
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto r = reduce_max_and_exp_sum(tile, d_in, cols);
    __brkpt();
}
void softmax_cuda(float *h_out, float *h_in, const size_t rows,
                  const size_t cols) {
    auto nbytes = rows * cols * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    dim3 block = 32;
    dim3 grid = (cols + block.x - 1) / block.x;
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    online_softmax<<<grid, block>>>(d_out, d_in, rows, cols);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
}
int main() {
    const int N = 64;
    float *h_out = new float[N];
    float *h_in = new float[N];
    std::iota(h_in, h_in + N, 0.f);
    softmax_cuda(h_out, h_in, 1, N);
    for (int i = 0; i < 10; i++) {
        std::cout << h_out[i] << " ";
    }
}
