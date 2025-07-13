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
// __device__ float2 reduce_max_and_exp_sum_tile(
//     const cooperative_groups::thread_block_tile<32> &tile,
//     const float *d_in_block, size_t N) {
//     float tile_max = -FLT_MAX;
//     float tile_max = tile_max =
//         cooperative_groups::reduce(tile, d_in_block[tile.thread_rank()],
//                                    cooperative_groups::greater<float>());
// }
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
int main() {
    const int rows = 5;
    const int cols = 512;
    const int N = rows * cols;
    float *h_data = new float[N];
    float *cpu_out = new float[N];
    float *cuda_out = new float[N];
    generateRandom(h_data, N, 0, 10);
    safesoftmax(cpu_out, h_data, rows, cols);
    safesoftmax_cuda(cuda_out, h_data, rows, cols);
    int i = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            i++;
            std::cout << "index = " << i << " cpu = " << cpu_out[r * cols + c]
                      << " cuda = " << cuda_out[r * cols + c] << "\n";
        }
    }
}
