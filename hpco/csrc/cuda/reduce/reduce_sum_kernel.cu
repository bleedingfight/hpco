// #include <cooperative_groups.h>
// #include <cooperative_groups/reduce.h>
// #include <cub/cub.cuh>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <numeric>
// #include <random>
// #include <vector>
// enum class OPTIMIZE : char {
//     BASELINE,
//     OPT1,
//     OPT2,
//     OPT3,
// };
// // 检查CUDA API调用错误的宏
// #define CUDA_CHECK_ERROR(ans) \
//     { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line,
//                       bool abort = true) {
//     if (code != cudaSuccess) {
//         fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code),
//                 file, line);
//         if (abort)
//             exit(code);
//     }
// }

// template <typename T, uint32_t BLOCK_SIZE>
// __global__ void reduce_sum_kernel(T *d_out, const T *d_in,
//                                   const uint32_t col_nums) {
//     using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     // 每个线程处理一个数据块，然后进行规约
//     T thread_sum = 0;
//     for (int i = 0; i < (col_nums + blockDim.x - 1) / blockDim.x; ++i) {
//         uint32_t data_idx =
//             blockIdx.x * col_nums + i * blockDim.x + threadIdx.x;
//         if (data_idx < (blockIdx.x + 1) * col_nums) {
//             thread_sum += d_in[data_idx];
//         }
//     }

//     // 在块内进行规约求和
//     T block_sum = BlockReduce(temp_storage).Sum(thread_sum);

//     // 只有线程0将结果写入输出数组
//     if (threadIdx.x == 0) {
//         d_out[blockIdx.x] = block_sum;
//     }
// }

// template <typename T, uint32_t BLOCK_SIZE>
// __global__ void reduce_sum_kernel_cub(T *d_out, const T *d_in,
//                                       const uint32_t col_nums) {
//     auto block = cooperative_groups::this_thread_block();
//     auto grid = cooperative_groups::this_grid();

//     using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     T thread_val = T();
//     for (int idx = threadIdx.x; idx < col_nums; idx += blockDim.x) {
//         const float x = d_in[blockIdx.x * col_nums + idx];
//         thread_val += x * x;
//     }

//     T block_sum =
//         BlockReduce(temp_storage).Reduce(thread_val, cub::Sum{}, blockDim.x);
//     if (threadIdx.x == 0) {
//         d_out[blockIdx.x] = block_sum;
//     }
// }
// // 使用block在数据上循环
// template <typename T, uint32_t BLOCK_SIZE>
// __global__ void reduce_sum_kernel_block(T *d_out, const T *d_in,
//                                         const uint32_t col_nums) {
//     auto block = cooperative_groups::this_thread_block();
//     auto grid = cooperative_groups::this_grid();

//     using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     T thread_sum = 0;
//     uint32_t repeat = (col_nums + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     const T *row = d_in + grid.block_rank() * col_nums;
//     for (int i = 0; i < repeat; i++) {
//         if (i * BLOCK_SIZE + block.thread_rank() < col_nums) {
//             thread_sum += d_in[i * BLOCK_SIZE + block.thread_rank()];
//         }
//     }
//     T block_sum = BlockReduce(temp_storage).Sum(thread_sum);
//     if (threadIdx.x == 0) {
//         d_out[blockIdx.x] = block_sum;
//     }
// }
// // template <typename T>
// // __device__ blockReduceSum(const cooperative_groups::tiled_partition<32>
// // &tile,
// //                           const T *d_in) {

// // }
// template <typename T, uint32_t BLOCK_SIZE>
// __global__ void reduce_sum_kernel_tile_merge(T *d_out, const T *d_in,
//                                              const uint32_t col_nums) {
//     auto block = cooperative_groups::this_thread_block();
//     auto grid = cooperative_groups::this_grid();
//     // auto tile = cooperative_groups::tiled_partition<32>(block);
//     __shared__ T smem[BLOCK_SIZE];
//     T tile_value = T();
//     for (int i = block.thread_rank(); i < col_nums; i += block.num_threads())
//     {
//         tile_value += d_in[grid.block_rank() * col_nums + i];
//     }
//     T block_sum = cooperative_groups::reduce(block, tile_value,
//                                              cooperative_groups::plus<T>());
//     if (block.thread_rank() == 0) {
//         d_out[grid.block_rank()] = block_sum;
//     }
// }
// template <typename T, uint32_t BLOCK_SIZE, uint32_t BENCH_NUM = 10>
// void reduce_sum_cuda(T *h_out, const T *h_in, const uint32_t batch_size,
//                      const uint32_t col_nums,
//                      OPTIMIZE opt = OPTIMIZE::BASELINE) {
//     T *d_in = nullptr;
//     T *d_out = nullptr;
//     const uint32_t nbytes_in = sizeof(T) * batch_size * col_nums;
//     const uint32_t nbytes_out = sizeof(T) * batch_size;

//     CUDA_CHECK_ERROR(cudaMalloc((void **)&d_in, nbytes_in));
//     CUDA_CHECK_ERROR(cudaMalloc((void **)&d_out, nbytes_out));
//     CUDA_CHECK_ERROR(cudaMemcpy(d_in, h_in, nbytes_in,
//     cudaMemcpyHostToDevice));

//     dim3 grid(batch_size);
//     dim3 block(BLOCK_SIZE);

//     auto fn = reduce_sum_kernel<T, BLOCK_SIZE>;
//     switch (opt) {
//     case OPTIMIZE::BASELINE:
//         std::cout << "BaseLine\n";
//         fn = reduce_sum_kernel_cub<T, BLOCK_SIZE>;
//         break;
//     case OPTIMIZE::OPT1:
//         std::cout << "OPT1\n";
//         fn = reduce_sum_kernel<T, BLOCK_SIZE>;
//         break;
//     case OPTIMIZE::OPT2:
//         std::cout << "OPT2\n";
//         fn = reduce_sum_kernel_block<T, BLOCK_SIZE>;
//         break;
//     default:
//         std::cout << "OPT3 tile merge\n";
//         fn = reduce_sum_kernel_tile_merge<T, BLOCK_SIZE>;
//     }
//     for (int i = 0; i < BENCH_NUM; i++) {
//         fn<<<grid, block>>>(d_out, d_in, col_nums);
//     }
//     float elaps = 0.f;
//     cudaEvent_t start, end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&end);

//     cudaEventRecord(start);
//     for (int i = 0; i < BENCH_NUM; i++) {
//         fn<<<grid, block>>>(d_out, d_in, col_nums);
//     }
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     cudaEventElapsedTime_v2(&elaps, start, end);
//     elaps /= BENCH_NUM;
//     float bandwidth =
//         (batch_size * col_nums + batch_size) * sizeof(T) / elaps / 1e6;
//     std::cout << "Reduce Cost:" << elaps << "(ms) "
//               << "brandwidth = " << bandwidth << "GB/s";

//     CUDA_CHECK_ERROR(cudaDeviceSynchronize());
//     CUDA_CHECK_ERROR(
//         cudaMemcpy(h_out, d_out, nbytes_out, cudaMemcpyDeviceToHost));

//     CUDA_CHECK_ERROR(cudaFree(d_in));
//     CUDA_CHECK_ERROR(cudaFree(d_out));
// }

// template <typename T>
// void generateRandom(T *h_data, const int N, int minVal, int maxVal,
//                     const int seed = 42) {
//     static std::mt19937 generator(seed);

//     // 创建一个在 [minVal, maxVal] 范围内的均匀整数分布
//     auto min = std::min(minVal, maxVal);
//     auto max = std::max(minVal, maxVal);
//     std::uniform_int_distribution<int> distribution(min, max);
//     for (int i = 0; i < N; i++) {
//         h_data[i] = static_cast<T>(distribution(generator));
//     }
// }
// template <typename T>
// void reduce_sum_cpu(T *h_out, const T *h_in, const uint32_t batch_size,
//                     const uint32_t col_nums) {
//     for (int b = 0; b < batch_size; b++) {
//         h_out[b] =
//             std::reduce(h_in + b * col_nums, h_in + b * col_nums + col_nums);
//     }
// }
// template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
// bool same_array(T *d_in, T *d_out, const int N) {
//     for (int i = 0; i < N; i++) {
//         if (d_out[i] == d_in[i]) {
//             return false;
//         }
//     }
//     return true;
// }

// int main() {
//     const uint32_t batch_size = 10;
//     const uint32_t col_nums = 1 << 20;
//     const uint32_t N = batch_size * col_nums;
//     int *h_in = new int[N];
//     int *h_out = new int[batch_size];
//     int *d_out = new int[batch_size];
//     generateRandom(h_in, N, 0, 10);
//     reduce_sum_cpu(h_out, h_in, batch_size, col_nums);
//     reduce_sum_cuda<int, 1024>(h_out, h_in, batch_size, col_nums,
//                                OPTIMIZE::OPT3);
//     // for (int i = 0; i < batch_size; i++) {
//     //     std::cout << h_out[i] << "\n";
//     // }
//     assert(same_array(d_out, h_out, batch_size));
//     delete[] h_out;
//     delete[] h_in;
// }
