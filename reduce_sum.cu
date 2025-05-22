#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <numeric>
#include <utility>
namespace cg = cooperative_groups;
#define WARP_SIZE 32

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(error));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
__global__ void reduce_sum_kernel(float *d_out, float *d_in, const int size) {
    extern __shared__ float shared_mem[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    shared_mem[threadIdx.x] = d_in[blockDim.x * blockIdx.x + threadIdx.x];
    __syncthreads();
    if (blockDim.x >= 1024 and threadIdx.x < 512) {
        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 and threadIdx.x < 256) {
        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 and threadIdx.x < 128) {
        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 and threadIdx.x < 64) {
        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + 64];
    }
    __syncthreads();
    if (blockDim.x >= 64 and threadIdx.x < 32) {
        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + 32];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        shared_mem[threadIdx.x] +=
            __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], 16);
        shared_mem[threadIdx.x] +=
            __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], 8);
        shared_mem[threadIdx.x] +=
            __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], 4);
        shared_mem[threadIdx.x] +=
            __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], 2);
        shared_mem[threadIdx.x] +=
            __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], 1);
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shared_mem[0];
    }
}
template <int N = 1024>
__global__ void reduce_sum_kernel1(float *d_out, float *d_in, const int size) {
    extern __shared__ float shared_mem[];
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    shared_mem[threadIdx.x] = d_in[blockDim.x * blockIdx.x + threadIdx.x];
#pragma unroll
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (blockDim.x >= N and threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
            __syncthreads();
        }
    }
    if (threadIdx.x < 32) {
#pragma unroll
        for (int span = WARP_SIZE / 2; span >= 1; span /= 2) {
            shared_mem[threadIdx.x] +=
                __shfl_xor_sync(__activemask(), shared_mem[threadIdx.x], span);
        }
    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = shared_mem[0];
    }
}
__global__ void reduce_sum_kernel2(float *d_out, float *d_in, int n) {
    extern __shared__ float sdata[];
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    unsigned int ctaSize = cta.size();
    unsigned int numCtas = gridDim.x;
    unsigned int threadRank = cta.thread_rank();
    unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;
    auto r = tile.meta_group_rank();
    printf("tile idx = %d\n", r);

    // float threadVal = 0.f;
    // {
    //     unsigned int i = threadIndex;
    //     unsigned int indexStride = (numCtas * ctaSize);
    //     while (i < n) {
    //         threadVal += d_in[i];
    //         i += indexStride;
    //     }
    //     sdata[threadRank] = threadVal;
    // }

    // // Wait for all tiles to finish and reduce within CTA
    // {
    //     unsigned int ctaSteps = tile.meta_group_size();
    //     unsigned int ctaIndex = ctaSize >> 1;
    //     while (ctaIndex >= 32) {
    //         cta.sync();
    //         if (threadRank < ctaIndex) {
    //             threadVal += sdata[threadRank + ctaIndex];
    //             sdata[threadRank] = threadVal;
    //         }
    //         ctaSteps >>= 1;
    //         ctaIndex >>= 1;
    //     }
    // }

    // // Shuffle redux instead of smem redux
    // {
    //     cta.sync();
    //     if (tile.meta_group_rank() == 0) {
    //         threadVal = cg::reduce(tile, threadVal, cg::plus<float>());
    //     }
    // }

    // if (threadRank == 0)
    //     d_out[blockIdx.x] = threadVal;
}

// __global__ void reduce_sum_kernel2(float *d_out, float *d_in, const int size)
// {
//     extern __shared__ float smem[];

//     auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= size) {
//         return;
//     }

//     auto tb = cooperative_groups::this_thread_block();
//     auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(tb);
//     cooperative_groups::memcpy_async(tb, smem, d_in + blockDim.x *
//     blockIdx.x,
//                                      blockDim.x * sizeof(float));
//     cooperative_groups::wait(tb);

//     float my_value = smem[threadIdx.x % WARP_SIZE]; // 使用 tile 内的相对索引

//     // 在每个 warp (tile) 内部进行求和
//     float sum_value = cooperative_groups::reduce(
//         tile, my_value, cooperative_groups::plus<float>());

//     // 获取当前 tile 在线程块中的索引
//     int tile_index = threadIdx.x / WARP_SIZE;

//     // 将每个 warp 的局部和写回到共享内存的特定位置
//     if (tile.thread_rank() == 0) {
//         smem[tile_index] = sum_value;
//     }
//     __syncthreads(); // 等待所有 warp 完成局部求和和写入

//     // 如果需要整个 block 的总和，可以在 block 的第一个 warp
//     // 的第一个线程进行最终的 reduce
//     float global_block_sum = 0.0f;
//     if (tile_index == 0) {
//         global_block_sum = cooperative_groups::reduce(
//             tile, smem[tile.thread_rank()],
//             cooperative_groups::plus<float>());
//         if (threadIdx.x == 0) {
//             printf("Block sum: %f\n", global_block_sum);
//         }
//     }
// }

float reduce_sum_with_cuda(const float *h_in, const int N) {
    dim3 block = {1024, 1, 1};
    dim3 grid = {(N + block.x - 1) / block.x, 1, 1};
    float *d_in, *d_out;
    auto in_size = N * sizeof(float);
    auto out_size = grid.x * sizeof(float);
    float *h_out = new float[grid.x];
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), in_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), out_size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, out_size, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_sum_kernel2<<<grid, block, sizeof(float) * 1024>>>(d_out, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    float elaps = 0.f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elaps, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // std::cout << "bandwith  " << (in_size + out_size) / 1e-9 / elaps << "\n";
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));
    auto v = std::reduce(h_out, h_out + grid.x, 0.f,
                         [](float x, float y) { return x + y; });
    delete[] h_out;
    return v;
}
float reduce_sum_with_cpu(const float *h_in, const int N) {
    return std::reduce(h_in, h_in + N, 0.f,
                       [](float x, float y) { return x + y; });
}
int main() {
    // const int N = 1 << 28;
    // for (int i = 20; i < 28; i++) {
    //     int N = 1 << i;
    //     float *h_in = new float[N];
    //     std::fill(h_in, h_in + N, 1.f);
    //     float cpu_out = reduce_sum_with_cpu(h_in, N);
    //     float cuda_out = reduce_sum_with_cuda(h_in, N);
    //     std::cout << "N = " << N << " CPU = " << cpu_out
    //               << " CUDA = " << cuda_out << "\n";
    // }

    int N = 1 << 10;
    float *h_in = new float[N];
    std::fill(h_in, h_in + N, 1.f);
    float cpu_out = reduce_sum_with_cpu(h_in, N);
    float cuda_out = reduce_sum_with_cuda(h_in, N);
    std::cout << "N = " << N << " CPU = " << cpu_out << " CUDA = " << cuda_out
              << "\n";
}
