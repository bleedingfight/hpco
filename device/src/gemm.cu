#include "common.h"
#include <algorithm>
#include <cuda_runtime.h>
#define BLOCK_SIZE 128
// template <typename T>
// __global__ void vector_add_kernel(const T *A, const T *B, T *C,
//                                   const int numElements) {
//     auto i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < numElements) {
//         C[i] = A[i] + B[i];
//     }
// }

template <typename T>
__global__ void vector_add_kernel(T *C, const T *A, const T *B,
                                  const int numElements) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ T amem[BLOCK_SIZE];
    __shared__ T bmem[BLOCK_SIZE];
    auto *block_a = A + BLOCK_SIZE * blockDim.x;
    auto *block_b = B + BLOCK_SIZE * blockDim.x;
    amem[threadIdx.x] = block_a[threadIdx.x];
    bmem[threadIdx.x] = block_b[threadIdx.x];
    if (i < numElements) {
        amem[threadIdx.x] = amem[threadIdx.x] + bmem[threadIdx.x];
    }
    __syncthreads();
    if (i < numElements) {
        C[i] = amem[threadIdx.x];
    }
}

// template <typename T>
// __global__ void matmul_kernel(T *C, const T *A, const T *B, const int M,
//                               const int K, const int N) {
//     auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//     auto idy = blockDim.y * blockIdx.y + threadIdx.y;
//     for
// }

// template <typename T>
// void matmul_kernek_with_cuda(T *h_C, const T *h_A, const T *h_B, const int M,
//                              const int K, const int N) {
//     auto nbytes_a = sizeof(T) * M * K;
//     auto nbytes_b = sizeof(T) * K * N;
//     auto nbytes_c = sizeof(T) * M * N;
//     T *d_A, *d_B, *d_C;

//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), nbytes_a));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), nbytes_b));
//     CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes_a, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_B, h_B, nbytes_b, cudaMemcpyHostToDevice));
//     dim3 block(DIM_BLOCK, DIM_BLOCK);
//     dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
//     matmul_kernel<<<grid, block>>>(d_C, d_A, d_B, M, K, N);
//     CUDA_CHECK(cudaMemcpy(h_C, d_C, nbytes_c, cudaMemcpyDeviceToHost));
// }
namespace cuda {

BINARY_OP(vector_add)
BINARY_OP_REGISTER_TYPE(vector_add, float)
BINARY_OP_REGISTER_TYPE(vector_add, int)
}; // namespace cuda
