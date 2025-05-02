// #include "common.h"
// #include <cuda_runtime.h>
// #define BLOCK_SIZE 128
// // template <typename T>
// // __global__ void vector_add_kernel(const T *A, const T *B, T *C,
// //                                   const int numElements) {
// //     auto i = blockDim.x * blockIdx.x + threadIdx.x;
// //     if (i < numElements) {
// //         C[i] = A[i] + B[i];
// //     }
// // }

// template <typename T>
// __global__ void vector_add_kernel(T *C, const T *A, const T *B,
//                                   const int numElements) {
//     auto i = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ T amem[BLOCK_SIZE];
//     __shared__ T bmem[BLOCK_SIZE];
//     auto *block_a = A + BLOCK_SIZE * blockDim.x;
//     auto *block_b = B + BLOCK_SIZE * blockDim.x;
//     amem[threadIdx.x] = block_a[threadIdx.x];
//     bmem[threadIdx.x] = block_b[threadIdx.x];
//     if (i < numElements) {
//         amem[threadIdx.x] = amem[threadIdx.x] + bmem[threadIdx.x];
//     }
//     __syncthreads();
//     if (i < numElements) {
//         C[i] = amem[threadIdx.x];
//     }
// }
// namespace cuda {
// BINARY_OP(vector_add)
// BINARY_OP_REGISTER_TYPE(vector_add, float)
// BINARY_OP_REGISTER_TYPE(vector_add, int)
// }; // namespace cuda
