#include "common.h"
#include <algorithm>
#include <cuda_runtime.h>
namespace cuda {
template <typename T, int BLOCK_SIZE>
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

BINARY_OP(vector_add, 512)
template void vector_add_with_cuda<float, 512>(float *, float *, float *,
                                               const int);
// BINARY_OP_REGISTER_TYPE(vector_add, float)
// BINARY_OP_REGISTER_TYPE(vector_add, int)
}; // namespace cuda
