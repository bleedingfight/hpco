#include "common.h"
#include <cuda_runtime.h>
template <typename T, size_t BLOCK_SIZE = 512>
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
namespace cuda {
BINARY_OP(vector_add, 512)
BINARY_OP_REGISTER_TYPE(vector_add, float, 512)
BINARY_OP_REGISTER_TYPE(vector_add, int, 512)
}; // namespace cuda
