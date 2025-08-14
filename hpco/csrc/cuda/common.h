#pragma once
#include <stdio.h>
// CUDA错误检查宏
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(error));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#define BINARY_OP_HEAD(op_name)                                                \
    template <typename T>                                                      \
    void op_name##_with_cuda(T *h_out, T *h_in1, T *h_in2, const int N);

#define BINARY_OP(op_name, block_size)                                         \
    template <typename T, int BLOCK_SIZE>                                      \
    void op_name##_with_cuda(T *h_out, T *h_in1, T *h_in2, const int N) {      \
        size_t nbytes = N * sizeof(T);                                         \
        T *d_out, *d_in1, *d_in2;                                              \
        CUDA_CHECK(cudaMalloc((void **)&d_out, nbytes));                       \
        CUDA_CHECK(cudaMalloc((void **)&d_in1, nbytes));                       \
        CUDA_CHECK(cudaMalloc((void **)&d_in2, nbytes));                       \
        CUDA_CHECK(cudaMemcpy(d_in1, h_in1, nbytes, cudaMemcpyHostToDevice));  \
        CUDA_CHECK(cudaMemcpy(d_in2, h_in2, nbytes, cudaMemcpyHostToDevice));  \
        dim3 block = {block_size};                                             \
        dim3 grid = {(N + block.x - 1) / block.x};                             \
        op_name##_kernel<T, block_size>                                        \
            <<<grid, block>>>(d_out, d_in1, d_in2, N);                         \
        CUDA_CHECK(cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost));  \
        cudaFree(d_in1);                                                       \
        cudaFree(d_in2);                                                       \
        cudaFree(d_out);                                                       \
    }
#define BINARY_OP_REGISTER_TYPE(op_name, T)                                    \
    template void op_name##_with_cuda(T *h_dst, T *h_src1, T *h_src2,          \
                                      const int);

namespace cuda {
template <typename T, int BLOCK_SIZE>
void vector_add_with_cuda(T *h_out, T *h_in1, T *h_in2, const int N);
// BINARY_OP_HEAD(vector_add);
template <typename T> T reduce_max_with_cuda(T *h_in, const int N);
} // namespace cuda
