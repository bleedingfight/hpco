#include "common.h"
#include "hpco/csrc/op_kernels.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace hpco::cuda {
template <typename T>
__global__ void embedding_kernel(T *d_out, const T *d_weight, const int *index,
                                 const int n, const int rows,
                                 const int embedding_size) {
    auto tb = cooperative_groups::this_thread_block();
    auto idx = index[blockIdx.x];
    if (tb.thread_rank() < embedding_size) {
        // Each thread copies one element of the embedding vector
        d_out[blockIdx.x * embedding_size + tb.thread_rank()] =
            d_weight[idx * embedding_size + tb.thread_rank()];
    }
}

template <typename T>
void embedding(T *h_out, const T *h_weight, const int *index, const int n,
               const int rows, const int embedding_size) {
    assert(embedding_size < 1024);
    const size_t nbytes = embedding_size * rows * sizeof(T);
    const size_t out_size = embedding_size * n * sizeof(T);
    T *d_out, *d_weight;
    int *d_index;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_weight), nbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), out_size));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_index), n * sizeof(int)));
    dim3 block = embedding_size;
    dim3 grid = n;
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_index, index, n * sizeof(T), cudaMemcpyHostToDevice));
    embedding_kernel<<<grid, block>>>(d_out, d_weight, d_index, n, rows,
                                      embedding_size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_index));
}

template void embedding<float>(float *h_out, const float *h_weight,
                               const int *index, const int index_size,
                               const int rows, const int embedding_size);
} // namespace hpco::cuda
