#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace hpco::embedding::cuda {
template <typename T>
__global__ void embedding_kernel(T *d_out, const T *d_weight, const int *index,
                                 const int n, const int embedding_size) {
    auto tb = cooperative_groups::this_thread_block();
    auto idx = index[blockIdx.x];
    if (tb.thread_rank() < embedding_size) {
        // Each thread copies one element of the embedding vector
        d_out[blockIdx.x * embedding_size + tb.thread_rank()] =
            d_weight[idx * embedding_size + tb.thread_rank()];
    }
}
} // namespace hpco::embedding::cuda
