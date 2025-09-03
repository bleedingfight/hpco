#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>

#include <limits>
#include <numeric>
enum OPT_MODE {
    VLLM,
    FLASHINFER,
    OPT,
    FAKE,
};
namespace hpco {
namespace cpu {
template <typename T>
void embedding_cpu(T *h_out, const T *h_weight, const int *index, const int n,
                   const int rows, const int embedding_size);
template <typename T>
void rms_norm(T *out, const T *input, const T *weight, const float eps,
              uint32_t batch_size, uint32_t d);

template <typename T>
void safesoftmax(T *h_out, const T *h_in, const size_t rows, const size_t cols);
template <typename T>
void onlinesoftmax(T *h_out, const T *h_in, const size_t rows,
                   const size_t cols);
template <typename T, int tile_size>
void onlinesoftmax_tile(T *h_out, const T *h_in, const size_t rows,
                        const size_t cols);

template <typename T>
void self_attention_cpu(T *h_o, const T *h_q /*M,K*/, const T *h_k /*N,K*/,
                        const T *h_v /*M,W*/, const int M, const int N,
                        const int K, const int W, const int tile);

template <typename T>
void flashattention_cpu(T *h_o, const T *h_q, const T *h_k, const int h_v,
                        const int M, const int N, const int K,
                        const int tile = 1);
} // namespace cpu

namespace cuda {
template <typename T>
void embedding(T *h_out, const T *h_weight, const int *index, const int n,
               const int rows, const int embedding_size);

template <typename scalar_t, size_t BLOCK_SIZE>
void rms_norm_interface(scalar_t *out,          // [..., hidden_size]
                        const scalar_t *input,  // [..., hidden_size]
                        const scalar_t *weight, // [hidden_size]
                        const float epsilon, const uint32_t num_tokens,
                        const uint32_t hidden_size, OPT_MODE mode);

template <typename T, int BLOCK_SIZE = 512>
void online_softmax_interface(T *h_out, const T *h_in, const uint32_t rows,
                              const uint32_t cols);
template<typename DType,typename IdType>
void top_k_top_p_sampling_from_probs_interface(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset);
void uniform(float *h_out,uint64_t batch_size,uint64_t nums,uint64_t philox_seed,uint64_t philox_offset);

} // namespace cuda
} // namespace hpco
