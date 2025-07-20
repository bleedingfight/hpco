#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

#include <limits>
#include <numeric>
namespace hpco {
namespace cpu {
template <typename T>
void embedding_cpu(T *h_out, const T *h_weight, const int *index, const int n,
                   const int rows, const int embedding_size);
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
                        const int K, const int W);
template <typename T>
void flashattention_cpu(T *h_o, const T *h_q, const T *h_k, const int h_v,
                        const int M, const int N, const int K);
} // namespace cpu

namespace cuda {
template <typename T>
void embedding(T *h_out, const T *h_weight, const int *index, const int n,
               const int rows, const int embedding_size);
void safesoftmax_cuda(float *h_out, const float *h_in, const size_t rows,
                      const size_t cols);
void onlinesoftmax_cuda(float *h_out, const float *h_in, const size_t rows,
                        const size_t cols);

} // namespace cuda
} // namespace hpco
