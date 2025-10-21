#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
namespace hpco::reduce_ops::cpu {
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N);
template <typename T>
void reduce_sum_with_cpu(T *d_out, const T *h_in, const uint32_t batch_size,
                         const uint32_t num_tokens);
}; // namespace hpco::reduce_ops::cpu
