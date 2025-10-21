#include "reduce_ops_cpu.h"
namespace hpco::reduce_ops::cpu {
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N) {
    return std::reduce(h_in, h_in + N, T(0),
                       [](int a, int b) { return std::max(a, b); });
}

template <typename T>
void reduce_sum_with_cpu(T *h_out, const T *h_in, const uint32_t batch_size,
                         const uint32_t num_tokens) {
    for (int i = 0; i < batch_size; i++) {
        h_out[i] =
            std::reduce(h_in + i * num_tokens, h_in + (i + 1) * num_tokens,
                        T(0), std::plus<T>{});
    }
}

template float reduce_max_with_cpu(const float *h_in, const int N);
template int reduce_max_with_cpu(const int *h_in, const int N);

template void reduce_sum_with_cpu(float *, const float *, const uint32_t,
                                  const uint32_t);
template void reduce_sum_with_cpu(int *, const int *, const uint32_t,
                                  const uint32_t);
} // namespace hpco::reduce_ops::cpu
