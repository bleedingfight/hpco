#include "hpco/csrc/op_kernels.h"
namespace hpco::unary_ops::cpu {
template <typename T>
void elu_cpu(T *h_out, T *h_in, const int N, float alpha) {
    std::transform(h_in, h_in + N, h_out, [&alpha](T x) {
        return x > 0 ? x
                     : static_cast<T>(alpha) *
                           (static_cast<T>(std::exp(x)) - static_cast<T>(1.f));
    });
}

template <typename T> void silu_cpu(T *h_out, T *h_in, const int N) {
    std::transform(h_in, h_in + N, h_out,
                   [](T x) { return x / (1 + std::exp(-x)); });
}

template void elu_cpu<float>(float *, float *, const int, float);
template void silu_cpu<float>(float *, float *, const int);
}; // namespace hpco::unary_ops::cpu
