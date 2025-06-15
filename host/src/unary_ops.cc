#include "unary_ops.h"
template <typename T>
void elu_cpu(T *h_out, T *h_in, const int N, float alpha) {
    std::transform(h_in, h_in + N, h_out, [&alpha](T x) {
        return x > 0 ? x
                     : static_cast<T>(alpha) *
                           (static_cast<T>(std::exp(x)) - static_cast<T>(1.f));
    });
}
template void elu_cpu<float>(float *, float *, const int, float);
