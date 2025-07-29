#include "hpco/csrc/op_kernels.h"
namespace hpco::cpu {

template <typename T>
void rms_norm(T *out, const T *input, const T *weight, size_t batch_size,
              size_t d, float eps) {
    for (size_t i = 0; i < batch_size; i++) {
        T sum = std::inner_product(input + i * d, input + (i + 1) * d,
                                   input + i * d, T());
        float rms_rcp = 1.f / (std::sqrt(sum / float(d)) + eps);
        std::transform(input + i * d, input + (i + 1) * d, out + i * d,
                       out + i * d,
                       [&rms_rcp](T a, T b) { return rms_rcp * a * b; });
    }
}
template void rms_norm<float>(float *, const float *, const float *, size_t,
                              size_t, float);
} // namespace hpco::cpu
