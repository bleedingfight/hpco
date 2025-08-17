#include "hpco/csrc/op_kernels.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace hpco::cpu {
template <typename T>
void rms_norm(T *out, const T *input, const T *weight, const float eps,
              uint32_t batch_size, uint32_t d) {
    for (size_t i = 0; i < batch_size; i++) {
        T sum = std::inner_product(input + i * d, input + (i + 1) * d,
                                   input + i * d, 0.0);
        float rms_rcp = 1.f / (std::sqrt(sum / float(d)) + eps);
        std::transform(input + i * d, input + (i + 1) * d, weight, out + i * d,
                       [&rms_rcp](T a, T b) { return rms_rcp * a * b; });
    }
}
template void rms_norm<float>(float *, const float *, const float *, float, uint32_t,
                              uint32_t);
} // namespace hpco::cpu
