#include "reduce_ops_cpu.h"
namespace hpco::reduce_ops::cpu {
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N) {
    return std::reduce(h_in, h_in + N, T(0),
                       [](int a, int b) { return std::max(a, b); });
}
template float reduce_max_with_cpu(const float *h_in, const int N);
template int reduce_max_with_cpu(const int *h_in, const int N);
} // namespace hpco::reduce_ops::cpu
