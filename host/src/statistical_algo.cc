#include "statistical_algo.h"
#include <algorithm>
#include <numeric>
namespace sci::stats::host {
// template <typename T> T vars1D_with_cpu(T *h_in, const unsigned int N) {
//     T mean = std::accumulate(h_in, h_in + N, T()) / N;
//     std::transform(h_in, h_in + N, h_in, [](int x) { return std::pow(x, 2);
//     }); return std::reduce(h_in, h_in + N, [](T x, T y) { return x + y; }) /
//     N;
// }
// template <typename T>
// T vars2D_with_cpu(T *h_in, const usize_t rows, const usize_t cols) {
//     T *mean = new T[rows];
//     T vars = T();
//     for (int r = 0; r < rows; r++) {
//         mean[r] =
//             std::accumulate(h_in + r * cols, h_in + cols * r + cols, 0) /
//             cols;
//     }
//     for (int r = 0; r < rows; r++) {
//         std::pow(h_in[r * cols] - mean[r], 2);
//     }
// }
template <typename T> void normalize(T *h_out, T *h_in, const int N) {
    T sum = std::accumulate(h_in, h_in + N, T());
    float mean = static_cast<float>(sum) / N;
    std::transform(h_in, h_in + N, h_out,
                   [&mean](T x) { return std::pow(x - mean, 2); });
    T den = std::sqrt(std::accumulate(h_out, h_out + N, T()));
    std::transform(h_in, h_in + N, h_out, [&den, &mean](T a) {
        return static_cast<T>((a - mean) / den);
    });
}
template void normalize(float *, float *, const int);
} // namespace sci::stats::host
