#include "statistical_algo.h"
#include <algorithm>
#include <cmath>
namespace sci::stats::host {
template <typename T> T vars1D_with_cpu(T *h_in, const unsigned int N) {
    T mean = std::accumulate(h_in, h_in + N, T()) / N;
    std::transform(h_in, h_in + N, h_in, [](int x) { return std::pow(x, 2); });
    return std::reduce(h_in, h_in + N, [](T x, T y) { return x + y; }) / N;
}

T vars2D_with_cpu(T *h_in, const usize_t rows, const usize_t cols) {
    T *mean = new T[rows];
    T vars = T();
    for (int r = 0; r < rows; r++) {
        mean[r] =
            std::accumulate(h_in + r * cols, h_in + cols * r + cols, 0) / cols;
    }
    for (int r = 0; r < rows; r++) {
        std::pow(h_in[r * cols] - mean[r], 2);
    }
}
} // namespace sci::stats::host
