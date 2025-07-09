#include "softmax.h"
#include <limits>
namespace hpco::activation::cpu {
template <typename T>
void safesoftmax(T *h_out, const T *h_in, const size_t rows,
                 const size_t cols) {
    for (int r = 0; r < rows; r++) {
        T max_value = std::reduce(h_in + r * cols, h_in + r * cols + cols,
                                  std::numeric_limits<T>::min(),
                                  [](T a, T b) { return std::max(a, b); });
        std::transform(h_in + r * cols, h_in + r * cols + cols,
                       h_out + r * cols,
                       [&](T x) { return std::exp(x - max_value); });
        T den = std::reduce(h_out + r * cols, h_out + r * cols + cols, T(),
                            [](T a, T b) { return a + b; });
        std::transform(h_out + r * cols, h_out + r * cols + cols,
                       h_out + r * cols, [&](T x) { return x / den; });
    }
}

template <typename T>
void onlinesoftmax(T *h_out, const T *h_in, const size_t rows,
                   const size_t cols) {
    for (int r = 0; r < rows; r++) {
        T max_value = std::numeric_limits<T>::min();
        T den = T();
        for (int c = 0; c < cols; c++) {
            T m_i = std::max(max_value, h_in[r * cols + c]);
            den =
                den * std::exp(max_value - m_i) + std::exp(r * cols + c - m_i);
            max_value = std::max(max_value, h_in[r * cols + c]);
        }
        std::transform(h_in + r * cols, h_in + r * cols + cols,
                       h_out + r * cols,
                       [&](T x) { return std::exp(x - max_value) / den; });
    }
};
template void safesoftmax(float *h_out, const float *h_in, const size_t rows,
                          const size_t cols);

template void onlinesoftmax(float *h_out, const float *h_in, const size_t rows,
                            const size_t cols);
} // namespace hpco::activation::cpu
