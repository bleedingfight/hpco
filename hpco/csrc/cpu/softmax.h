#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
namespace hpco::activation::cpu {
template <typename T>
void safesoftmax(T *h_out, const T *h_in, const size_t rows, const size_t cols);
template <typename T>
void onlinesoftmax(T *h_out, const T *h_in, const size_t rows,
                   const size_t cols);
} // namespace hpco::activation::cpu
