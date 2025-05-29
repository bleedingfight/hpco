#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
namespace sci::stats::host {
// template <typename T> T vars1D_with_cpu(T *h_in, const unsigned int N);
// T std_with_cpu(T *h_in, const usize_t n, const usize_t);
template <typename T> void normalize(T *h_out, T *h_in, const int N);
} // namespace sci::stats::host
