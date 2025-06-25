#pragma once
#include <algorithm>
#include <cmath>
namespace hpco::unary_ops::cpu {
template <typename T>
void elu_cpu(T *h_out, T *h_in, const int N, float alpha = 1.f);
template <typename T> void silu_cpu(T *h_out, T *h_in, const int N);
} // namespace hpco::unary_ops::cpu
