#pragma once
namespace hpco::unary_ops::cuda {
template <typename T> void elu_cuda(T *h_out, const T *h_in, const int N);
template <typename T> void silu_cuda(T *h_out, const T *h_in, const int N);
} // namespace hpco::unary_ops::cuda
