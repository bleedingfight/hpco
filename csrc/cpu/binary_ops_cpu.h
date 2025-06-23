#pragma once
#include <algorithm>
#include <numeric>
namespace hpco::binary_ops::cpu {
template <typename T>
void vector_add_with_cpu(T *h_dst, T *h_src1, T *h_src2, const int N);
}
