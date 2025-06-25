#pragma once
#include <algorithm>
#include <numeric>
namespace hpco::reduce_ops::cpu {
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N);
};
