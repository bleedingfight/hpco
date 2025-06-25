#pragma once
namespace hpco::embedding::cpu {
template <typename T>
void embedding_cpu(T *d_out, const T *d_weight, const int *index, const int n,
                   const int embedding_size);
} // namespace hpco::embedding::cpu
