#pragma once
namespace hpco::gemm::cpu {
template <typename T>
void matmul_with_cpu(T *h_c, const T *h_a, const T *h_b, const int M,
                     const int N, const int K, const int tile);
}
