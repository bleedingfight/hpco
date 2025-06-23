#include "gemm_cpu.h"
namespace hpco::gemm::cpu {
template <typename T>
void matmul_with_cpu(T *h_c, const T *h_a, const T *h_b, const int M,
                     const int N, const int K, const int tile) {
    T acc = T();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            acc = T();
            for (int k = 0; k < K; k++) {
                acc += h_a[m * K + k] * h_b[k * N + n];
            }
            h_c[N * m + n] = acc;
        }
    }
}

template void matmul_with_cpu(float *h_c, const float *h_a, const float *h_b,
                              const int M, const int N, const int K,
                              const int tile);
template void matmul_with_cpu(int *h_c, const int *h_a, const int *h_b,
                              const int M, const int N, const int K,
                              const int tile);
} // namespace hpco::gemm::cpu
