#include "host_ops.h"
template <typename T>
void vector_add_with_cpu(T *h_dst, T *h_src1, T *h_src2, const int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        h_dst[i] = h_src1[i] + h_src2[i];
}
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
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N) {
    return std::reduce(h_in, h_in + N, T(0),
                       [](int a, int b) { return std::max(a, b); });
}

template void vector_add_with_cpu(float *h_dst, float *h_src1, float *h_src2,
                                  const int);
template void vector_add_with_cpu(int *h_dst, int *h_src1, int *h_src2,
                                  const int);

template void matmul_with_cpu(float *h_c, const float *h_a, const float *h_b,
                              const int M, const int N, const int K,
                              const int tile);
template void matmul_with_cpu(int *h_c, const int *h_a, const int *h_b,
                              const int M, const int N, const int K,
                              const int tile);
template float reduce_max_with_cpu(const float *h_in, const int N);
template int reduce_max_with_cpu(const int *h_in, const int N);
