#include "binary_ops_cpu.h"
namespace hpco::binary_ops::cpu {
template <typename T>
void vector_add_with_cpu(T *h_dst, T *h_src1, T *h_src2, const int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        h_dst[i] = h_src1[i] + h_src2[i];
}

template void vector_add_with_cpu(float *h_dst, float *h_src1, float *h_src2,
                                  const int);
template void vector_add_with_cpu(int *h_dst, int *h_src1, int *h_src2,
                                  const int);
} // namespace hpco::binary_ops::cpu
