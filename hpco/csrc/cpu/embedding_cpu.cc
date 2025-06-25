#include "embedding_cpu.h"
namespace hpco::embedding::cpu {
template <typename T>
void embedding_cpu(T *d_out, const T *d_weight, const int *index, const int n,
                   const int embedding_size) {
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < embedding_size; j++) {
                d_out[i * embedding_size + j] =
                    d_weight[index[i] * embedding_size + j];
            }
        }
    }
}
template void embedding_cpu<float>(float *d_out, const float *d_weight,
                                   const int *index, const int n,
                                   const int embedding_size);

} // namespace hpco::embedding::cpu
