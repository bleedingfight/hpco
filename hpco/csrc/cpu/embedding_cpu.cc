#include "hpco/csrc/op_kernels.h"
#include <cassert>
namespace hpco::cpu {
template <typename T>
void embedding_cpu(T *h_out, const T *h_weight, const int *index, const int n,
                   const int rows, const int embedding_size) {
    {
        for (int i = 0; i < n; i++) {

            assert(index[i] < rows);
            for (int j = 0; j < embedding_size; j++) {
                h_out[i * embedding_size + j] =
                    h_weight[index[i] * embedding_size + j];
            }
        }
    }
}
template void embedding_cpu<float>(float *h_out, const float *h_weight,
                                   const int *index, const int index_size,
                                   const int rows, const int embedding_size);

} // namespace hpco::cpu
