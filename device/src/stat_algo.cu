#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace common {
// template <typename T>
// __global__ void normalize_kernel(T *d_out, T *d_in, const int N) {
//     extern __shared__ T sdata[];
//     auto tid = threadIdx.x;
//     auto tb = cooperative_groups::this_thread_block();
//     auto tile = cooperative_groups::tiled_partition<32>(tb);
//     T *cooperative_groups::reduce(tile, smem, cooperative_groups::plus<T>());
// }

} // namespace common
