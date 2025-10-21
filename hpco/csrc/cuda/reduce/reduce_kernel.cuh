#include "hpco/csrc/cuda/common.h"
#include <algorithm>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
template <typename T, uint32_t WARP_SIZE>
__global__ void reduce_max_kernel(T *d_out, const T *d_in, const int N);
template <typename T, uint32_t WARP_SIZE>
__global__ void reduce_max_kernel_opt(T *d_out, const T *d_in, const int N);
template <typename T> T reduce_max_with_cuda(T *h_in, const int N);
