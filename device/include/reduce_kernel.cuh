#include "common.h"
#include <algorithm>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda_runtime.h>
#define WARP_SIZE 32
template <typename T>
__global__ void reduce_max_kernel(T *d_out, const T *d_in, const int N);
template <typename T>
__global__ void reduce_max_kernel_opt(T *d_out, const T *d_in, const int N);
template <typename T> T reduce_max_with_cuda(T *h_in, const int N);
