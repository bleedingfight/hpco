
#pragma once
#include <algorithm>
// #include <cooperative_groups.h>

// #include <cuda_runtime.h>
#include <iostream>
#include <numeric>
void elu_fp32_cuda(float *h_out, const float *h_in, const int N);
