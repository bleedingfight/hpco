
#include "common.h"
#include "host_ops.h"
#include "statistical_algo.h"
#include "tests/include/common.h"
#include "timer.h"
#include "unary_operators.cuh"
#include "unary_ops.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
TEST(TestElu, CUDAUnaryOperator) {

    const int N = 1 << 10;
    float *h_in = new float[N];
    float *h_out = new float[N];
    float *gpu_out = new float[N];
    std::iota(h_in, h_in + N, -512.f);
    elu_fp32_cuda(h_out, h_in, N);
    elu_cpu(gpu_out, h_in, N);
    EXPECT_TRUE(same_array(h_out, gpu_out, N, 1e-6));
    // for (int i = 0; i < 10; i++) {
    //     std::cout << h_out[i] << " ";
    // }
    delete[] h_in;
    delete[] h_out;
    delete[] gpu_out;
}
