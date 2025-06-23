
#include "csrc/cpu/unary_ops_cpu.h"
#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
TEST(TestElu, CUDAUnaryOperator) {
    const int N = 1 << 10;
    float *h_in = new float[N];
    float *h_out = new float[N];
    float *gpu_out = new float[N];
    generateRandom(h_in, N, -100, 100);
    hpco::unary_ops::cuda::elu_fp32_cuda(h_out, h_in, N);
    hpco::unary_ops::cpu::elu_cpu(gpu_out, h_in, N);
    EXPECT_TRUE(same_array(h_out, gpu_out, N, 1e-6));
    delete[] h_in;
    delete[] h_out;
    delete[] gpu_out;
}
