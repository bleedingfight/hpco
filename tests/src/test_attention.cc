#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "hpco/csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
using namespace hpco::cpu;
using namespace hpco::cuda;
TEST(TestFlashAttention, CUDAUnaryOperator) {
    const int M = 32;
    const int N = 32;
    const int K = 32;
    const int W = 32;
    const int tile = 1;
    float *h_q = new float[M * N];
    float *h_k = new float[N * K];
    float *h_v = new float[M * W];
    float *h_o = new float[M * W];

    generateRandom(h_q, M * N, -10, 10);
    generateRandom(h_k, N * K, -10, 10, 0);
    generateRandom(h_v, M * W, -10, 10, 123);

    self_attention_cpu(h_o, h_q /*M,K*/, h_k /*N,K*/, h_v /*M,W*/, M, N, K, W);
    // EXPECT_TRUE(same_array(h_out, gpu_out, N, 1e-6));
    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_o;
}
