#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "hpco/csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
using namespace hpco::cpu;
using namespace hpco::cuda;
TEST(TestRMSNorm, CUDAUnaryOperator) {
    size_t batch = 10;
    size_t d = 1024;
    const size_t N = batch * d;
    float *h_in = new float[N];
    float *out = new float[N];
    float *weight = new float[N];
    std::fill(weight, weight + N, 2.f);
    generateRandom(h_in, N, 0, 10);
    rms_norm(out, h_in, weight, batch, d);
    delete[] h_in;
    delete[] out;
    delete[] weight;
}
