#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "hpco/csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
using namespace hpco;
TEST(TestRmsNorm, CUDAOps) {
    const uint32_t hidden_size = 1 << 19;
    const uint32_t num_tokens = 10;
    const int N = num_tokens * hidden_size;
    const float epsilon = 1e-6f;
    float *h_input = new float[N];
    float *cpu_out = new float[N];
    float *gpu_out = new float[N];
    float *h_weight = new float[hidden_size];
    generateRandom(h_input, N, 0, 10);
    std::fill(h_weight, h_weight + hidden_size, 1.0f);
    cpu::rms_norm(cpu_out, h_input, h_weight, epsilon, num_tokens, hidden_size);

    hpco::cuda::rms_norm_interface<float, 512>(gpu_out, h_input, h_weight,
                                               epsilon, num_tokens, hidden_size,
                                               OPT_MODE::OPT);
hpco::cuda::rms_norm_interface<float, 512>(gpu_out, h_input, h_weight,
                                               epsilon, num_tokens, hidden_size,
                                               OPT_MODE::FLASHINFER);

    EXPECT_TRUE(same_array(cpu_out, gpu_out, N, 1e-6));
    delete[] h_weight;
    delete[] h_input;
    delete[] cpu_out;
    delete[] gpu_out;
}
