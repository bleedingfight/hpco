#include "csrc/cpu/embedding_cpu.h"
#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <algorithm>
#include <gtest/gtest.h>
using namespace hpco::embedding;
TEST(TestEmbedding, CUDAUnaryOperator) {
    const int rows = 1 << 10;
    const int embedding_size = 768;
    const int out_size = rows * embedding_size;
    constexpr int N = rows * embedding_size;
    const int index_size = 10;
    int *h_index = new int[index_size];
    generateRandom(h_index, index_size, 0, rows - 1);

    float *h_weight = new float[N];
    float *h_out = new float[out_size];
    generateRandom(h_weight, N, -100, 100);

    // float *gpu_out = new float[N];

    cpu::embedding_cpu(h_out, h_weight, h_index, index_size, embedding_size);
    // cpu::embedding_cpu(h_out, h_weight, n, embeding_size);
    // cuda::elu_fp32_cuda(h_out, h_in, N);
    // EXPECT_TRUE(same_array(h_out, gpu_out, N, 1e-6));
    delete[] h_weight;
    delete[] h_out;
    delete[] h_index;
}
