#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "hpco/csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
using namespace hpco::cpu;
using namespace hpco::cuda;
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
    float *gpu_out = new float[out_size];
    generateRandom(h_weight, N, -100, 100);

    embedding_cpu(h_out, h_weight, h_index, index_size, rows, embedding_size);
    embedding(gpu_out, h_weight, h_index, index_size, rows, embedding_size);
    EXPECT_TRUE(same_array(h_out, gpu_out, N, 1e-6));
    delete[] h_weight;
    delete[] h_out;
    delete[] gpu_out;
    delete[] h_index;
}
