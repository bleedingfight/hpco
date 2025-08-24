#include "csrc/cuda/common.h"
#include "csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
using namespace hpco::cpu;
TEST(TestSoftmax, CUDADeviceSuits) {
    const int rows = 6;
    const int cols = 1024;
    const int N = rows * cols;
    float *h_in = new float[N];
    float *h_out = new float[N];
    float *h_out1 = new float[N];
    float *h_out2 = new float[N];
    generateRandom(h_in, N, 1, 100);
    safesoftmax(h_out, h_in, rows, cols);
    onlinesoftmax(h_out1, h_in, rows, cols);
    onlinesoftmax_tile<float, 32>(h_out2, h_in, rows, cols);
    auto fin = std::ofstream("/tmp/softmax_input.txt");
    for (int i = 0; i < N; i++) {
        fin << h_in[i] << " ";
    }
    hpco::cuda::online_softmax_interface(h_out2, h_in, rows, cols);
    EXPECT_TRUE(same_array(h_out, h_out1, N, 1e-6));
    EXPECT_TRUE(same_array(h_out, h_out2, N, 1e-6));
    delete[] h_in;
    delete[] h_out;
    delete[] h_out1;
    delete[] h_out2;
}
