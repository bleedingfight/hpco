#include "csrc/cpu/softmax.h"
#include "csrc/cuda/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
using namespace hpco::activation::cpu;
TEST(TestSoftmax, CUDADeviceSuits) {
    const int N = 10;
    float *h_in = new float[N];
    float *h_out = new float[N];
    float *h_out1 = new float[N];
    std::iota(h_in, h_in + N, 0.f);
    safesoftmax(h_out, h_in, 2, 5);
    onlinesoftmax(h_out1, h_in, 2, 5);
    for (int i = 0; i < 10; i++) {
        std::cout << "h_out[" << i << "]=" << h_out[i] << " " << h_out1[i]
                  << "\n";
    }
    delete[] h_in;
    delete[] h_out;
    delete[] h_out1;
}
