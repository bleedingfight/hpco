#include "csrc/cpu/statistical_algo.h"
#include "csrc/cuda/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
TEST(TestNormalize, CUDADeviceSuits) {
    // float x = {1, 2, 3, 4, 5, 6};
    int const N = 10;
    float *h_in = new float[N];
    float *h_out = new float[N];
    std::iota(h_in, h_in + N, 1.f);
    hpco::stats::cpu::normalize(h_out, h_in, 6);
    for (int i = 0; i < 6; i++) {
        std::cout << "max cpu = " << h_out[i] << "\n";
    }
}
