#include "csrc/cuda/common.h"
#include "hpco/csrc/op_kernels.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
TEST(TestMatmul, CUDADeviceSuits) {
    int M = 512;
    int N = 512;
    int K = 512;
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];
    std::fill(h_a, h_a + M * K, 1.f);
    std::fill(h_b, h_b + K * N, 2.f);
    // matmul_with_cuda(h_c, h_a, h_b, M, N, K);
    hpco::gemm::cpu::matmul_with_cpu(h_c, h_a, h_b, M, N, K, 1);
    for (int i = 0; i < 10; i++)
        std::cout << h_c[i] << " ";
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}
