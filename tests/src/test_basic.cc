
#include "common.h"
#include "host_ops.h"
#include "statistical_algo.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
#include <utility>
using namespace cuda;
TEST(TestVecAdd, CUDADeviceSuits) {

    const int N = 1 << 20;
    float *h_src1 = new float[N];
    float *h_src2 = new float[N];
    float *h_dst = new float[N];
    float *h_gpu = new float[N];
    std::fill(h_src1, h_src1 + N, 1.f);
    std::fill(h_src2, h_src2 + N, 3.f);
    auto timer = Timer();
    FUNC_COST_TIME(vector_add_with_cuda, h_gpu, h_src1, h_src2, N);
    // vector_add_with_cuda(h_gpu, h_src1, h_src2, N);
    vector_add_with_cpu(h_dst, h_src1, h_src2, N);
    EXPECT_TRUE(std::equal(h_dst, h_dst + N, h_gpu));
    auto elaps = timer.elapsed_nanoseconds();
    // std::cout << "cost time:" << elaps << "(ns)\n";
    // for (int i = 0; i < 10; i++)
    //     std::cout << h_dst[i] << " ";
    delete[] h_src1;
    delete[] h_src2;
    delete[] h_dst;
    delete[] h_gpu;
}

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
    matmul_with_cpu(h_c, h_a, h_b, M, N, K, 1);
    for (int i = 0; i < 10; i++)
        std::cout << h_c[i] << " ";
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}
TEST(TestReduceMax, CUDADeviceSuits) {
    for (int i = 10; i < 18; i++) {
        const int N = 1 << i;
        int *data = new int[N];
        generateRandomInt(data, N, 1, 100);
        int max_cpu = reduce_max_with_cpu(data, N);
        int max_cuda = reduce_max_with_cuda(data, N);
        EXPECT_EQ(max_cpu, max_cuda);
        delete[] data;
        std::cout << "max cpu = " << max_cpu << " cuda max:" << max_cuda
                  << "\n";
    }
}
TEST(TestNormalize, CUDADeviceSuits) {
    // float x = {1, 2, 3, 4, 5, 6};
    int const N = 10;
    float *h_in = new float[N];
    float *h_out = new float[N];
    std::iota(h_in, h_in + N, 1.f);

    sci::stats::host::normalize(h_out, h_in, 6);
    for (int i = 0; i < 6; i++) {
        std::cout << "max cpu = " << h_out[i] << "\n";
    }
}
