#include "csrc/cpu/binary_ops_cpu.h"
#include "csrc/cuda/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
#include <utility>
using namespace hpco::binary_ops::cpu;
// using namespace hpco::binary_ops::cuda;
TEST(TestVecAdd, CUDADeviceSuits) {

    const int N = 1 << 20;
    float *h_src1 = new float[N];
    float *h_src2 = new float[N];
    float *h_dst = new float[N];
    float *h_gpu = new float[N];
    std::fill(h_src1, h_src1 + N, 1.f);
    std::fill(h_src2, h_src2 + N, 3.f);
    auto timer = Timer();
    FUNC_COST_TIME(cuda::vector_add_with_cuda, h_gpu, h_src1, h_src2, N);
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
