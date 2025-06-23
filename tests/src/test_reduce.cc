#include "csrc/cpu/reduce_ops_cpu.h"
#include "csrc/cpu/statistical_algo.h"
#include "csrc/cuda/common.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <numeric>
#include <utility>
using namespace hpco::reduce_ops::cpu;
TEST(TestReduceMax, CUDADeviceSuits) {
    for (int i = 10; i < 18; i++) {
        const int N = 1 << i;
        int *data = new int[N];
        generateRandom(data, N, 1, 100);
        int max_cpu = reduce_max_with_cpu(data, N);
        int max_cuda = cuda::reduce_max_with_cuda(data, N);
        EXPECT_EQ(max_cpu, max_cuda);
        delete[] data;
        std::cout << "max cpu = " << max_cpu << " cuda max:" << max_cuda
                  << "\n";
    }
}
