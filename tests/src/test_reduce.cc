#include "csrc/cuda/common.h"
#include "hpco/csrc/op_kernels.h"
#include "timer.h"
#include "utils.h"
#include <gtest/gtest.h>
using namespace hpco::reduce_ops::cpu;
TEST(TestReduceMax, CUDADeviceSuits) {
    for (int i = 10; i < 18; i++) {
        const int N = 1 << i;
        int *data = new int[N];
        generateRandom(data, N, 1, 100);
        int max_cpu = reduce_max_with_cpu(data, N);
        int max_cuda = cuda::reduce_max_with_cuda<int, 512>(data, N);
        EXPECT_EQ(max_cpu, max_cuda);
        delete[] data;
        std::cout << "max cpu = " << max_cpu << " cuda max:" << max_cuda
                  << "\n";
    }
}
