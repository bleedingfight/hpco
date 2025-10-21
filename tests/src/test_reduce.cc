#include "csrc/cpu/reduce_ops_cpu.h"
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

TEST(TestReduceSum, CUDADeviceSuits) {
    const uint32_t batch_size = 10;
    const uint32_t num_tokens = 1 << 10;
    const int N = batch_size * num_tokens;
    int *data = new int[N];
    int *h_out = new int[batch_size];
    int *d_out = new int[batch_size];
    generateRandom(data, N, 1, 100);
    reduce_sum_with_cpu(h_out, data, batch_size, num_tokens);
    cuda::reduce_sum_with_cuda(d_out, data, batch_size, num_tokens);
    for (int i = 0; i < batch_size; i++) {
        std::cout << "batch value = " << d_out[i] << "\n";
    }
    delete[] data;
    delete[] d_out;
    delete[] h_out;
}
