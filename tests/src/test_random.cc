#include "csrc/cuda/common.h"
#include "csrc/cuda/ops.h"
#include "hpco/csrc/op_kernels.h"
#include "tests/include/common.h"
#include "timer.h"
#include "utils.h"
#include <cstdlib>
#include <gtest/gtest.h>
using namespace hpco::cpu;
// using namespace hpco::cuda;
TEST(TestRandom, CUDAOperator) {
    const int batch_size = 32;
    const int nums = 32;
    constexpr int N = batch_size*nums;
    float *h_out = new float[N];
        hpco::cuda::uniform(h_out,batch_size,nums,42,N);
    for(int i=0;i<N;i++){
        std::cout<<h_out[i]<<"\n";
    }
    delete[] h_out;
}
