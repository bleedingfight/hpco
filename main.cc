#include "tools/include/timer.h"
#include <algorithm>
#include <iostream>
int main() {
    // using namespace cuda;
    const int N = 1 << 20;
    float *h_src1 = new float[N];
    float *h_src2 = new float[N];
    float *h_dst = new float[N];
    std::fill(h_src1, h_src1 + N, 1.f);
    std::fill(h_src2, h_src2 + N, 3.f);
    auto timer = Timer();
    // vector_add_with_cuda(h_dst, h_src1, h_src2, N);
    // auto elaps = timer.elapsed_nanoseconds();
    // std::cout << "cost time:" << elaps << "(ns)\n";
    // for (int i = 0; i < 10; i++)
    //     std::cout << h_dst[i] << " ";
}
