#pragma once
#include <cmath>
#include <random>
void generateRandomInt(int *h_data, const int N, int minVal, int maxVal,
                       const int seed = 42) {
    static std::mt19937 generator(seed);

    // 创建一个在 [minVal, maxVal] 范围内的均匀整数分布
    auto min = std::min(minVal, maxVal);
    auto max = std::max(minVal, maxVal);
    std::uniform_int_distribution<int> distribution(min, max);
    for (int i = 0; i < N; i++) {
        h_data[i] = distribution(generator);
    }
}
