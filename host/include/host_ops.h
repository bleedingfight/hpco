#pragma once
#include <algorithm>
#include <numeric>
template <typename T>
void vector_add_with_cpu(T *h_dst, T *h_src1, T *h_src2, const int N);
template <typename T>
void matmul_with_cpu(T *h_c, const T *h_a, const T *h_b, const int M,
                     const int N, const int K, const int tile);
template <typename T> T reduce_max_with_cpu(const T *h_in, const int N);
