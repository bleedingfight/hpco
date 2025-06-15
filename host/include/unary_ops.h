#pragma once
#include <algorithm>
#include <cmath>
template <typename T>
void elu_cpu(T *h_out, T *h_in, const int N, float alpha = 1.f);
