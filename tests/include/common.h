#pragma once
#include <cmath>
template <typename T>
bool same_array(const T *h_out, const T *d_out, const int N, double tol) {
    for (int i = 0; i < N; i++) {
        auto error = static_cast<T>(abs(d_out[i] - h_out[i]));
        if (error > tol) {
            return false;
        }
    }
    return true;
}
