#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <string>
#include <vector>
using namespace std;
// Q:[M,K]
// K:[N,K] -> K^T:[KxN]
// V:[N,L]
template <typename T>
void flashattention(T *out_ptr, T *q_ptr, T *k_ptr, T *v_ptr, const int M,
                    const int N, const int K, const int L) {
    T *qk_row = new T[N];
    for (int m = 0; m < M; m++) {
        T *q_row = q_ptr + m * K;
        T d = T();
        T g_m = std::numeric_limits<T>::min();
        for (int n = 0; n < N; n++) {
            T *k_row = k_ptr + n * K;
            T acc = T();
            // 计算QK^T 第m行
            for (int k = 0; k < K; k++) {
                acc += q_row[k] * k_row[k] / std::sqrt(K);
            }
            d = d * std::exp(g_m - std::max(acc, g_m)) +
                std::exp(acc - std::max(g_m, acc));
            g_m = std::max(g_m, acc);
            // 将QK^T的结果写入缓存
            qk_row[n] = acc;
        }
        // 计算softmax的结果
        for (int n = 0; n < N; n++) {
            qk_row[n] = std::exp(qk_row[n] - g_m) / d;
        }
        // 计算QK^T的第i行和v矩阵的第l列
        for (int l = 0; l < L; l++) {

            T acc = T();
            for (int n = 0; n < N; n++) {
                acc += qk_row[n] * v_ptr[n * L + l];
            }
            out_ptr[m * L + l] = acc;
        }
    }
    delete[] qk_row;
}

template <typename T>
void flashattention_vector(T *out_ptr, T *q_ptr, T *k_ptr, T *v_ptr,
                           const int M, const int N, const int K, const int L) {
    T *qk_row = new T[N];
    for (int m = 0; m < M; m++) {
        T *q_row = q_ptr + m * K;
        T d = T();
        T g_m = std::numeric_limits<T>::min();
        for (int n = 0; n < N; n++) {
            T *k_row = k_ptr + n * K;
            T acc =
                std::inner_product(q_row, q_row + K, k_row, T()) / std::sqrt(K);
            d = d * std::exp(g_m - std::max(acc, g_m)) +
                std::exp(acc - std::max(g_m, acc));
            g_m = std::max(g_m, acc);
            // 将QK^T的结果写入缓存
            qk_row[n] = acc;
        }
        std::transform(qk_row, qk_row + N, qk_row,
                       [&](T v) { return std::exp(v - g_m) / d; });
        // 计算QK^T的第i行和v矩阵的第l列
        for (int l = 0; l < L; l++) {

            T acc = T();
            for (int n = 0; n < N; n++) {
                acc += qk_row[n] * v_ptr[n * L + l];
            }
            out_ptr[m * L + l] = acc;
        }
    }
    delete[] qk_row;
}
vector<float> file_to_vector(const string &filename) {
    ifstream file(filename);
    vector<float> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        float value;
        while (ss >> value) {
            data.push_back(value);
        }
    }
    return data;
}
int main() {
    int M = 4;
    int K = 5;
    int N = 8;
    int L = 4;
    // float *Q = new float[M * K];
    // float *K = new float[N * K];
    // float *V = new float[N * L];
    auto Q_array = file_to_vector("Q.txt");
    auto K_array = file_to_vector("K.txt");
    auto V_array = file_to_vector("V.txt");
    auto Out_array = vector<float>(M * L, 0.0f);
    flashattention_vector(Out_array.data(), Q_array.data(), K_array.data(),
                          V_array.data(), M, N, K, L);
    for (auto e : Out_array) {
        std::cout << e << " ";
    }
    std::cout << Q_array.size() << std::endl;
}
