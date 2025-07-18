#include "hpco/csrc/op_kernels.h"
#include <functional>
#include <numeric>
// 转置操作只修改索引
#define IDX2D(idx, rows, cols) ((idx / rows) + (idx % rows) * cols)
namespace hpco::cpu {
// q MxK
// K NxK
template <typename T>
void self_attention_cpu(T *h_o, const T *h_q /*M,K*/, const T *h_k /*N,K*/,
                        const T *h_v /*M,W*/, const int M, const int N,
                        const int K, const int W, const int tile) {
    float scale = std::sqrt(K);
    T *v_rows = new T[M];
    for (int m = 0; m < M; m++) {
        // q 的第m行
        T *q_row = h_q + m * K;
        T *v_row = h_v + m * K;
        T *o_row = h_o + m * K;
        T m_prev = std::numeric_limits<T>::min();
        T den = 0;
        for (int i = 0; i < K; i++) {
            T *k_row = h_q + i * K;
            T t =
                std::transform_reduce(q_row, q_row + K, k_row, k_row + K, T(0),
                                      std::plus<T>(), std::multiplies<T>());
            T m_i = std::max(t, m_prev);
            den = den * std::exp(m_prev - m_i) + std::exp(t);
            o_row[i] = t;
            m_prev = m_i;
        }
        std::transform(o_row, o_row + K, o_row,
                       [&](T x) { return (x - m_prev) / den / scale; });
        // 计算和输出的矩阵乘法
        for (int col = 0; col < W; col++) {
            T o_i = T();
            for (int row = 0; row < N; row++) {
                o_i += o_row[row] * h_v[row * N + col];
            }
            h_o[m * W + col] = o_i;
        }
    }
}

// q MxK
// K NxK
template <typename T>
void flashattention_cpu(T *h_o, const T *h_q /*M,K*/, const T *h_k /*N,K*/,
                        const int h_v /*M,N*/, const int M, const int N,
                        const int K, const int tile) {
    float scale = std::sqrt(K);
    for (int m = 0; m < M; m++) {
        // Q 的第k行
        T *q_row = h_q + m * K;
        T *o_row = h_o + m * K;
        T *v_row = h_v + m * K;
        T m_prev = std::numeric_limits<T>::min();
        T den = 0;
        for (int i = 0; i < N; i++) {
            T t = std::transform_reduce(q_row, q_row + K, h_k, h_k + K, T(0),
                                        std::plus<T>(), std::multiplies<T>());
            T m_i = std::max(t, m_prev);
            den = den * std::exp(m_prev - m_i) + std::exp(t);
            o_row[i] = t;
        }
        std::transform(o_row, o_row + K, o_row,
                       [&](T x) { return (x - m_prev) / den / scale; });
        for (int i = 0; i < N; i++) {
            T o_val =
                std::transform_reduce(o_row, o_row + N, v_row, v_row + N, T(),
                                      std::plus<T>(), std::multiplies<T>());
            h_o[m * N + i] = o_val;
        }
    }
}

} // namespace hpco::cpu
