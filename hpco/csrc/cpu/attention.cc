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
                        const int K, const int W) {
    float scale = std::sqrt(K);
    T *v_rows = new T[M];
    for (int i = 0; i < M; i++) {
        // q 的第i行
        const T *q_row = h_q + i * K;
        const T *v_row = h_v + i * K;
        T *o_row = h_o + i * K;
        T m_prev = std::numeric_limits<T>::min();
        T den = 0;
        for (int n = 0; n < N; n++) {
            T acc = T();
            for (int k = 0; k < K; k++) {
                acc += h_k[N * k + i] * q_row[k];
            }
            o_row[n] = acc;
            T m_i = std::max(acc, m_prev);
            den = den * std::exp(m_prev - m_i) + std::exp(acc);
            m_prev = m_i;
        }
        std::transform(o_row, o_row + K, o_row,
                       [&](T x) { return (x - m_prev) / den / scale; });
        // 计算和输出的矩阵乘法
        T o_i = T();
        for (int r = 0; r < N; r++) {
            o_i += o_row[r] * h_v[r * W + i];
        }
        o_row[i] = o_i;
    }
}

// q MxK
// K NxK
// template <typename T>
// void flashattention_cpu(T *h_o, const T *h_q /*M,K*/, const T *h_k /*N,K*/,
//                         const int *h_v /*N,W*/, const int M, const int N,
//                         const int K, const int W, const int tile) {
//     float scale = std::sqrt(K);
//     for (int m = 0; m < M; m++) {
//         // Q 的第k行
//         T *q_row = h_q + m * K;
//         T *o_row = h_o + m * K;
//         T *v_row = h_v + m * K;
//         T m_prev = std::numeric_limits<T>::min();
//         T den = 0;
//         for (int i = 0; i < N; i++) {
//             T t = std::transform_reduce(q_row, q_row + K, h_k, h_k + K, T(0),
//                                         std::plus<T>(),
//                                         std::multiplies<T>());
//             T m_i = std::max(t, m_prev);
//             den = den * std::exp(m_prev - m_i) + std::exp(t);
//             // 计算结果先缓存起来
//             o_row[i] = t;
//         }
//         // 计算softmax结果
//         std::transform(o_row, o_row + K, o_row,
//                        [&](T x) { return (x - m_prev) / den / scale; });
//         T acc = 0;
//         for (int i = 0; i < W; i++) {
//             acc += o_row[i] * h_v[i * W + m];
//         }
//         h_o[m * N + i] = o_val;
//     }
// }
template void self_attention_cpu(float *h_o, const float *h_q /*M,K*/,
                                 const float *h_k /*N,K*/,
                                 const float *h_v /*M,W*/, const int M,
                                 const int N, const int K, const int W);
} // namespace hpco::cpu
