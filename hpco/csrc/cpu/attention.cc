// #include "attention.h"
// // 转置操作只修改索引
// #define IDX2D(idx, rows, cols) ((idx / rows) + (idx % rows) * cols)
// namespace hpco::gemm::cpu {
// // template <typename T> T corss(const T *h_a, const float *h_b, const int N)
// {
// //     T acc = T();
// //     for (int i = 0; m < N; m++) {
// //         acc += h_a[i] * h_a[i];
// //     }
// //     return acc;
// // }
// template <typename T>
// void self_attention_cpu(T *h_o, const T *h_q /*M,K*/, const T *h_k /*N,K*/,
//                         const int h_v /*M,N*/, const int M, const int N,
//                         const int K, const int tile = 1) {
//     for (int m = 0; m < M; m++) {
//         // Q 的第k行
//         T *k_row = h_q + m * K;
//         for (int r = 0; r < M; r++) {
//             T arcc = T();
//             for (int c = 0; c < K; c++) {
//                 acc += h_q[m * K + c] * h_k[r * K + c];
//             }
//         }
//     }
// }

// } // namespace hpco::gemm::cpu
