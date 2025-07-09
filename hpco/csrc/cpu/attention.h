#pragma once
namespace hpco::gemm::cpu {
template <typename T>
void self_attention_cpu(T *h_o, const T *h_q, const T *h_k, const int h_v,
                        const int M, const int N, const int K,
                        const int tile = 1);
}
