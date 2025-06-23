#pragma once
namespace hpco::unary_ops::cuda {
void elu_fp32_cuda(float *h_out, const float *h_in, const int N);
}
