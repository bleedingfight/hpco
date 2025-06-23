#include "unary_ops.cuh"
namespace hpco::unary_ops::cuda {
void elu_fp32_cuda(float *h_out, const float *h_in, const int N) {
    auto nbytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(reinterpret_cast<void **>(&d_in), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    dim3 block = {512, 1, 1};
    dim3 grid = {(N + block.x - 1) / block.x, 1, 1};
    elu_kernel_fp32<<<grid, block, block.x * sizeof(float)>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
} // namespace hpco::unary_ops::cuda
