#include "hpco/csrc/op_kernels.h"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
__global__ void uniform_kernel(float *d_out,uint64_t batch_size,uint64_t nums,uint64_t philox_seed,uint64_t philox_offset){
    curandStatePhilox4_32_10_t state;
  curand_init(philox_seed,  blockIdx.x, philox_offset, &state);
    float u = curand_uniform(&state);
    d_out[blockIdx.x*blockDim.x+threadIdx.x] = u;
}
void uniform(float *h_out,uint64_t batch_size,uint64_t nums,uint64_t philox_seed,uint64_t philox_offset){
    const int N  = batch_size*nums;
    float *d_out;
    cudaMalloc(reinterpret_cast<void**>(&d_out),sizeof(float)*N);
    dim3 block = 512;
    dim3 grid = (N+block.x-1)/block.x;
    uniform_kernel<<<grid,block>>>(d_out, batch_size, nums, philox_seed, philox_offset);
    cudaMemcpy(h_out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);

}
