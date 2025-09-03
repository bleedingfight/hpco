#include "hpco/csrc/cuda/cuda_kernels.cuh"
#include "hpco/csrc/op_kernels.h"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>



template<typename DType,typename IdType>
__global__ void top_k_top_p_sampling_from_probs_kernel(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset){
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];
  float u = curand_uniform(&state) ;

};
