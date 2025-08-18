#include "hpco/csrc/cuda/cuda_kernels.cuh"
#include "hpco/csrc/op_kernels.h"
#include <cstdio>
#include <iostream>
#include <type_traits>
using namespace flashinfer;

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10) {
    cudaEvent_t start, stop;
    float time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i) {
        bound_function(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    // CHECK_LAST_CUDA_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

namespace hpco::cuda {
namespace fashinfer_ops {
template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
    return (x + y - 1) / y;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void
RMSNormKernel(const T *__restrict__ input, const T *__restrict__ weight,
              T *__restrict__ output, const uint32_t d,
              const uint32_t stride_input, const uint32_t stride_output,
              float weight_bias, float eps) {
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
    const uint32_t thread_id = tx + ty * warp_size;
    const uint32_t num_threads = num_warps * warp_size;
    const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
    extern __shared__ float smem[];

    float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    for (uint32_t i = 0; i < rounds; i++) {
        vec_t<T, VEC_SIZE> input_vec;
        input_vec.fill(0.f);
        if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            input_vec.load(input + bx * stride_input +
                           i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            sum_sq += float(input_vec[j]) * float(input_vec[j]);
        }
    }

    // first, warp reduce sum
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }

    smem[ty] = sum_sq;
    __syncthreads();
    // then, cross warp reduce sum using only the first warp
    if (ty == 0) {
        sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
            sum_sq += math::shfl_xor_sync(sum_sq, offset);
        }
        smem[0] = sum_sq;
    }
    __syncthreads();

    float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

    for (uint32_t i = 0; i < rounds; i++) {
        vec_t<T, VEC_SIZE> input_vec;
        vec_t<T, VEC_SIZE> weight_vec;
        vec_t<T, VEC_SIZE> output_vec;
        input_vec.fill(0.f);
        weight_vec.fill(0.f);
        if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            input_vec.load(input + bx * stride_input +
                           i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
            weight_vec.load(weight + i * num_threads * VEC_SIZE +
                            thread_id * VEC_SIZE);
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            output_vec[j] = float(input_vec[j]) * rms_rcp *
                            (weight_bias + float(weight_vec[j]));
        }
        if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            output_vec.store(output + bx * stride_output +
                             i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t VEC_SIZE, typename T>
__global__ void
RMSNormKernelFake(const T *__restrict__ input, const T *__restrict__ weight,
                  T *__restrict__ output, const uint32_t d,
                  const uint32_t stride_input, const uint32_t stride_output,
                  float weight_bias, float eps) {
    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
    const uint32_t thread_id = tx + ty * warp_size;
    const uint32_t num_threads = num_warps * warp_size;
    const uint32_t rounds =
        (d + VEC_SIZE * num_threads - 1) / (VEC_SIZE * num_threads);
    extern __shared__ float smem[];

    float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    for (uint32_t i = 0; i < rounds; i++) {
        // input float4
        T *input_vec = input + bx * stride_input / VEC_SIZE + i * num_threads +
                       grid.thread_rank();
        // float4 模拟向量
        sum_sq += float(input_vec.x) * float(input_vec.x) +
                  float(input_vec.y) * float(input_vec.y) +
                  float(input_vec.z) * float(input_vec.z) +
                  float(input_vec.w) * float(input_vec.w);
    }

    // first, warp reduce sum
    sum_sq += cooperative_groups::reduce(tile, sum_sq,
                                         cooperative_groups::plus<float>());

    smem[ty] = sum_sq;
    tb.sync();
    // then, cross warp reduce sum using only the first warp
    if (ty == 0) {
        sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
        sum_sq += cooperative_groups::reduce(tile, sum_sq,
                                             cooperative_groups::plus<float>());
        smem[0] = sum_sq;
    }
    tb.sync();

    float rms_rcp = rsqrtf(smem[0] / float(d) + eps);

    for (uint32_t i = 0; i < rounds; i++) {
        T *input_vec = input + bx * stride_input / VEC_SIZE + i * num_threads +
                       grid.thread_rank();

        T *output_vec = output + bx * stride_input / VEC_SIZE +
                        i * num_threads + grid.thread_rank();
        T *weight_vec = weight + i * num_threads + grid.thread_rank();
        if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            output_vec[thread_id].x =
                float(input_vec[thread_id]) * rms_rcp *
                (weight_bias + float(weight_vec[thread_id]));
            output_vec[thread_id].y =
                float(input_vec[thread_id]) * rms_rcp *
                (weight_bias + float(weight_vec[thread_id]));
            output_vec[thread_id].z =
                float(input_vec[thread_id]) * rms_rcp *
                (weight_bias + float(weight_vec[thread_id]));
            output_vec[thread_id].w =
                float(input_vec[thread_id]) * rms_rcp *
                (weight_bias + float(weight_vec[thread_id]));
            if ((i * num_threads + thread_id) * VEC_SIZE < d) {
                output_vec.store(output + bx * stride_output +
                                 i * num_threads * VEC_SIZE +
                                 thread_id * VEC_SIZE);
            }
        }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
     (__CUDA_ARCH__ >= 900))
        asm volatile("griddepcontrol.launch_dependents;");
#endif
    }

    template <typename T>
    cudaError_t RMSNorm(const T *input, const T *weight, T *output,
                        uint32_t batch_size, uint32_t d, uint32_t stride_input,
                        uint32_t stride_output, float eps = 1e-5,
                        bool enable_pdl = false, cudaStream_t stream = 0) {
        const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

        const uint32_t block_size = std::min<uint32_t>(512, d / vec_size);
        const uint32_t num_warps = ceil_div(block_size, 32);
        dim3 nblks(batch_size);
        dim3 nthrs(32, num_warps);
        const uint32_t smem_size = num_warps * sizeof(float);
        float weight_bias = 0.f;
        void *args[] = {&input,        &weight,        &output,      &d,
                        &stride_input, &stride_output, &weight_bias, &eps};

        cudaLaunchConfig_t config;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
        config.numAttrs = 1;
        config.attrs = attrs;

        DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
            auto kernel = RMSNormKernel<VEC_SIZE, T>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
                &config, kernel, input, weight, output, d, stride_input,
                stride_output, weight_bias, eps));
        });
        return cudaSuccess;
    }
} // namespace fashinfer_ops

template <typename scalar_t>
__global__ void
rms_norm_kernel_opt(scalar_t *__restrict__ out,          // [..., hidden_size]
                    const scalar_t *__restrict__ input,  // [..., hidden_size]
                    const scalar_t *__restrict__ weight, // [hidden_size]
                    const float epsilon, const int num_tokens,
                    const int hidden_size) {
    auto tb = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto grid = cooperative_groups::this_grid();
    auto in_row = input + blockIdx.x * hidden_size;
    auto out_row = out + blockIdx.x * hidden_size;
    auto weight_row = weight + blockIdx.x * hidden_size;
    scalar_t thread_value = scalar_t();
    for (int idx = tile.thread_rank(); idx < hidden_size;
         idx += tile.num_threads()) {
        thread_value += in_row[idx] * in_row[idx];
    }
    scalar_t total_sum = scalar_t();
    scalar_t sum_tile = cooperative_groups::reduce(
        tile, thread_value, cooperative_groups::plus<scalar_t>());
    scalar_t scale = rsqrtf(sum_tile / hidden_size + epsilon);
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = (float)input[blockIdx.x * hidden_size + idx];
        out[blockIdx.x * hidden_size + idx] =
            ((scalar_t)(static_cast<float>(in_row[idx]) * scale)) *
            weight_row[idx];
    }
}

template <typename scalar_t>
__global__ void
rms_norm_kernel_smem(scalar_t *__restrict__ out,          // [..., hidden_size]
                     const scalar_t *__restrict__ input,  // [..., hidden_size]
                     const scalar_t *__restrict__ weight, // [hidden_size]
                     const float epsilon, const int num_tokens,
                     const int hidden_size) {

    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto in_row = input + blockIdx.x * hidden_size;
    auto out_row = out + blockIdx.x * hidden_size;

    scalar_t thread_value = scalar_t();
    extern __shared__ scalar_t smem[];
#pragma unroll
    for (int idx = tb.thread_rank(); idx < hidden_size;
         idx += tb.num_threads()) {
        thread_value += in_row[idx] * in_row[idx];
    }
    scalar_t tile_sum = cooperative_groups::reduce(
        tile, thread_value, cooperative_groups::plus<scalar_t>());

    if (tile.thread_rank() == 0) {
        smem[tile.meta_group_rank()] = tile_sum;
    }
    tb.sync();
    scalar_t block_value_sum = scalar_t();
    scalar_t scale;

    if (tile.meta_group_rank() == 0) {
        int warp_nums = tb.num_threads() / tile.num_threads();
        if (tile.thread_rank() < warp_nums) {
            block_value_sum = smem[tile.thread_rank()];
#pragma unroll
            for (int span = warp_nums / 2; span > 0; span >>= 1) {
                block_value_sum +=
                    __shfl_xor_sync(__activemask(), block_value_sum, span);
            }
            scale = rsqrt(block_value_sum / hidden_size + epsilon);
            smem[0] = scale;
        }
    }
    tb.sync();
    scale = smem[0];
    const float4 *in_row4 = reinterpret_cast<const float4 *>(in_row);
    const float4 *weight4 = reinterpret_cast<const float4 *>(weight);
    float4 *out_row4 = reinterpret_cast<float4 *>(out_row);
#pragma unroll
    for (int idx = tb.thread_rank(); idx < hidden_size / 4;
         idx += tb.num_threads()) {
        out_row4[idx].x = in_row4[idx].x * scale * weight4[idx].x;
        out_row4[idx].y = in_row4[idx].y * scale * weight4[idx].y;
        out_row4[idx].z = in_row4[idx].z * scale * weight4[idx].z;
        out_row4[idx].w = in_row4[idx].w * scale * weight4[idx].w;
    }
}

template <typename scalar_t, size_t BLOCK_SIZE, size_t ROUND_NUMS = 10>
void rms_norm_cuda(scalar_t *__restrict__ out,          // [..., hidden_size]
                   const scalar_t *__restrict__ input,  // [..., hidden_size]
                   const scalar_t *__restrict__ weight, // [hi
                   const uint32_t num_tokens, const uint32_t hidden_size,
                   const float epsilon, OPT_MODE mode) {

    const uint32_t vec_size = std::gcd(16 / sizeof(scalar_t), hidden_size);

    const uint32_t block_size =
        std::min<uint32_t>(BLOCK_SIZE, hidden_size / vec_size);
    const uint32_t num_warps = ceil_div(block_size, 32);
    dim3 nblks(num_tokens);
    dim3 nthrs(32, num_warps);
    const uint32_t smem_size = num_warps * sizeof(scalar_t);

    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.dynamicSmemBytes = smem_size;
    cudaLaunchAttribute attrs[1];
    config.numAttrs = 1;
    config.attrs = attrs;
    auto fn = rms_norm_kernel_smem<scalar_t>;
    switch (mode) {
    case OPT_MODE::VLLM:
        fn = rms_norm_kernel_opt<scalar_t>;
        break;
    case OPT_MODE::OPT:
        fn = rms_norm_kernel_smem<scalar_t>;
        break;

    case OPT_MODE::FLASHINFER:
        fashinfer_ops::RMSNorm(input, weight, out, num_tokens, hidden_size,
                               hidden_size, hidden_size, epsilon, false, 0);
        break;
    }

    fn<<<config.gridDim, config.blockDim, sizeof(scalar_t) * num_warps>>>(
        out, input, weight, epsilon, num_tokens, hidden_size);

    if (ROUND_NUMS > 0) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ROUND_NUMS; i++) {
            fn<<<config.gridDim, config.blockDim,
                 sizeof(scalar_t) * num_warps>>>(out, input, weight, epsilon,
                                                 num_tokens, hidden_size);
        }

        float elaps = 0.f;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elaps, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        const size_t size = num_tokens * hidden_size * sizeof(scalar_t);
        size_t numCrossMemoryBound = 2 * size;
        float latency = elaps / ROUND_NUMS;
        float bandwidth = (numCrossMemoryBound / latency) / 1e6;
        size_t nbytes =
            num_tokens * hidden_size * sizeof(scalar_t) / 1024.f / 1024.f;
        std::cout << "Elaps = " << latency << "(ms) Bandwidth = " << bandwidth
                  << "(GB)/s"
                  << " data size:" << nbytes << "MiB\n";
    }
} // namespace fashinfer_ops

template <typename scalar_t, size_t BLOCK_SIZE>
void rms_norm_interface(scalar_t *out,          // [..., hidden_size]
                        const scalar_t *input,  // [..., hidden_size]
                        const scalar_t *weight, // [hidden_size]
                        const float epsilon, const uint32_t num_tokens,
                        const uint32_t hidden_size, OPT_MODE mode) {
    const size_t nbytes = sizeof(scalar_t) * hidden_size * num_tokens;
    scalar_t *d_input, *d_out, *d_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_input), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_weight),
               hidden_size * sizeof(scalar_t));
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    CUDA_CHECK(cudaMemcpy(d_input, input, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight, sizeof(scalar_t) * hidden_size,
                          cudaMemcpyHostToDevice));

    rms_norm_cuda<scalar_t, BLOCK_SIZE, 10>(
        d_out, d_input, d_weight, num_tokens, hidden_size, epsilon, mode);
    CUDA_CHECK(cudaMemcpy(out, d_out, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
}

// template <typename scalar_t, size_t BLOCK_SIZE>
// void rms_norm_interface4(scalar_t *out,          // [..., hidden_size]
//                          const scalar_t *input,  // [..., hidden_size]
//                          const scalar_t *weight, // [hidden_size]
//                          const float epsilon, const int num_tokens,
//                          const int hidden_size, OPT_MODE mode) {
//     const size_t nbytes = sizeof(float) * hidden_size * num_tokens;
//     float *d_input, *d_out, *d_weight;
//     cudaMalloc(reinterpret_cast<void **>(&d_input), nbytes);
//     cudaMalloc(reinterpret_cast<void **>(&d_weight), nbytes);
//     cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
//     cudaMemcpy(d_input, input, nbytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_weight, weight, nbytes, cudaMemcpyHostToDevice);
//     rms_norm_cuda4<float4, BLOCK_SIZE, 10>(
//         reinterpret_cast<float4 *>(d_out), reinterpret_cast<float4
//         *>(d_input), reinterpret_cast<float4 *>(d_weight), epsilon,
//         num_tokens, hidden_size, mode);
//     cudaMemcpy(out, d_out, nbytes, cudaMemcpyDeviceToHost);
// }

template void rms_norm_cuda<float, 1024, 10>(
    float *__restrict__ out,          // [..., hidden_size]
    const float *__restrict__ input,  // [..., hidden_size]
    const float *__restrict__ weight, // [hidden_size]
    const uint32_t num_tokens, const uint32_t hidden_size, const float epsilon,
    OPT_MODE mode);
// template void
// rms_norm_interface<float, 1024>(float *out,          // [..., hidden_size]
//                                 const float *input,  // [..., hidden_size]
//                                 const float *weight, // [hidden_size]
//                                 const float epsilon, const int num_tokens,
//                                 const int hidden_size, OPT_MODE mode);

template void
rms_norm_interface<float, 512>(float *out,          // [..., hidden_size]
                               const float *input,  // [..., hidden_size]
                               const float *weight, // [hidden_size]
                               const float epsilon, const uint32_t num_tokens,
                               const uint32_t hidden_size, OPT_MODE mode);
} // namespace hpco::cuda
