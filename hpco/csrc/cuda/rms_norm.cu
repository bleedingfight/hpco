#include "hpco/csrc/cuda/cuda_kernels.cuh"
#include "hpco/csrc/cuda/math.cuh"
#include "hpco/csrc/cuda/utils.cuh"
#include "hpco/csrc/cuda/vec_dtypes.cuh"
#include "hpco/csrc/op_kernels.h"
#include <cstdio>
#include <iostream>
#include <type_traits>
using namespace flashinfer;
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

template <typename T>
cudaError_t RMSNorm(const T *input, const T *weight, T *output,
                    uint32_t batch_size, uint32_t d, uint32_t stride_input,
                    uint32_t stride_output, float eps = 1e-5,
                    bool enable_pdl = false, cudaStream_t stream = 0) {
    const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
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
    attrs[0].id = cudaLaunchAttributeProgrammat

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = (float)input[blockIdx.x * hidden_size + idx];
        out[blockIdx.x * hidden_size + idx] =
            ((scalar_t)(x * s_variance)) * weight[idx];
    }
}

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
        // 使用一个warp对元素reduce
        // TODO 这里有问题，似乎只累加了前4个元素
                      const int hidden_size) {
    auto tb = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();
    auto tile = cooperative_groups::tiled_partition<32>(tb);
    auto in_row = input + blockIdx.x * hidden_size / 4;
    auto out_row = out + blockIdx.x * hidden_size / 4;
    auto weight_row = weight + blockIdx.x * hidden_size / 4;
    float thread_value = 0.f;
    extern __shared__ float smem[];
#pragma unroll
    for (int idx = tb.thread_rank(); idx < hidden_size / 4;
         idx += tb.num_threads()) {
        thread_value +=
            in_row[idx].x * in_row[idx].x + in_row[idx].y * in_row[idx].y +
            in_row[idx].z * in_row[idx].z + in_row[idx].w * in_row[idx].w;
    }
    float tile_sum = cooperative_groups::reduce(
        tile, thread_value, cooperative_groups::plus<float>());

    if (tile.thread_rank() == 0) {
        smem[tile.meta_group_rank()] = tile_sum;
    }
    tb.sync();
    float block_value_sum = 0.f;
    float scale = 0.f;

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
#pragma unroll
    for (int idx = tb.thread_rank(); idx < hidden_size / 4;
         idx += tb.num_threads()) {
        out_row[idx].x = in_row[idx].x * scale * weight[idx].x;
        out_row[idx].y = in_row[idx].y * scale * weight[idx].y;
        out_row[idx].z = in_row[idx].z * scale * weight[idx].z;
        out_row[idx].w = in_row[idx].w * scale * weight[idx].w;
    }
}
template <typename scalar_t, size_t BLOCK_SIZE, size_t ROUND_NUMS>
void rms_norm_cuda(scalar_t *__restrict__ out,          // [..., hidden_size]
                   const scalar_t *__restrict__ input,  // [..., hidden_size]
                   const scalar_t *__restrict__ weight, // [hi
        }
        float elaps = 0.f;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elaps, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        float latency = elaps / ROUND_NUMS;
        std::cout << "FlashInfer Cost time:" << latency << "\n";
        return;
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
}

template <typename scalar_t, size_t BLOCK_SIZE>
void rms_norm_interface(scalar_t *out,          // [..., hidden_size]
                        const scalar_t *input,  // [..., hidden_size]
                        const scalar_t *weight, // [hidden_size]
                        const float epsilon, const int num_tokens,
                        const int hidden_size, OPT_MODE mode) {
    const size_t nbytes = sizeof(scalar_t) * hidden_size * num_tokens;
    scalar_t *d_input, *d_out, *d_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_input), nbytes);
    cudaMalloc(rei
    config.attrs = attrs;

    auto fn = rms_norm_kernel_smem4<scalar_t>;
    fn<<<config.gridDim, config.blockDim, sizeof(float) * num_warps>>>(
        out, input, weight, epsilon, num_tokens, hidden_size);

    if (ROUND_NUMS > 0) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ROUND_NUMS; i++) {
            fn<<<config.gridDim, config.blockDim,
                 sizeof(scalar_t) * num_warps / 4>>>(
                out, input, weight, epsilon, num_tokens, hidden_size);
        }
        float elaps = 0.f;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elaps, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        const size_t size = num_tokens * hidden_size * sizeof(float);
        size_t numCrossMemoryBound = 2 * size;
        float latency = elaps / ROUND_NUMS;
        float bandwidth = (numCrossMemoryBound / latency) / 1e6;
        std::cout << "Elaps = " << latency << "(ms) Bandwidth = " << bandwidth
                  << "(GB)/s\n";
    }
}

template <typename scalar_t, size_t BLOCK_SIZE>
void rms_norm_interface4(scalar_t *out,          // [..., hidden_size]
                         const scalar_t *input,  // [..., hidden_size]
                         const scalar_t *weight, // [hidden_size]
                         const float epsilon, const int num_tokens,
                         const int hidden_size, OPT_MODE mode) {
    const size_t nbytes = sizeof(float) * hidden_size * num_tokens;
    float *d_input, *d_out, *d_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_input), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_weight), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes);
    cudaMemcpy(d_input, input, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, nbytes, cudaMemcpyHostToDevice);
    rms_norm_cuda4<float4, BLOCK_SIZE, 10>(
        reinterpret_cast<float4 *>(d_out), reinterpret_cast<float4 *>(d_input),
        reinterpret_cast<float4 *>(d_weight), epsilon, num_tokens, hidden_size,
        mode);
    cudaMemcpy(out, d_out, nbytes, cudaMemcpyDeviceToHost);
}
template void rms_norm_cuda<float, 1024, 10>(float *, const float *,
                                             const float *, const float,
                                             const int, c
//     static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
//     static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) *
//     width);

//     const int vec_hidden_size = hidden_size / width;
//     __shared__ float s_variance;
//     float variance = 0.0f;
//     /* These and the argument pointers are all declared `restrict` as they
//     are
//        not aliased in practice. Argument pointers should not be dereferenced
//        in this kernel as that would be undefined behavior */
//     auto *__restrict__ input_v =
//         reinterpret_cast<_f16Vec<scalar_t, width> *>(input);
//     auto *__restrict__ residual_v =
//         reinterpret_cast<_f16Vec<scalar_t, width> *>(residual);
//     auto *__restrict__ weight_v =
//         reinterpret_cast<const _f16Vec<scalar_t, width> *>(weight);

//     for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
//         int id = blockIdx.x * vec_hidden_size + idx;
//         _f16Vec<scalar_t, width> temp = input_v[id];
//         temp += residual_v[id];
//         variance += temp.sum_squares();
//         residual_v[id] = temp;
//     }

//     using BlockReduce = cub::BlockReduce<float, 1024>;
//     __shared__ typename BlockReduce::TempStorage reduceStore;
//     variance =
//         BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

//     if (threadIdx.x == 0) {
//         s_variance = rsqrtf(variance / hidden_size + epsilon);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
//         int id = blockIdx.x * vec_hidden_size + idx;
//         _f16Vec<scalar_t, width> temp = residual_v[id];
//         temp *= s_variance;
//         temp *= weight_v[idx];
//         input_v[id] = temp;
//     }
// }

// /* Generic fused_add_rms_norm_kernel
//    The width field is not used here but necessary for other specializations.
//  */
// template <typename scalar_t, int width>
// __global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
// fused_add_rms_norm_kernel(scalar_t *__restrict__ input,    // [...,
// hidden_size]
//                           scalar_t *__restrict__ residual, // [...,
//                           hidden_size] const scalar_t *__restrict__ weight,
//                           // [hidden_size] const float epsilon, const int
//                           num_tokens, const int hidden_size) {
//     __shared__ float s_variance;
//     float variance = 0.0f;

//     for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
//         scalar_t z = input[blockIdx.x * hidden_size + idx];
//         z += residual[blockIdx.x * hidden_size + idx];
//         float x = (float)z;
//         variance += x * x;
//         residual[blockIdx.x * hidden_size + idx] = z;
//     }

//     using BlockReduce = cub::BlockReduce<float, 1024>;
//     __shared__ typename BlockReduce::TempStorage reduceStore;
//     variance =
//         BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

//     if (threadIdx.x == 0) {
//         s_variance = rsqrtf(variance / hidden_size + epsilon);
//     }
//     __syncthreads();

//     for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
//         float x = (float)residual[blockIdx.x * hidden_size + idx];
//         input[blockIdx.x * hidden_size + idx] =
//             ((scalar_t)(x * s_variance)) * weight[idx];
//     }
// }

} // namespace hpco::cuda
