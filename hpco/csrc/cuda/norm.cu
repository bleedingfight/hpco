#include "hpco/csrc/op_kernels.h"
#include <cooperative_groups.h>
namespace hpco::cuda {
// #define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

// template <typename float_t, size_t vec_size> struct vec_t {
//     FLASHINFER_INLINE float_t &operator[](size_t i);
//     FLASHINFER_INLINE const float_t &operator[](size_t i) const;
//     FLASHINFER_INLINE void fill(float_t val);
//     FLASHINFER_INLINE void load(const float_t *ptr);
//     FLASHINFER_INLINE void store(float_t *ptr) const;
//     FLASHINFER_INLINE void load_global_acquire(float *addr);
//     FLASHINFER_INLINE void store_global_release(float *addr) const;
//     FLASHINFER_INLINE void load_global_volatile(float *addr);
//     FLASHINFER_INLINE void store_global_volatile(float *addr) const;
//     template <typename T>
//     FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src);
//     template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr);
//     template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const;
//     FLASHINFER_INLINE static void memcpy(float_t *dst, const float_t *src);
//     FLASHINFER_INLINE float_t *ptr();
// };

// template <uint32_t VEC_SIZE, typename T>
// __global__ void RMSNormKernel(T *__restrict__ input, T *__restrict__ weight,
//                               T *__restrict__ output, const uint32_t d,
//                               const uint32_t stride_input,
//                               const uint32_t stride_output, float
//                               weight_bias, float eps) {
//     const uint32_t bx = blockIdx.x;
//     const uint32_t tx = threadIdx.x, ty = threadIdx.y;
//     constexpr uint32_t warp_size = 32;
//     const uint32_t num_warps = blockDim.y;
//     // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
//     const uint32_t thread_id = tx + ty * warp_size;
//     const uint32_t num_threads = num_warps * warp_size;
//     const uint32_t rounds =
//         (d + VEC_SIZE * num_threads - 1) / (VEC_SIZE * num_threads);
//     extern __shared__ float smem[];

//     float sum_sq = 0.f;

// #if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) &&                   \
//      (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.wait;");
// #endif

//     for (uint32_t i = 0; i < rounds; i++) {
//         vec_t<T, VEC_SIZE> input_vec;
//         input_vec.fill(0.f);
//         if ((i * num_threads + thread_id) * VEC_SIZE < d) {
//             input_vec.load(input + bx * stride_input +
//                            i * num_threads * VEC_SIZE + thread_id *
//                            VEC_SIZE);
//         }
// #pragma unroll
//         for (uint32_t j = 0; j < VEC_SIZE; j++) {
//             sum_sq += float(input_vec[j]) * float(input_vec[j]);
//         }
//     }

//     // first, warp reduce sum
// #pragma unroll
//     for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
//         sum_sq += shfl_xor_sync(sum_sq, offset);
//     }

//     smem[ty] = sum_sq;
//     __syncthreads();
//     // then, cross warp reduce sum using only the first warp
//     if (ty == 0) {
//         sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
// #pragma unroll
//         for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
//             sum_sq += shfl_xor_sync(sum_sq, offset);
//         }
//         smem[0] = sum_sq;
//     }
//     __syncthreads();

//     float rms_rcp = __rsqrt(smem[0] / float(d) + eps);

//     for (uint32_t i = 0; i < rounds; i++) {
//         vec_t<T, VEC_SIZE> input_vec;
//         vec_t<T, VEC_SIZE> weight_vec;
//         vec_t<T, VEC_SIZE> output_vec;
//         input_vec.fill(0.f);
//         weight_vec.fill(0.f);
//         if ((i * num_threads + thread_id) * VEC_SIZE < d) {
//             input_vec.load(input + bx * stride_input +
//                            i * num_threads * VEC_SIZE + thread_id *
//                            VEC_SIZE);
//             weight_vec.load(weight + i * num_threads * VEC_SIZE +
//                             thread_id * VEC_SIZE);
//         }
// #pragma unroll
//         for (uint32_t j = 0; j < VEC_SIZE; j++) {
//             output_vec[j] = float(input_vec[j]) * rms_rcp *
//                             (weight_bias + float(weight_vec[j]));
//         }
//         if ((i * num_threads + thread_id) * VEC_SIZE < d) {
//             output_vec.store(output + bx * stride_output +
//                              i * num_threads * VEC_SIZE + thread_id *
//                              VEC_SIZE);
//         }
//     }
// #if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
//      (__CUDA_ARCH__ >= 900))
//     asm volatile("griddepcontrol.launch_dependents;");
// #endif
// }

template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(T *__restrict__ input, T *__restrict__ weight,
                              T *__restrict__ output, const uint32_t d,
                              const uint32_t stride_input,
                              const uint32_t stride_output, float weight_bias,
                              float eps) {
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
    const uint32_t thread_id = tx + ty * warp_size;
    // num_warps列个线程束，一个block中总共的线程
    const uint32_t num_threads = num_warps * warp_size;
    // 每个线程负责处理VEC_SIZE个数据
    const uint32_t rounds =
        (d + VEC_SIZE * num_threads - 1) / (VEC_SIZE * num_threads);
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
            // bx+stride_input找到对应的行
            // i*num_threads*VEC_SIZE 找到对应的block负责的段
            // thread_id*VEC_SIZE 找到对应的线程负责的向量
            input_vec.load(input + bx * stride_input +
                           i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        }
        // 对向量中元素求平方和，vec类似tile
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            sum_sq += float(input_vec[j]) * float(input_vec[j]);
        }
    }

    // 每个warp计算除了自己负责的32个向量的平方,对这些平方求和
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += shfl_xor_sync(sum_sq, offset);
    }

    // 每个warp
    // reduce求和之后的结果存放到共享内存,这样整个block负责的数据计算完成
    smem[ty] = sum_sq;
    __syncthreads();
    // then, cross warp reduce sum using only the first warp
    // 第一个warp执行跨线程的reduce将结果更新到第一个位置
    if (ty == 0) {
        sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
            sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        smem[0] = sum_sq;
    }
    __syncthreads();

    // 计算缩放系数
    float rms_rcp = __rsqrt(smem[0] / float(d) + eps);

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
cudaError_t RMSNorm(T *input, T *weight, T *output, uint32_t batch_size,
                    uint32_t d, uint32_t stride_input, uint32_t stride_output,
                    float eps = 1e-5, bool enable_pdl = false,
                    cudaStream_t stream = 0) {
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
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        auto kernel = RMSNormKernel<VEC_SIZE, T>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernelEx(&config, kernel, input, weight, output, d,
                               stride_input, stride_output, weight_bias, eps));
    });
    return cudaSuccess;
}

} // namespace hpco::cuda
