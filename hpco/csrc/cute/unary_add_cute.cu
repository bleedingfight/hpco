#include <cmath> // For std::abs
#include <cuda_runtime.h>
#include <iostream>
#include <numeric> // For std::iota
#include <vector>

// 包含 Cute 库
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

// CUDA 错误检查宏
#define CUDA_CHECK_ERROR(ans)                                                  \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}
namespace hpco::cutlass {
// -----------------------------------------------------------
// 设备端核函数
// -----------------------------------------------------------
template <typename T>
__global__ void matrix_add_cute_kernel(T *d_A, T *d_B, T *d_C, int M, int N) {
    // 1. 定义全局内存中矩阵的布局 (Layout)
    // 假设是 M 行 N 列的行主序矩阵
    // cute::Shape 的模板参数现在是类型 (int)，表示维度是整数类型
    // 在构造函数中传入实际的运行时值 M 和 N
    using MatrixShape = cute::Shape<int, int>;
    using MatrixStride = cute::Stride<int, int>; // 行步长N，列步长1

    cute::Layout<MatrixShape, MatrixStride> matrix_layout(
        MatrixShape(M, N), // 形状：M行N列，这里M和N是传入的int类型变量
        MatrixStride(N, 1) // 步长：行步长N，列步长1 (行主序)
    );

    // 2. 创建 TensorView (张量视图)
    // TensorView 将原始指针与布局关联起来，提供维度感知的数据访问
    cute::Tensor input_A =
        cute::make_tensor(cute::make_gmem_ptr(d_A), matrix_layout);
    cute::Tensor input_B =
        cute::make_tensor(cute::make_gmem_ptr(d_B), matrix_layout);
    cute::Tensor output_C =
        cute::make_tensor(cute::make_gmem_ptr(d_C), matrix_layout);

    // 3. 计算当前线程的全局索引
    // 每个线程处理一个元素
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 4. 遍历并执行元素级加法 (Grid-striding 模式)
    // 使用简单的 for 循环，让每个线程处理其负责的元素子集
    // 确保处理所有 N*M 个元素
    size_t total_elements = (size_t)M * N;
    size_t total_threads = (size_t)gridDim.x * blockDim.x;

    for (size_t i = global_tid; i < total_elements; i += total_threads) {
        // Cute 允许通过线性索引访问元素，它会根据 Layout 自动计算多维坐标
        // 例如，如果 Layout 是行主序，i 会被映射到 (row, col)
        float val_A = input_A(i);
        float val_B = input_B(i);
        float res_C = val_A + val_B;
        output_C(i) = res_C;
    }
}
} // namespace hpco::cutlass

// -----------------------------------------------------------
// 主机端代码
// -----------------------------------------------------------
int main() {
    const int M = 1024; // 矩阵行数
    const int N = 2048; // 矩阵列数
    const size_t TOTAL_ELEMENTS = static_cast<size_t>(M * N);

    // 根据 GPU 资源和性能调整 Block 和 Grid 大小
    // 通常 BLOCK_SIZE 设为 256 或 512
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (TOTAL_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. 分配主机内存并初始化数据
    std::vector<float> h_A(TOTAL_ELEMENTS);
    std::vector<float> h_B(TOTAL_ELEMENTS);
    std::vector<float> h_C(TOTAL_ELEMENTS);     // 用于存储 GPU 结果
    std::vector<float> h_C_ref(TOTAL_ELEMENTS); // 用于存储 CPU 参考结果

    // 初始化 A 和 B 矩阵，便于验证
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        h_A[i] = static_cast<float>(i + 1);
        h_B[i] = static_cast<float>(i + 0.5);
    }

    // 2. CPU 参考计算 (用于验证)
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK_ERROR(cudaMalloc(&d_A, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_B, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_C, TOTAL_ELEMENTS * sizeof(float)));

    // 4. 将数据从主机拷贝到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A.data(), TOTAL_ELEMENTS * sizeof(float),
                                cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B.data(), TOTAL_ELEMENTS * sizeof(float),
                                cudaMemcpyHostToDevice));

    // 5. 启动核函数
    std::cout << "Launching kernel for matrix addition:" << std::endl;
    std::cout << "  Matrix dimensions: " << M << " x " << N << std::endl;
    std::cout << "  Total elements: " << TOTAL_ELEMENTS << std::endl;
    std::cout << "  Block size: " << BLOCK_SIZE << std::endl;
    std::cout << "  Grid size: " << GRID_SIZE << std::endl;

    hpco::cutlass::matrix_add_cute_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B,
                                                                     d_C, M, N);
    CUDA_CHECK_ERROR(cudaGetLastError()); // 检查核函数启动错误

    // 6. 将结果从设备拷贝回主机
    CUDA_CHECK_ERROR(cudaMemcpy(h_C.data(), d_C, TOTAL_ELEMENTS * sizeof(float),
                                cudaMemcpyDeviceToHost));

    // 7. 验证结果
    bool success = true;
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        if (std::abs(h_C[i] - h_C_ref[i]) > 1e-6) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_C[i]
                      << ", CPU=" << h_C_ref[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cerr << "Verification FAILED!" << std::endl;
    }

    // 8. 释放设备内存
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    return 0;
}
