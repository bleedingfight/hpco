file(GLOB BENCHMARK_CU_FILES "*.cu")
include_directories(${CMAKE_SOURCE_DIR}/device/include)
foreach(BENCH_FILE ${BENCHMARK_CU_FILES})
    get_filename_component(BENCH_NAME "${BENCH_FILE}" NAME_WE)

    # 定义可执行文件的名称，例如：bench_add_benchmark
    set(EXECUTABLE_NAME "bench_${BENCH_NAME}")
    message(STATUS "Creating benchmark executable: ${EXECUTABLE_NAME} from ${BENCH_FILE}")
    # 添加基准测试可执行文件
    add_executable(${EXECUTABLE_NAME} ${BENCH_FILE})
    # 链接 nvbench 库
    target_link_libraries(${EXECUTABLE_NAME}
        PRIVATE
            nvbench::nvbench deviceop cudart
    )

    # 确保基准测试可执行文件能够找到 device_kernels 的头文件
    target_include_directories(${EXECUTABLE_NAME}
        PRIVATE
            $<TARGET_PROPERTY:deviceop,INTERFACE_INCLUDE_DIRECTORIES>
    )

    # ------------------------------------------------------------------
    # 设置 CUDA 编译属性
    # ------------------------------------------------------------------

    # 确保基准测试代码使用正确的 CUDA 架构进行编译
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    # (可选) 设置输出目录，将可执行文件放到统一的 bin 目录下
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
