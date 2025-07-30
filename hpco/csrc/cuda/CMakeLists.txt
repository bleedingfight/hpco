file(GLOB_RECURSE SRCS *.cu)
add_library(hpco_cuda SHARED ${SRCS})
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories(SYSTEM ${CUTLASS_ROOT_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/include)
target_compile_options(hpco_cuda PRIVATE
    # $<IF:...> 是一个生成器表达式，它在生成构建系统时进行评估
    # 如果 CMAKE_BUILD_TYPE 是 "Debug"，则添加 -g
    # 否则不添加任何内容（空字符串）
    $<$<CONFIG:Debug>:-g> 
    $<$<CONFIG:Debug>:-G> 
)
set_target_properties(hpco_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
