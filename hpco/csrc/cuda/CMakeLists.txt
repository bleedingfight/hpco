file(GLOB_RECURSE SRCS *.cu)
add_library(ops SHARED ${SRCS})
include_directories(SYSTEM ${CUTLASS_ROOT_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ops PRIVATE
    $<$<CONFIG:Debug>:--device-debug -G>
    $<$<CONFIG:Release>:-O3>
)
set_target_properties(ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
