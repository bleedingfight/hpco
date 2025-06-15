file(GLOB_RECURSE SRCS *.cu)
include_directories(SYSTEM ${CUTLASS_ROOT_DIR}/include include)
add_library(ops SHARED ${SRCS})
set_target_properties(ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
