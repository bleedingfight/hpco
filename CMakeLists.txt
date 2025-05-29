cmake_minimum_required(VERSION 3.18)
project(hpc LANGUAGES CXX CUDA)
include(FetchContent)

add_subdirectory(host)
include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
option(CUDA_ENABLED "CUDA_ENABLED" ON)
if(CUDA_ENABLED)
  enable_language(CUDA)
  add_subdirectory(device)
endif()

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG        main # 建议使用稳定版本标签，或者 GIT_BRANCH main
    #GIT_SUBMODULES OFF # 如果nvbench有submodules但你不想下载，可以设为OFF
)

# 获取nvbench并将其添加到构建中
FetchContent_MakeAvailable(nvbench)

# 检查nvbench是否被成功添加并提供了目标
if (TARGET nvbench::nvbench)
    message(STATUS "nvbench found and ready to use.")
else()
    message(FATAL_ERROR "nvbench target not found after FetchContent_MakeAvailable.")
endif()

option(BUILD_EIGEN "build with eigen" OFF)
# 开启cutlass支持
option(USE_CUTLASS "USE cutlass" OFF)
if(USE_CUTLASS)
  if(DEFINED ENV{CUTLASS_HOME})
    message(STATUS "Using CUTLASS_HOME environment variable: $ENV{CUTLASS_HOME}")
    set(CUTLASS_ROOT_DIR $ENV{CUTLASS_HOME} CACHE STRING "设置全局CUTLASS")
  else()
    # Define a default path if CUTLASS_HOME is not set
    set(DEFAULT_CUTLASS_PATH "/usr/local/cutlass/") # Change this to your desired default path
    message(STATUS "CUTLASS_HOME not set, using default path: ${DEFAULT_CUTLASS_PATH}")
    # set(CUTLASS_ROOT_DIR ${DEFAULT_CUTLASS_PATH})
    set(CUTLASS_ROOT_DIR ${DEFAULT_CUTLASS_PATH} CACHE STRING "设置全局CUTLASS")
  endif()
  include_directories(${CUTLASS_ROOT_DIR}/include)
endif()

# 是否编译工具
option(BUILD_TOOLS "Build the Tools" OFF)
if(BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# 是否开启测试
option(BUILD_TESTS "Build the tests" ON)
if(BUILD_TESTS)
  message(STATUS "CUDA Test: ${BUILD_TESTS}")
  enable_testing()
  add_subdirectory(tests)
endif()

option(BUILD_BENCHES "Build the tests" ON)
if(BUILD_BENCHES)
  message(STATUS "CUDA NVBench: ${BUILD_BENCHES}")
  enable_testing()
  add_subdirectory(benches)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
    message(STATUS "Debug CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
endif()
# 根据选项选择是否包含device的库
if(CUDA_ENABLED)
    set(link_libs "deviceop hostop")
else()
    set(link_libs "hostop")
endif()

# if(OpenMP_CXX_FOUND)
#     message(STATUS "OpenMP enabled: ${OpenMP_CXX_FOUND}")
#     add_compile_options(-fopenmp)
#     target_link_libraries(hostop PUBLIC OpenMP::OpenMP_CXX)
# endif()
message(STATUS "CUDA_ENABLED: ${CUDA_ENABLED}")
include_directories(SYSTEM AFTER ${CMAKE_SOURCE_DIR}/host/include
                    ${CMAKE_SOURCE_DIR}/device/include ${EIGEN_INCLUDE_DIR})
add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE hostop deviceop)
