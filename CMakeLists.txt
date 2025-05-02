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

# # 添加spdlog
# FetchContent_Declare(
#   spdlog
#   GIT_REPOSITORY https://github.com/gabime/spdlog.git
#   GIT_TAG v2.x)
# add_compile_options(-fPIC)
# FetchContent_MakeAvailable(spdlog)

# # cache spdlog to share with different directory
# set(SPDLOG_SOURCE_INCLUDE
#     "${spdlog_SOURCE_DIR}/include"
#     CACHE STRING "spdlog head directory ")
# set(SPDLOG_LIBRARY_DIR
#     "${spdlog_BINARY_DIR}"
#     CACHE STRING "spdlog library directory ")

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
target_link_libraries(${PROJECT_NAME} hostop deviceop)
