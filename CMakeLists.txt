cmake_minimum_required(VERSION 3.20)
project(hpc LANGUAGES CXX CUDA)
include(FetchContent)

include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
option(CUDA_ENABLED "CUDA_ENABLED" ON)
if(CUDA_ENABLED)
  enable_language(CUDA)
  add_subdirectory(hpco)
endif()

include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

option(BUILD_EIGEN "build with eigen" OFF)
# 开启cutlass支持
option(USE_CUTLASS "USE cutlass" ON)
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
option(BUILD_TOOLS "Build the Tools" ON)
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
    set(link_libs "ops hostop")
else()
    set(link_libs "hostop")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
message(STATUS "CUDA_ENABLED: ${CUDA_ENABLED}")
include_directories(SYSTEM AFTER ${CMAKE_SOURCE_DIR}/host/include
                    ${CMAKE_SOURCE_DIR}/device/include ${EIGEN_INCLUDE_DIR})
add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE hostop ops)
