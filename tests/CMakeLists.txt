include_directories(${CMAKE_SOURCE_DIR}/device/include)
include_directories(${CMAKE_SOURCE_DIR}/host/include)
include_directories(${CMAKE_SOURCE_DIR}/tools/include ${CMAKE_SOURCE_DIR}/csrc/cuda/unary)

file(GLOB TEST_SRCS "src/*.cc")
enable_testing()

include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_build_tests OFF CACHE BOOL "" FORCE)
set(gtest_build_samples OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4) # need master for benchmark::benchmark

FetchContent_MakeAvailable(googletest googlebenchmark)
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
foreach(filename ${TEST_SRCS})
  # message("Filename = ${filename}")
  get_filename_component(TEST_NAME ${filename} NAME_WE)
  set(TARGET_NAME benchmark_${TEST_NAME})
  add_executable(${TARGET_NAME} ${filename})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${TEST_NAME})
  target_link_libraries(
    ${TARGET_NAME}
    benchmark::benchmark 
    gtest_main
    hostop
    ops
  )
  add_test(${TARGET_NAME} ${TEST_NAME})
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
