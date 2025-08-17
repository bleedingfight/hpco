#pragma once
#include "hpco/csrc/cuda/math.cuh"
#include "hpco/csrc/cuda/utils.cuh"
#include "hpco/csrc/cuda/vec_dtypes.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector_types.h>
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(error));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
