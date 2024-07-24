#pragma once

#include <stdint.h>
#include <type_traits>
#include <limits>

#define KERNEL extern "C" __global__

template <typename T> struct SharedMemory {
    __device__ inline static T *get() {
        extern __shared__ int shared[];
        return (T *) shared;
    }
};

template <> struct SharedMemory<double> {
    __device__ inline static double *get() {
        extern __shared__ double shared_d[];
        return shared_d;
    }
};
