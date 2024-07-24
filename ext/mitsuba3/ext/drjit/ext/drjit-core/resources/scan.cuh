/*
    kernels/mkperm.cuh -- CUDA, exclusive scan for 32 bit unsigned integers

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

KERNEL void scan_small_u32(const uint32_t *in, uint32_t *out, uint32_t size) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t values[4];
    *(uint4 *) values = ((const uint4 *) in)[threadIdx.x];

    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t v = values[i];
        values[i] = sum_local;
        sum_local += v;
    }

    // Reduce using shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += blockDim.x;
    shared[si] = sum_local;

    uint32_t sum_block = sum_local;
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
        shared[si] = sum_block;
    }

    sum_block -= sum_local;
    for (int i = 0; i < 4; ++i)
        values[i] += sum_block;

    ((uint4 *) out)[threadIdx.x] = *(const uint4 *) values;
}

__device__ __forceinline__ void store_cg(uint64_t *ptr, uint64_t val) {
    asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val));
}

__device__ __forceinline__ uint64_t load_cg(uint64_t *ptr) {
    uint64_t retval;
    asm volatile("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr));
    return retval;
}

KERNEL void scan_large_u32_init(uint64_t *out, uint32_t size) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        out[i] = (i < 32) ? 2 : 0;
}

KERNEL void scan_large_u32(const uint32_t *in, uint32_t *out, uint64_t *scratch) {
    uint32_t *shared = SharedMemory<uint32_t>::get();
    uint32_t thread_count = 128;

    /* Transpose inputs in shared memory using blocks of 16 values */ {
        uint4 v[4];
        for (int i = 0; i < 4; ++i)
            v[i] = ((const uint4 *) in)[(blockIdx.x * 4 + i) * thread_count + threadIdx.x];

        for (int i = 0; i < 4; ++i)
            ((uint4 *) shared)[i * thread_count + threadIdx.x] = v[i];
    }

    __syncthreads();

    // Fetch input from shared memory
    uint32_t values[16];
    for (int i = 0; i < 4; ++i)
        ((uint4 *) values)[i] = ((const uint4 *) shared)[threadIdx.x * 4 + i];

    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t v = values[i];
        values[i] = sum_local;
        sum_local += v;
    }

    __syncthreads();

    // Block-level reduction of partial sum over 16 elements via shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += thread_count;
    shared[si] = sum_local;

    uint32_t sum_block = sum_local;
    for (uint32_t offset = 1; offset < thread_count; offset <<= 1) {
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
        shared[si] = sum_block;
    }

    // Store block-level partial inclusive scan value in global memory
    scratch += blockIdx.x;
    if (threadIdx.x == thread_count - 1)
        store_cg(scratch, (((uint64_t) sum_block) << 32) | 1ull);

    uint32_t lane = threadIdx.x & (warpSize - 1);
    uint32_t prefix = 0;
    int32_t shift = lane - warpSize;

    /* Compute prefix due to previous blocks using warp-level primitives.
       Based on "Single-pass Parallel Prefix Scan with Decoupled Look-back"
       by Duane Merrill and Michael Garland */
    while (true) {
        /// Prevent loop invariant code motion of loads
        uint64_t temp = load_cg(scratch + shift);
        uint32_t flag = (uint32_t) temp;

        if (__any_sync(0xFFFFFFFF, flag == 0))
            continue;

        uint32_t mask  = __ballot_sync(0xFFFFFFFF, flag == 2),
                 value = (uint32_t) (temp >> 32);
        if (mask == 0) {
            prefix += value;
            shift -= warpSize;
        } else {
            uint32_t index = 31 - __clz(mask);
            if (lane >= index)
                prefix += value;
            break;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        prefix += __shfl_down_sync(0xFFFFFFFF, prefix, offset, 32);
    sum_block += __shfl_sync(0xFFFFFFFF, prefix, 0);

    // Store block-level complete inclusive scan value in global memory
    if (threadIdx.x == thread_count - 1)
        store_cg(scratch, (((uint64_t) sum_block) << 32) | 2ull);

    sum_block -= sum_local;
    for (int i = 0; i < 16; ++i)
        values[i] += sum_block;

    // Store input into shared memory
    for (int i = 0; i < 4; ++i)
        ((uint4 *) shared)[threadIdx.x*4 + i] = ((const uint4 *) values)[i];

    __syncthreads();

    /* Transpose inputs in shared memory using blocks of 16 values */ {
        for (int i = 0; i < 4; ++i) {
            uint4 v = ((const uint4 *) shared)[i * thread_count + threadIdx.x];
            ((uint4 *) out)[(blockIdx.x * 4 + i) * thread_count + threadIdx.x] = v;
        }
    }
}
