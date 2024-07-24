/*
    kernels/misc.cuh -- Miscellaneous CUDA kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

KERNEL void poke_u8(uint8_t *out, uint8_t value) {
    *out = value;
}

KERNEL void poke_u16(uint16_t *out, uint16_t value) {
    *out = value;
}

KERNEL void poke_u32(uint32_t *out, uint32_t value) {
    *out = value;
}

KERNEL void poke_u64(uint64_t *out, uint64_t value) {
    *out = value;
}

KERNEL void fill_64(uint64_t *out, uint32_t size, uint64_t value) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}

struct VCallDataRecord {
    uint32_t offset;
    uint8_t size;
    bool literal;
    uint16_t unused;
    uintptr_t value;
};
static_assert(sizeof(VCallDataRecord) == 16, "VCallDataRecord has unexpected size!");

KERNEL void vcall_prepare(void *out, const VCallDataRecord *rec_, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    VCallDataRecord rec = rec_[idx];

    bool literal = rec.literal;
    uintptr_t value = rec.value;
    void *dst = (uint8_t *) out + rec.offset;

    switch (rec.size) {
        case 1: *(uint8_t *)  dst = literal ? (uint8_t)  value : *(uint8_t *)  value; break;
        case 2: *(uint16_t *) dst = literal ? (uint16_t) value : *(uint16_t *) value; break;
        case 4: *(uint32_t *) dst = literal ? (uint32_t) value : *(uint32_t *) value; break;
        case 8: *(uint64_t *) dst = literal ? (uint64_t) value : *(uint64_t *) value; break;
    }
}

#if __CUDA_ARCH__ <= 600
__device__ double atomicAdd(double *ptr_, double value) {
    unsigned long long int *ptr = (unsigned long long int *) ptr_;
    unsigned long long int old = *ptr, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            ptr, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#define BLOCK_KERNEL(Suffix, Type)                                             \
    KERNEL void block_copy_##Suffix(const Type *in, Type *out, uint32_t size,  \
                                    uint32_t block_size) {                     \
        uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;              \
        if (out_idx >= size)                                                   \
            return;                                                            \
        uint32_t in_idx = out_idx / block_size;                                \
        out[out_idx] = in[in_idx];                                             \
    }                                                                          \
                                                                               \
    KERNEL void block_sum_##Suffix(const Type *in, Type *out, uint32_t size,   \
                                   uint32_t block_size) {                      \
        uint32_t in_idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (in_idx >= size)                                                    \
            return;                                                            \
        uint32_t out_idx = in_idx / block_size;                                \
        atomicAdd(out + out_idx, in[in_idx]);                                  \
    }

BLOCK_KERNEL(u32, unsigned);
BLOCK_KERNEL(u64, unsigned long long);
BLOCK_KERNEL(f32, float);
BLOCK_KERNEL(f64, double);
