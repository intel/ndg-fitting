/*
    src/util.h -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include "cuda_api.h"

/// Descriptive names for the various reduction operations
extern const char *reduction_name[(int) ReduceOp::Count];

/// Fill a device memory region with constants of a given type
extern void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size, uint32_t isize,
                              const void *src);

/// Reduce the given array to a single value
extern void jitc_reduce(JitBackend backend, VarType type, ReduceOp rtype,
                        const void *ptr, uint32_t size, void *out);

/// 'All' reduction for boolean arrays
extern bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size);

/// 'Any' reduction for boolean arrays
extern bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size);

/// Exclusive prefix sum
extern void jitc_scan_u32(JitBackend backend, const uint32_t *in, uint32_t size,
                          uint32_t *out);

/// Mask compression
extern uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size,
                              uint32_t *out);

/// Compute a permutation to reorder an integer array into discrete groups
extern uint32_t jitc_mkperm(JitBackend backend, const uint32_t *values, uint32_t size,
                            uint32_t bucket_count, uint32_t *perm,
                            uint32_t *offsets);

/// Perform a synchronous copy operation
extern void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size);

/// Perform an assynchronous copy operation
extern void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size);

/// Replicate individual input elements to larger blocks
extern void jitc_block_copy(JitBackend backend, enum VarType type, const void *in,
                            void *out, uint32_t size, uint32_t block_size);

/// Sum over elements within blocks
extern void jitc_block_sum(JitBackend backend, enum VarType type, const void *in,
                           void *out, uint32_t size, uint32_t block_size);

/// Asynchronously update a single element in memory
extern void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size);

struct VCallDataRecord;
/// Initialize the data block consumed by a vcall
extern void jitc_vcall_prepare(JitBackend backend, void *dst,
                               VCallDataRecord *rec, uint32_t size);
