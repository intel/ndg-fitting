/*
    kernels/mkperm.cuh -- Efficient CUDA kernels for sorting arrays
    with low-valued entries

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

// Determine bit mask of lanes with a matching value
__device__ __inline__ uint32_t get_peers(uint32_t active, uint32_t value) {
#if __CUDA_ARCH__ >= 700
    return __match_any_sync(active, value);
#else
    /* Emulate __match_any_sync intrinsics. Based on "Voting And
       Shuffling For Fewer Atomic Operations" by Elmar Westphal. */
    do {
        // Find lowest-numbered active lane
        int first_active = __ffs(active) - 1;

        // Fetch its value and compare to ours
        bool match = (value == __shfl_sync(active, value, first_active));

        // Determine, which lanes had a match
        uint32_t peers = __ballot_sync(active, match);

        // Key of the current lane was chosen, return the active mask
        if (match)
            return peers;

        // Remove lanes with matching values from the pool
        active ^= peers;
    } while (true);
#endif
}

/// Accumulate 'value' into histogram 'buckets', using a minimal number of memory operations
inline __device__ uint32_t reduce(uint32_t active, uint32_t value, uint32_t *buckets) {
    uint32_t peers = get_peers(active, value);

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (warpSize - 1);

    // Designate a leader thread within the set of peers
    uint32_t leader_idx  = __ffs(peers) - 1;

    // If the current thread is the leader, perform atomic op.
    uint32_t offset = 0;
    if (lane_idx == leader_idx) {
        offset = buckets[value];
        buckets[value] = offset + __popc(peers);
    }

    // Fetch offset into output array from leader
    offset = __shfl_sync(peers, offset, leader_idx);

    // Determine current thread's position within peer group
    uint32_t rel_pos = __popc(peers << (32 - lane_idx));

    return offset + rel_pos;
}

/// Atomically accumulate 'value' into histogram 'buckets', using a minimal number of atomic operations
inline __device__ uint32_t reduce_atomic(uint32_t active, uint32_t value, uint32_t *buckets) {
    uint32_t peers = get_peers(active, value);

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (warpSize - 1);

    // Designate a leader thread within the set of peers
    uint32_t leader_idx  = __ffs(peers) - 1;

    // If the current thread is the leader, perform atomic op.
    uint32_t offset = 0;
    if (lane_idx == leader_idx)
        offset = atomicAdd(buckets + value, __popc(peers));

    // Fetch offset into output array from leader
    offset = __shfl_sync(peers, offset, leader_idx);

    // Determine current thread's position within peer group
    uint32_t rel_pos = __popc(peers << (32 - lane_idx));

    return offset + rel_pos;
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Tiny" variant, which uses shared memory atomics to produce a stable
 * permutation. Handles up to 512 buckets with 64KiB of shared memory. Should be
 * combined with \ref mkperm_phase_4_tiny.
 */
KERNEL void mkperm_phase_1_tiny(const uint32_t *values,
                                uint32_t *buckets,
                                uint32_t size,
                                uint32_t size_per_block,
                                uint32_t bucket_count) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block,
             warp_count   = thread_count / warpSize,
             warp_id      = thread_id / warpSize;

    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        shared[i] = 0;

    __syncthreads();

    uint32_t *shared_warp = shared + warp_id * bucket_count;

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active)
            reduce(active_mask, values[i], shared_warp);

        __syncwarp();
    }

    __syncthreads();

    uint32_t *out = buckets + blockIdx.x * bucket_count * warp_count;
    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        out[i] = shared[i];
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Small" variant, which uses shared memory atomics and handles up to 16K
 * buckets with 64KiB of shared memory. The permutation can be somewhat
 * unstable due to scheduling variations when performing atomic operations
 * (although some effort is made to keep it stable within each group of 32
 * elements by performing an intra-warp reduction.) Should be combined with
 * \ref mkperm_phase_4_small.
 */
KERNEL void mkperm_phase_1_small(const uint32_t *values,
                                 uint32_t *buckets,
                                 uint32_t size,
                                 uint32_t size_per_block,
                                 uint32_t bucket_count) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block;

    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        shared[i] = 0;

    __syncthreads();

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active)
            reduce_atomic(active_mask, values[i], shared);
    }

    __syncthreads();

    uint32_t *out = buckets + blockIdx.x * bucket_count;
    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        out[i] = shared[i];
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Large" variant, which uses global memory atomics and handles arbitrarily
 * many elements (though this is somewhat slower than the previous two shared
 * memory variants). The permutation can be somewhat unstable due to scheduling
 * variations when performing atomic operations (although some effort is made
 * to keep it stable within each group of 32 elements by performing an
 * intra-warp reduction.) Should be combined with \ref mkperm_phase_4_large.
 */
KERNEL void mkperm_phase_1_large(const uint32_t *values,
                                 uint32_t *buckets_, uint32_t size,
                                 uint32_t size_per_block, uint32_t bucket_count) {
    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block;

    uint32_t *buckets = buckets_ + blockIdx.x * bucket_count;

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active)
            reduce_atomic(active_mask, values[i], buckets);
    }
}

/// Detect non-empty buckets and record their offsets
KERNEL void mkperm_phase_3(uint32_t *buckets,
                           uint32_t bucket_count,
                           uint32_t bucket_count_rounded,
                           uint32_t perm_size,
                           uint32_t *counter,
                           uint4 *offsets) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (warpSize - 1);

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < bucket_count_rounded; i += blockDim.x * gridDim.x) {

        uint32_t offset_a, offset_b;

        offset_a = (i < bucket_count) ? buckets[i] : perm_size;
        shared[threadIdx.x] = offset_a;
        __syncthreads();

        if (threadIdx.x + 1 < blockDim.x)
            offset_b = shared[threadIdx.x + 1];
        else
            offset_b = (i + 1 < bucket_count) ? buckets[i + 1] : perm_size;

        // Did we find a non-empty bucket?
        bool found = offset_a != offset_b;

        // Peers within the same warp that also found one
        uint32_t peers = __ballot_sync(0xFFFFFFFF, found);

        if (found) {
            // Designate a leader thread within the set of peers
            uint32_t leader_idx  = __ffs(peers) - 1;

            // If the current thread is the leader, perform atomic op.
            uint32_t offset = 0;
            if (lane_idx == leader_idx)
                offset = atomicAdd(counter, __popc(peers));

            // Fetch offset into output array from leader
            offset = __shfl_sync(peers, offset, leader_idx);

            // Determine current thread's position within peer group
            offset += __popc(peers << (32 - lane_idx));

            offsets[offset] = make_uint4(i, offset_a, offset_b - offset_a, 0);
        }
    }
}

/// Generate a sorting permutation based on offsets generated by \ref mkperm_phase_1_tiny()
KERNEL void mkperm_phase_4_tiny(const uint32_t *values,
                                const uint32_t *buckets_,
                                uint32_t *perm,
                                uint32_t size,
                                uint32_t size_per_block,
                                uint32_t bucket_count) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block,
             warp_count   = thread_count / warpSize,
             warp_id      = thread_id / warpSize;

    const uint32_t *buckets = buckets_ + blockIdx.x * bucket_count * warp_count;
    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        shared[i] = buckets[i];

    __syncthreads();

    uint32_t *shared_warp = shared + warp_id * bucket_count;

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active) {
            uint32_t offset = reduce(active_mask, values[i], shared_warp);
            perm[offset] = i;
        }
    }
}

/// Generate a sorting permutation based on offsets generated by \ref mkperm_phase_1_small()
KERNEL void mkperm_phase_4_small(const uint32_t *values,
                                 const uint32_t *buckets_,
                                 uint32_t *perm,
                                 uint32_t size,
                                 uint32_t size_per_block,
                                 uint32_t bucket_count) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block;

    const uint32_t *buckets = buckets_ + blockIdx.x * bucket_count;
    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        shared[i] = buckets[i];

    __syncthreads();

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active) {
            uint32_t offset = reduce_atomic(active_mask, values[i], shared);
            perm[offset] = i;
        }
    }
}

/// Generate a sorting permutation based on offsets generated by \ref mkperm_phase_1_large()
KERNEL void mkperm_phase_4_large(const uint32_t *values,
                                 uint32_t *buckets_,
                                 uint32_t *perm,
                                 uint32_t size,
                                 uint32_t size_per_block,
                                 uint32_t bucket_count) {
    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             block_start  = blockIdx.x * size_per_block,
             block_end    = block_start + size_per_block;

    uint32_t *buckets = buckets_ + blockIdx.x * bucket_count;

    for (uint32_t i = block_start + thread_id; i < block_end; i += thread_count) {
        bool active = i < size;

        /* This assumes that the whole warp does an iteration or exits
           (i.e. size_per_block must be a multiple of 32) */
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, active);

        if (active) {
            uint32_t offset = reduce_atomic(active_mask, values[i], buckets);
            perm[offset] = i;
        }
    }
}

KERNEL void transpose(const uint32_t *in,
                      uint32_t *out,
                      uint32_t i_rows,
                      uint32_t i_cols) {
    uint32_t *shared = SharedMemory<uint32_t>::get();
    const uint32_t dim = 16;

    uint32_t i_c = blockIdx.x * dim + threadIdx.x,
             i_r = blockIdx.y * dim + threadIdx.y;

    bool valid = i_r < i_rows && i_c < i_cols;

    if (valid)
        shared[threadIdx.y * (dim + 1) + threadIdx.x] = in[i_r * i_cols + i_c];

    __syncthreads();

    uint32_t o_rows = i_cols,
             o_cols = i_rows,
             o_c = blockIdx.y * dim + threadIdx.x,
             o_r = blockIdx.x * dim + threadIdx.y;

    valid = o_r < o_rows && o_c < o_cols;

    if (valid)
        out[o_r * o_cols + o_c] = shared[threadIdx.x * (dim + 1) + threadIdx.y];
}
