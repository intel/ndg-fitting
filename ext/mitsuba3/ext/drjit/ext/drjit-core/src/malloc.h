/*
    src/malloc.h -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include "hash.h"

/// Data structure characterizing a memory allocation
#pragma pack(push, 1)
struct AllocInfo {
    uint64_t size : 48;
    uint64_t type : 8;
    uint64_t device : 8;

    AllocInfo() { memset(this, 0, sizeof(AllocInfo)); }

    AllocInfo(size_t size, AllocType type, int device)
        : size(size), type((unsigned) type), device(device) { }

    bool operator==(const AllocInfo &at) const {
        return type == at.type && device == at.device &&
               size == at.size;
    }

    bool operator!=(const AllocInfo &at) const {
        return type != at.type || device != at.device ||
               size != at.size;
    }
};
#pragma pack(pop)

/// Custom hasher for \ref AllocInfo
struct AllocInfoHasher {
    size_t operator()(const AllocInfo &at) const {
        size_t result = std::hash<size_t>()(at.size);
        hash_combine(result, std::hash<uint32_t>()((uint32_t) at.device << 8) + at.type);
        return result;
    }
};

using AllocInfoMap = tsl::robin_map<AllocInfo, std::vector<void *>, AllocInfoHasher>;
using AllocUsedMap = tsl::robin_pg_map<const void *, AllocInfo>;
struct ThreadState;

/// Round to the next power of two
extern size_t round_pow2(size_t x);
extern uint32_t round_pow2(uint32_t x);

/// Descriptive names for the various allocation types
extern const char *alloc_type_name[(int) AllocType::Count];
extern const char *alloc_type_name_short[(int) AllocType::Count];

/// Allocate the given flavor of memory
extern void *jitc_malloc(AllocType type, size_t size) JIT_MALLOC;

/// Release the given pointer
extern void jitc_free(void *ptr);

/// Schedule a function that will reclaim memory from pending jitc_free()s
extern void jitc_free_flush(ThreadState *ts);

/// Change the flavor of an allocated memory region
extern void* jitc_malloc_migrate(void *ptr, AllocType type, int move);

/// Asynchronously prefetch a memory region
extern void jitc_malloc_prefetch(void *ptr, int device);

/// Release all unused memory to the GPU / OS
extern void jitc_flush_malloc_cache(bool flush_local, bool warn);

/// Shut down the memory allocator (calls \ref jitc_flush_malloc_cache() and reports leaks)
extern void jitc_malloc_shutdown();

/// Query the flavor of a memory allocation made using \ref jitc_malloc()
extern AllocType jitc_malloc_type(void *ptr);

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
extern int jitc_malloc_device(void *ptr);

/// Clear the peak memory usage statistics
extern void jitc_malloc_clear_statistics();
