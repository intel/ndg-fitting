/*
    src/hash.h -- Functionality for hashing kernels and other data (via xxHash)

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined (_MSC_VER)
#  pragma warning (push)
#  pragma warning (disable:26451)
#endif

#include <tsl/robin_map.h>
#include <xxh3.h>

#if defined (_MSC_VER)
#  pragma warning (pop)
#endif

#include <string.h>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

#if defined(__GNUC__)
    __attribute__((noreturn, __format__(__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
extern void jitc_fail(const char* fmt, ...);

struct UInt32Hasher {
    size_t operator()(uint32_t v) const {
        // fmix32 from MurmurHash by Austin Appleby (public domain)
        v ^= v >> 16;
        v *= 0x85ebca6b;
        v ^= v >> 13;
        v *= 0xc2b2ae35;
        v ^= v >> 16;
        return (size_t) v;
    }
};

struct UInt64Hasher {
    size_t operator()(uint64_t v) const {
        // fmix32 from MurmurHash by Austin Appleby (public domain)
        v ^= v >> 33;
        v *= (uint64_t) 0xff51afd7ed558ccdull;
        v ^= v >> 33;
        v *= (uint64_t) 0xc4ceb9fe1a85ec53ull;
        v ^= v >> 33;
        return (size_t) v;
    }
};

inline void hash_combine(size_t& seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

struct pair_hash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2> &x) const {
        size_t result = std::hash<T1>()(x.first);
        hash_combine(result, std::hash<T2>()(x.second));
        return result;
    }
};

inline size_t hash(const void *ptr, size_t size, size_t seed) {
    return (size_t) XXH3_64bits_withSeed(ptr, size, (unsigned long long) seed);
}

inline size_t hash(const void *ptr, size_t size) {
    return (size_t) XXH3_64bits(ptr, size);
}

inline size_t hash_str(const char *str, size_t seed) {
    return hash(str, strlen(str), seed);
}

inline size_t hash_str(const char *str) {
    return hash(str, strlen(str));
}

inline XXH128_hash_t hash_kernel(const char *str) {
    const char *offset = strstr(str, "body:");
    if (unlikely(!offset)) {
        offset = strchr(str, '{');
        if (unlikely(!offset))
            jitc_fail("hash_kernel(): invalid input!");
    }

    return XXH128(offset, strlen(offset), 0);
}
