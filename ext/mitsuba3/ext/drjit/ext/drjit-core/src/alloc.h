/*
    src/alloc.h -- Aligned memory allocator

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdlib>

/// Immediately terminate the application due to a fatal internal error
#if defined(__GNUC__)
    __attribute__((noreturn, __format__(__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
extern void jitc_fail(const char* fmt, ...);

template <typename T, size_t Align = alignof(T)>
struct aligned_allocator {
public:
    template <typename T2> struct rebind {
        using other = aligned_allocator<T2, Align>;
    };

    using value_type      = T;
    using reference       = T &;
    using const_reference = const T &;
    using pointer         = T *;
    using const_pointer   = const T *;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    aligned_allocator() = default;
    aligned_allocator(const aligned_allocator &) = default;

    template <typename T2, size_t Align2>
    aligned_allocator(const aligned_allocator<T2, Align2> &) { }

    value_type *allocate(size_t count) {
        void *ptr;
#if !defined(_WIN32)
        if (posix_memalign(&ptr, Align, sizeof(T) * count) != 0)
            jitc_fail("aligned_allocator::allocate(): out of memory!");
#else
        if ((ptr = _aligned_malloc(sizeof(T) * count, Align)) == nullptr)
            jitc_fail("aligned_allocator::allocate(): out of memory!");
#endif
        return (value_type *) ptr;
    }

    void deallocate(value_type *ptr, size_t) {
#if !defined(_WIN32)
        free(ptr);
#else
        _aligned_free(ptr);
#endif
    }

    template <typename T2, size_t Align2>
    bool operator==(const aligned_allocator<T2, Align2> &) const {
        return Align == Align2;
    }

    template <typename T2, size_t Align2>
    bool operator!=(const aligned_allocator<T2, Align2> &) const {
        return Align != Align2;
    }
};
