/*
    src/log.h -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <stdarg.h>
#include "common.h"

#define DRJIT_DISABLE_TRACE 1

#if DRJIT_DISABLE_TRACE
#  define jitc_trace(...) do { } while (0)
#else
#  define jitc_trace(...) jitc_log(Trace, __VA_ARGS__)
#endif

/// Print a log message with the specified log level and message
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
extern void jitc_log(LogLevel level, const char* fmt, ...);

/// Print a log message with the specified log level and message
extern void jitc_vlog(LogLevel level, const char* fmt, va_list args);

/// Raise a std::runtime_error with the given message
#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
extern void jitc_raise(const char* fmt, ...);

/// Raise a std::runtime_error with the given message
[[noreturn]] extern void jitc_vraise(const char* fmt, va_list args);

/// Immediately terminate the application due to a fatal internal error
#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
   [[noreturn]]
#endif
extern void jitc_fail(const char* fmt, ...);

/// Immediately terminate the application due to a fatal internal error
[[noreturn]] extern void jitc_vfail(const char* fmt, va_list args);

/// Return and clear the log buffer
extern char *jitc_log_buffer();

/// Convert a number of bytes into a human-readable string (returns static buffer!)
extern const char *jitc_mem_string(size_t size);

/// Convert a time in microseconds into a human-readable string (returns static buffer!)
extern const char *jitc_time_string(float us);

/// Return the number of microseconds since the previous timer() call
extern float timer();

 JIT_MALLOC inline void* malloc_check(size_t size) {
    void *ptr = malloc(size);
    if (unlikely(!ptr))
        jitc_fail("malloc_check(): failed to allocate %zu bytes!", size);
    return ptr;
}

 JIT_MALLOC inline void* realloc_check(void *orig, size_t size) {
    void *ptr = realloc(orig, size);
    if (unlikely(!ptr))
        jitc_fail("realloc_check(): could not resize memory region to %zu bytes!", size);
    return ptr;
}
