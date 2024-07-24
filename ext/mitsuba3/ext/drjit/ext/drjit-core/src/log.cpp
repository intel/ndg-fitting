/*
    src/log.cpp -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cstdio>
#include <stdexcept>
#include <ctime>
#include "internal.h"
#include "log.h"

#if defined(_WIN32)
#  include <windows.h>
#endif

static Buffer log_buffer{128};
static char jitc_string_buf[64];

void jitc_log(LogLevel log_level, const char* fmt, ...) {
    if (unlikely(log_level <= state.log_level_stderr)) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    if (unlikely(log_level <= state.log_level_callback && state.log_callback)) {
        va_list args;
        va_start(args, fmt);
        log_buffer.clear();
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(log_level, log_buffer.get());
    }
}

void jitc_vlog(LogLevel log_level, const char* fmt, va_list args_) {
    if (unlikely(log_level <= state.log_level_stderr)) {
        va_list args;
        va_copy(args, args_);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    if (unlikely(log_level <= state.log_level_callback && state.log_callback)) {
        va_list args;
        va_copy(args, args_);
        log_buffer.clear();
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(log_level, log_buffer.get());
    }
}

void jitc_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);
    va_end(args);

    throw std::runtime_error(log_buffer.get());
}

void jitc_vraise(const char* fmt, va_list args) {
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);

    throw std::runtime_error(log_buffer.get());
}

void jitc_fail(const char* fmt, ...) {
    fprintf(stderr, "\n\nCritical Dr.Jit compiler failure: ");

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    lock_release(state.lock);
    abort();
}

void jitc_vfail(const char* fmt, va_list args) {
    fprintf(stderr, "Critical Dr.Jit compiler failure: ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    abort();
}

/// Generate a string representing a floating point followed by a unit
static void print_float_with_unit(char *buf, size_t bufsize, double value,
                                  bool accurate, const char *unit) {
    int digits_after_comma = accurate ? 5 : 3;

    digits_after_comma =
        std::max(digits_after_comma - int(std::log10(value)), 0);

    int pos = snprintf(buf, bufsize, "%.*f", digits_after_comma, value);

    // Remove trailing zeros
    char c;
    pos--;
    while (c = jitc_string_buf[pos], pos > 0 && (c == '0' || c == '.'))
        pos--;
    pos++;

    // Append unit if there is space
    if (pos + 1 < (int) bufsize)
        buf[pos++] = ' ';

    uint32_t i = 0;
    while (unit[i] != '\0' && pos + 1 < (int) bufsize)
        buf[pos++] = unit[i++];

    buf[pos] = '\0';
}

const char *jitc_mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };

    double value = (double) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.0; ++i)
        value /= 1024.0;

    print_float_with_unit(jitc_string_buf, sizeof(jitc_string_buf),
                          value, false, orders[i]);

    return jitc_string_buf;
}

const char *jitc_time_string(float value_) {
    double value = (double) value_;

    struct Order { double factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { 52.1429, "y" } };

    int i = 0;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    print_float_with_unit(jitc_string_buf, sizeof(jitc_string_buf),
                          value, true, orders[i].suffix);

    return jitc_string_buf;
}

Buffer::Buffer(size_t size) : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    m_start = (char *) malloc_check(size);
    m_end = m_start + size;
    clear();
}

size_t Buffer::fmt(const char *format, ...) {
    size_t written;
    do {
        size_t size = remain();
        va_list args;
        va_start(args, format);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written + 1 < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);

    return written;
}

size_t Buffer::vfmt(const char *format, va_list args_) {
    size_t written;
    va_list args;
    do {
        size_t size = remain();
        va_copy(args, args_);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written + 1 < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);
    return written;
}

void Buffer::expand(size_t minval) {
    size_t old_alloc_size = m_end - m_start,
           new_alloc_size = 2 * old_alloc_size + minval,
           used_size      = m_cur - m_start;

    m_start = (char *) realloc_check(m_start, new_alloc_size);
    m_end = m_start + new_alloc_size;
    m_cur = m_start + used_size;
}

#if !defined(_WIN32)
static timespec timer_value { 0, 0 };

float timer() {
    timespec timer_value_2;
    clock_gettime(CLOCK_MONOTONIC, &timer_value_2);
    float result = (timer_value_2.tv_sec - timer_value.tv_sec) * 1e6f +
                   (timer_value_2.tv_nsec - timer_value.tv_nsec) * 1e-3f;
    timer_value = timer_value_2;
    return result;
}
#else
static LARGE_INTEGER timer_value {};
float timer_frequency_scale;

float timer() {
    LARGE_INTEGER timer_value_2;
    QueryPerformanceCounter(&timer_value_2);
    float result = timer_frequency_scale *
                   (timer_value_2.QuadPart - timer_value.QuadPart);
    timer_value = timer_value_2;
    return result;
}
#endif
