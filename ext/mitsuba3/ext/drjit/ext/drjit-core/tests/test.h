#pragma once

#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>

using namespace drjit;

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), const char *flags = nullptr);
extern "C" void log_level_callback(LogLevel cb, const char *msg);

using FloatC  = CUDAArray<float>;
using Int32C  = CUDAArray<int32_t>;
using UInt32C = CUDAArray<uint32_t>;
using MaskC   = CUDAArray<bool>;
using FloatL  = LLVMArray<float>;
using Int32L  = LLVMArray<int32_t>;
using UInt32L = LLVMArray<uint32_t>;
using MaskL   = LLVMArray<bool>;

#define TEST_CUDA(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    int test##name##_c =                                                       \
        test_register("test" #name "_cuda",                                    \
                      test##name<JitBackend::CUDA, FloatC, Int32C, UInt32C,    \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);                                          \
    int test##name##_o =                                                       \
        test_register("test" #name "_optix",                                   \
                      test##name<JitBackend::CUDA, FloatC, Int32C, UInt32C,    \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);                                          \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_LLVM(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    int test##name##_l =                                                       \
        test_register("test" #name "_llvm",                                    \
                      test##name<JitBackend::LLVM, FloatL, Int32L, UInt32L,    \
                                 MaskL, LLVMArray>,                            \
                      ##__VA_ARGS__);                                          \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_BOTH(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    int test##name##_c =                                                       \
        test_register("test" #name "_cuda",                                    \
                      test##name<JitBackend::CUDA, FloatC, Int32C, UInt32C,    \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);                                          \
    int test##name##_o =                                                       \
        test_register("test" #name "_optix",                                   \
                      test##name<JitBackend::CUDA, FloatC, Int32C, UInt32C,    \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);                                          \
    int test##name##_l =                                                       \
        test_register("test" #name "_llvm",                                    \
                      test##name<JitBackend::LLVM, FloatL, Int32L, UInt32L,    \
                                 MaskL, LLVMArray>,                            \
                      ##__VA_ARGS__);                                          \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define jit_assert(cond)                                                      \
    do {                                                                       \
        if (!(cond))                                                           \
            jit_fail("Assertion failure: %s in line %u.", #cond, __LINE__);   \
    } while (0)

/// RAII helper for temporarily decreasing the log level
struct scoped_set_log_level {
public:
    scoped_set_log_level(LogLevel level) {
        m_cb_level = jit_log_level_callback();
        m_stderr_level = jit_log_level_stderr();
        jit_set_log_level_callback(level < m_cb_level ? level : m_cb_level,
                                    log_level_callback);
        jit_set_log_level_stderr(level < m_stderr_level ? level
                                                         : m_stderr_level);
    }

    ~scoped_set_log_level() {
        jit_set_log_level_callback(m_cb_level, log_level_callback);
        jit_set_log_level_stderr(m_stderr_level);
    }

private:
    LogLevel m_cb_level;
    LogLevel m_stderr_level;
};
