/*
    drjit-core/jit.h -- Self-contained JIT compiler for CUDA & LLVM.

    This library implements a self-contained tracing JIT compiler that supports
    both CUDA PTX and LLVM IR as intermediate representations. It takes care of
    many tricky aspects, such as recording of arithmetic and higher-level
    operations (loops, virtual function calls), asynchronous memory allocation
    and release, multi-device computation, kernel caching and reuse, common
    subexpression elimination, etc.

    While the library is internally implemented using C++14, this header file
    provides a compact C99-compatible API that can be used to access all
    functionality. The library is thread-safe: multiple threads can
    simultaneously dispatch computation to one or more CPUs/GPUs.

    As an alternative to the fairly low-level API defined here, you may prefer
    the interface in 'include/drjit-core/array.h', which provides a header-only
    C++ array abstraction with operator overloading that dispatches to the C
    API. The Dr.Jit parent project (https://github.com/mitsuba-renderer/drjit)
    can also be interpreted as continuation of this kind of abstraction, which
    adds further components like a library of transcendental mathematical
    operations, automatic differentiation support, Python bindings, etc.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdlib.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  if defined(DRJIT_BUILD)
#    define JIT_EXPORT    __declspec(dllexport)
#  else
#    define JIT_EXPORT    __declspec(dllimport)
#  endif
#  define JIT_MALLOC
#  define JIT_INLINE    __forceinline
#  define JIT_NOINLINE  __declspec(noinline)
#else
#  define JIT_EXPORT    __attribute__ ((visibility("default")))
#  define JIT_MALLOC    __attribute__((malloc))
#  define JIT_INLINE    __attribute__ ((always_inline)) inline
#  define JIT_NOINLINE  __attribute__ ((noinline))
#endif

#if defined(__cplusplus)
#  define JIT_CONSTEXPR constexpr
#  define JIT_DEF(x) = x
#  define JIT_NOEXCEPT noexcept(true)
#  define JIT_ENUM ::
#  if !defined(NAMESPACE_BEGIN)
#    define NAMESPACE_BEGIN(name) namespace name {
#  endif
#  if !defined(NAMESPACE_END)
#    define NAMESPACE_END(name) }
#  endif
#else
#  define JIT_CONSTEXPR inline
#  define JIT_DEF(x)
#  define JIT_NOEXCEPT
#  define JIT_ENUM enum
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// ====================================================================
//         Initialization, device enumeration, and management
// ====================================================================

/**
 * \brief List of backends that can be targeted by Dr.Jit
 *
 * Dr.Jit can perform computation using one of several computational
 * backends. Before use, a backend must be initialized via \ref jit_init().
 */
#if defined(__cplusplus)
enum class JitBackend : uint32_t {
    /// CUDA backend (requires CUDA >= 10, generates PTX instructions)
    CUDA = (1 << 0),

    /// LLVM backend targeting the CPU (generates LLVM IR)
    LLVM = (1 << 1)
};
#else
enum JitBackend {
    JitBackendCUDA = (1 << 0),
    JitBackendLLVM = (1 << 1)
};
#endif

/**
 * \brief Initialize a JIT compiler backend
 *
 * The function <tt>jit_init()</tt> must be called before using the JIT
 * compiler. It takes a bit-wise OR of elements of the \ref JitBackend
 * enumeration and tries to initialize each specified backend. Query \ref
 * jit_has_backend() following this operation to check if a backend was
 * initialized successfully. This function does nothing when initialization has
 * already occurred. It is possible to re-initialize the JIT following a call
 * to \ref jit_shutdown(), which can be useful to reset the state, e.g., in
 * testcases.
 */
extern JIT_EXPORT void
jit_init(uint32_t backends JIT_DEF((uint32_t) JitBackend::CUDA |
                                   (uint32_t) JitBackend::LLVM));

/**
 * \brief Launch an asynchronous thread that will execute jit_init() and
 * return immediately
 *
 * On machines with several GPUs, \ref jit_init() will set up a CUDA
 * environment on all devices when <tt>cuda=true</tt> is specified. This can be
 * a rather slow operation (e.g. 1 second). This function provides a convenient
 * alternative to hide this latency, for instance when importing this library
 * from an interactive Python session which doesn't need the JIT right away.
 *
 * The \c llvm and \c cuda arguments should be set to \c 1 to initialize the
 * corresponding backend, and \c 0 otherwise.
 *
 * Note that it is safe to call <tt>jit_*</tt> API functions following
 * initialization via \ref jit_init_async(), since it acquires a lock to the
 * internal data structures.
 */
extern JIT_EXPORT void
jit_init_async(uint32_t backends JIT_DEF((uint32_t) JitBackend::CUDA |
                                         (uint32_t) JitBackend::LLVM));

/// Check whether the LLVM backend was successfully initialized
extern JIT_EXPORT int jit_has_backend(JIT_ENUM JitBackend backend);

/**
 * \brief Release resources used by the JIT compiler, and report reference leaks.
 *
 * If <tt>light=1</tt>, this function performs a "light" shutdown, which
 * flushes any still running computation and releases unused memory back to the
 * OS or GPU. It will also warn about leaked variables and memory allocations.
 *
 * If <tt>light=0</tt>, the function furthermore completely unloads the LLVM and
 * CUDA backends. This frees up more memory but means that a later call to \ref
 * jit_init() or \ref jit_init_async() will be slow.
 */
extern JIT_EXPORT void jit_shutdown(int light JIT_DEF(0));

/**
 * \brief Wait for all computation scheduled by the current thread to finish
 *
 * Each thread using Dr.Jit will issue computation to an independent queue.
 * This function only synchronizes with computation issued to the queue of the
 * calling thread.
 */
extern JIT_EXPORT void jit_sync_thread();

/// Wait for all computation on the current device to finish
extern JIT_EXPORT void jit_sync_device();

/// Wait for all computation on the *all devices* to finish
extern JIT_EXPORT void jit_sync_all_devices();

// ====================================================================
//                    CUDA/LLVM-specific functionality
// ====================================================================

/// Return the no. of available CUDA devices that are compatible with Dr.Jit.
extern JIT_EXPORT int jit_cuda_device_count();

/**
 * \brief Set the active CUDA device.
 *
 * The argument must be between 0 and <tt>jit_cuda_device_count() - 1</tt>,
 * which only accounts for Dr.Jit-compatible devices. This is a per-thread
 * property: independent threads can optionally issue computation to different
 * GPUs.
 */
extern JIT_EXPORT void jit_cuda_set_device(int device);

/**
 * \brief Return the CUDA device ID associated with the current thread
 *
 * The result is in the range of 0 and <tt>jit_cuda_device_count() - 1</tt>.
 * When the machine contains CUDA devices that are incompatible with Dr.Jit (due
 * to a lack of 64-bit addressing, uniform address space, or managed memory),
 * this number may differ from the default CUDA device ID. Use
 * <tt>jit_cuda_device_raw()</tt> in that case.
 */
extern JIT_EXPORT int jit_cuda_device();

/// Return the raw CUDA device associated with the current thread
extern JIT_EXPORT int jit_cuda_device_raw();

/// Return the CUDA stream associated with the current thread
extern JIT_EXPORT void* jit_cuda_stream();

/// Return the CUDA context associated with the current thread
extern JIT_EXPORT void* jit_cuda_context();

/// Push a new CUDA context to be associated with the current thread
extern JIT_EXPORT void jit_cuda_push_context(void *);

/// Pop the CUDA context associated to the current thread and return it
extern JIT_EXPORT void* jit_cuda_pop_context();

/// Query the compute capability of the current device (e.g. '52')
extern JIT_EXPORT int jit_cuda_compute_capability();

/**
 * \brief Override generated PTX version and compute capability
 *
 * Dr.Jit generates code that runs on a wide variety of platforms supporting
 * at least the PTX version and compute capability of 60, and 50, respectively.
 * Those versions can both be bumped via this function---there is no
 * performance advantage in doing so, though some more recent features (e.g.
 * atomic operations involving double precision values) require specifying a
 * newer compute capability.
 */
extern JIT_EXPORT void jit_cuda_set_target(uint32_t ptx_version,
                                           uint32_t compute_capability);

/// Look up an CUDA driver function by name
extern JIT_EXPORT void *jit_cuda_lookup(const char *name);

/**
 * \brief Override the target CPU, features, and vector width of the LLVM backend
 *
 * The LLVM backend normally generates code for the detected native hardware
 * architecture akin to compiling with <tt>-march=native</tt>. This function
 * can be used to change the following code generation-related parameters:
 *
 * \param target_cpu
 *     Target CPU (e.g. <tt>haswell</tt>)
 *
 * \param target_features
 *     Comma-separated list of LLVM feature flags (e.g. <tt>+avx512f</tt>).
 *     This should be set to <tt>nullptr</tt> if you do not wish to specify
 *     individual features.
 *
 * \param vector_width
 *     Width of vector registers (e.g. 8 for AVX). Must be a power of two, and
 *     can be a multiple of the hardware register size to enable unrolling.
 */
extern JIT_EXPORT void jit_llvm_set_target(const char *target_cpu,
                                           const char *target_features,
                                           uint32_t vector_width);

/// Get the CPU that is currently targeted by the LLVM backend
extern JIT_EXPORT const char *jit_llvm_target_cpu();

/// Get the list of CPU features currently used by the LLVM backend
extern JIT_EXPORT const char *jit_llvm_target_features();

/// Get the major, minor and patch version of the LLVM library
extern JIT_EXPORT void jit_llvm_version(int *major, int *minor, int *patch);

/// Get the vector width of the LLVM backend
extern JIT_EXPORT uint32_t jit_llvm_vector_width();

/// Specify the number of threads that are used to parallelize the computation
extern JIT_EXPORT void jit_llvm_set_thread_count(uint32_t size);

/**
 * \brief Convenience function for intrinsic function selection
 *
 * Returns \c 1 if the current vector width is is at least as large as a
 * provided value, and when the host CPU provides a given target feature (e.g.
 * "+avx512f").
 */
extern JIT_EXPORT int jit_llvm_if_at_least(uint32_t vector_width,
                                           const char *feature);

// ====================================================================
//                        Logging infrastructure
// ====================================================================

#if defined(__cplusplus)
enum class LogLevel : uint32_t {
    Disable, Error, Warn, Info, InfoSym, Debug, Trace
};
#else
enum LogLevel {
    LogLevelDisable, LogLevelError, LogLevelWarn, LogLevelInfo,
    LogLevelInfoSym, LogLevelDebug, LogLevelTrace
};
#endif

/**
 * \brief Control the destination of log messages (stderr)
 *
 * By default, this library prints all log messages to the console (\c stderr).
 * This function can be used to control the minimum log level for such output
 * or prevent it entirely. In the latter case, you may wish to enable logging
 * via a callback in \ref jit_set_log_level_callback(). Both destinations can also
 * be enabled simultaneously, potentially using different log levels.
 */
extern JIT_EXPORT void jit_set_log_level_stderr(JIT_ENUM LogLevel level);

/// Return the currently set minimum log level for output to \c stderr
extern JIT_EXPORT JIT_ENUM LogLevel jit_log_level_stderr();


/**
 * \brief Control the destination of log messages (callback)
 *
 * This function can be used to specify an optional callback that will be
 * invoked with the contents of library log messages, whose severity matches or
 * exceeds the specified \c level.
 */
typedef void (*LogCallback)(JIT_ENUM LogLevel, const char *);
extern JIT_EXPORT void jit_set_log_level_callback(JIT_ENUM LogLevel level,
                                                  LogCallback callback);

/// Return the currently set minimum log level for output to a callback
extern JIT_EXPORT JIT_ENUM LogLevel jit_log_level_callback();

/// Print a log message with the specified log level and message
extern JIT_EXPORT void jit_log(JIT_ENUM LogLevel level, const char* fmt, ...);

/// Raise an exception message with the specified message
extern JIT_EXPORT void jit_raise(const char* fmt, ...);

/// Terminate the application due to a non-recoverable error
extern JIT_EXPORT void jit_fail(const char* fmt, ...);

// ====================================================================
//                         Memory allocation
// ====================================================================

#if defined(__cplusplus)
enum class AllocType : uint32_t {
    /**
     * Memory that is located on the host (i.e., the CPU). When allocated via
     * \ref jit_malloc(), host memory is immediately ready for use, and
     * its later release via \ref jit_free() also occurs instantaneously.
     *
     * Note, however, that released memory is kept within a cache and not
     * immediately given back to the operating system. Call \ref
     * jit_flush_malloc_cache() to also flush this cache.
     */
    Host,

    /**
     * Like \c Host memory, except that it may only be used *asynchronously*
     * within a computation performed by drjit-core.
     *
     * In particular, host-asynchronous memory obtained via \ref jit_malloc()
     * should not be written to directly (i.e. outside of drjit-core), since it
     * may still be used by a currently running kernel. Releasing
     * host-asynchronous memory via \ref jit_free() also occurs
     * asynchronously.
     *
     * This type of memory is used internally when running code via the LLVM
     * backend, and when this process is furthermore parallelized using Dr.Jit's
     * internal thread pool.
     */
    HostAsync,

    /**
     * Memory on the host that is "pinned" and thus cannot be paged out.
     * Host-pinned memory is accessible (albeit slowly) from CUDA-capable GPUs
     * as part of the unified memory model, and it also can be a source or
     * destination of asynchronous host <-> device memcpy operations.
     *
     * Host-pinned memory has asynchronous semantics similar to \c HostAsync.
     */
    HostPinned,

    /**
     * Memory that is located on a device (i.e., one of potentially several
     * GPUs).
     *
     * Device memory has asynchronous semantics similar to \c HostAsync.
     */
    Device,

    /**
     * Memory that is mapped in the address space of the host & all GPUs.
     *
     * Managed memory has asynchronous semantics similar to \c HostAsync.
     */
    Managed,

    /**
     * Like \c Managed, but more efficient when accesses are mostly reads. In
     * this case, the system will distribute multiple read-only copies instead
     * of moving memory back and forth.
     *
     * This type of memory has asynchronous semantics similar to \c HostAsync.
     */
    ManagedReadMostly,

    /// Number of possible allocation types
    Count
};
#else
enum AllocType {
    AllocTypeHost,
    AllocTypeHostPinned,
    AllocTypeDevice,
    AllocTypeManaged,
    AllocTypeManagedReadMostly,
    AllocTypeCount
};
#endif

/**
 * \brief Allocate memory of the specified type
 *
 * Under the hood, Dr.Jit implements a custom allocation scheme that tries to
 * reuse allocated memory regions instead of giving them back to the OS/GPU.
 * This eliminates inefficient synchronization points in the context of CUDA
 * programs, and it can also improve performance on the CPU when working with
 * large allocations.
 *
 * The returned pointer is guaranteed to be sufficiently aligned for any kind
 * of use.
 *
 */
extern JIT_EXPORT void *jit_malloc(JIT_ENUM AllocType type, size_t size)
    JIT_MALLOC;

/**
 * \brief Release a given pointer asynchronously
 *
 * For CPU-only arrays (\ref AllocType::Host), <tt>jit_free()</tt> is
 * synchronous and very similar to <tt>free()</tt>, except that the released
 * memory is placed in Dr.Jit's internal allocation cache instead of being
 * returned to the OS. The function \ref jit_flush_malloc_cache() can optionally
 * be called to also clear this cache.
 *
 * When \c ptr is an asynchronous host pointer (\ref AllocType::HostAsync) or
 * GPU-accessible pointer (\ref AllocType::Device, \ref AllocType::HostPinned,
 * \ref AllocType::Managed, \ref AllocType::ManagedReadMostly), the associated
 * memory region is possibly still being used by a running kernel, and it is
 * therefore merely *scheduled* to be reclaimed once this kernel finishes.
 *
 * Kernel launches and memory-related operations (malloc, free) occur
 * asynchronously but using a linear ordering when they are scheduled by the
 * same thread (they will be placed into the same <i>stream</i> in CUDA
 * terminology). Extra care must be taken in the context of multi-threaded
 * software: it is not permissible to e.g. allocate memory on one thread,
 * launch a kernel using it, then immediately release that memory from a
 * different thread, because a valid ordering is not guaranteed in that case.
 * Operations like \ref jit_sync_thread(), \ref jit_sync_device(), and \ref
 * jit_sync_all_devices() can be used to defuse such situations.
 */
extern JIT_EXPORT void jit_free(void *ptr);

/// Release all currently unused memory to the GPU / OS
extern JIT_EXPORT void jit_flush_malloc_cache();

/// Clear the peak memory usage statistics
extern JIT_EXPORT void jit_malloc_clear_statistics();

/// Flush internal kernel cache
extern JIT_EXPORT void jit_flush_kernel_cache();

/**
 * \brief Asynchronously prefetch a managed memory region allocated using \ref
 * jit_malloc() so that it is available on a specified device
 *
 * This operation prefetches a memory region so that it is available on the CPU
 * (<tt>device==-1</tt>) or specified CUDA device (<tt>device&gt;=0</tt>). This
 * operation only make sense for allocations of type <tt>AllocType::Managed<tt>
 * and <tt>AllocType::ManagedReadMostly</tt>. In the former case, the memory
 * region will be fully migrated to the specified device, and page mappings
 * established elsewhere are cleared. For the latter, a read-only copy is
 * created on the target device in addition to other copies that may exist
 * elsewhere.
 *
 * The function also takes a special argument <tt>device==-2</tt>, which
 * creates a read-only mapping on *all* available GPUs.
 *
 * The prefetch operation is enqueued on the current device and thread and runs
 * asynchronously with respect to the CPU, hence a \ref jit_sync_thread()
 * operation is advisable if data is <tt>target==-1</tt> (i.e. prefetching into
 * CPU memory).
 */
extern JIT_EXPORT void jit_malloc_prefetch(void *ptr, int device);

/// Query the flavor of a memory allocation made using \ref jit_malloc()
extern JIT_EXPORT JIT_ENUM AllocType jit_malloc_type(void *ptr);

/// Query the device associated with a memory allocation made using \ref jit_malloc()
extern JIT_EXPORT int jit_malloc_device(void *ptr);

/**
 * \brief Asynchronously change the flavor of an allocated memory region and
 * return the new pointer
 *
 * The operation is *always* asynchronous and, hence, will need to be followed
 * by an explicit synchronization via \ref jit_sync_thread() if memory is
 * migrated from the GPU to the CPU and expected to be accessed on the CPU
 * before the transfer has finished. Nothing needs to be done in the other
 * direction, e.g. when migrating memory that is subsequently accessed by
 * a GPU kernel.
 *
 * When no migration is necessary, the function simply returns the input
 * pointer. If migration is necessary, the behavior depends on the supplied
 * <tt>move</tt> parameter. When <tt>move==0</tt>, the implementation schedules
 * an asynchronous copy and leaves the old pointer undisturbed. If
 * <tt>move==1</tt>, the old pointer is asynchronously freed once the copy
 * operation finishes.
 *
 * When both source and target are of type \ref AllocType::Device, and
 * when the currently active device (determined by the last call to \ref
 * jit_set_device()) does not match the device associated with the allocation,
 * a peer-to-peer migration is performed.
 */
extern JIT_EXPORT void *jit_malloc_migrate(void *ptr, JIT_ENUM AllocType type,
                                           int move JIT_DEF(1));

// ====================================================================
//                          Pointer registry
// ====================================================================

/**
 * \brief Register a pointer with Dr.Jit's pointer registry
 *
 * Dr.Jit provides a central registry that maps registered pointer values to
 * low-valued 32-bit IDs. The main application is efficient virtual function
 * dispatch via \ref jit_var_vcall(), through the registry could be used for other
 * applications as well.
 *
 * This function registers the specified pointer \c ptr with the registry,
 * returning the associated ID value, which is guaranteed to be unique within
 * the specified domain \c domain. The domain is normally an identifier that is
 * associated with the "flavor" of the pointer (e.g. instances of a particular
 * class), and which ensures that the returned ID values are as low as
 * possible.
 *
 * Caution: for reasons of efficiency, the \c domain parameter is assumed to a
 * static constant that will remain alive. The RTTI identifier
 * <tt>typeid(MyClass).name()<tt> is a reasonable choice that satisfies this
 * requirement.
 *
 * Returns zero if <tt>ptr == nullptr</tt> and throws if the pointer is already
 * registered (with *any* domain).
 */
extern JIT_EXPORT uint32_t jit_registry_put(JIT_ENUM JitBackend backend,
                                            const char *domain, void *ptr);

/**
 * \brief Remove a pointer from the registry
 *
 * No-op if <tt>ptr == nullptr</tt>. Throws an exception if the pointer is not
 * currently registered.
 */
extern JIT_EXPORT void jit_registry_remove(JIT_ENUM JitBackend backend,
                                           void *ptr);

/**
 * \brief Query the ID associated a registered pointer
 *
 * Returns 0 if <tt>ptr==nullptr</tt> and throws if the pointer is not known.
 */
extern JIT_EXPORT uint32_t jit_registry_get_id(JIT_ENUM JitBackend backend,
                                               const void *ptr);

/**
 * \brief Query the domain associated a registered pointer
 *
 * Returns \c nullptr if <tt>ptr==nullptr</tt> and throws if the pointer is not
 * known.
 */
extern JIT_EXPORT const char *jit_registry_get_domain(JIT_ENUM JitBackend backend,
                                                      const void *ptr);

/**
 * \brief Query the pointer associated a given domain and ID
 *
 * Returns \c nullptr if <tt>id==0</tt>, or when the (domain, ID) combination
 * is not known.
 */
extern JIT_EXPORT void *jit_registry_get_ptr(JIT_ENUM JitBackend backend,
                                             const char *domain, uint32_t id);

/// Provide a bound (<=) on the largest ID associated with a domain
extern JIT_EXPORT uint32_t jit_registry_get_max(JIT_ENUM JitBackend backend, const char *domain);

/**
 * \brief Compact the registry and release unused IDs and attributes
 *
 * It's a good idea to call this function following a large number of calls to
 * \ref jit_registry_remove().
 */
extern JIT_EXPORT void jit_registry_trim();

/**
 * \brief Clear the registry and release all IDs and attributes
 *
 * Extra care must be taken when calling this function as it might result in
 * undefined behaviour and crashes if instances are still alive and used
 * afterward.
 */
extern JIT_EXPORT void jit_registry_clear();

/**
 * \brief Set a custom per-pointer attribute
 *
 * The pointer registry can optionally associate one or more read-only
 * attribute with each pointer that can be set using this function. Such
 * pointer attributes provide an efficient way to avoid expensive vectorized
 * method calls (via \ref jit_var_vcall()) for simple getter-like functions. In
 * particular, this feature would be used in conjunction with \ref
 * jit_registry_attr_data(), which returns a pointer to a linear array
 * containing all attributes. A vector of 32-bit IDs (returned by \ref
 * jit_registry_put() or \ref jit_registry_get_id()) can then be used to
 * gather from this address.
 *
 * \param ptr
 *     Pointer, whose attribute should be set. Must have been previously
 *     registered using \ref jit_registry_put()
 *
 * \param name
 *     Name of the attribute to be set.
 *
 * \param value
 *     Pointer to the attribute value (in CPU memory)
 *
 * \param size
 *     Size of the pointed-to region.
 */
extern JIT_EXPORT void jit_registry_set_attr(JIT_ENUM JitBackend backend,
                                             void *ptr,
                                             const char *name,
                                             const void *value,
                                             size_t size);

/**
 * \brief Return a pointer to a contiguous array containing a specific
 * attribute associated with a specific domain
 *
 * \sa jit_registry_set_attr
 */
extern JIT_EXPORT const void *jit_registry_attr_data(JIT_ENUM JitBackend backend,
                                                     const char *domain,
                                                     const char *name);

// ====================================================================
//                        Variable management
// ====================================================================

#if defined(__cplusplus)
/**
 * \brief Variable types supported by the JIT compiler.
 *
 * A type promotion routine in the Dr.Jit Python bindings depends on on this
 * exact ordering, so please don't change.
 */
enum class VarType : uint32_t {
    Void, Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Pointer, Float16, Float32, Float64, Count
};
#else
enum VarType {
    VarTypeVoid, VarTypeBool, VarTypeInt8, VarTypeUInt8,
    VarTypeInt16, VarTypeUInt16, VarTypeInt32, VarTypeUInt32,
    VarTypeInt64, VarTypeUInt64, VarTypePointer, VarTypeFloat16,
    VarTypeFloat32, VarTypeFloat64, VarTypeCount
};
#endif

/**
 * \brief Create a variable representing a literal constant
 *
 * <b>Advanced usage</b>: When \c eval is nonzero, the variable is directly
 * created in evaluated form, which means that subsequent usage will access the
 * contents via memory instead of including the actual constant value in
 * generated PTX/LLVM code. This is particularly useful for loops: suppose a
 * loop references a literal constant that keeps changing (e.g. an iteration
 * counter). This change causes each iteration to generate different code,
 * requiring repeated compilation steps. By preemptively evaluating this
 * constant, Dr.Jit can reuse a single kernel for all steps.
 *
 * The parameter \c is_class specifies whether the variable represents an
 * instance index of a class, which may trigger further optimizations within
 * virtual function calls.
 */
extern JIT_EXPORT uint32_t jit_var_new_literal(JIT_ENUM JitBackend backend,
                                               JIT_ENUM VarType type,
                                               const void *value,
                                               size_t size JIT_DEF(1),
                                               int eval JIT_DEF(0),
                                               int is_class JIT_DEF(0));

/**
 * \brief Create a counter variable
 *
 * This operation creates a variable of type \ref VarType::UInt32 that will
 * evaluate to <tt>0, ..., size - 1</tt>.
 */
extern JIT_EXPORT uint32_t jit_var_new_counter(JIT_ENUM JitBackend backend,
                                               size_t size);

/**
 * \brief Create a new variable representing the result of a LLVM/PTX statement
 *
 * This function takes a statement in an intermediate representation (CUDA PTX or
 * LLVM IR) and registers it in the global variable list. It returns the index
 * of the variable that will store the result of the statement, whose external
 * reference count is initialized to \c 1.
 *
 * You will probably want to access this function through the wrappers \ref
 * jit_var_new_stmt_0() to \ref jit_var_new_stmt_4() that take an explicit
 * list of parameter indices and assume that it's not necessary to make a copy
 * of \c stmt (i.e. <tt>stmt_static == 1</tt>).
 *
 * The string \c stmt may contain special dollar-prefixed expressions
 * (<tt>$rN</tt>, <tt>$tN</tt>, or <tt>$bN</tt>, where <tt>N</tt> ranges from
 * 0-4) to refer to operands and their types. During compilation, these will
 * then be rewritten into a register name of the variable (<tt>r</tt>), its
 * type (<tt>t</tt>), or a generic binary type of matching size (<tt>b</tt>).
 * Index <tt>0</tt> refers to the variable being generated, while indices
 * <tt>1<tt>-<tt>3</tt> refer to the operands. For instance, a PTX integer
 * addition would be encoded as follows:
 *
 * \code
 * uint32_t result = jit_var_new_stmt_2(JitBackend:::CUDA, VarType::Int32,
 *                                      "add.$t0 $r0, $r1, $r2",
 *                                      op1, op2);
 * \endcode
 *
 * \param backend
 *    Specifies whether 'stmt' contains a CUDA PTX or LLVM IR instruction.
 *
 * \param vt
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param stmt
 *    Intermediate language statement.
 *
 * \param stmt_static
 *    When 'stmt' is a static string stored in the data segment of the
 *    executable, it is not necessary to make a copy. In this case, set
 *    <tt>stmt_static == 1</tt>, and <tt>0</tt> otherwise.
 *
 * \param n_dep
 *    Number of dependencies (between 0 and 4)
 *
 * \param dep
 *    Pointer to a list of \c n_dep valid variable indices
 */
extern JIT_EXPORT uint32_t jit_var_new_stmt(JIT_ENUM JitBackend backend,
                                            JIT_ENUM VarType vt,
                                            const char *stmt,
                                            int stmt_static,
                                            uint32_t n_dep,
                                            const uint32_t *dep);

// Create a new variable with 0 dependencies (wraps \c jit_var_new_stmt())
static inline uint32_t jit_var_new_stmt_0(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt,
                                          const char *stmt) {
    return jit_var_new_stmt(backend, vt, stmt, 1, 0, NULL);
}

// Create a new variable with 1 dependency (wraps \c jit_var_new_stmt())
static inline uint32_t jit_var_new_stmt_1(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt, const char *stmt,
                                          uint32_t dep0) {
    return jit_var_new_stmt(backend, vt, stmt, 1, 1, &dep0);
}

// Create a new variable with 2 dependencies (wraps \c jit_var_new_stmt())
static inline uint32_t jit_var_new_stmt_2(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt, const char *stmt,
                                          uint32_t dep0, uint32_t dep1) {
    const uint32_t dep[] = { dep0, dep1 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 2, dep);
}

// Create a new variable with 3 dependencies (wraps \c jit_var_new_stmt())
static inline uint32_t jit_var_new_stmt_3(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt, const char *stmt,
                                          uint32_t dep0, uint32_t dep1,
                                          uint32_t dep2) {
    const uint32_t dep[] = { dep0, dep1, dep2 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 3, dep);
}

// Create a new variable with 4 dependencies (wraps \c jit_var_new_stmt())
static inline uint32_t jit_var_new_stmt_4(JIT_ENUM JitBackend backend,
                                          JIT_ENUM VarType vt, const char *stmt,
                                          uint32_t dep0, uint32_t dep1,
                                          uint32_t dep2, uint32_t dep3) {
    const uint32_t dep[] = { dep0, dep1, dep2, dep3 };
    return jit_var_new_stmt(backend, vt, stmt, 1, 4, dep);
}

#if defined(__cplusplus)
/// List of operations supported by \ref jit_var_new_op()
enum class JitOp : uint32_t {
    // ---- Unary ----
    Not, Neg, Abs, Sqrt, Rcp, Rsqrt, Ceil, Floor, Round, Trunc, Exp2, Log2, Sin, Cos,
    Popc, Clz, Ctz,
    // ---- Binary ----
    Add, Sub, Mul, Mulhi, Div, Mod, Min, Max, And, Or, Xor, Shl, Shr,
    // ---- Comparisons ----
    Eq, Neq, Lt, Le, Gt, Ge,
    // ---- Ternary ----
    Fmadd, Select,

    Count
};
#else
enum JitOp {
    JitOpNot, JitOpNeg, JitOpAbs, JitOpSqrt, JitOpRcp, JitOpRsqrt, JitOpCeil,
    JitOpFloor, JitOpRound, JitOpTrunc, JitOpExp2, JitOpLog2, JitOpSin,
    JitOpCos, JitOpPopc, JitOpClz, JitOpCtz, JitOpAdd, JitOpSub, JitOpMul,
    JitOpMulhi, JitOpDiv, JitOpMod, JitOpMin, JitOpMax, JitOpAnd, JitOpOr,
    JitOpXor, JitOpShl, JitOpShr, JitOpEq, JitOpNeq, JitOpLt, JitOpLe, JitOpGt,
    JitOpGe, JitOpFmadd, JitOpSelect, JitOpCount
};
#endif


/**
 * \brief Perform an arithmetic operation involving one or more variables
 *
 * This function can perform a large range of unary, binary, and ternary
 * arithmetic operations. It automatically infers the necessary LLVM or PTX
 * instructions and performs constant propagation if possible (when one or
 * more input are literals).
 *
 * You will probably want to access this function through the wrappers \ref
 * jit_var_new_op_0() to \ref jit_var_new_op_4() that take an explicit
 * list of parameter indices.
 *
 * \param op
 *    The operation to be performed
 *
 * \param n_dep
 *    Number of dependencies (between 0 and 4)
 *
 * \param dep
 *    Pointer to a list of \c n_dep valid variable indices
 */
extern JIT_EXPORT uint32_t jit_var_new_op(JIT_ENUM JitOp op,
                                          uint32_t n_dep, const uint32_t *dep);

// Perform an operation with 1 input (wraps \c jit_var_new_op())
static inline uint32_t jit_var_new_op_1(JIT_ENUM JitOp op, uint32_t dep0) {
    return jit_var_new_op(op, 1, &dep0);
}

// Perform an operation with 2 inputs (wraps \c jit_var_new_op())
static inline uint32_t jit_var_new_op_2(JIT_ENUM JitOp op, uint32_t dep0,
                                        uint32_t dep1) {
    const uint32_t dep[] = { dep0, dep1 };
    return jit_var_new_op(op, 2, dep);
}

// Perform an operation with 3 inputs (wraps \c jit_var_new_op())
static inline uint32_t jit_var_new_op_3(JIT_ENUM JitOp op, uint32_t dep0,
                                        uint32_t dep1, uint32_t dep2) {
    const uint32_t dep[] = { dep0, dep1, dep2 };
    return jit_var_new_op(op, 3, dep);
}

// Perform an operation with 4 inputs (wraps \c jit_var_new_op())
static inline uint32_t jit_var_new_op_4(JIT_ENUM JitOp op, uint32_t dep0,
                                        uint32_t dep1, uint32_t dep2,
                                        uint32_t dep3) {
    const uint32_t dep[] = { dep0, dep1, dep2, dep3 };
    return jit_var_new_op(op, 4, dep);
}

/**
 * \brief Perform an ordinary or reinterpreting cast of a variable
 *
 * This function casts the variable \c index to a different type
 * \c target_type. When \c reinterpret is zero, this is like a C-style cast (i.e.,
 * <tt>new_value = (Type) old_value;</tt>). When \c reinterpret is nonzero, the
 * value is reinterpreted without converting the value (i.e.,
 * <tt>memcpy(&new_value, &old_value, sizeof(Type));</tt>), which requires that
 * source and target type are of the same size.
 */
extern JIT_EXPORT uint32_t jit_var_new_cast(uint32_t index,
                                            JIT_ENUM VarType target_type,
                                            int reinterpret);

/**
 * \brief Create a variable that refers to a memory region
 *
 * This function creates a 64 bit unsigned integer literal that refers to a
 * memory region. Optionally, if \c dep is nonzero, the created variable will
 * hold a reference to the variable \c dep until the pointer is destroyed,
 * which is useful when implementing operations that access global memory.
 *
 * A nonzero value should be passed to the \c write parameter if the pointer is
 * going to be used to perform write operations. Dr.Jit needs to know about
 * this to infer whether a future scatter operation to \c dep requires making a
 * backup copy first.
 */
extern JIT_EXPORT uint32_t jit_var_new_pointer(JIT_ENUM JitBackend backend,
                                               const void *value,
                                               uint32_t dep,
                                               int write);
/**
 * \brief Create a variable that reads from another variable
 *
 * This operation creates a variable that performs a <em>masked gather</em>
 * operation equivalent to <tt>mask ? source[index] : 0</tt>. The variable
 * \c index must be an integer array, and \c mask must be a boolean array.
 */
extern JIT_EXPORT uint32_t jit_var_new_gather(uint32_t source, uint32_t index,
                                              uint32_t mask);

#if defined(__cplusplus)
/// Reduction operations for \ref jit_var_new_scatter() \ref jit_reduce()
enum class ReduceOp : uint32_t { None, Add, Mul, Min, Max, And, Or, Count };
#else
enum ReduceOp {
    ReduceOpNone, ReduceOpAdd, ReduceOpMul, ReduceOpMin, ReduceOpMax,
    ReduceOpAnd, ReduceOpOr, ReduceOpCount
};
#endif

/**
 * \brief Schedule a scatter or atomic read-modify-write operation
 *
 * This operation schedules a side effect that will perform an operation
 * equivalent to <tt>if (mask) target[index] = value</tt>. The variable \c
 * index must be an integer array, and \c mask must be a boolean array.
 *
 * A direct write may not be safe (e.g. if unevaluated computation references
 * the array \c target). The function thus returns the index of a new array
 * (which may happen to be identical to \c target), whose external reference
 * count is increased by 1.
 *
 * For performance reasons, sequences involving multiple scatters to the same
 * array are exempt from this safety check, and these may furthermore execute
 * in arbitrary order due to the inherent parallelism. This is fine if the
 * written addresses do not overlap. Otherwise, explicit evaluation via
 * <tt>jit_var_eval(target)</tt> is necessary to ensure a fixed ordering.
 *
 * If <t>op != ReduceOp::None</tt>, an atomic read-modify-write operation will
 * be used instead of simply overwriting entries of 'target'.
 */
extern JIT_EXPORT uint32_t jit_var_new_scatter(uint32_t target, uint32_t value,
                                               uint32_t index, uint32_t mask,
                                               JIT_ENUM ReduceOp reduce_op);

/**
 * \brief Create an identical copy of the given variable
 *
 * This function creates an exact copy of the variable \c index and returns the
 * index of the copy, whose external reference count is initialized to 1.
 */
extern JIT_EXPORT uint32_t jit_var_copy(uint32_t index);


/**
 * Register an existing memory region as a variable in the JIT compiler, and
 * return its index. Its external reference count is initialized to \c 1.
 *
 * \param type
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param cuda
 *    Is this a CUDA variable?
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements (and *not* the size in bytes)
 *
 * \param free
 *    If free != 0, the JIT compiler will free the memory region via
 *    \ref jit_free() once its reference count reaches zero.
 *
 * \sa jit_var_mem_copy()
 */
extern JIT_EXPORT uint32_t jit_var_mem_map(JIT_ENUM JitBackend backend,
                                           JIT_ENUM VarType type, void *ptr,
                                           size_t size, int free);

/**
 * Copy a memory region onto the device and return its variable index. Its
 * external reference count is initialized to \c 1.
 *
 * \param atype
 *    Enumeration characterizing the "flavor" of the source memory.
 *
 * \param cuda
 *    Is this a CUDA variable?
 *
 * \param vtype
 *    Type of the variable to be created, see \ref VarType for details.
 *
 * \param ptr
 *    Point of the memory region
 *
 * \param size
 *    Number of elements (and *not* the size in bytes)
 *
 * \sa jit_var_mem_map()
 */
extern JIT_EXPORT uint32_t jit_var_mem_copy(JIT_ENUM JitBackend backend,
                                            JIT_ENUM AllocType atype,
                                            JIT_ENUM VarType vtype,
                                            const void *ptr,
                                            size_t size);

/// Increase the external reference count of a given variable
extern JIT_EXPORT void jit_var_inc_ref_ext_impl(uint32_t index) JIT_NOEXCEPT;

/// Decrease the external reference count of a given variable
extern JIT_EXPORT void jit_var_dec_ref_ext_impl(uint32_t index) JIT_NOEXCEPT;

#if defined(__GNUC__)
JIT_INLINE void jit_var_inc_ref_ext(uint32_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to jit_var_dec_ref_ext */
    if (!__builtin_constant_p(index) || index != 0)
        jit_var_inc_ref_ext_impl(index);
}
JIT_INLINE void jit_var_dec_ref_ext(uint32_t index) JIT_NOEXCEPT {
    if (!__builtin_constant_p(index) || index != 0)
        jit_var_dec_ref_ext_impl(index);
}
#else
#define jit_var_dec_ref_ext jit_var_dec_ref_ext_impl
#define jit_var_inc_ref_ext jit_var_inc_ref_ext_impl
#endif

/// Check if a variable with a given index exists
extern JIT_EXPORT int jit_var_exists(uint32_t index);

/// Query the a variable's internal reference count (used by the test suite)
extern JIT_EXPORT uint32_t jit_var_ref_int(uint32_t index);

/// Query the a variable's external reference count (used by the test suite)
extern JIT_EXPORT uint32_t jit_var_ref_ext(uint32_t index);

/// Query the pointer variable associated with a given variable
extern JIT_EXPORT void *jit_var_ptr(uint32_t index);

/// Query the size of a given variable
extern JIT_EXPORT size_t jit_var_size(uint32_t index);

/// Query the IR string of a given variable
extern JIT_EXPORT const char *jit_var_stmt(uint32_t index);

/// Query the type of a given variable
extern JIT_EXPORT JIT_ENUM VarType jit_var_type(uint32_t index);

/// Check if a variable is a constant literal
extern JIT_EXPORT int jit_var_is_literal(uint32_t index);

/// Check if a variable is evaluated
extern JIT_EXPORT int jit_var_is_evaluated(uint32_t index);

/// Check if a variable is a special placeholder value used to record computation
extern JIT_EXPORT int jit_var_is_placeholder(uint32_t index);

/**
 * \brief Resize a scalar variable to a new size
 *
 * This function takes a scalar variable as input and changes its size to \c
 * size, potentially creating a new copy in case something already depends on
 * \c index. The returned copy is symbolic form.
 *
 * The function increases the external reference count of the returned value.
 * When \c index is not a scalar variable and its size exactly matches \c size,
 * the function does nothing and just increases the external reference count of
 * \c index. Otherwise, it fails.
 */
extern JIT_EXPORT uint32_t jit_var_resize(uint32_t index, size_t size);

/**
 * \brief Asynchronously migrate a variable to a different flavor of memory
 *
 * Returns the resulting variable index and increases its external reference
 * count by one. When source and target type are identical, this function does
 * not perform a migration and simply returns the input index (though it
 * increases the reference count even in this case). When the source and target
 * types are different, the implementation schedules an asynchronous copy and
 * generates a new variable index.
 *
 * When both source & target are of type \ref AllocType::Device, and if the
 * current device (\ref jit_set_device()) does not match the device associated
 * with the allocation, a peer-to-peer migration is performed.
 */
extern JIT_EXPORT uint32_t jit_var_migrate(uint32_t index,
                                           JIT_ENUM AllocType type);

/// Query the current (or future, if unevaluated) allocation flavor of a variable
extern JIT_EXPORT JIT_ENUM AllocType jit_var_alloc_type(uint32_t index);

/// Query the device (or future, if not yet evaluated) associated with a variable
extern JIT_EXPORT int jit_var_device(uint32_t index);

/**
 * \brief Mark a variable as a scatter operation
 *
 * This function informs the JIT compiler that the variable 'index' has side
 * effects. It then steals an external reference, includes the variable in the
 * next kernel launch, and de-references it following execution.
 */
extern JIT_EXPORT void jit_var_mark_side_effect(uint32_t index);

/**
 * \brief Return a human-readable summary of the contents of a variable
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_str(uint32_t index);

/**
 * \brief Read a single element of a variable and write it to 'dst'
 *
 * This function fetches a single entry from the variable with \c index at
 * offset \c offset and writes it to the CPU output buffer \c dst.
 *
 * This function is convenient to spot-check entries of an array, but it should
 * never be used to extract complete array contents due to its low performance
 * (every read will be performed via an individual transaction). This operation
 * fully synchronizes the host CPU & device.
 */
extern JIT_EXPORT void jit_var_read(uint32_t index, size_t offset,
                                    void *dst);

/**
 * \brief Copy 'dst' to a single element of a variable
 *
 * This function implements the reverse of jit_var_read(). This function is
 * convenient for testing, and to change localized entries of an array, but it
 * should never be used to access the complete contents of an array due to its
 * low performance (every write will be performed via an individual
 * asynchronous transaction).
 *
 * A direct write may not be safe (e.g. if unevaluated computation references
 * the array \c index). The function thus returns the index of a new array
 * (which may happen to be identical to \c index), whose external reference
 * count is increased by 1.
 */
extern JIT_EXPORT uint32_t jit_var_write(uint32_t index, size_t offset,
                                         const void *src);

/**
 * \brief Print the specified variable contents from the kernel
 *
 * This function inserts a print statement directly into the kernel being
 * generated. Note that this may produce a very large volume of output,
 * and a nonzero \c mask parameter can be supplied to suppress it based
 * on condition.
 *
 * Example: <tt>jit_var_printf(JIT_ENUM JitBackend::CUDA, 0, "Hello world: %f\n", 1,
 * &my_variable_id);</tt>
 */
extern JIT_EXPORT void jit_var_printf(JIT_ENUM JitBackend backend,
                                      uint32_t mask, const char *fmt,
                                      uint32_t narg, const uint32_t *arg);

/**
 * \brief Create a new variable representing an array containing a specific
 * attribute associated with a specific domain in the registry.
 *
 * This function is very similar to jit_registry_attr_data but returns
 * a variable instead of a data pointer.
 *
 * \sa jit_registry_attr_data
 */
extern JIT_EXPORT uint32_t jit_var_registry_attr(JIT_ENUM JitBackend backend,
                                                 JIT_ENUM VarType type,
                                                 const char *domain,
                                                 const char *name);

// ====================================================================
//                 Kernel compilation and evaluation
// ====================================================================

/**
 * \brief Schedule a variable \c index for future evaluation via \ref jit_eval()
 *
 * Returns \c 1 if anything was scheduled, and \c 0 otherwise.
 */
extern JIT_EXPORT int jit_var_schedule(uint32_t index);

/**
 * \brief Evaluate the variable \c index right away, if it is unevaluated/dirty.
 *
 * Returns \c 1 if anything was evaluated, and \c 0 otherwise.
 */
extern JIT_EXPORT int jit_var_eval(uint32_t index);

/// Evaluate all scheduled computation
extern JIT_EXPORT void jit_eval();

/**
 * \brief Assign a callback function that is invoked when the variable is
 * evaluated or freed.
 *
 * The provided function should have the signature <tt>void callback(uint32_t
 * index, int free, void *callback_data)</tt>, where \c index is the variable
 * index, \c free == 0 indicates that the variable is evaluated, \c free == 1
 * indicates that it is freed, and \c callback_data is a user-specified value
 * that will additionally be supplied to the callback.
 *
 * Passing \c callback == nullptr will remove a previously set callback if any.
 */
extern JIT_EXPORT void
jit_var_set_callback(uint32_t index, void (*callback)(uint32_t, int, void *),
                     void *callback_data);

// ====================================================================
//      Functionality for debug output and GraphViz visualizations
// ====================================================================

/**
 * \brief Assign a descriptive label to a given variable
 *
 * The label is shown in the output of \ref jit_var_whos() and \ref
 * jit_var_graphviz()
 */
extern JIT_EXPORT uint32_t jit_var_set_label(uint32_t index, const char *label);

/// Query the descriptive label associated with a given variable
extern JIT_EXPORT const char *jit_var_label(uint32_t index);

/**
 * \brief Return a human-readable summary of registered variables
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_whos();

/**
 * \brief Return a GraphViz representation of registered variables and their
 * dependencies
 *
 * Note: the return value points into a static array, whose contents may be
 * changed by later calls to <tt>jit_*</tt> API functions. Either use it right
 * away or create a copy.
 */
extern JIT_EXPORT const char *jit_var_graphviz();

/**
 * \brief Push a string onto the label stack
 *
 * Dr.Jit maintains a per-thread label stack that is initially empty and
 * inactive. If values are pushed onto it, they will be used to initialize the
 * labels of any newly created variables.
 *
 * For example, if <tt>"prefix"</tt> and <tt>"prefix2"</tt> are pushed via this
 * function, any newly created variable \c index will be labeled
 * <tt>"prefix1/prefix2/"</tt>. A subsequent call to <tt>jit_var_set_label(index,
 * "name")</tt>; will change the label to <tt>"prefix1/prefix2/name"</tt>.
 *
 * This feature works hand-in-hand with \ref jit_var_graphviz(), which can
 * de-clutter large graph visualizations by drawing boxes around variables with
 * a common prefix.
 */
extern JIT_EXPORT void jit_prefix_push(JIT_ENUM JitBackend backend,
                                       const char *value);

/// Pop a string from the label stack
extern JIT_EXPORT void jit_prefix_pop(JIT_ENUM JitBackend backend);

// ====================================================================
//  JIT compiler status flags
// ====================================================================

/**
 * \brief Status flags to adjust/inspect the eagerness of the JIT compiler
 *
 * Certain Dr.Jit operations can operate in two different ways: they can be
 * executed at once, or they can be recorded to postpone evaluation to a later
 * point. The latter is generally more efficient because it enables fusion of
 * multiple operations that will then exchange information via registers
 * instead of global memory. The downside is that recording computation is
 * generally more complex/fragile and less suitable to interactive software
 * development (one e.g. cannot simply print array contents while something is
 * being recorded). The following list of flags can be used to control the
 * behavior of these features.
 *
 * The default set of flags is:
 *
 * <tt>ConstProp | ValueNumbering | LoopRecord | LoopOptimize |
 * VCallRecord | VCallOptimize | ADOptimize</tt>
 */
#if defined(__cplusplus)
enum class JitFlag : uint32_t {
    /// Constant propagation: don't generate code for arithmetic involving literals
    ConstProp = 1,

    /// Local value numbering (cheap form of common subexpression elimination)
    ValueNumbering = 2,

    /// Record loops instead of unrolling them into wavefronts
    LoopRecord = 4,

    /// Try to detect and remove unnecessary (constant/unreferenced) loop variables
    LoopOptimize = 8,

    /// Record virtual function calls instead of splitting them into many small kernel launches
    VCallRecord = 16,

    /// De-duplicate virtual function calls that produce the same code
    VCallDeduplicate = 32,

    /// Enable constant propagation and elide unnecessary function arguments
    VCallOptimize = 64,

    /**
     * \brief Inline calls if there is only a single instance? (off by default,
     * inlining can make kernels so large that they actually run slower in
     * CUDA/OptiX).
     */
    VCallInline = 128,

    /// Force execution through OptiX even if a kernel doesn't use ray tracing
    ForceOptiX = 256,

    /// Temporarily postpone evaluation of statements with side effects
    Recording = 512,

    /// Print the intermediate representation of generated programs
    PrintIR = 1024,

    /// Enable writing of the kernel history
    KernelHistory = 2048,

    /* Force synchronization after every kernel launch. This is useful to
       isolate crashes to a specific kernel, and to benchmark kernel runtime
       along with the KernelHistory feature. */
    LaunchBlocking = 4096,

    /// Exploit literal constants during AD (used in the Dr.Jit parent project)
    ADOptimize = 8192,

    /// Default flags
    Default = (uint32_t) ConstProp | (uint32_t) ValueNumbering |
              (uint32_t) LoopRecord | (uint32_t) LoopOptimize |
              (uint32_t) VCallRecord | (uint32_t) VCallDeduplicate |
              (uint32_t) VCallOptimize | (uint32_t) ADOptimize
};
#else
enum JitFlag {
    JitFlagConstProp           = 1,
    JitFlagValueNumbering      = 2,
    JitFlagLoopRecord          = 4,
    JitFlagLoopOptimize        = 8,
    JitFlagVCallRecord         = 16,
    JitFlagVCallDeduplicate    = 32,
    JitFlagVCallOptimize       = 64,
    JitFlagVCallInline         = 128,
    JitFlagForceOptiX          = 256,
    JitFlagRecording           = 512,
    JitFlagPrintIR             = 1024,
    JitFlagKernelHistory       = 2048,
    JitFlagLaunchBlocking      = 4096,
    JitFlagADOptimize          = 8192
};
#endif

/// Set the JIT compiler status flags (see \ref JitFlags)
extern JIT_EXPORT void jit_set_flags(uint32_t flags);

/// Retrieve the JIT compiler status flags (see \ref JitFlags)
extern JIT_EXPORT uint32_t jit_flags();

/// Selectively enables/disables flags
extern JIT_EXPORT void jit_set_flag(JIT_ENUM JitFlag flag, int enable);

/// Checks whether a given flag is active. Returns zero or one.
extern JIT_EXPORT int jit_flag(JIT_ENUM JitFlag flag);

// ====================================================================
//  Advanced JIT usage: recording loops, virtual function calls, etc.
// ====================================================================

/**
 * \brief Begin a recording session
 *
 * Dr.Jit can record virtual function calls and loops to preserve them 1:1 in
 * the generated code. This function indicates to Dr.Jit that the program is
 * starting to record computation. The function sets \ref JitFlag.Recording and
 * returns information that will later enable stopping or canceling a recording
 * session via \ref jit_record_end(). When recording a virtual function call,
 * the index of the last variable created to this point will be set to
 * \c vcall_bound_index (which should be a valid pointer). This will be used to
 * identify literal variables external to the virtual function when tracing.
 *
 * Recording sessions can be nested.
 */
extern JIT_EXPORT uint32_t jit_record_begin(JIT_ENUM JitBackend backend,
                                            uint32_t *vcall_bound_index);

/// Return a checkpoint within a recorded computation for resumption via jit_record_end
extern JIT_EXPORT uint32_t jit_record_checkpoint(JIT_ENUM JitBackend backend);

/**
 * \brief End a recording session
 *
 * The parameter \c state should be the return value from a prior call to \ref
 * jit_record_begin(). This function cleans internal data structures and
 * recovers the previous setting of the flag \ref JitFlag.Recording. When ending
 * the recording of a virtual function call, the parameter \c vcall_bound_index
 * should point to the value set during a prior call to \ref jit_record_begin().
 */
extern JIT_EXPORT void jit_record_end(JIT_ENUM JitBackend backend,
                                      uint32_t state,
                                      uint32_t *vcall_bound_index);

/**
 * \brief Wrap an input variable of a virtual function call before recording
 * computation
 *
 * Creates a copy of a virtual function call input argument. The copy has a
 * 'placeholder' bit set that propagates into any computation referencing it.
 * Placeholder variables trigger an error when the user tries to evaluate or
 * print them (these operations are not allowed in a recording session).
 */
extern JIT_EXPORT uint32_t jit_var_wrap_vcall(uint32_t index);

/**
 * \brief Wrap a loop state variable before recording computation
 *
 * Creates a copy of a loop state variable. The copy has a 'placeholder' bit
 * set that propagates into any computation referencing it. Placeholder
 * variables trigger an error when the user tries to evaluate or print them
 * (these operations are not allowed in a recording session).
 */
extern JIT_EXPORT uint32_t jit_var_wrap_loop(uint32_t index, uint32_t cond, uint32_t size);

/**
 * \brief Inform the JIT compiler about the current instance while
 * o virtual function calls
 *
 * Following a call to \ref jit_vcall_set_self(), the JIT compiler will
 * intercept constant literals referring to the instance ID 'value'. In this
 * case, it will return the variable ID 'index'.
 *
 * This feature is crucial to avoid merging instance IDs into generated code.
 */
extern JIT_EXPORT void jit_vcall_set_self(JIT_ENUM JitBackend backend,
                                          uint32_t value, uint32_t index);

/// Query the information set via \ref jit_vcall_set_self
extern JIT_EXPORT void jit_vcall_self(JIT_ENUM JitBackend backend,
                                      uint32_t *value, uint32_t *index);

/**
 * \brief Record a virtual function call
 *
 * This function inserts a virtual function call into into the computation
 * graph. This works like a giant demultiplexer-multiplexer pair: depending on
 * the value of the \c self argument, information will flow through one of \c
 * n_inst computation graphs that are provided via the `out_nested` argument.
 *
 * \param name
 *     A descriptive name that will be used to label various created nodes in
 *     the computation graph.
 *
 * \param self
 *     Instance index (a variable of type <tt>VarType::UInt32</tt>), where
 *     0 indicates that the function call should be masked. All outputs
 *     will be zero-valued in that case.
 *
 * \param n_inst
 *     The number of instances (must be >= 1)
 *
 * \param n_in
 *     The number of input variables
 *
 * \param in
 *     Pointer to an array of input variable indices of size \c n_in
 *
 * \param n_out_nested
 *     Total number of output variables, where <tt>n_out_nested = (# of
 *     outputs) * n_inst</tt>
 *
 * \param out_nested
 *     Pointer to an array of output variable indices of size \c n_out_nested
 *
 * \param se_offset
 *     Indicates the size of the side effects queue (obtained from \ref
 *     jit_side_effects_scheduled()) before each instance call, and
 *     after the last one. <tt>n_inst + 1</tt> entries.
 *
 * \param out
 *     The final output variables representing the result of the operation
 *     are written into this argument (size <tt>n_out_nested / n_inst</tt>)
 */
extern JIT_EXPORT uint32_t jit_var_vcall(const char *name, uint32_t self,
                                         uint32_t mask, uint32_t n_inst,
                                         const uint32_t *inst_id,
                                         uint32_t n_in, const uint32_t *in,
                                         uint32_t n_out_nested,
                                         const uint32_t *out_nested,
                                         const uint32_t *se_offset, uint32_t *out);

/**
 * \brief Initialize a set of loop state variables
 *
 * When recording an Dr.Jit loop
 *
 * \return A variable index representing the start of the loop. It must be
 * passed to the \c loop_start argument of \ref jit_var_loop()
 */
extern JIT_EXPORT uint32_t jit_var_loop_init(size_t n_indices,
                                             uint32_t **indices);

extern JIT_EXPORT uint32_t jit_var_loop_cond(uint32_t loop_init,
                                             uint32_t cond,
                                             size_t n_indices,
                                             uint32_t **indices);

extern JIT_EXPORT uint32_t jit_var_loop(const char *name, uint32_t loop_init,
                                        uint32_t loop_cond, size_t n_indices,
                                        uint32_t *indices_in,
                                        uint32_t **indices, uint32_t checkpoint,
                                        int first_round);

/**
 * \brief Pushes a new mask variable onto the mask stack
 *
 * In advanced usage of Dr.Jit (e.g. recorded loops, virtual function calls,
 * etc.), it may be necessary to mask scatter and gather operations to prevent
 * undefined behavior and crashes. This function can be used to push a mask
 * onto a mask stack.  While on the stack, Dr.Jit will hold an internal
 * reference to \c index to keep it from being freed.
 */
extern JIT_EXPORT void jit_var_mask_push(JIT_ENUM JitBackend backend, uint32_t index);

/// Pop the mask stack
extern JIT_EXPORT void jit_var_mask_pop(JIT_ENUM JitBackend backend);

/**
 * \brief Return the top entry of the mask stack and increase its external
 * reference count. Returns zero when the stack is empty.
 */
extern JIT_EXPORT uint32_t jit_var_mask_peek(JIT_ENUM JitBackend backend);

/// Return the default mask for a wavefront of the given \c size
extern JIT_EXPORT uint32_t jit_var_mask_default(JIT_ENUM JitBackend backend,
                                                uint32_t size);

/**
 * \brief Combine the given mask 'index' with the mask stack
 *
 * On the LLVM backend, a default mask will be created when the mask stack is empty.
 * The \c size parameter determines the size of the associated wavefront.
 */
extern JIT_EXPORT uint32_t jit_var_mask_apply(uint32_t index, uint32_t size);

// ====================================================================
//                          Horizontal reductions
// ====================================================================

/// Reduce (And) a boolean array to a single value, synchronizes.
extern JIT_EXPORT int jit_var_all(uint32_t index);

/// Reduce (Or) a boolean array to a single value, synchronizes.
extern JIT_EXPORT int jit_var_any(uint32_t index);

/// Reduce a variable to a single value
extern JIT_EXPORT uint32_t jit_var_reduce(uint32_t index, JIT_ENUM ReduceOp reduce_op);

// ====================================================================
//  Assortment of tuned kernels for initialization, reductions, etc.
// ====================================================================

/**
 * \brief Fill a device memory region with constants of a given type
 *
 * This function writes \c size values of size \c isize to the output array \c
 * ptr. The specific value is taken from \c src, which must be a CPU pointer to
 * a single int, float, double, etc. (\c isize can be 1, 2, 4, or 8).
 * Runs asynchronously.
 */
extern JIT_EXPORT void jit_memset_async(JIT_ENUM JitBackend backend, void *ptr, uint32_t size,
                                        uint32_t isize, const void *src);

/// Perform a synchronous copy operation
extern JIT_EXPORT void jit_memcpy(JIT_ENUM JitBackend backend, void *dst, const void *src, size_t size);

/// Perform an asynchronous copy operation
extern JIT_EXPORT void jit_memcpy_async(JIT_ENUM JitBackend backend, void *dst, const void *src,
                                        size_t size);

/**
 * \brief Reduce the given array to a single value
 *
 * This operation reads \c size values of type \type from the input array \c
 * ptr and performs an specified operation (e.g., addition, multiplication,
 * etc.) to combine them into a single value that is written to the device
 * variable \c out.
 *
 * Runs asynchronously.
 */
extern JIT_EXPORT void jit_reduce(JIT_ENUM JitBackend backend, JIT_ENUM VarType type,
                                  JIT_ENUM ReduceOp rtype,
                                  const void *ptr, uint32_t size, void *out);

/**
 * \brief Perform an exclusive scan / prefix sum over an unsigned 32 bit integer
 * array
 *
 * If desired, the scan can be performed in-place (i.e. <tt>in == out</tt>).
 * Note that the CUDA implementation will round up \c size to the maximum of
 * the following three values for performance reasons:
 *
 * - the value 4,
 * - the next highest power of two (when size <= 4096),
 * - the next highest multiple of 2K (when size > 4096),
 *
 * For this reason, the the supplied memory regions must be sufficiently large
 * to avoid both out-of-bounds reads and writes. This is not an issue for
 * memory obtained using \ref jit_malloc(), which internally rounds
 * allocations to the next largest power of two and enforces a 64 byte minimum
 * allocation size.
 *
 * Runs asynchronously.
 */
extern JIT_EXPORT void jit_scan_u32(JIT_ENUM JitBackend backend, const uint32_t *in,
                                    uint32_t size, uint32_t *out);

/**
 * \brief Compress a mask into a list of nonzero indices
 *
 * This function takes an 8-bit mask array \c in with size \c size as input,
 * whose entries are required to equal either zero or one. It then writes the
 * indices of nonzero entries to \c out (in increasing order), and it
 * furthermore returns the total number of nonzero mask entries.
 *
 * The internals resemble \ref jit_scan_u32(), and the CUDA implementation may
 * similarly access regions beyond the end of \c in and \c out.
 *
 * This function internally performs a synchronization step.
 */
extern JIT_EXPORT uint32_t jit_compress(JIT_ENUM JitBackend backend, const uint8_t *in,
                                        uint32_t size, uint32_t *out);


/**
 * \brief Compute a permutation to reorder an integer array into a sorted
 * configuration
 *
 * Given an unsigned integer array \c values of size \c size with entries in
 * the range <tt>0 .. bucket_count - 1</tt>, compute a permutation that can be
 * used to reorder the inputs into a sorted (but non-stable) configuration.
 * When <tt>bucket_count</tt> is relatively small (e.g. < 10K), the
 * implementation is much more efficient than the alternative of actually
 * sorting the array.
 *
 * \param perm
 *     The permutation is written to \c perm, which must point to a buffer in
 *     device memory having size <tt>size * sizeof(uint32_t)</tt>.
 *
 * \param offsets
 *     When \c offset is non-NULL, the parameter should point to a host (LLVM)
 *     or host-pinned (CUDA) memory region with a size of at least
 *     <tt>(bucket_count * 4 + 1) * sizeof(uint32_t)<tt> bytes that will be
 *     used to record the details of non-empty buckets. It will contain
 *     quadruples <tt>(index, start, size, unused)<tt> where \c index is the
 *     bucket index, and \c start and \c end specify the associated entries of
 *     the \c perm array. The 'unused' field is padding for 16 byte alignment.
 *
 * \return
 *     When \c offsets != NULL, the function returns the number of unique
 *     values found in \c values. Otherwise, it returns zero.
 */
extern JIT_EXPORT uint32_t jit_mkperm(JIT_ENUM JitBackend backend, const uint32_t *values,
                                      uint32_t size, uint32_t bucket_count,
                                      uint32_t *perm, uint32_t *offsets);

/// Helper data structure for vector method calls, see \ref jit_var_vcall()
struct VCallBucket {
    /// Resolved pointer address associated with this bucket
    void *ptr;

    /// Variable index of a uint32 array storing a partial permutation
    uint32_t index;

    /// Original instance ID
    uint32_t id;
};

/**
 * \brief Compute a permutation to reorder an array of registered pointers
 * in preparation for a vectorized method call
 *
 * This function expects an array of integers, whose entries correspond to
 * pointers that have previously been registered by calling \ref
 * jit_registry_put() with domain \c domain. It then invokes \ref jitc_mkperm()
 * to compute a permutation that reorders the array into coherent buckets. The
 * buckets are returned using an array of type \ref VCallBucket, which contains
 * both the resolved pointer address (obtained via \ref
 * jit_registry_get_ptr()) and the variable index of an unsigned 32 bit array
 * containing the corresponding entries of the input array. The total number of
 * buckets is returned via the \c bucket_count_out argument.
 *
 * The memory region accessible via the \c VCallBucket pointer will remain
 * accessible until the variable \c index is itself freed (i.e. when its
 * internal and external reference counts both become equal to zero). Until
 * then, additional calls to \ref jit_var_vcall() will return the previously
 * computed result. This is an important optimization in situations where
 * multiple vector function calls are executed on the same set of instances.
 */
extern JIT_EXPORT struct VCallBucket *
jit_var_vcall_reduce(JIT_ENUM JitBackend backend, const char *domain,
                     uint32_t index, uint32_t *bucket_count_out);

/**
 * \brief Replicate individual input elements across larger blocks
 *
 * This function copies each element of the input array \c to a contiguous
 * block of size \c block_size in the output array \c out. For example, <tt>a,
 * b, c</tt> turns into <tt>a, a, b, b, c, c</tt> when the \c block_size is set
 * to \c 2. The input array must contain <tt>size</tt> elements, and the output
 * array must have space for <tt>size * block_size</tt> elements.
 */
extern JIT_EXPORT void jit_block_copy(JIT_ENUM JitBackend backend, JIT_ENUM VarType type,
                                      const void *in, void *out,
                                      uint32_t size, uint32_t block_size);

/**
 * \brief Sum over elements within blocks
 *
 * This function adds all elements of contiguous blocks of size \c block_size
 * in the input array \c in and writes them to \c out. For example, <tt>a, b,
 * c, d, e, f</tt> turns into <tt>a+b, c+d, e+f</tt> when the \c block_size is
 * set to \c 2. The input array must contain <tt>size * block_size</tt> elements,
 * and the output array must have space for <tt>size</tt> elements.
 */
extern JIT_EXPORT void jit_block_sum(JIT_ENUM JitBackend backend, JIT_ENUM VarType type,
                                     const void *in, void *out, uint32_t size,
                                     uint32_t block_size);
/**
 * \brief Insert a function call to a ray tracing functor into the LLVM program
 *
 * The \c args list should contain a list of variable indices corresponding to
 * the 13 required function arguments
 * - active_mask (32 bit integer with '-1' for active, and '0' for inactive)
 * - ox, oy, oz
 * - tmin
 * - dx, dy, dz
 * - time
 * - tfar
 * - mask, id, flags
 * </tt>.
 */
extern JIT_EXPORT void jit_llvm_ray_trace(uint32_t func, uint32_t scene,
                                          int shadow_ray, const uint32_t *in,
                                          uint32_t *out);

/**
 * \brief Set a new scope identifier to limit the effect of common
 * subexpression elimination
 *
 * Dr.Jit implements a very basic approximation of common subexpression
 * elimination based on local value numbering (LVN): an attempt to create a
 * variable, whose statement and dependencies match a previously created
 * variable will sidestep creation and instead reuse the old variable via
 * reference counting. However, this approach of collapsing variables does not
 * play well with more advanced constructs like recorded loops, where variables
 * in separate scopes should be kept apart.
 *
 * This function sets a unique scope identifier (a simple 32 bit integer)
 * isolate the effects of this optimization.
 */
extern JIT_EXPORT void jit_new_cse_scope(JIT_ENUM JitBackend backend);

/// Queries the CSE scope identifier (see \ref jit_new_cse_scope())
extern JIT_EXPORT uint32_t jit_cse_scope(JIT_ENUM JitBackend backend);

/// Manually sets a CSE scope identifier (see \ref jit_new_cse_scope())
extern JIT_EXPORT void jit_set_cse_scope(JIT_ENUM JitBackend backend, uint32_t domain);

// ====================================================================
//                            Kernel History
// ====================================================================

/**
 * \brief List of kernel types that can be launched by Dr.Jit
 *
 * Dr.Jit sometimes launches kernels that are not generated by the JIT itself
 * (e.g. precompiled CUDA kernels for horizontal reductions). The kernel history
 * identifies them using a field of type \c KernelType.
 */
enum KernelType : uint32_t {
    /// JIT-compiled kernel
    JIT,

    /// Kernel responsible for a horizontal reduction operation (e.g. hsum)
    Reduce,

    /// Permutation kernel produced by \ref jit_mkperm()
    VCallReduce,

    /// Any other kernel
    Other
};

/// Data structure for preserving kernel launch information (debugging, testing)
struct KernelHistoryEntry {
    /// Jit backend, for which the kernel was compiled
    JitBackend backend;

    /// Kernel type
    KernelType type;

    /// Stores the low/high 64 bits of the 128-bit hash kernel identifier
    uint64_t hash[2];

    /// Copy of the kernel IR string buffer
    char *ir;

    /// Does the kernel contain any OptiX (ray tracing) operations?
    int uses_optix;

    /// Whether the kernel was reused from the kernel cache
    int cache_hit;

    /// Whether the kernel was loaded from the cache on disk
    int cache_disk;

    /// Launch width / number of array entries that were processed
    uint32_t size;

    /// Number of input arrays
    uint32_t input_count;

    /// Number of output arrays + side effects
    uint32_t output_count;

    /// Number of IR operations
    uint32_t operation_count;

    /// Time (ms) spent generating the kernel intermediate representation
    float codegen_time;

    /// Time (ms) spent compiling the kernel (\c 0 if \c cache_hit is \c true)
    float backend_time;

    /// Time (ms) spent executing the kernel
    float execution_time;

    // Dr.Jit internal portion, will be cleared by jit_kernel_history()
    // ================================================================

    /// CUDA events for measuring the runtime of the kernel
    void *event_start, *event_end;

    /// nanothread task handle
    void *task;
};

/// Clear the kernel history
extern JIT_EXPORT void jit_kernel_history_clear();

/**
 * \brief Return a pointer to the first entry of the kernel history
 *
 * When \c JitFlag.KernelHistory is set to \c true, every kernel launch will add
 * and entry in the history which can be accessed via this function.
 *
 * The caller is responsible for freeing the returned data structure via the
 * following construction:
 *
 *     KernelHistoryEntry *data = jit_kernel_history();
 *     KernelHistoryEntry *e = data;
 *     while (e->ir) {
 *         free(e->ir);
 *         e++;
 *     }
 *     free(data);
 *
 * When the kernel history is empty, the function will return a null pointer.
 * Otherwise, the size of the kernel history can be inferred by iterating over
 * the entries until one reaches a entry with an invalid \c backend (e.g.
 * initialized to \c 0).
 */
extern JIT_EXPORT struct KernelHistoryEntry *jit_kernel_history();

#if defined(__cplusplus)
}

static inline void jit_init(JIT_ENUM JitBackend backend)       { jit_init((uint32_t) backend); }
static inline void jit_init_async(JIT_ENUM JitBackend backend) { jit_init((uint32_t) backend); }
#endif
