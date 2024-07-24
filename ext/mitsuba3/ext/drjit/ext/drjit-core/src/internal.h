/*
    src/internal.h -- Central data structure definitions

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "malloc.h"
#include "cuda_api.h"
#include "llvm_api.h"
#include "alloc.h"
#include "io.h"
#include <deque>
#include <string.h>
#include <inttypes.h>
#include <nanothread/nanothread.h>

/// Number of entries to process per work unit in the parallel LLVM backend
#define DRJIT_POOL_BLOCK_SIZE 16384

/// Can't pass more than 4096 bytes of parameter data to a CUDA kernel
#define DRJIT_CUDA_ARG_LIMIT 512

#define DRJIT_PTR "<0x%" PRIxPTR ">"

static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel InfoSym = LogLevel::InfoSym;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

#pragma pack(push, 1)

/// Central variable data structure, which represents an assignment in SSA form
struct Variable {
    #if defined(__GNUC__)
    #  pragma GCC diagnostic push
    #  if defined(__has_warning)
    #    if __has_warning("-Wclass-memaccess")
    #      pragma GCC diagnostic ignored "-Wclass-memaccess"
    #    endif
    #  else
    #    pragma GCC diagnostic ignored "-Wclass-memaccess"
    #  endif
    #endif

    /// Zero-initialize by default
    Variable() { memset(this, 0, sizeof(Variable)); }

    #if defined(__GNUC__)
    #  pragma GCC diagnostic pop
    #endif

    // ===================   References, reference counts   ===================

    /// External reference count (by application using DrJit)
    uint64_t ref_count_ext : 24;

    /// Internal reference count (dependencies within computation graph)
    uint64_t ref_count_int : 24;

    /// Number of queued side effects
    uint64_t ref_count_se : 16;

    /// Up to 4 dependencies of this instruction (further possible via 'extra')
    uint32_t dep[4];

    // =============  Encoded instruction / data pointer / size   =============

    union {
        // If literal == 0: Intermediate language (PTX, LLVM IR) statement
        char *stmt;

        // If literal == 1, floating point/integer value (reinterpreted as u64)
        uint64_t value;
    };

    /// Pointer to device memory, equals NULL if the variable is not evaluated
    void *data;

    /// Number of entries
    uint32_t size;

    // ============  Essential flags, basic block ID (-> LVN key)  ============

    /// Data type of this variable
    uint32_t type : 4;

    /// Backend associated with this variable
    uint32_t backend : 2;

    /// Is this a pointer variable that is used to write to some array?
    uint32_t write_ptr : 1;

    /// Does this variable store a number literal?
    uint32_t literal : 1;

    /// Free the 'stmt' variables at destruction time?
    uint32_t free_stmt : 1;

    /// Basic block ID of this variable
    uint32_t cse_scope : 23;

    // ========================  Miscellaneous flags  =========================

    /// Don't deallocate 'data' when this variable is destructed?
    uint32_t retain_data : 1;

    /// Does evaluation of this variable have side effects on other variables?
    uint32_t side_effect : 1;

    /// Is this a placeholder variable used to record arithmetic symbolically?
    uint32_t placeholder : 1;

    /// Is this a placeholder variable used to record arithmetic symbolically?
    uint32_t vcall_iface : 1;

    /// Is this variable associated with extra information?
    uint32_t extra : 1;

    /// Do the variable contents have irregular alignment? (e.g. due to jitc_var_mem_map())
    uint32_t unaligned : 1;

    /// Does this variable perform an OptiX operation?
    uint32_t optix : 1;

    // ===============   Temporarily used during jitc_eval()   ================

    /// Argument type
    uint32_t param_type : 2;

    /// Is this variable marked as an output?
    uint32_t output_flag : 1;

    /// Offset of the argument in the list of kernel parameters
    uint32_t param_offset : 22;

    /// Register index
    uint32_t reg_index;
};

/// Abbreviated version of the Variable data structure
struct VariableKey {
    uint32_t dep[4];
    uint32_t size;
    uint32_t backend     : 2;
    uint32_t type        : 4;
    uint32_t write_ptr   : 1;
    uint32_t literal     : 1;
    uint32_t free_stmt   : 1;
    uint32_t cse_scope   : 23;

    union {
        char *stmt;
        uint64_t value;
    };

    VariableKey(const Variable &v) {
        memcpy(dep, v.dep, sizeof(uint32_t) * 4);
        size = v.size;
        type = v.type;
        backend = v.backend;
        write_ptr = v.write_ptr;
        literal = v.literal;
        cse_scope = v.cse_scope;
        free_stmt = v.free_stmt;
        if (literal)
            value = v.value;
        else
            stmt = v.stmt;
    }

    bool operator==(const VariableKey &v) const {
        if (memcmp(this, &v, 6 * sizeof(uint32_t)) != 0)
            return false;
        if (literal)
            return value == v.value;
        else if (!free_stmt)
            return stmt == v.stmt;
        else
            return stmt == v.stmt || strcmp(stmt, v.stmt) == 0;
    }
};

#pragma pack(pop)

/// Helper class to hash VariableKey instances
struct VariableKeyHasher {
    size_t operator()(const VariableKey &k) const {
        uint64_t hash_1;
        if (k.literal)
            hash_1 = k.value;
        else if (!k.free_stmt)
            hash_1 = (uintptr_t) k.stmt;
        else
            hash_1 = hash_str(k.stmt);

        return hash(k.dep, 6 * sizeof(uint32_t), hash_1);
    }
};

/// Cache data structure for common subexpression elimination
using CSECache =
    tsl::robin_map<VariableKey, uint32_t, VariableKeyHasher,
                   std::equal_to<VariableKey>,
                   std::allocator<std::pair<VariableKey, uint32_t>>,
                   /* StoreHash = */ true>;

/// Caches basic information about a CUDA device
struct Device {
    // CUDA device context
    CUcontext context;

    /// CUDA device ID
    int id;

    /// Device compute capability (major * 10 + minor)
    int compute_capability;

    /// Number of SMs
    uint32_t num_sm;

    /// Max. bytes of shared memory per SM
    uint32_t shared_memory_bytes;

    // Support for stream ordered memory allocations (async alloc/free)
    bool memory_pool_support;

    /// Is this a WDDM-style driver where long launches will freeze the OS graphics?
    bool wddm_driver;

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX device context
    void *optix_context = nullptr;
#endif

    /// Compute a good configuration for a grid-stride loop
    void get_launch_config(uint32_t *blocks_out, uint32_t *threads_out,
                           uint32_t size, uint32_t max_threads = 1024,
                           uint32_t max_blocks_per_sm = 4) const {
        uint32_t blocks_avail  = (size + max_threads - 1) / max_threads;

        uint32_t blocks;
        if (blocks_avail < num_sm) {
            // Not enough work for 1 full wave
            blocks = blocks_avail;
        } else {
            // Don't produce more than 4 blocks per SM
            uint32_t blocks_per_sm = blocks_avail / num_sm;
            if (blocks_per_sm > max_blocks_per_sm)
                blocks_per_sm = max_blocks_per_sm;
            blocks = blocks_per_sm * num_sm;
        }

        uint32_t threads = max_threads;
        if (blocks <= 1 && size < max_threads)
            threads = size;

        if (blocks_out)
            *blocks_out  = blocks;

        if (threads_out)
            *threads_out = threads;
    }
};

/// Keeps track of asynchronous deallocations via jitc_free()
struct ReleaseChain {
    AllocInfoMap entries;

    /// Pointer to next linked list entry
    ReleaseChain *next = nullptr;
};

/// A few forward declarations for OptiX
#if defined(DRJIT_ENABLE_OPTIX)
using OptixProgramGroup = void *;

struct OptixShaderBindingTable {
    void* raygenRecord;
    void* exceptionRecord;
    void* missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    void* hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    void* callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
};

struct OptixPipelineData {
    OptixPipelineCompileOptions compile_options;
    OptixModule module;
    std::vector<OptixProgramGroup> program_groups;
};
#endif

/// Represents a single stream of a parallel communication
struct ThreadState {
    /// Backend type
    JitBackend backend;

    /**
     * Memory regions that were freed via jitc_free(), but which might still be
     * used by a currently running kernel. They will be safe to re-use once the
     * currently running kernel has finished.
     */
    ReleaseChain *release_chain = nullptr;

    /**
     * List of variables that are scheduled for evaluation (via
     * jitc_var_schedule()) that will take place at the next call to jitc_eval().
     */
    std::vector<uint32_t> scheduled;

    /**
     * List of special variables of type VarType::Void, whose evaluation will
     * cause side effects that modify other variables. They will be evaluated
     * at the next call to jitc_eval().
     */
    std::vector<uint32_t> side_effects;

    /// When recording loops or virtual function calls, side effects go here.
    std::vector<uint32_t> side_effects_recorded;

    /**
     * Stack of variable indices indicating the list of active SIMD lanes.
     * This is used to constrain the behavior of gather/scatter operations.
     */
    std::vector<uint32_t> mask_stack;

    /// Stack of variable name prefixes, mainly useful for GraphViz exports
    std::vector<char *> prefix_stack;

    /// Combined version of the elements of 'prefix_stack'
    char *prefix = nullptr;

    /// Index used to isolate CSE from other parts of the program
    uint32_t cse_scope = 0;

    /// Registry index of the 'self' pointer of the vcall being recorded
    uint32_t vcall_self_value = 0;

    /// .. and the JIT variable that it will be mapped to
    uint32_t vcall_self_index = 0;

    /// Index of the last variable created before recording the current vcall
    uint32_t vcall_bound_index = 0;

    /// ---------------------------- LLVM-specific ----------------------------

    // Currently active task within the thread pool
    Task *task = nullptr;

    /// ---------------------------- CUDA-specific ----------------------------

    /// Redundant copy of the device context
    CUcontext context = nullptr;

    /// Associated CUDA stream handle
    CUstream stream = nullptr;

    /// A CUDA event for synchronization purposes
    CUevent event = nullptr;

    /**
     * \brief DrJit device ID associated with this device
     *
     * This value may differ from the CUDA device ID if the machine contains
     * CUDA devices that are incompatible with DrJit.
     *
     * Equals -1 for LLVM ThreadState instances.
     */
    int device = 0;

    /// Targeted compute compatibility
    uint32_t compute_capability = 50;

    /// Targeted PTX version (major * 10 + minor)
    uint32_t ptx_version = 60;

#if defined(DRJIT_ENABLE_OPTIX)
    /// OptiX pipeline associated with the next kernel launch
    OptixPipelineData *optix_pipeline;
    OptixShaderBindingTable *optix_sbt;
#endif
};

/// Maps from variable ID to a Variable instance
using VariableMap =
    tsl::robin_map<uint32_t, Variable, UInt32Hasher,
                   std::equal_to<uint32_t>,
                   aligned_allocator<std::pair<uint32_t, Variable>, 64>,
                   /* StoreHash = */ false>;

/// Key data structure for kernel source code & device ID
struct KernelKey {
    char *str = nullptr;
    int device = 0;
    uint64_t flags = 0;

    KernelKey(char *str, int device, uint64_t flags) : str(str), device(device), flags(flags) { }

    bool operator==(const KernelKey &k) const {
        return strcmp(k.str, str) == 0 && device == k.device && flags == k.flags;
    }
};

/// Helper class to hash KernelKey instances
struct KernelHash {
    size_t operator()(const KernelKey &k) const {
        return compute_hash(hash_kernel(k.str).high64, k.device, k.flags);
    }

    static size_t compute_hash(size_t kernel_hash, int device, uint64_t flags) {
        size_t hash = kernel_hash;
        hash_combine(hash, (size_t) flags + size_t(device + 1));
        return hash;
    }
};

/// Data structure, which maps from kernel source code to compiled kernels
using KernelCache =
    tsl::robin_map<KernelKey, Kernel, KernelHash, std::equal_to<KernelKey>,
                   std::allocator<std::pair<KernelKey, Kernel>>,
                   /* StoreHash = */ true>;

/// Data structure to store a history of the launched kernels
struct KernelHistory {
    KernelHistory();
    ~KernelHistory();

    void append(const KernelHistoryEntry &entry);
    KernelHistoryEntry *get();
    void clear();

private:
    KernelHistoryEntry *m_data;
    size_t m_size;
    size_t m_capacity;
};

// Key associated with a pointer registered in DrJit's pointer registry
struct RegistryKey {
    const char *domain;
    uint32_t id;
    RegistryKey(const char *domain, uint32_t id) : domain(domain), id(id) { }

    bool operator==(const RegistryKey &k) const {
        return id == k.id && strcmp(domain, k.domain) == 0;
    }

    /// Helper class to hash RegistryKey instances
    struct Hasher {
        size_t operator()(const RegistryKey &k) const {
            return hash_str(k.domain, k.id);
        }
    };
};

struct AttributeKey {
    const char *domain;
    const char *name;

    AttributeKey(const char *domain, const char *name) : domain(domain), name(name) { }

    bool operator==(const AttributeKey &k) const {
        return strcmp(domain, k.domain) == 0 && strcmp(name, k.name) == 0;
    }

    /// Helper class to hash AttributeKey instances
    struct Hasher {
        size_t operator()(const AttributeKey &k) const {
            return hash_str(k.domain, hash_str(k.name));
        }
    };
};

struct AttributeValue {
    uint32_t isize = 0;
    uint32_t count = 0;
    void *ptr = nullptr;
};

struct Registry {
    using RegistryFwdMap =
        tsl::robin_map<RegistryKey, void *, RegistryKey::Hasher,
                       std::equal_to<RegistryKey>,
                       std::allocator<std::pair<RegistryKey, void *>>,
                       /* StoreHash = */ true>;

    using RegistryRevMap = tsl::robin_pg_map<const void *, RegistryKey>;

    using AttributeMap =
        tsl::robin_map<AttributeKey, AttributeValue, AttributeKey::Hasher,
                       std::equal_to<AttributeKey>,
                       std::allocator<std::pair<AttributeKey, AttributeValue>>,
                       /* StoreHash = */ true>;

    /// Two-way mapping that can be used to associate pointers with unique 32 bit IDs
    RegistryFwdMap fwd;
    RegistryRevMap rev;

    /// Per-pointer attributes provided by the pointer registry
    AttributeMap attributes;
};

struct Extra {
    /// Optional descriptive label
    char *label = nullptr;

    /// Additional references
    uint32_t *dep = nullptr;
    uint32_t n_dep = 0;

    /// Callback to be invoked when the variable is evaluated/deallocated
    void (*callback)(uint32_t, int, void *) = nullptr;
    void *callback_data = nullptr;
    bool callback_internal = false;

    /// Bucket decomposition for virtual function calls
    uint32_t vcall_bucket_count = 0;
    VCallBucket *vcall_buckets = nullptr;

    /// Code generation callback
    void (*assemble)(const Variable *v, const Extra &extra) = nullptr;
};

using ExtraMap = tsl::robin_map<uint32_t, Extra, UInt32Hasher>;

/// Records the full JIT compiler state (most frequently two used entries at top)
struct State {
    /// Must be held to access members
    Lock lock;

    /// Must be held to access 'stream->release_chain' and 'state.alloc_free'
    Lock malloc_lock;

    /// Stores the mapping from variable indices to variables
    VariableMap variables;

    /// Unique counter to create new CSE domains
    uint32_t cse_scope_ctr = 0;

    /// Maps from a key characterizing a variable to its index
    CSECache cse_cache;

    /// Must be held to execute jitc_eval()
    Lock eval_lock;

    /// Log level (stderr)
    LogLevel log_level_stderr = LogLevel::Info;

    /// Log level (callback)
    LogLevel log_level_callback = LogLevel::Disable;

    /// Callback for log messages
    LogCallback log_callback = nullptr;

    /// Bit-mask of successfully initialized backends
    uint32_t backends = 0;

    /// Available devices and their CUDA IDs
    std::vector<Device> devices;

    /// State associated with each DrJit thread
    std::vector<ThreadState *> tss;

    /// Pointer registries for LLVM and CUDA backends
    Registry registry_cpu, registry_gpu;

    /// Map of currently allocated memory regions
    AllocUsedMap alloc_used;

    /// Map of currently unused memory regions
    AllocInfoMap alloc_free;

    /// Keep track of current memory usage and a maximum watermark
    size_t alloc_usage    [(int) AllocType::Count] { 0 },
           alloc_allocated[(int) AllocType::Count] { 0 },
           alloc_watermark[(int) AllocType::Count] { 0 };

    /// Keep track of the number of created JIT variables
    uint32_t variable_watermark = 0;

    /// Maps from variable ID to extra information for a fraction of variables
    ExtraMap extra;

    /// Current variable index
    uint32_t variable_index = 1;

    /// Limit the output of jit_var_str()?
    uint32_t print_limit = 20;

    /// Statistics on kernel launches
    size_t kernel_hard_misses = 0;
    size_t kernel_soft_misses = 0;
    size_t kernel_hits = 0;
    size_t kernel_launches = 0;

    /// Cache of previously compiled kernels
    KernelCache kernel_cache;

    /// Kernel launch history
    KernelHistory kernel_history = KernelHistory();

#if defined(DRJIT_ENABLE_OPTIX)
    /// Default OptiX pipeline for testcases etc.
    OptixPipelineData *optix_default_pipeline = nullptr;
    /// Default OptiX Shader Binding Table for testcases etc.
    OptixShaderBindingTable *optix_default_sbt = nullptr;
    /// Index of the JIT variable handling the lifetime of the default Optix SBT
    uint32_t optix_default_sbt_index = 0;
#endif

    /// Return a pointer to the registry corresponding to the specified backend
    Registry *registry(JitBackend backend) {
        return backend == JitBackend::CUDA ? &registry_gpu : &registry_cpu;
    }

    State() {
        lock_init(lock);
        lock_init(malloc_lock);
        lock_init(eval_lock);
    }

    ~State() {
        lock_destroy(lock);
        lock_destroy(malloc_lock);
        lock_destroy(eval_lock);
    }
};

struct Buffer {
public:
    Buffer(size_t size);

    // Disable copy/move constructor and assignment
    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;

    ~Buffer() {
        free(m_start);
    }

    const char *get() { return m_start; }
    const char *cur() { return m_cur; }

    void clear() {
        m_cur = m_start;
        if (m_start != m_end)
            m_start[0] = '\0';
    }

    template <size_t N> void put(const char (&str)[N]) {
        put(str, N - 1);
    }

    /// Append a string with the specified length
    void put(const char *str, size_t size) {
        if (unlikely(m_cur + size >= m_end))
            expand(size + 1 - remain());

        memcpy(m_cur, str, size);
        m_cur += size;
        *m_cur = '\0';
    }

    /// Append an unsigned 32 bit integer
    void put_uint32(uint32_t value) {
        const int digits = 10;
        const char *num = "0123456789";
        char buf[digits];
        int i = digits;

        do {
            buf[--i] = num[value % 10];
            value /= 10;
        } while (value);

        return put(buf + i, digits - i);
    }

    /// Append an unsigned 64 bit integer
    void put_uint64(uint64_t value) {
        const int digits = 20;
        const char *num = "0123456789";
        char buf[digits];
        int i = digits;

        do {
            buf[--i] = num[value % 10];
            value /= 10;
        } while (value);

        return put(buf + i, digits - i);
    }

    /// Append an unsigned 64 bit integer (hex version)
    void put_uint64_hex(uint64_t value) {
        const int digits = 18;
        const char *num = "0123456789abcdef";
        char buf[digits];
        int i = digits;

        do {
            buf[--i] = num[value & 0xF];
            value >>= 4;
        } while (value);

        return put(buf + i, digits - i);
    }

    /// Append a single character to the buffer
    void putc(char c) {
        if (unlikely(m_cur + 1 >= m_end))
            expand();
        *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Append multiple copies of a single character to the buffer
    void putc(char c, size_t count) {
        if (unlikely(m_cur + count >= m_end))
            expand(count + 1 - remain());
        for (size_t i = 0; i < count; ++i)
            *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Remove the last 'n' characters
    void rewind(size_t n) {
        if (m_cur < m_start + n)
            m_cur = m_start;
        else
            m_cur -= n;
        *m_cur = '\0';
    }

    /// Check if the buffer contains a given substring
    bool contains(const char *needle) const {
        return strstr(m_start, needle) != nullptr;
    }

    /// Append a formatted (printf-style) string to the buffer
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    size_t fmt(const char *format, ...);

    /// Like \ref fmt, but specify arguments through a va_list.
    size_t vfmt(const char *format, va_list args_);

    size_t size() const { return m_cur - m_start; }
    size_t remain() const { return m_end - m_cur; }

    void swap(Buffer &b) {
        std::swap(m_start, b.m_start);
        std::swap(m_cur, b.m_cur);
        std::swap(m_end, b.m_end);
    }
    void expand(size_t minval = 2);

private:
    char *m_start, *m_cur, *m_end;
};

enum ParamType { Register, Input, Output };

/// State specific to threads
#if defined(_MSC_VER)
  extern __declspec(thread) ThreadState* thread_state_llvm;
  extern __declspec(thread) ThreadState* thread_state_cuda;
#else
  extern __thread ThreadState* thread_state_llvm;
  extern __thread ThreadState* thread_state_cuda;
#endif

extern ThreadState *jitc_init_thread_state(JitBackend backend);

inline ThreadState *thread_state(JitBackend backend) {
    ThreadState *result = (backend == JitBackend::CUDA) ? thread_state_cuda : thread_state_llvm;
    if (unlikely(!result))
        result = jitc_init_thread_state(backend);
    return result;
}

inline ThreadState *thread_state(uint32_t backend) {
    return thread_state((JitBackend) backend);
}

extern State state;
extern Buffer buffer;

#if !defined(_WIN32)
  extern char *jitc_temp_path;
#else
  extern wchar_t *jitc_temp_path;
#endif

/// Initialize core data structures of the JIT compiler
extern void jitc_init(uint32_t backends);

/// Release all resources used by the JIT compiler, and report reference leaks.
extern void jitc_shutdown(int light);

/// Set the currently active device & stream
extern void jitc_cuda_set_device(int device);

/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread();

/// Wait for all computation on the current stream to finish
extern void jitc_sync_thread(ThreadState *stream);

/// Wait for all computation on the current device to finish
extern void jitc_sync_device();

/// Wait for all computation on *all devices* to finish
extern void jitc_sync_all_devices();

/// Search for a shared library and dlopen it if possible
void *jitc_find_library(const char *fname, const char *glob_pat,
                        const char *env_var);

/// Return a pointer to the CUDA stream associated with the currently active device
extern void* jitc_cuda_stream();

/// Return a pointer to the CUDA context associated with the currently active device
extern void* jitc_cuda_context();

/// Push a new CUDA context to the currently active device
extern void jitc_cuda_push_context(void *);

/// Pop the current CUDA context and return it
extern void* jitc_cuda_pop_context();

extern void jitc_set_flags(uint32_t flags);

extern uint32_t jitc_flags();

/// Push a new label onto the prefix stack
extern void jitc_prefix_push(JitBackend backend, const char *label);

/// Pop a label from the prefix stack
extern void jitc_prefix_pop(JitBackend backend);
