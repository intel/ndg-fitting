/*
    src/init.cpp -- Initialization and shutdown of the core parts of DrJit

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "malloc.h"
#include "internal.h"
#include "log.h"
#include "registry.h"
#include "var.h"
#include "profiler.h"
#include <sys/stat.h>

#if defined(DRJIT_ENABLE_OPTIX)
#  include "optix_api.h"
#endif

#if defined(_WIN32)
#  include <windows.h>
#  include <direct.h>
#else
#  include <glob.h>
#  include <dlfcn.h>
#endif

State state;
Buffer buffer{1024};

#if !defined(_WIN32)
  char* jitc_temp_path = nullptr;
#else
  wchar_t* jitc_temp_path = nullptr;
#endif

#if defined(_MSC_VER)
  __declspec(thread) ThreadState* thread_state_cuda = nullptr;
  __declspec(thread) ThreadState* thread_state_llvm = nullptr;
  __declspec(thread) uint32_t jitc_flags_v = (uint32_t) JitFlag::Default;
#else
  __thread ThreadState* thread_state_cuda = nullptr;
  __thread ThreadState* thread_state_llvm = nullptr;
  __thread uint32_t jitc_flags_v = (uint32_t) JitFlag::Default;
#endif

#if defined(DRJIT_ENABLE_ITTNOTIFY)
__itt_domain *drjit_domain = __itt_domain_create("drjit");
#endif

static_assert(
    sizeof(VariableKey) == 8 * sizeof(uint32_t),
    "VariableKey: incorrect size, likely an issue with padding/packing!");

static_assert(
    sizeof(tsl::detail_robin_hash::bucket_entry<VariableMap::value_type, false>) == 64,
    "VariableMap: incorrect bucket size, likely an issue with padding/packing!");

static ProfilerRegion profiler_region_init("jit_init");

#if defined(_WIN32)
extern float timer_frequency_scale;
#endif

/// Initialize core data structures of the JIT compiler
void jitc_init(uint32_t backends) {
    ProfilerPhase profiler(profiler_region_init);

#if defined(_WIN32)
    // Initialize frequency scale for performance counters
    LARGE_INTEGER timer_frequency;
    QueryPerformanceFrequency(&timer_frequency);
    timer_frequency_scale = 1e6f / timer_frequency.QuadPart;
#endif

#if defined(__APPLE__)
    backends &= ~(uint32_t) JitBackend::CUDA;
#endif

    if ((backends & ~state.backends) == 0)
        return;

#if !defined(_WIN32)
    char temp_path[512];
    snprintf(temp_path, sizeof(temp_path), "%s/.drjit", getenv("HOME"));
    struct stat st = {};
    int rv = stat(temp_path, &st);
    size_t temp_path_size = (strlen(temp_path) + 1) * sizeof(char);
    jitc_temp_path = (char*) malloc(temp_path_size);
    memcpy(jitc_temp_path, temp_path, temp_path_size);
#else
    wchar_t temp_path_w[512];
    char temp_path[512];
    if (GetTempPathW(sizeof(temp_path_w) / sizeof(wchar_t), temp_path_w) == 0)
        jitc_fail("jit_init(): could not obtain path to temporary directory!");
    wcsncat(temp_path_w, L"drjit", sizeof(temp_path) / sizeof(wchar_t));
    struct _stat st = {};
    int rv = _wstat(temp_path_w, &st);
    size_t temp_path_size = (wcslen(temp_path_w) + 1) * sizeof(wchar_t);
    jitc_temp_path = (wchar_t*) malloc(temp_path_size);
    memcpy(jitc_temp_path, temp_path_w, temp_path_size);
    wcstombs(temp_path, temp_path_w, sizeof(temp_path));
#endif

    if (rv == -1) {
        jitc_log(Info, "jit_init(): creating directory \"%s\" ..", temp_path);
#if !defined(_WIN32)
        if (mkdir(temp_path, 0700) == -1)
#else
        if (_wmkdir(temp_path_w) == -1)
#endif
            jitc_fail("jit_init(): creation of directory \"%s\" failed: %s",
                temp_path, strerror(errno));
    }

    // Enumerate CUDA devices and collect suitable ones
    jitc_log(Info, "jit_init(): detecting devices ..");

    if ((backends & ~state.backends) == 0)
        return;

    if ((backends & (uint32_t) JitBackend::LLVM) && jitc_llvm_init())
        state.backends |= (uint32_t) JitBackend::LLVM;

    if ((backends & (uint32_t) JitBackend::CUDA) && jitc_cuda_init())
        state.backends |= (uint32_t) JitBackend::CUDA;

    bool has_cuda = state.backends & (uint32_t) JitBackend::CUDA;
    for (int i = 0; has_cuda && i < jitc_cuda_devices; ++i) {
        int pci_bus_id = 0, pci_dom_id = 0, pci_dev_id = 0, num_sm = 0,
            unified_addr = 0, managed = 0, shared_memory_bytes = 0,
            cc_minor = 0, cc_major = 0, memory_pool_support = 0,
            tcc_driver = 1;

        size_t mem_total = 0;
        char name[256];

        cuda_check(cuDeviceTotalMem(&mem_total, i));
        cuda_check(cuDeviceGetName(name, sizeof(name), i));
        cuda_check(cuDeviceGetAttribute(&pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, i));
        cuda_check(cuDeviceGetAttribute(&pci_dev_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, i));
        cuda_check(cuDeviceGetAttribute(&pci_dom_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, i));
        cuda_check(cuDeviceGetAttribute(&num_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i));
        cuda_check(cuDeviceGetAttribute(&unified_addr, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, i));
        cuda_check(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, i));
        cuda_check(cuDeviceGetAttribute(&shared_memory_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, i));
        cuda_check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
        cuda_check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));

        #if defined(_WIN32)
            // Distinguish WDDM and TCM-style drivers on Windows. The default for other OSes is tcc_driver=1
            cuda_check(cuDeviceGetAttribute(&tcc_driver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i));
        #endif

        if (jitc_cuda_version_major > 11 || (jitc_cuda_version_major == 11 && jitc_cuda_version_minor >= 2))
            cuda_check(cuDeviceGetAttribute(&memory_pool_support, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, i));

        jitc_log(Info,
                " - Found CUDA device %i: \"%s\" "
                "(PCI ID %02x:%02x.%i, compute cap. %i.%i, %i SMs w/%s shared mem., %s global mem.%s)",
                i, name, pci_bus_id, pci_dev_id, pci_dom_id, cc_major, cc_minor, num_sm,
                std::string(jitc_mem_string(shared_memory_bytes)).c_str(),
                std::string(jitc_mem_string(mem_total)).c_str(),
                tcc_driver == 0 ? ", WDDM driver" : "");

        if (unified_addr == 0) {
            jitc_log(Warn, " - Warning: device does *not* support unified addressing, skipping ..");
            continue;
        } else if (managed == 0) {
            jitc_log(Warn, " - Warning: device does *not* support managed memory, skipping ..");
            continue;
        }

        Device device;
        device.id = i;
        device.compute_capability = cc_major * 10 + cc_minor;
        device.shared_memory_bytes = (uint32_t) shared_memory_bytes;
        device.num_sm = (uint32_t) num_sm;
        device.memory_pool_support = memory_pool_support != 0;
        device.wddm_driver = tcc_driver == 0;
        cuda_check(cuDevicePrimaryCtxRetain(&device.context, i));
        state.devices.push_back(device);
    }

    // Enable P2P communication if possible
    for (auto &a : state.devices) {
        for (auto &b : state.devices) {
            if (a.id == b.id)
                continue;

            int peer_ok = 0;
            scoped_set_context guard(a.context);
            cuda_check(cuDeviceCanAccessPeer(&peer_ok, a.id, b.id));
            if (peer_ok) {
                jitc_log(Debug, " - Enabling peer access from device %i -> %i",
                        a.id, b.id);
                CUresult rv_2 = cuCtxEnablePeerAccess(b.context, 0);
                if (rv_2 == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
                    continue;
                cuda_check(rv_2);
            }
        }
    }

    state.variable_index = 1;
    state.variable_watermark = 0;

    state.kernel_hard_misses = state.kernel_soft_misses = 0;
    state.kernel_hits = state.kernel_launches = 0;
}

void* jitc_cuda_stream() {
    return (void*) thread_state(JitBackend::CUDA)->stream;
}

void* jitc_cuda_context() {
    return (void*) thread_state(JitBackend::CUDA)->context;
}

void jitc_cuda_push_context(void* ctx) {
    cuda_check(cuCtxPushCurrent((CUcontext) ctx));
}

void* jitc_cuda_pop_context() {
    CUcontext out;
    cuda_check(cuCtxPopCurrent(&out));
    return out;
}

/// Release all resources used by the JIT compiler, and report reference leaks.
void jitc_shutdown(int light) {
    // Synchronize with everything
    for (ThreadState *ts : state.tss) {
        jitc_free_flush(ts);
        if (ts->backend == JitBackend::CUDA) {
            scoped_set_context guard(ts->context);
            cuda_check(cuStreamSynchronize(ts->stream));
        } else {
            task_wait_and_release(ts->task);
        }
        if (!ts->mask_stack.empty())
            jitc_log(Warn, "jit_shutdown(): leaked %zu active masks!",
                    ts->mask_stack.size());
    }

    if (!state.kernel_cache.empty()) {
        jitc_log(Info, "jit_shutdown(): releasing %zu kernel%s ..",
                state.kernel_cache.size(),
                state.kernel_cache.size() > 1 ? "s" : "");

        for (auto &v : state.kernel_cache) {
            jitc_kernel_free(v.first.device, v.second);
            free(v.first.str);
        }

        state.kernel_cache.clear();
    }

    state.kernel_history.clear();

    // CUDA: Try to already free some memory asynchronously (faster)
    if (thread_state_cuda && state.devices[thread_state_cuda->device].memory_pool_support) {
        ThreadState *ts = thread_state_cuda;
        scoped_set_context guard2(ts->context);

        lock_guard guard(state.malloc_lock);
        for (auto it = state.alloc_free.begin(); it != state.alloc_free.end(); ++it) {
            AllocInfo ai = it->first;
            if ((AllocType) ai.type != AllocType::Device)
                continue;

            std::vector<void *> &entries = it.value();
            state.alloc_allocated[int(AllocType::Device)] -= ai.size * entries.size();

            for (void *ptr : entries)
                cuda_check(cuMemFreeAsync((CUdeviceptr) ptr, ts->stream));
            entries.clear();
        }
    }

#if defined(DRJIT_ENABLE_OPTIX)
    // Free the default OptiX shader binding table and pipeline (ref counting)
    if (state.optix_default_sbt_index) {
        jitc_var_dec_ref_ext(state.optix_default_sbt_index);
        state.optix_default_sbt_index = 0;
    }
#endif

    if (!state.tss.empty()) {
        jitc_log(Info, "jit_shutdown(): releasing %zu thread state%s ..",
                state.tss.size(), state.tss.size() > 1 ? "s" : "");

        for (ThreadState *ts : state.tss) {
            for (uint32_t i : ts->side_effects) {
                if (state.variables.find(i) != state.variables.end())
                    jitc_var_dec_ref_ext(i);
            }
            jitc_free_flush(ts);
            if (ts->backend == JitBackend::CUDA) {
                scoped_set_context guard(ts->context);
                cuda_check(cuEventDestroy(ts->event));
                cuda_check(cuStreamSynchronize(ts->stream));
                cuda_check(cuStreamDestroy(ts->stream));
            }

            if (!ts->prefix_stack.empty()) {
                for (char *s : ts->prefix_stack)
                    free(s);
                jitc_log(Warn,
                         "jit_shutdown(): leaked %zu prefix stack entries.",
                         ts->prefix_stack.size());
                free(ts->prefix);
            }

            delete ts->release_chain;
            delete ts;
        }

        if (state.variables.empty() && !state.cse_cache.empty()) {
            for (auto &kv: state.cse_cache)
                jitc_log(Warn,
                        " - id=%u: size=%u, type=%s, literal=%u, dep=[%u, "
                        "%u, %u, %u], stmt=\"%s\", value=%lli",
                        kv.second, kv.first.size,
                        type_name[kv.first.type], kv.first.literal,
                        kv.first.dep[0], kv.first.dep[1], kv.first.dep[2],
                        kv.first.dep[3], kv.first.literal ? "" : kv.first.stmt,
                        kv.first.literal ? (long long) kv.first.value : 0);

            jitc_log(Warn, "jit_shutdown(): detected a common subexpression "
                          "elimination cache leak (see above).");
        }

        pool_destroy();
        state.tss.clear();
    }

    thread_state_llvm = nullptr;
    thread_state_cuda = nullptr;

    if (std::max(state.log_level_stderr, state.log_level_callback) >= LogLevel::Warn) {
        uint32_t n_leaked = 0;
        for (auto &var : state.variables) {
            if (n_leaked == 0)
                jitc_log(Warn, "jit_shutdown(): detected variable leaks:");
            if (n_leaked < 10)
                jitc_log(Warn,
                         " - variable r%u is still being referenced! "
                         "(int_ref=%u, ext_ref=%u, se_ref=%u, type=%s, size=%u, "
                         "stmt=\"%s\", dep=[%u, %u, %u, %u])",
                         var.first,
                         (uint32_t) var.second.ref_count_int,
                         (uint32_t) var.second.ref_count_ext,
                         (uint32_t) var.second.ref_count_se,
                         type_name[var.second.type],
                         var.second.size,
                         var.second.literal
                             ? "<literal>"
                             : (var.second.stmt ? var.second.stmt : "<null>"),
                         var.second.dep[0], var.second.dep[1],
                         var.second.dep[2], var.second.dep[3]);
            else if (n_leaked == 10)
                jitc_log(Warn, " - (skipping remainder)");
            ++n_leaked;
        }

        if (n_leaked > 0)
            jitc_log(Warn, "jit_shutdown(): %u variables are still referenced!", n_leaked);

        if (state.variables.empty() && !state.extra.empty()) {
            jitc_log(Warn,
                    "jit_shutdown(): %zu 'extra' records were not cleaned up:",
                    state.extra.size());
            n_leaked = 0;
            for (const auto &kv : state.extra) {
                jitc_log(Warn, "- variable r%u", kv.first);
                if (++n_leaked == 10) {
                    jitc_log(Warn, " - (skipping remainder)");
                    break;
                }
            }
        }
    }

    jitc_registry_shutdown();
    jitc_malloc_shutdown();

    if (state.backends & (uint32_t) JitBackend::CUDA) {
        for (auto &v : state.devices) {
#if defined(DRJIT_ENABLE_OPTIX)
            jitc_optix_context_destroy(v);
#endif
            cuda_check(cuDevicePrimaryCtxRelease(v.id));
        }
        state.devices.clear();
    }

    jitc_log(Info, "jit_shutdown(light=%u): done", (uint32_t) light);

    if (light == 0) {
        jitc_llvm_shutdown();
#if defined(DRJIT_ENABLE_OPTIX)
        jitc_optix_shutdown();
#endif
        jitc_cuda_shutdown();
    }

    free(jitc_temp_path);
    jitc_temp_path = nullptr;

    state.backends = 0;
}


ThreadState *jitc_init_thread_state(JitBackend backend) {
    ThreadState *ts = new ThreadState();

    if (backend == JitBackend::CUDA) {
        if ((state.backends & (uint32_t) JitBackend::CUDA) == 0) {
            #if defined(_WIN32)
                const char *cuda_fname = "nvcuda.dll";
            #elif defined(__linux__)
                const char *cuda_fname  = "libcuda.so";
            #else
                const char *cuda_fname  = "libcuda.dylib";
            #endif

            delete ts;
            if (jitc_cuda_cuinit_result == CUDA_ERROR_NOT_INITIALIZED) {
                jitc_raise(
                    "jit_init_thread_state(): the CUDA backend hasn't been "
                    "initialized. Make sure to call jit_init(JitBackend::CUDA) "
                    "to properly initialize this backend.");
            } else if (jitc_cuda_cuinit_result != CUDA_SUCCESS) {
                const char *msg = nullptr;
                cuGetErrorString(jitc_cuda_cuinit_result, &msg);
                jitc_raise("jit_cuda_init(): the CUDA backend is not available "
                           "because cuInit() failed.\nThere are two common "
                           "explanations for this type of failure:\n\n 1. your "
                           "computer simply does not contain a graphics card "
                           "that supports CUDA.\n\n 2. your CUDA kernel module "
                           "and CUDA library are out of sync. Try to see if "
                           "you\n    can run a utility like 'nvida-smi'. If "
                           "not, a reboot will likely fix this\n    issue. "
                           "Otherwise reinstall your graphics driver. \n\n "
                           "The specific error message produced by cuInit was\n"
                           "   \"%s\"", msg);
            } else {
                jitc_raise(
                    "jit_init_thread_state(): the CUDA backend is inactive "
                    "because it has not been initialized via jit_init(), or "
                    "because the CUDA driver library (\"%s\") could not be "
                    "found! Set the DRJIT_LIBCUDA_PATH environment variable to "
                    "specify its path.",
                    cuda_fname);
            }
        }

        if (state.devices.empty()) {
            delete ts;
            jitc_raise("jit_init_thread_state(): the CUDA backend is inactive "
                       "because no compatible CUDA devices were found on your "
                       "system.");
        }

        const Device &device = state.devices[0];
        ts->device = 0;
        ts->context = device.context;
        ts->compute_capability = device.compute_capability >= 60 ? 60 : 50;
        scoped_set_context guard(ts->context);
        cuda_check(cuStreamCreate(&ts->stream, CU_STREAM_DEFAULT));
        cuda_check(cuEventCreate(&ts->event, CU_EVENT_DISABLE_TIMING));
        thread_state_cuda = ts;
    } else {
        if ((state.backends & (uint32_t) JitBackend::LLVM) == 0) {
            delete ts;
            #if defined(_WIN32)
                const char *llvm_fname = "LLVM-C.dll";
            #elif defined(__linux__)
                const char *llvm_fname  = "libLLVM.so";
            #else
                const char *llvm_fname  = "libLLVM.dylib";
            #endif

            jitc_raise("jit_init_thread_state(): the LLVM backend is inactive "
                      "because the LLVM shared library (\"%s\") could not be "
                      "found! Set the DRJIT_LIBLLVM_PATH environment "
                      "variable to specify its path.",
                      llvm_fname);
        }
        thread_state_llvm = ts;
        ts->device = -1;
    }

    ts->backend = backend;
    ts->cse_scope = ++state.cse_scope_ctr;
    state.tss.push_back(ts);
    return ts;
}

void jitc_cuda_set_device(int device_id) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    if (ts->device == device_id)
        return;

    if ((size_t) device_id >= state.devices.size())
        jitc_raise("jit_cuda_set_device(%i): must be in the range 0..%i!",
                  device_id, (int) state.devices.size() - 1);

    jitc_log(Info, "jit_cuda_set_device(%i)", device_id);

    Device &device = state.devices[device_id];
    CUcontext new_context = device.context;

    /* Disassociate from old context */ {
        scoped_set_context guard(ts->context);
        cuda_check(cuStreamSynchronize(ts->stream));
        cuda_check(cuEventDestroy(ts->event));
        cuda_check(cuStreamDestroy(ts->stream));
    }

    /* Associate with new context */ {
        ts->context = new_context;
        ts->device = device_id;
        ts->compute_capability = device.compute_capability >= 60 ? 60 : 50;
        scoped_set_context guard(ts->context);

        cuda_check(cuStreamCreate(&ts->stream, CU_STREAM_DEFAULT));
        cuda_check(cuEventCreate(&ts->event, CU_EVENT_DISABLE_TIMING));
    }
}

void jitc_sync_thread(ThreadState *ts) {
    if (!ts)
        return;
    if (ts->backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuStreamSynchronize(ts->stream));
    } else {
        task_wait_and_release(ts->task);
        ts->task = nullptr;
    }
}

/// Wait for all computation on the current stream to finish
void jitc_sync_thread() {
    unlock_guard guard(state.lock);
    jitc_sync_thread(thread_state_cuda);
    jitc_sync_thread(thread_state_llvm);
}

/// Wait for all computation on the current device to finish
void jitc_sync_device() {
    ThreadState *ts = thread_state_cuda;
    if (ts) {
        /* Release lock while synchronizing */ {
            unlock_guard guard(state.lock);
            scoped_set_context guard2(ts->context);
            cuda_check(cuCtxSynchronize());
        }
    }

    if (thread_state_llvm) {
        std::vector<ThreadState *> tss = state.tss;
        // Release lock while synchronizing */
        unlock_guard guard(state.lock);
        for (ThreadState *ts_2 : tss) {
            if (ts_2->backend == JitBackend::LLVM)
                jitc_sync_thread(ts_2);
        }
    }
}

/// Wait for all computation on *all devices* to finish
void jitc_sync_all_devices() {
    std::vector<ThreadState *> tss = state.tss;
    unlock_guard guard(state.lock);
    for (ThreadState *ts : tss)
        jitc_sync_thread(ts);
}

static void jitc_rebuild_prefix(ThreadState *ts) {
    free(ts->prefix);

    if (!ts->prefix_stack.empty()) {
        size_t size = 1;
        for (const char *s : ts->prefix_stack)
            size += strlen(s) + 1;
        ts->prefix = (char *) malloc(size);
        char *p = ts->prefix;

        for (const char *s : ts->prefix_stack) {
            size_t len = strlen(s);
            memcpy(p, s, len);
            p += len;
            *p++ = '/';
        }
        *p++ = '\0';
    } else {
        ts->prefix = nullptr;
    }
}

void jitc_prefix_push(JitBackend backend, const char *label) {
    if (strchr(label, '\n') || strchr(label, '/'))
        jitc_raise("jit_prefix_push(): invalid string (may not contain newline "
                   "or '/' characters)");

    ThreadState *ts = thread_state(backend);
    ts->prefix_stack.push_back(strdup(label));
    jitc_rebuild_prefix(ts);
}

void jitc_prefix_pop(JitBackend backend) {
    ThreadState *ts = thread_state(backend);
    auto &stack = ts->prefix_stack;
    if (stack.empty())
        jitc_raise("jit_prefix_pop(): stack underflow!");
    free(stack.back());
    stack.pop_back();
    jitc_rebuild_prefix(ts);
}

/// Glob for a shared library and try to load the most recent version
void *jitc_find_library(const char *fname, const char *glob_pat,
                       const char *env_var) {
#if !defined(_WIN32)
    const char* env_var_val = env_var ? getenv(env_var) : nullptr;
    if (env_var_val != nullptr && strlen(env_var_val) == 0)
        env_var_val = nullptr;

    void* handle = dlopen(env_var_val ? env_var_val : fname, RTLD_LAZY);

    if (!handle) {
        if (env_var_val) {
            jitc_log(Warn, "jit_find_library(): Unable to load \"%s\": %s!",
                    env_var_val, dlerror());
            return nullptr;
        }

        glob_t g;
        if (glob(glob_pat, GLOB_BRACE, nullptr, &g) == 0) {
            const char *chosen = nullptr;
            if (g.gl_pathc > 1) {
                jitc_log(Info, "jit_find_library(): Multiple versions of "
                              "%s were found on your system!\n", fname);
                std::sort(g.gl_pathv, g.gl_pathv + g.gl_pathc,
                          [](const char *a, const char *b) {
                              while (a != nullptr && b != nullptr) {
                                  while (*a == *b && *a != '\0' && !isdigit(*a)) {
                                      ++a; ++b;
                                  }
                                  if (isdigit(*a) && isdigit(*b)) {
                                      char *ap, *bp;
                                      int ai = strtol(a, &ap, 10);
                                      int bi = strtol(b, &bp, 10);
                                      if (ai != bi)
                                          return ai < bi;
                                      a = ap;
                                      b = bp;
                                  } else {
                                      return strcmp(a, b) < 0;
                                  }
                              }
                              return false;
                          });
                uint32_t counter = 1;
                for (int j = 0; j < 2; ++j) {
                    for (size_t i = 0; i < g.gl_pathc; ++i) {
                        struct stat buf;
                        // Skip symbolic links at first
                        if (j == 0 && (lstat(g.gl_pathv[i], &buf) || S_ISLNK(buf.st_mode)))
                            continue;
                        jitc_log(Info, " %u. \"%s\"", counter++, g.gl_pathv[i]);
                        chosen = g.gl_pathv[i];
                    }
                    if (chosen)
                        break;
                }
                jitc_log(Info,
                        "\nChoosing the last one. Specify a path manually "
                        "using the environment\nvariable '%s' to override this "
                        "behavior.\n", env_var);
            } else if (g.gl_pathc == 1) {
                chosen = g.gl_pathv[0];
            }
            if (chosen)
                handle = dlopen(chosen, RTLD_LAZY);
            globfree(&g);
        }
    }
#else
    (void) glob_pat;

    wchar_t buf[1024];
    mbstowcs(buf, env_var, sizeof(buf) / sizeof(wchar_t));

    const wchar_t* env_var_val = env_var ? _wgetenv(buf) : nullptr;
    if (env_var_val != nullptr && wcslen(env_var_val) == 0)
        env_var_val = nullptr;

    mbstowcs(buf, fname, sizeof(buf) / sizeof(wchar_t));
    void* handle = (void *) LoadLibraryW(env_var_val ? env_var_val : buf);
#endif

    return handle;
}

void jitc_set_flags(uint32_t flags) {
    pool_set_profile(int(flags & (uint32_t) JitFlag::KernelHistory));
    jitc_flags_v = flags;
}

uint32_t jitc_flags() {
    return jitc_flags_v;
}

/// ==========================================================================

KernelHistory::KernelHistory() : m_data(nullptr), m_size(0), m_capacity(0) { }

KernelHistory::~KernelHistory() { free(m_data); }

void KernelHistory::append(const KernelHistoryEntry &value) {
    /* Expand kernel history buffer if necessary. There should always be
       enough memory for an additional end-of-list marker at the end */

    if (m_size + 2 > m_capacity) {
        m_capacity = (m_size + 2) * 2;
        void *tmp = malloc_check(m_capacity * sizeof(KernelHistoryEntry));
        memcpy(tmp, m_data, m_size * sizeof(KernelHistoryEntry));
        free(m_data);
        m_data = (KernelHistoryEntry *) tmp;
    }

    m_data[m_size++] = value;
    memset(m_data + m_size, 0, sizeof(KernelHistoryEntry));
}

KernelHistoryEntry *KernelHistory::get() {
    KernelHistoryEntry *data = m_data;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventElapsedTime(&k.execution_time,
                               (CUevent) k.event_start,
                               (CUevent) k.event_end);
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
            k.event_start = k.event_end = 0;
        } else {
            k.execution_time = task_time((Task *) k.task);
            task_release((Task *) k.task);
            k.task = nullptr;
        }
    }

    m_data = nullptr;
    m_size = m_capacity = 0;

    return data;
}

void KernelHistory::clear() {
    if (m_size == 0)
        return;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = m_data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
        } else {
            task_release((Task *) k.task);
        }
    }

    free(m_data);
    m_data = nullptr;
    m_size = m_capacity = 0;
}
