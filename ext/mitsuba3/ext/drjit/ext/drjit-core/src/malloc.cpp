/*
    src/malloc.cpp -- Asynchronous memory allocation system + cache

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "util.h"
#include "profiler.h"

#if !defined(_WIN32)
#  include <sys/mman.h>
#endif

// Try to use huge pages for allocations > 2M (only on Linux)
#if defined(__linux__)
#  define DRJIT_HUGEPAGE 1
#else
#  define DRJIT_HUGEPAGE 0
#endif

#define DRJIT_HUGEPAGE_SIZE (2 * 1024 * 1024)

static_assert(
    sizeof(tsl::detail_robin_hash::bucket_entry<AllocUsedMap::value_type, false>) == 24,
    "AllocUsedMap: incorrect bucket size, likely an issue with padding/packing!");

const char *alloc_type_name[(int) AllocType::Count] = {
    "host",   "host-async", "host-pinned",
    "device", "managed",    "managed-read-mostly"
};

const char *alloc_type_name_short[(int) AllocType::Count] = {
    "host       ",
    "host-async ",
    "host-pinned",
    "device     ",
    "managed    ",
    "managed/rm "
};

// Round an unsigned integer up to a power of two
size_t round_pow2(size_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;  x |= x >> 32;
    return x + 1;
}

uint32_t round_pow2(uint32_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}


static void *aligned_malloc(size_t size) {
    /* Temporarily release the main lock */
    unlock_guard guard(state.lock);
#if !defined(_WIN32)
    // Use posix_memalign for small allocations and mmap() for big ones
    if (size < DRJIT_HUGEPAGE_SIZE) {
        void *ptr = nullptr;
        int rv = posix_memalign(&ptr, 64, size);
        return rv == 0 ? ptr : nullptr;
    } else {
        void *ptr;

#if DRJIT_HUGEPAGE
        // Attempt to allocate a 2M page directly
        ptr = mmap(0, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_HUGETLB, -1, 0);
        if (ptr != MAP_FAILED)
            return ptr;
#endif

        // Allocate 4K pages
        ptr = mmap(0, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);

#if DRJIT_HUGEPAGE
        // .. and advise the OS to convert to 2M pages
        if (ptr != MAP_FAILED)
            madvise(ptr, size, MADV_HUGEPAGE);
#endif

        return ptr;
    }
#else
    return _aligned_malloc(size, 64);
#endif
}

static void aligned_free(void *ptr, size_t size) {
#if !defined(_WIN32)
    if (size < DRJIT_HUGEPAGE_SIZE)
        free(ptr);
    else
        munmap(ptr, size);
#else
    (void) size;
    _aligned_free(ptr);
#endif
}

void* jitc_malloc(AllocType type, size_t size) {
    if (size == 0)
        return nullptr;

    if ((type != AllocType::Host && type != AllocType::HostAsync) ||
        jitc_llvm_vector_width < 16) {
        // Round up to the next multiple of 64 bytes
        size = (size + 63) / 64 * 64;
    } else {
        size_t packet_size = jitc_llvm_vector_width * sizeof(double);
        size = (size + packet_size - 1) / packet_size * packet_size;
    }

    /* Round 'size' to the next larger power of two. This is somewhat
       wasteful, but reduces the number of different sizes that an allocation
       can have to a manageable amount that facilitates re-use. */
    size = round_pow2(size);

    AllocInfo ai(size, type, 0);

    const char *descr = nullptr;
    void *ptr = nullptr;
    JitBackend backend =
        (type != AllocType::Host && type != AllocType::HostAsync)
            ? JitBackend::CUDA
            : JitBackend::LLVM;
    ThreadState *ts = nullptr;

    if (type != AllocType::Host)
        ts = thread_state(backend);

    /* Acquire lock protecting ts->release_chain contents and state.alloc_free */ {
        lock_guard guard(state.malloc_lock);

        if (type == AllocType::Device)
            ai.device = ts->device;

        if (type == AllocType::Device || type == AllocType::HostAsync) {
            /* Check for arrays with a pending free operation on the current
               stream. This only works for device or host-async memory, as other
               allocation flavors (host-pinned, shared, shared-read-mostly) can be
               accessed from both CPU & GPU and might still be used. */

            ReleaseChain *chain = ts->release_chain;
            while (chain) {
                auto it = chain->entries.find(ai);
                if (it != chain->entries.end()) {
                    auto &list = it.value();
                    if (!list.empty()) {
                        ptr = list.back();
                        list.pop_back();
                        descr = "reused local";
                        break;
                    }
                }

                chain = chain->next;
            }
        }

        // Look globally. Are there suitable freed arrays?
        if (ptr == nullptr) {
            auto it = state.alloc_free.find(ai);

            if (it != state.alloc_free.end()) {
                std::vector<void *> &list = it.value();
                if (!list.empty()) {
                    ptr = list.back();
                    list.pop_back();
                    descr = "reused global";
                }
            }
        }
    }

    // 3. Looks like we will have to allocate some memory..
    if (unlikely(ptr == nullptr)) {
        if (type == AllocType::Host || type == AllocType::HostAsync) {
            ptr = aligned_malloc(ai.size);
            if (!ptr) {
                jitc_flush_malloc_cache(true, true);
                ptr = aligned_malloc(ai.size);
            }
        } else {
            scoped_set_context guard(ts->context);
            CUresult (*alloc) (CUdeviceptr *, size_t) = nullptr;

            auto cuMemAllocAsync_ = [](CUdeviceptr *ptr_, size_t size_) {
                return cuMemAllocAsync(ptr_, size_, thread_state_cuda->stream);
            };

            auto cuMemAllocManaged_ = [](CUdeviceptr *ptr_, size_t size_) {
                return cuMemAllocManaged(ptr_, size_, CU_MEM_ATTACH_GLOBAL);
            };

            auto cuMemAllocManagedReadMostly_ = [](CUdeviceptr *ptr_, size_t size_) {
                CUresult ret = cuMemAllocManaged(ptr_, size_, CU_MEM_ATTACH_GLOBAL);
                if (ret == CUDA_SUCCESS)
                    cuda_check(cuMemAdvise(*ptr_, size_, CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
                return ret;
            };

            bool hasAllocAsync = state.devices[ts->device].memory_pool_support;
            switch (type) {
                case AllocType::HostPinned:        alloc = (decltype(alloc)) cuMemAllocHost; break;
                case AllocType::Device:            alloc = hasAllocAsync ? cuMemAllocAsync_ : cuMemAlloc; break;
                case AllocType::Managed:           alloc = cuMemAllocManaged_; break;
                case AllocType::ManagedReadMostly: alloc = cuMemAllocManagedReadMostly_; break;
                default:
                    jitc_fail("jit_malloc(): internal-error unsupported allocation type!");
            }

            CUresult ret;
            /* Temporarily release the main lock */ {
                unlock_guard guard_2(state.lock);
                ret = alloc((CUdeviceptr *) &ptr, ai.size);
            }

            if (ret != CUDA_SUCCESS) {
                jitc_flush_malloc_cache(true, true);

                /* Temporarily release the main lock */ {
                    unlock_guard guard_2(state.lock);
                    ret = alloc((CUdeviceptr *) &ptr, ai.size);
                }

                if (ret != CUDA_SUCCESS)
                    ptr = nullptr;
            }
        }
        descr = "new allocation";

        size_t &allocated = state.alloc_allocated[ai.type],
               &watermark = state.alloc_watermark[ai.type];

        allocated += ai.size;
        watermark = std::max(allocated, watermark);
    }

    if (unlikely(ptr == nullptr))
        jitc_raise("jit_malloc(): out of memory! Could not "
                  "allocate %zu bytes of %s memory.",
                  size, alloc_type_name[ai.type]);

    state.alloc_used.emplace(ptr, ai);

    (void) descr; // don't warn if tracing is disabled
    if ((AllocType) ai.type == AllocType::Device)
        jitc_trace("jit_malloc(type=%s, device=%u, size=%zu): " DRJIT_PTR " (%s)",
                  alloc_type_name[ai.type], (uint32_t) ai.device, (size_t) ai.size,
                  (uintptr_t) ptr, descr);
    else
        jitc_trace("jit_malloc(type=%s, size=%zu): " DRJIT_PTR " (%s)",
                  alloc_type_name[ai.type], (size_t) ai.size, (uintptr_t) ptr,
                  descr);

    state.alloc_usage[ai.type] += ai.size;

    return ptr;
}

static uint32_t free_ctr = 0;

void jitc_free(void *ptr) {
    if (ptr == nullptr)
        return;

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_free(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);

    AllocInfo ai = it.value();

    if ((AllocType) ai.type == AllocType::Host) {
        // Acquire lock protecting 'state.alloc_free'
        lock_guard guard(state.malloc_lock);
        state.alloc_free[ai].push_back(ptr);
    } else {
        ThreadState *ts = (AllocType) ai.type == AllocType::HostAsync
                              ? thread_state_llvm
                              : thread_state_cuda;
        if (likely(ts)) {
            /* Acquire lock protecting 'ts->release_chain' contents */ {
                lock_guard guard(state.malloc_lock);
                ReleaseChain *chain = ts->release_chain;
                if (unlikely(!chain))
                    chain = ts->release_chain = new ReleaseChain();
                chain->entries[ai].push_back(ptr);
            }

            if (free_ctr++ > 64) {
                jitc_free_flush(ts);
                free_ctr = 0;
            }
        } else {
            /* This is bad -- freeing a pointer outside of an active
               stream, or with the wrong backend activated. That pointer may
               still be used in a kernel that is currently being executed
               asynchronously. The only thing we can do at this point is to
               flush all streams. */
            jitc_sync_all_devices();
            lock_guard guard(state.malloc_lock);
            state.alloc_free[ai].push_back(ptr);
        }
    }

    if ((AllocType) ai.type == AllocType::Device)
        jitc_trace("jit_free(" DRJIT_PTR ", type=%s, device=%u, size=%zu)",
                  (uintptr_t) ptr, alloc_type_name[ai.type],
		  (uint32_t) ai.device, (size_t) ai.size);
    else
        jitc_trace("jit_free(" DRJIT_PTR ", type=%s, size=%zu)", (uintptr_t) ptr,
                  alloc_type_name[ai.type], (size_t) ai.size);

    state.alloc_usage[ai.type] -= ai.size;
    state.alloc_used.erase(it);
}

void jitc_malloc_clear_statistics() {
    for (int i = 0; i < (int) AllocType::Count; ++i)
        state.alloc_watermark[i] = state.alloc_allocated[i];
}

static void jitc_free_chain(void *ptr) {
    /* Acquire lock protecting ts->release_chain contents and
       state.alloc_free */
    lock_guard guard(state.malloc_lock);
    ReleaseChain *chain0 = (ReleaseChain *) ptr,
                 *chain1 = chain0->next;

    for (auto &kv : chain1->entries) {
        const AllocInfo &ai = kv.first;
        std::vector<void *> &target = state.alloc_free[ai];
        target.insert(target.end(), kv.second.begin(),
                      kv.second.end());
    }

    delete chain1;
    chain0->next = nullptr;
}

void jitc_free_flush(ThreadState *ts) {
    if (unlikely(!ts))
        return;

    ReleaseChain *chain = ts->release_chain;
    if (chain == nullptr || chain->entries.empty())
        return;

    size_t n_dealloc = 0;
    for (auto &kv: chain->entries)
        n_dealloc += kv.second.size();

    if (n_dealloc == 0)
        return;

    ReleaseChain *chain_new = new ReleaseChain();
    chain_new->next = chain;
    ts->release_chain = chain_new;

    jitc_trace("jit_free_flush(): scheduling %zu deallocation%s",
              n_dealloc, n_dealloc > 1 ? "s" : "");

    if (ts->backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuLaunchHostFunc(ts->stream, jitc_free_chain, chain_new));
    } else {
        Task *new_task = task_submit_dep(
            nullptr, &ts->task, 1, 1,
            [](uint32_t, void *ptr) { jitc_free_chain(ptr); }, chain_new);
        task_release(ts->task);
        ts->task = new_task;
    }
}

void* jitc_malloc_migrate(void *ptr, AllocType type, int move) {
    if (ptr == nullptr)
        return nullptr;

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_migrate(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);

    AllocInfo ai = it.value();

    // Maybe nothing needs to be done..
    if ((AllocType) ai.type == type &&
        (type != AllocType::Device || ai.device == thread_state(JitBackend::CUDA)->device)) {
        if (move) {
            return ptr;
        } else {
            void *ptr_new = jitc_malloc(type, ai.size);
            if (type == AllocType::Host) {
                memcpy(ptr_new, ptr, ai.size);
            } else {
                JitBackend backend =
                    (type != AllocType::Host && type != AllocType::HostAsync)
                        ? JitBackend::CUDA
                        : JitBackend::LLVM;
                jitc_memcpy_async(backend, ptr_new, ptr, ai.size);
            }
            return ptr_new;
        }
    }

    if (((AllocType) ai.type == AllocType::Host && type == AllocType::HostAsync) ||
        ((AllocType) ai.type == AllocType::HostAsync && type == AllocType::Host)) {
        if (move) {
            state.alloc_usage[ai.type] -= ai.size;
            state.alloc_usage[(int) type] += ai.size;
            state.alloc_allocated[ai.type] -= ai.size;
            state.alloc_allocated[(int) type] += ai.size;
            it.value().type = (uint32_t) type;
            return ptr;
        } else {
            void *ptr_new = jitc_malloc(type, ai.size);
            jitc_memcpy_async(JitBackend::LLVM, ptr_new, ptr, ai.size);
            if ((AllocType) ai.type == AllocType::Host)
                jitc_sync_thread(); // be careful when copying from host
            return ptr_new;
        }
    }

    if (type == AllocType::HostAsync || (AllocType) ai.type == AllocType::HostAsync)
        jitc_raise("jit_malloc_migrate(): migrations between CUDA and "
                   "host-asynchronous memory are not supported.");

    /// At this point, source or destination is a GPU array, get assoc. state
    ThreadState *ts = thread_state(JitBackend::CUDA);

    if (type == AllocType::Host) // Upgrade from host to host-pinned memory
        type = AllocType::HostPinned;

    void *ptr_new = jitc_malloc(type, ai.size);
    jitc_trace("jit_malloc_migrate(" DRJIT_PTR " -> " DRJIT_PTR ", %s -> %s)",
              (uintptr_t) ptr, (uintptr_t) ptr_new,
              alloc_type_name[ai.type], alloc_type_name[(int) type]);

    scoped_set_context guard(ts->context);
    if ((AllocType) ai.type == AllocType::Host) {
        // Host -> Device memory, create an intermediate host-pinned array
        void *tmp = jitc_malloc(AllocType::HostPinned, ai.size);
        memcpy(tmp, ptr, ai.size);
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, ai.size,
                                 ts->stream));
        jitc_free(tmp);
    } else {
        cuda_check(cuMemcpyAsync((CUdeviceptr) ptr_new,
                                 (CUdeviceptr) ptr, ai.size,
                                 ts->stream));
    }

    if (move)
        jitc_free(ptr);

    return ptr_new;
}

/// Asynchronously prefetch a memory region
void jitc_malloc_prefetch(void *ptr, int device) {
    if (device == -1) {
        device = CU_DEVICE_CPU;
    } else {
        if ((size_t) device >= state.devices.size())
            jitc_raise("jit_malloc_prefetch(): invalid device ID!");
        device = state.devices[device].id;
    }

    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_prefetch(): unknown address " DRJIT_PTR "!",
                  (uintptr_t) ptr);

    AllocInfo ai = it.value();

    if ((AllocType) ai.type != AllocType::Managed &&
        (AllocType) ai.type != AllocType::ManagedReadMostly)
        jitc_raise("jit_malloc_prefetch(): invalid memory type, expected "
                  "Managed or ManagedReadMostly.");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);
    if (device == -2) {
        for (const Device &d : state.devices)
            cuda_check(cuMemPrefetchAsync((CUdeviceptr) ptr, ai.size, d.id,
                                          ts->stream));
    } else {
        cuda_check(cuMemPrefetchAsync((CUdeviceptr) ptr, ai.size, device,
                                      ts->stream));
    }
}

static bool jitc_flush_malloc_cache_warned = false;

static ProfilerRegion profiler_region_flush_malloc_cache("jit_flush_malloc_cache");

/// Release all unused memory to the GPU / OS
void jitc_flush_malloc_cache(bool flush_local, bool warn) {
    if (warn && !jitc_flush_malloc_cache_warned) {
        jitc_log(
            Warn,
            "jit_flush_malloc_cache(): Dr.Jit exhausted the available memory and had "
            "to flush its allocation cache to free up additional memory. This "
            "is an expensive operation and will have a negative effect on "
            "performance. You may want to change your computation so that it "
            "uses less memory. This warning will only be displayed once.");

        jitc_flush_malloc_cache_warned = true;
    }
    ProfilerPhase profiler(profiler_region_flush_malloc_cache);

    if (flush_local) {
        if (thread_state_cuda)
            jitc_free_flush(thread_state_cuda);
        if (thread_state_llvm)
            jitc_free_flush(thread_state_llvm);
        // Ensure that all computation has completed
        jitc_sync_all_devices();
    }

    AllocInfoMap alloc_free;

    /* Critical section */ {
        lock_guard guard(state.malloc_lock);
        alloc_free = std::move(state.alloc_free);
    }

    // Another synchronization to be sure that 'alloc_free' can be released
    jitc_sync_all_devices();

    size_t trim_count[(int) AllocType::Count] = { 0 },
           trim_size [(int) AllocType::Count] = { 0 };

    /* Temporarily release the main lock */ {
        unlock_guard guard(state.lock);

        for (auto& kv : alloc_free) {
            const std::vector<void *> &entries = kv.second;

            trim_count[(int) kv.first.type] += entries.size();
            trim_size[(int) kv.first.type] += kv.first.size * entries.size();

            switch ((AllocType) kv.first.type) {
                case AllocType::Device:
                    if (state.backends & (uint32_t) JitBackend::CUDA) {
                        ThreadState *ts = thread_state_cuda;
                        if (ts && state.devices[ts->device].memory_pool_support) {
                            scoped_set_context guard2(ts->context);
                            for (void *ptr : entries)
                                cuda_check(cuMemFreeAsync((CUdeviceptr) ptr, ts->stream));
                        } else {
                            for (void *ptr : entries)
                                cuda_check(cuMemFree((CUdeviceptr) ptr));
                        }
                    }
                    break;

                case AllocType::Managed:
                case AllocType::ManagedReadMostly:
                    if (state.backends & (uint32_t) JitBackend::CUDA) {
                        for (void *ptr : entries)
                            cuda_check(cuMemFree((CUdeviceptr) ptr));
                    }
                    break;

                case AllocType::HostPinned:
                    if (state.backends & (uint32_t) JitBackend::CUDA) {
                        for (void *ptr : entries)
                            cuda_check(cuMemFreeHost(ptr));
                    }
                    break;

                case AllocType::Host:
                case AllocType::HostAsync:
                    for (void *ptr : entries)
                        aligned_free(ptr, kv.first.size);
                    break;

                default:
                    jitc_fail("jit_flush_malloc_cache(): unsupported allocation type!");
            }
        }
    }

    for (int i = 0; i < (int) AllocType::Count; ++i)
        state.alloc_allocated[i] -= trim_size[i];

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += trim_count[i];

    if (total > 0) {
        jitc_log(Debug, "jit_flush_malloc_cache(): freed");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (trim_count[i] == 0)
                continue;
            jitc_log(Debug, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(trim_size[i]),
                    trim_count[i], trim_count[i] > 1 ? "s" : "");
        }
    }
}

/// Query the flavor of a memory allocation made using \ref jitc_malloc()
AllocType jitc_malloc_type(void *ptr) {
    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jit_malloc_type(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    return (AllocType) it->second.type;
}

/// Query the device associated with a memory allocation made using \ref jitc_malloc()
int jitc_malloc_device(void *ptr) {
    auto it = state.alloc_used.find(ptr);
    if (unlikely(it == state.alloc_used.end()))
        jitc_raise("jitc_malloc_device(): unknown address " DRJIT_PTR "!", (uintptr_t) ptr);
    const AllocInfo &ai = it.value();
    if (ai.type == (int) AllocType::Host || ai.type == (int) AllocType::HostAsync)
        return -1;
    else
        return ai.device;
}

void jitc_malloc_shutdown() {
    jitc_flush_malloc_cache(false, false);

    size_t leak_count[(int) AllocType::Count] = { 0 },
           leak_size [(int) AllocType::Count] = { 0 };
    for (auto kv : state.alloc_used) {
        leak_count[(int) kv.second.type]++;
        leak_size[(int) kv.second.type] += kv.second.size;
    }

    size_t total = 0;
    for (int i = 0; i < (int) AllocType::Count; ++i)
        total += leak_count[i];

    if (total > 0) {
        jitc_log(Warn, "jit_malloc_shutdown(): leaked");
        for (int i = 0; i < (int) AllocType::Count; ++i) {
            if (leak_count[i] == 0)
                continue;

            jitc_log(Warn, " - %s memory: %s in %zu allocation%s",
                    alloc_type_name[i], jitc_mem_string(leak_size[i]),
                    leak_count[i], leak_count[i] > 1 ? "s" : "");
        }
    }
}
