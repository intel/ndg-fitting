/*
    src/registry.cpp -- Pointer registry for vectorized method calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"

static_assert(sizeof(void*) == 8, "32 bit architectures are not supported!");

/// Register a pointer with Dr.Jit's pointer registry
uint32_t jitc_registry_put(JitBackend backend, const char *domain, void *ptr) {
    if (unlikely(ptr == nullptr))
        jitc_raise("jit_registry_put(): cannot register the null pointer!");

    Registry* registry = state.registry(backend);

    // Create the rev. map. first and throw if the pointer is already registered
    auto it_rev = registry->rev.try_emplace(ptr, RegistryKey(domain, 0));
    if (unlikely(!it_rev.second))
        jitc_raise("jit_registry_put(): pointer %p was already registered!", ptr);

    // Get or create the bookkeeping record associated with the domain
    auto it_head = registry->fwd.try_emplace(RegistryKey(domain, 0), nullptr);
    uintptr_t &value_head = (uintptr_t &) it_head.first.value();

    uint32_t next_avail = (uint32_t) (value_head >> 32),
             counter    = (uint32_t)  value_head;

    if (next_avail) {
        // Case 1: some previously released IDs are available, reuse them
        auto it_next = registry->fwd.find(RegistryKey(domain, next_avail));
        if (unlikely(it_next == registry->fwd.end()))
            jitc_fail("jit_registry_put(): data structure corrupted (1)!");

        uintptr_t &value_next = (uintptr_t &) it_next.value();

        uint32_t next_avail_2 = (uint32_t) (value_next >> 32),
                 unused       = (uint32_t)  value_next;

        if (unlikely(unused != 0))
            jitc_fail("jit_registry_put(): data structure corrupted (2)!");

        // Update bookkeeping record with next element from linked list
        value_head = ((uintptr_t) next_avail_2 << 32) | counter;

        // Initialize reused entry
        value_next = (uintptr_t) ptr;

        // Finally, update reverse mapping
        it_rev.first.value().id = next_avail;

        jitc_trace("jit_registry_put(" DRJIT_PTR ", domain=\"%s\"): %u (reused)",
                  (uintptr_t) ptr, domain, next_avail);

        return next_avail;
    } else {
        // Case 2: need to create a new record

        // Increment counter and update bookkeeping record
        value_head = ++counter;

        // Create new record
        auto it_new =
            registry->fwd.try_emplace(RegistryKey(domain, counter), ptr);
        if (unlikely(!it_new.second))
            jitc_fail("jit_registry_put(): data structure corrupted (3)!");

        // Finally, update reverse mapping
        it_rev.first.value().id = counter;

        jitc_trace("jit_registry_put(" DRJIT_PTR ", domain=\"%s\"): %u (new)",
                  (uintptr_t) ptr, domain, counter);

        return counter;
    }
}

/// Remove a pointer from the registry
void jitc_registry_remove(JitBackend backend, void *ptr) {
    if (ptr == nullptr)
        return;

    jitc_trace("jit_registry_remove(" DRJIT_PTR ")", (uintptr_t) ptr);

    Registry* registry = state.registry(backend);
    auto it_rev = registry->rev.find(ptr);
    if (unlikely(it_rev == registry->rev.end()))
        jitc_raise("jit_registry_remove(): pointer %p could not be found!", ptr);

    RegistryKey key = it_rev.value();

    // Get the forward record associated with the pointer
    auto it_fwd = registry->fwd.find(RegistryKey(key.domain, key.id));
    if (unlikely(it_fwd == registry->fwd.end()))
        jitc_raise("jit_registry_remove(): data structure corrupted (1)!");

    // Get the bookkeeping record associated with the domain
    auto it_head = registry->fwd.find(RegistryKey(key.domain, 0));
    if (unlikely(it_head == registry->fwd.end()))
        jitc_raise("jit_registry_remove(): data structure corrupted (2)!");

    // Update the head node
    uintptr_t &value_head = (uintptr_t &) it_head.value();
    uint32_t next_avail = (uint32_t) (value_head >> 32),
             counter    = (uint32_t)  value_head;
    value_head = ((uintptr_t) key.id << 32) | counter;

    // Update the current node
    uintptr_t &value_fwd = (uintptr_t &) it_fwd.value();
    value_fwd = (uintptr_t) next_avail << 32;

    // Remove reverse mapping
    registry->rev.erase(it_rev);
}

/// Query the ID associated a registered pointer
uint32_t jitc_registry_get_id(JitBackend backend, const void *ptr) {
    if (ptr == nullptr)
        return 0;

    Registry* registry = state.registry(backend);
    auto it = registry->rev.find(ptr);
    if (unlikely(it == registry->rev.end()))
        jitc_raise("jit_registry_get_id(): pointer %p could not be found!", ptr);
    return it.value().id;
}

/// Query the domain associated a registered pointer
const char *jitc_registry_get_domain(JitBackend backend, const void *ptr) {
    if (ptr == nullptr)
        return nullptr;

    Registry* registry = state.registry(backend);
    auto it = registry->rev.find(ptr);
    if (unlikely(it == registry->rev.end()))
        jitc_raise("jit_registry_get_domain(): pointer %p could not be found!", ptr);
    return it.value().domain;
}

/// Query the pointer associated a given domain and ID
void *jitc_registry_get_ptr(JitBackend backend, const char *domain, uint32_t id) {
    if (id == 0)
        return nullptr;

    Registry* registry = state.registry(backend);
    auto it = registry->fwd.find(RegistryKey(domain, id));
    if (unlikely(it == registry->fwd.end()))
        return nullptr;

    uintptr_t value = (uintptr_t) it.value();
    if ((value & 0xFFFFFFFFu) == 0)
        return nullptr;

    return (void *) value;
}

/// Compact the registry and release unused IDs and attributes
void jitc_registry_trim() {
    auto trim_registry = [](JitBackend backend) {
        Registry* registry = state.registry(backend);
        Registry tmp_registry;

        for (auto &kv : registry->fwd) {
            const char *domain = kv.first.domain;
            uint32_t id = kv.first.id;
            void *ptr = kv.second;

            if (id != 0 &&
                ((uint32_t) (uintptr_t) ptr != 0u ||
                registry->rev.find(ptr) != registry->rev.end())) {
                tmp_registry.fwd.insert(kv);

                auto it_head =
                    tmp_registry.fwd.try_emplace(RegistryKey(domain, 0), nullptr);

                uintptr_t &value_head = (uintptr_t &) it_head.first.value();
                value_head = std::max(value_head, (uintptr_t) id);
            }
        }

        if (registry->fwd.size() != tmp_registry.fwd.size()) {
            jitc_trace("jit_registry_trim(): removed %zu / %zu entries.",
                    registry->fwd.size() - tmp_registry.fwd.size(),
                    registry->fwd.size());

            registry->fwd = std::move(tmp_registry.fwd);
        }

        for (auto &kv : registry->attributes) {
            if (registry->fwd.find(RegistryKey(kv.first.domain, 0)) != registry->fwd.end()) {
                tmp_registry.attributes.insert(kv);
            } else {
                if (backend == JitBackend::CUDA)
                    cuda_check(cuMemFree((CUdeviceptr) kv.second.ptr));
                else
                    free(kv.second.ptr);
            }
        }

        if (registry->attributes.size() != tmp_registry.attributes.size()) {
            jitc_trace("jit_registry_trim(): removed %zu / %zu attributes.",
                    registry->attributes.size() - tmp_registry.attributes.size(),
                    registry->attributes.size());
            registry->attributes = std::move(tmp_registry.attributes);
        }
    };

    if (state.backends & (uint32_t) JitBackend::CUDA)
        trim_registry(JitBackend::CUDA);
    if (state.backends & (uint32_t) JitBackend::LLVM)
        trim_registry(JitBackend::LLVM);
}

/// Clear the registry and release all IDs and attributes
void jitc_registry_clean() {
    auto clean_registry = [](JitBackend backend) {
        Registry* registry = state.registry(backend);
        for (auto &kv : registry->attributes) {
            if (backend == JitBackend::CUDA)
                cuda_check(cuMemFree((CUdeviceptr) kv.second.ptr));
            else
                free(kv.second.ptr);
        }
        registry->fwd.clear();
        registry->rev.clear();
        registry->attributes.clear();
    };

    if (state.backends & (uint32_t) JitBackend::CUDA)
        clean_registry(JitBackend::CUDA);
    if (state.backends & (uint32_t) JitBackend::LLVM)
        clean_registry(JitBackend::LLVM);
}

/// Provide a bound (<=) on the largest ID associated with a domain
uint32_t jitc_registry_get_max(JitBackend backend, const char *domain) {
    Registry* registry = state.registry(backend);
    // Get the bookkeeping record associated with the domain
    auto it_head = registry->fwd.find(RegistryKey(domain, 0));
    if (unlikely(it_head == registry->fwd.end()))
        return 0;

    uintptr_t value_head = (uintptr_t) it_head.value();
    return (uint32_t) value_head; // extract counter field
}

void jitc_registry_shutdown() {
    jitc_registry_trim();

    if (state.backends & (uint32_t) JitBackend::CUDA) {
        Registry* registry = state.registry(JitBackend::CUDA);
        if (!registry->fwd.empty() || !registry->rev.empty())
            jitc_log(Warn, "jit_registry_shutdown(): CUDA registry leaked %zu "
                    "forward and %zu reverse mappings!",
                    registry->fwd.size(), registry->rev.size());

        if (!registry->attributes.empty())
            jitc_log(Warn, "jit_registry_shutdown(): CUDA registry leaked "
                    "%zu attributes!",
                    registry->attributes.size());
    }

    if (state.backends & (uint32_t) JitBackend::LLVM) {
        Registry* registry = state.registry(JitBackend::LLVM);
        if (!registry->fwd.empty() || !registry->rev.empty())
            jitc_log(Warn, "jit_registry_shutdown(): LLVM registry leaked %zu "
                    "forward and %zu reverse mappings!",
                    registry->fwd.size(), registry->rev.size());

        if (!registry->attributes.empty())
            jitc_log(Warn, "jit_registry_shutdown(): LLVM registry leaked "
                    "%zu attributes!",
                    registry->attributes.size());
    }
}

void jitc_registry_set_attr(JitBackend backend, void *ptr, const char *name,
                            const void *value, size_t isize) {
    ThreadState *ts = thread_state(backend);
    Registry* registry = state.registry(backend);
    auto it = registry->rev.find(ptr);
    if (unlikely(it == registry->rev.end()))
        jitc_raise("jit_registry_set_attr(): pointer %p could not be found!", ptr);

    const char *domain = it.value().domain;
    uint32_t id = it.value().id;

    jitc_trace("jit_registry_set_attr(" DRJIT_PTR ", id=%u, name=\"%s\", size=%zu)",
              (uintptr_t) ptr, id, name, isize);

    AttributeValue &attr = registry->attributes[AttributeKey(domain, name)];
    if (attr.isize == 0)
        attr.isize = (uint32_t) isize;
    else if (attr.isize != isize)
        jitc_raise("jit_registry_set_attr(): incompatible size!");

    if (id >= attr.count) {
        uint32_t new_count = std::max(id + 1, std::max(8u, attr.count * 2u));
        size_t old_size = (size_t) attr.count * (size_t) isize;
        size_t new_size = (size_t) new_count * (size_t) isize;
        void *new_ptr;

        if (backend == JitBackend::CUDA) {
            scoped_set_context guard(state.devices[0].context);
            CUresult ret = cuMemAlloc((CUdeviceptr *) &new_ptr, new_size);
            if (ret != CUDA_SUCCESS) {
                jitc_flush_malloc_cache(true, true);
                cuda_check(cuMemAlloc((CUdeviceptr *) &new_ptr, new_size));
            }

            if (old_size != 0)
                cuda_check(cuMemcpyAsync((CUdeviceptr) new_ptr,
                                         (CUdeviceptr) attr.ptr, old_size,
                                         ts->stream));
            cuda_check(
                cuMemsetD8Async((CUdeviceptr)((uint8_t *) new_ptr + old_size), 0,
                                new_size - old_size, ts->stream));

            cuda_check(cuMemFree((CUdeviceptr) attr.ptr));
        } else {
            new_ptr = malloc_check(new_size);

            if (old_size != 0)
                memcpy(new_ptr, attr.ptr, old_size);
            memset((uint8_t *) new_ptr + old_size, 0, new_size - old_size);

            free(attr.ptr);
        }

        attr.ptr = new_ptr;
        attr.count = new_count;
    }

    if (backend == JitBackend::CUDA) {
        cuda_check(
            cuMemcpyAsync((CUdeviceptr)((uint8_t *) attr.ptr + id * isize),
                          (CUdeviceptr) value, isize, ts->stream));
    } else {
        memcpy((uint8_t *) attr.ptr + id * isize, value, isize);
    }
}

const void *jitc_registry_attr_data(JitBackend backend, const char *domain,
                                    const char *name) {
    Registry* registry = state.registry(backend);
    auto it = registry->attributes.find(AttributeKey(domain, name));
    if (unlikely(it == registry->attributes.end())) {
        if (jitc_registry_get_max(backend, domain) > 0) {
            jitc_log(Warn,
                     "jit_registry_attr_data(): entry with domain=\"%s\", "
                     "name=\"%s\" not found!",
                     domain, name);
        }
        return nullptr;
    }
    return it.value().ptr;
}
