/*
    src/registry.h -- Pointer registry for vectorized method calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>

/// Register a pointer with Dr.Jit's pointer registry
extern uint32_t jitc_registry_put(JitBackend backend, const char *domain,
                                  void *ptr);

/// Remove a pointer from the registry
extern void jitc_registry_remove(JitBackend backend, void *ptr);

/// Provide a bound (<=) on the largest ID associated with a domain
extern uint32_t jitc_registry_get_max(JitBackend backend, const char *domain);

/// Query the ID associated a registered pointer
extern uint32_t jitc_registry_get_id(JitBackend backend, const void *ptr);

/// Query the domain associated a registered pointer
extern const char *jitc_registry_get_domain(JitBackend backend,
                                            const void *ptr);

/// Query the pointer associated a given domain and ID
extern void *jitc_registry_get_ptr(JitBackend backend, const char *domain,
                                   uint32_t id);

/// Compact the registry and release unused IDs
extern void jitc_registry_trim();

/// Clear the registry and release all IDs and attributes
extern void jitc_registry_clean();

/// Shut down the pointer registry (reports leaks)
extern void jitc_registry_shutdown();

/// Set a custom per-pointer attribute
extern void jitc_registry_set_attr(JitBackend backend, void *ptr,
                                   const char *name, const void *value,
                                   size_t size);

/// Retrieve a pointer to a buffer storing a specific attribute
extern const void *jitc_registry_attr_data(JitBackend backend,
                                           const char *domain,
                                           const char *name);
