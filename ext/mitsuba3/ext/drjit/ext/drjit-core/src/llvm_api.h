/*
    src/llvm_api.h -- Low-level interface to LLVM driver API

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdlib.h>
#include <stdint.h>

struct Kernel;

/// Target CPU string used by the LLVM backend
extern char *jitc_llvm_target_cpu;

/// Target feature string used by the LLVM backend
extern char *jitc_llvm_target_features;

/// Vector width used by the LLVM backend
extern uint32_t jitc_llvm_vector_width;

/// LLVM version
extern uint32_t jitc_llvm_version_major;
extern uint32_t jitc_llvm_version_minor;
extern uint32_t jitc_llvm_version_patch;

/// Strings related to the vector width, used by template engine
extern char *jitc_llvm_vector_width_str;
extern char *jitc_llvm_counter_str;
extern char **jitc_llvm_ones_str;

/// Try to load the LLVM backend
extern bool jitc_llvm_init();

/// Compile an IR string
extern void jitc_llvm_compile(const char *str, size_t size,
                              const char *kernel_name, Kernel &kernel);

/// Dump disassembly for the given kernel
extern void jitc_llvm_disasm(const Kernel &kernel);

/// Fully unload LLVM
extern void jitc_llvm_shutdown();

/// Override the target architecture
extern void jitc_llvm_set_target(const char *target_cpu,
                                 const char *target_features,
                                 uint32_t vector_width);

/// Convenience function for intrinsic function selection
extern int jitc_llvm_if_at_least(uint32_t vector_width,
                                 const char *feature);

/// Insert a ray tracing function call into the LLVM program
extern void jitc_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                                const uint32_t *in, uint32_t *out);
