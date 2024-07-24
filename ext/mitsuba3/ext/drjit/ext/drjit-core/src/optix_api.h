/*
    src/optix_api.h -- Low-level interface to OptiX

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "cuda_api.h"

using OptixDeviceContext = void *;
using OptixProgramGroup = void*;
using OptixModule = void*;
struct OptixPipelineCompileOptions;
struct OptixShaderBindingTable;
struct ThreadState;

struct OptixPipelineData;

/// Create an OptiX device context on the current ThreadState
extern OptixDeviceContext jitc_optix_context();

/// Destroy an OptiX device context
struct Device;
extern void jitc_optix_context_destroy(Device &d);

/// Look up an OptiX function by name
extern void *jitc_optix_lookup(const char *name);

/// Unload the OptiX library
extern void jitc_optix_shutdown();

/// Inform Dr.Jit about a partially created OptiX pipeline
extern uint32_t jitc_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                              OptixModule module,
                                              const OptixProgramGroup *pg,
                                              uint32_t pg_count);

/// Inform Dr.Jit about an OptiX Shader Binding Table
extern uint32_t jitc_optix_configure_sbt(const OptixShaderBindingTable *sbt,
                                         uint32_t pipeline);

/// Overwrite existing OptiX Shader Binding Table given an index
extern void jitc_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt);

/// Insert a function call to optixTrace into the program
extern void jitc_optix_ray_trace(uint32_t nargs, uint32_t *args, uint32_t mask,
                                 uint32_t pipeline, uint32_t sbt);

/// Compile an OptiX kernel
extern bool jitc_optix_compile(ThreadState *ts, const char *buffer,
                               size_t buffer_size, const char *kernel_name,
                               Kernel &kernel);

/// Free a compiled OptiX kernel
extern void jitc_optix_free(const Kernel &kernel);

/// Perform an OptiX kernel launch
extern void jitc_optix_launch(ThreadState *ts, const Kernel &kernel,
                              uint32_t size, const void *args, uint32_t n_args);

/// Mark a variable as an expression requiring compilation via OptiX
extern void jitc_optix_mark(uint32_t index);

/// Optional: set the desired launch size
extern void jitc_optix_set_launch_size(uint32_t width, uint32_t height, uint32_t samples);
