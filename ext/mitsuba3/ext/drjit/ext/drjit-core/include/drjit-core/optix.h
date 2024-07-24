/*
    drjit-core/optix.h -- JIT-compilation of kernels that use OptiX ray tracing

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "jit.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef void *OptixDeviceContext;
typedef void *OptixProgramGroup;
typedef void *OptixModule;
struct OptixPipelineCompileOptions;
struct OptixShaderBindingTable;

/// Return the OptiX device context associated with the currently active device
extern JIT_EXPORT OptixDeviceContext jit_optix_context();

/// Look up an OptiX function by name
extern JIT_EXPORT void *jit_optix_lookup(const char *name);

/**
 * \brief Check the return value of an OptiX function and terminate the
 * application with a helpful error message upon failure
 */
#define jit_optix_check(err) jit_optix_check_impl((err), __FILE__, __LINE__)
extern JIT_EXPORT void jit_optix_check_impl(int errval, const char *file,
                                            const int line);

/**
 * \brief Inform Dr.Jit about a partially created OptiX pipeline
 *
 * This function creates a JIT variable responsible for the lifetime management
 * of the OptiX pipeline and returns its corresponding index. Once the reference
 * count of this variable reaches zero, the OptiX resources related to this
 * pipeline will be freed.
 *
 * The returned index should be passed as argument to subsequent calls to
 * \c jit_optix_ray_trace in order to use this pipeline for the ray tracing
 * operations. See the docstring of \c jit_optix_ray_trace for a small example
 * of how those functions relate to each other.
 */
extern JIT_EXPORT uint32_t
jit_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                             OptixModule module,
                             const OptixProgramGroup *pg,
                             uint32_t pg_count);

/**
 * \brief Inform Dr.Jit about an OptiX Shader Binding Table
 *
 * This function creates a JIT variable responsible for the lifetime management
 * of the OptiX Shader Binding Table and returns its corresponding index. Once
 * the reference count of this variable reaches zero, the OptiX resources
 * related to this Shader Binding Table will be freed.
 *
 * The returned index should be passed as argument to subsequent calls to
 * \c jit_optix_ray_trace in order to use this Shader Binding Table for the ray
 * tracing operations. See the docstring of \c jit_optix_ray_trace for a small
 * example of how those functions relate to each other.
 */
extern JIT_EXPORT uint32_t
jit_optix_configure_sbt(const OptixShaderBindingTable *sbt, uint32_t pipeline);

/**
 * \brief  Update existing OptiX Shader Binding Table data
 *
 * This function updates the Shader Binding Table data held by the JIT
 * variable \c index previously created using \c jit_optix_configure_sbt. This
 * update is necessary when adding more geometry to an existing scene or when
 * sharing the OptiX pipeline and SBT across multiple scenes (e.g. ray tracing
 * against different scenes within the same megakernel).
 */
extern JIT_EXPORT void
jit_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt);

/**
  * \brief Insert a function call to optixTrace into the program
  *
  * The \c args list should contain a list of variable indices corresponding to
  * the 15 required function arguments <tt>handle, ox, oy, oz, dx, dy, dz, tmin,
  * tmax, time, mask, flags, sbt_offset, sbt_stride, miss_sbt_index</tt>.
  *
  * Up to 32 payload values can optionally be provided by setting \c n_args to a
  * value greater than 15. In this case, the corresponding elements will be
  * overwritten with the new variable indices with external reference count 1
  * containing the final payload value.
  *
  * The \c pipeline JIT variable index specifies the OptiX pipeline to be used
  * in the kernel executing this ray tracing operation. This index should be
  * computed using the \c jit_optix_configure_pipeline function.
  *
  * The \c sbt JIT variable index specifies the OptiX Shader Binding Table to be
  * used in the kernel executing this ray tracing operation. This index should
  * be computed using the \c jit_optix_configure_sbt function.
  *
  * Here is a small example of how to use those functions together:
  * <tt>
  *   OptixPipelineCompileOptions pco = ...;
  *   OptixModule mod = ...;
  *   OptixProgramGroup pgs = ...;
  *   uint32_t pg_count = ...;
  *   uint32_t pipeline_idx = jit_optix_configure_pipeline(pco, mod, pgs, pg_count);
  *
  *   OptixShaderBindingTable sbt = ...;
  *   uint32_t sbt_idx = jit_optix_configure_sbt(&sbt, pipeline_idx);
  *
  *   active_idx = ...;
  *   trace_args = ...;
  *   jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
  *                       active_idx, pipeline_idx, sbt_idx);
  * </tt>
  */
extern JIT_EXPORT void
jit_optix_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask,
                    uint32_t pipeline, uint32_t sbt);

/// Mark a variable as an expression requiring compilation via OptiX
extern JIT_EXPORT void jit_optix_mark(uint32_t index);

#if defined(__cplusplus)
}
#endif
