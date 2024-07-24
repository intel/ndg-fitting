#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>

#include "optix_stubs.h"

/// A simple hit/miss shader combo
const char *miss_and_closesthit_ptx = R"(
.version 6.0
.target sm_50
.address_size 64

.entry __miss__ms() {
	.reg .b32 %r<1>;
	mov.b32 %r0, 0;
	call _optix_set_payload, (%r0, %r0);
	ret;
}

.entry __closesthit__ch() {
	.reg .b32 %r<2>;
	mov.b32 %r0, 0;
	mov.b32 %r1, 1;
	call _optix_set_payload, (%r0, %r1);
	ret;
})";

namespace dr = drjit;

using Float = dr::CUDAArray<float>;
using UInt32 = dr::CUDAArray<uint32_t>;
using UInt64 = dr::CUDAArray<uint64_t>;
using Mask = dr::CUDAArray<bool>;

void demo() {
    OptixDeviceContext context = jit_optix_context();

    // =====================================================
    // Copy vertex data to device
    // =====================================================
    float h_vertices[9] = { -0.8f, -0.8f, 0.0f,
                             0.8f, -0.8f, 0.0f,
                             0.0f,  0.8f, 0.0f };
    void *d_vertices = jit_malloc(AllocType::Device, sizeof(h_vertices));
    jit_memcpy(JitBackend::CUDA, d_vertices, h_vertices, sizeof(h_vertices));

    // =====================================================
    // Build geometric acceleration data structure
    // =====================================================

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput triangle_input { };
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = 3;
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    OptixAccelBufferSizes gas_buffer_sizes;
    jit_optix_check(optixAccelComputeMemoryUsage(
        context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    void *d_gas_temp =
        jit_malloc(AllocType::Device, gas_buffer_sizes.tempSizeInBytes);

    void *d_gas =
        jit_malloc(AllocType::Device, gas_buffer_sizes.outputSizeInBytes);

    size_t *d_compacted_size = (size_t *) jit_malloc(AllocType::Device, sizeof(size_t)),
            h_compacted_size = 0;
    OptixAccelEmitDesc build_property = {};
    build_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    build_property.result = d_compacted_size;

    OptixTraversableHandle gas_handle;
    jit_optix_check(optixAccelBuild(
        context, 0, &accel_options, &triangle_input, 1, d_gas_temp,
        gas_buffer_sizes.tempSizeInBytes, d_gas,
        gas_buffer_sizes.outputSizeInBytes, &gas_handle, &build_property, 1));

    jit_free(d_gas_temp);
    jit_free(d_vertices);

    // =====================================================
    // Compactify geometric acceleration data structure
    // =====================================================

    jit_memcpy(JitBackend::CUDA, &h_compacted_size, d_compacted_size, sizeof(size_t));

    if (h_compacted_size < gas_buffer_sizes.outputSizeInBytes) {
        void *d_gas_compact = jit_malloc(AllocType::Device, h_compacted_size);
        jit_optix_check(optixAccelCompact(context, jit_cuda_stream(),
                                           gas_handle, d_gas_compact,
                                           h_compacted_size, &gas_handle));
        jit_free(d_gas);
        d_gas = d_gas_compact;
    }

    jit_free(d_compacted_size);

    // =====================================================
    // Configure options for OptiX pipeline
    // =====================================================

    OptixModuleCompileOptions module_compile_options { };
    module_compile_options.debugLevel =
        OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_compile_options { };
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; // <-- Use this when possible
    pipeline_compile_options.numPayloadValues = 1;
    pipeline_compile_options.numAttributeValues = 0;
#if defined(NDEBUG)
    pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_NONE;
#else
    pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        (unsigned) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // <-- Use this when possible

    // =====================================================
    // Create Optix module from supplemental PTX code
    // =====================================================

    char log[1024];
    size_t log_size = sizeof(log);

    OptixModule mod;
    int rv = optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        miss_and_closesthit_ptx, strlen(miss_and_closesthit_ptx), log,
        &log_size, &mod);
    if (rv) {
        fputs(log, stderr);
        jit_optix_check(rv);
    }

    // =====================================================
    // Create program groups (raygen provided by Dr.Jit..)
    // =====================================================

    OptixProgramGroupOptions pgo { };
    OptixProgramGroupDesc pgd[2] { };
    OptixProgramGroup pg[2];

    pgd[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgd[0].miss.module                  = mod;
    pgd[0].miss.entryFunctionName       = "__miss__ms";
    pgd[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgd[1].hitgroup.moduleCH            = mod;
    pgd[1].hitgroup.entryFunctionNameCH = "__closesthit__ch";

    log_size = sizeof(log);
    jit_optix_check(optixProgramGroupCreate(context, pgd, 2 /* two at once */,
                                            &pgo, log, &log_size, pg));

    // =====================================================
    // Shader binding table setup
    // =====================================================
    OptixShaderBindingTable sbt {};

    sbt.missRecordBase =
        jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
    sbt.missRecordStrideInBytes     = OPTIX_SBT_RECORD_HEADER_SIZE;
    sbt.missRecordCount             = 1;

    sbt.hitgroupRecordBase =
        jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
    sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
    sbt.hitgroupRecordCount         = 1;

    jit_optix_check(optixSbtRecordPackHeader(pg[0], sbt.missRecordBase));
    jit_optix_check(optixSbtRecordPackHeader(pg[1], sbt.hitgroupRecordBase));

    sbt.missRecordBase =
        jit_malloc_migrate(sbt.missRecordBase, AllocType::Device, 1);
    sbt.hitgroupRecordBase =
        jit_malloc_migrate(sbt.hitgroupRecordBase, AllocType::Device, 1);

    // =====================================================
    // Let Dr.Jit know about all of this
    // =====================================================

    UInt32 pipeline_handle = UInt32::steal(jit_optix_configure_pipeline(
        &pipeline_compile_options, // <-- these pointers must stay
        mod,
        pg, 2
    ));

    UInt32 sbt_handle = UInt32::steal(jit_optix_configure_sbt(
        &sbt, //     alive while Dr.Jit runs
        pipeline_handle.index()
    ));

    // Do four times to verify caching, with mask in it. 3 + 4
    for (int i = 0; i < 2; ++i) {
        // =====================================================
        // Generate a camera ray
        // =====================================================

        int res = 16;
        UInt32 index = dr::arange<UInt32>(res * res),
               x     = index % res,
               y     = index / res;

        Float ox = Float(x) * (2.0f / res) - 1.0f,
              oy = Float(y) * (2.0f / res) - 1.0f,
              oz = -1.f;

        Float dx = 0.f, dy = 0.f, dz = 1.f;

        Float mint = 0.f, maxt = 100.f, time = 0.f;

        UInt64 handle = dr::opaque<UInt64>(gas_handle);
        UInt32 ray_mask(255), ray_flags(0), sbt_offset(0), sbt_stride(1),
            miss_sbt_index(0);

        UInt32 payload_0(0);
        Mask mask = true;

        // =====================================================
        // Launch a ray tracing call
        // =====================================================

        uint32_t trace_args[] {
            handle.index(),
            ox.index(), oy.index(), oz.index(),
            dx.index(), dy.index(), dz.index(),
            mint.index(), maxt.index(), time.index(),
            ray_mask.index(), ray_flags.index(),
            sbt_offset.index(), sbt_stride.index(),
            miss_sbt_index.index(), payload_0.index()
        };

        jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
                            mask.index(), pipeline_handle.index(), sbt_handle.index());

        payload_0 = UInt32::steal(trace_args[15]);

        jit_var_eval(payload_0.index());

        UInt32 payload_0_host =
            UInt32::steal(jit_var_migrate(payload_0.index(), AllocType::Host));
        jit_sync_thread();

        for (int k = 0; k < res; ++k) {
            for (int j = 0; j < res; ++j)
                printf("%i ", payload_0_host.data()[k*res + j]);
            printf("\n");
        }
        printf("\n");
    }

    // =====================================================
    // Cleanup
    // =====================================================

    jit_free(d_gas);
}

int main(int, char **) {
    jit_init((int) JitBackend::CUDA);
    // jit_set_log_level_stderr(LogLevel::Trace);
    init_optix_api();
    demo();

    // Shut down Dr.Jit (will destroy OptiX device context)
    jit_shutdown();
    return 0;
}
