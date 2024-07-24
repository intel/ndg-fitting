#include <drjit-core/optix.h>
#define OPTIX_STUBS_IMPL
#include "optix_stubs.h"

void init_optix_api() {
    jit_optix_context(); // Ensure OptiX is initialized

    #define L(name) name = (decltype(name)) jit_optix_lookup(#name);

    L(optixAccelComputeMemoryUsage);
    L(optixAccelBuild);
    L(optixAccelCompact);
    L(optixModuleCreateFromPTX);
    L(optixModuleDestroy)
    L(optixProgramGroupCreate);
    L(optixProgramGroupDestroy)
    L(optixSbtRecordPackHeader);

    #undef L
}
