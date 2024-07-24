#include <drjit-core/optix.h>
#include <tsl/robin_map.h>
#include "optix_api.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "var.h"
#include "op.h"
#include "util.h"

#define OPTIX_ABI_VERSION 55

#define ENABLE_OPTIX_VALIDATION_MODE 0

#if defined(_WIN32)
#  include <windows.h>
#  include <cfgmgr32.h>
#else
#  include <dlfcn.h>
#endif

#define OPTIX_FUNCTION_TABLE_SIZE 43
static void *jitc_optix_table[OPTIX_FUNCTION_TABLE_SIZE] { };
static const char *jitc_optix_table_names[OPTIX_FUNCTION_TABLE_SIZE] = {
    "optixGetErrorName",
    "optixGetErrorString",
    "optixDeviceContextCreate",
    "optixDeviceContextDestroy",
    "optixDeviceContextGetProperty",
    "optixDeviceContextSetLogCallback",
    "optixDeviceContextSetCacheEnabled",
    "optixDeviceContextSetCacheLocation",
    "optixDeviceContextSetCacheDatabaseSizes",
    "optixDeviceContextGetCacheEnabled",
    "optixDeviceContextGetCacheLocation",
    "optixDeviceContextGetCacheDatabaseSizes",
    "optixModuleCreateFromPTX",
    "optixModuleCreateFromPTXWithTasks",
    "optixModuleGetCompilationState",
    "optixModuleDestroy",
    "optixBuiltinISModuleGet",
    "optixTaskExecute",
    "optixProgramGroupCreate",
    "optixProgramGroupDestroy",
    "optixProgramGroupGetStackSize",
    "optixPipelineCreate",
    "optixPipelineDestroy",
    "optixPipelineSetStackSize",
    "optixAccelComputeMemoryUsage",
    "optixAccelBuild",
    "optixAccelGetRelocationInfo",
    "optixAccelCheckRelocationCompatibility",
    "optixAccelRelocate",
    "optixAccelCompact",
    "optixConvertPointerToTraversableHandle",
    "reserved1",
    "reserved2",
    "optixSbtRecordPackHeader",
    "optixLaunch",
    "optixDenoiserCreate",
    "optixDenoiserDestroy",
    "optixDenoiserComputeMemoryResources",
    "optixDenoiserSetup",
    "optixDenoiserInvoke",
    "optixDenoiserComputeIntensity",
    "optixDenoiserComputeAverageColor",
    "optixDenoiserCreateWithUserModel"
};

using OptixResult = int;
using OptixProgramGroupKind = int;
using OptixCompileDebugLevel = int;
using OptixDeviceContext = void*;
using OptixLogCallback = void (*)(unsigned int, const char *, const char *, void *);
using OptixTask = void*;

#define OPTIX_EXCEPTION_FLAG_NONE                     0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW           1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH              2
#define OPTIX_EXCEPTION_FLAG_DEBUG                    8
#define OPTIX_ERROR_VALIDATION_FAILURE                7053
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE                0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL             0x2351
#define OPTIX_COMPILE_DEBUG_LEVEL_MODERATE            0x2353
#define OPTIX_COMPILE_DEBUG_LEVEL_FULL                0x2352
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0            0x2340
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_1            0x2341
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_2            0x2342
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_3            0x2343
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF      0
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL      ((int) 0xFFFFFFFF)
#define OPTIX_MODULE_COMPILE_STATE_COMPLETED          0x2364
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN               0x2421
#define OPTIX_PROGRAM_GROUP_KIND_CALLABLES            0x2425
#define OPTIX_PROGRAM_GROUP_KIND_MISS                 0x2422
#define OPTIX_SBT_RECORD_HEADER_SIZE                  32
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1

struct OptixDeviceContextOptions {
    OptixLogCallback logCallbackFunction;
    void *logCallbackData;
    int logCallbackLevel;
    int validationMode;
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    OptixPayloadType *payloadTypes;
};

struct OptixPipelineLinkOptions {
    unsigned int maxTraceDepth;
    OptixCompileDebugLevel debugLevel;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupCallables {
    OptixModule moduleDC;
    const char* entryFunctionNameDC;
    OptixModule moduleCC;
    const char* entryFunctionNameCC;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupCallables callables;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixStackSizes {
    unsigned int cssRG;
    unsigned int cssMS;
    unsigned int cssCH;
    unsigned int cssAH;
    unsigned int cssIS;
    unsigned int cssCC;
    unsigned int dssDC;
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

OptixResult (*optixQueryFunctionTable)(int, unsigned int, void *, const void **,
                                       void *, size_t) = nullptr;
const char *(*optixGetErrorName)(OptixResult r) = nullptr;
const char *(*optixGetErrorString)(OptixResult ) = nullptr;
OptixResult (*optixDeviceContextCreate)(
    CUcontext fromContext, const OptixDeviceContextOptions *,
    OptixDeviceContext *context) = nullptr;
OptixResult (*optixDeviceContextDestroy)(OptixDeviceContext) = nullptr;
OptixResult (*optixDeviceContextSetCacheEnabled)(OptixDeviceContext, int) = nullptr;
OptixResult (*optixDeviceContextSetCacheLocation)(OptixDeviceContext, const char *) = nullptr;
OptixResult (*optixModuleCreateFromPTX)(OptixDeviceContext,
                                        const OptixModuleCompileOptions *,
                                        const OptixPipelineCompileOptions *,
                                        const char *, size_t, char *, size_t *,
                                        OptixModule *) = nullptr;
OptixResult (*optixModuleCreateFromPTXWithTasks)(OptixDeviceContext,
                                                const OptixModuleCompileOptions *,
                                                const OptixPipelineCompileOptions *,
                                                const char *, size_t,
                                                char *, size_t *,
                                                OptixModule *,
                                                OptixTask *firstTask);
OptixResult (*optixModuleGetCompilationState)(OptixModule, int *);
OptixResult (*optixModuleDestroy)(OptixModule) = nullptr;
OptixResult (*optixTaskExecute)(OptixTask task, OptixTask *, unsigned int,
                                unsigned int *);
OptixResult (*optixProgramGroupCreate)(OptixDeviceContext,
                                       const OptixProgramGroupDesc *,
                                       unsigned int,
                                       const OptixProgramGroupOptions *, char *,
                                       size_t *, OptixProgramGroup *) = nullptr;
OptixResult (*optixProgramGroupDestroy)(OptixProgramGroup) = nullptr;
OptixResult (*optixPipelineCreate)(OptixDeviceContext,
                                   const OptixPipelineCompileOptions *,
                                   const OptixPipelineLinkOptions *,
                                   const OptixProgramGroup *, unsigned int,
                                   char *, size_t *, OptixPipeline *) = nullptr;
OptixResult (*optixPipelineDestroy)(OptixPipeline) = nullptr;
OptixResult (*optixLaunch)(OptixPipeline, CUstream, CUdeviceptr, size_t,
                           const OptixShaderBindingTable *, unsigned int,
                           unsigned int, unsigned int) = nullptr;
OptixResult (*optixSbtRecordPackHeader)(OptixProgramGroup, void*) = nullptr;
OptixResult (*optixPipelineSetStackSize)(OptixPipeline, unsigned int,
                                         unsigned int, unsigned int,
                                         unsigned int);
OptixResult (*optixProgramGroupGetStackSize)(OptixProgramGroup,
                                             OptixStackSizes *);

#define jitc_optix_check(err) jitc_optix_check_impl((err), __FILE__, __LINE__)
extern void jitc_optix_check_impl(OptixResult errval, const char *file, const int line);

#if defined(_WIN32)
void *jitc_optix_win32_load_alternative();
#endif

static bool jitc_optix_init_attempted = false;
static bool jitc_optix_init_success = false;
static void *jitc_optix_handle = nullptr;
static bool jitc_optix_cache_hit = false;
static bool jitc_optix_cache_global_disable = false;

bool jitc_optix_init() {
    if (jitc_optix_init_attempted)
        return jitc_optix_init_success;

    if (jitc_cuda_version_major == 11 && jitc_cuda_version_minor == 5)
        jitc_raise(
            "jit_optix_init(): DrJit considers the driver of your graphics "
            "card buggy and prone to miscompilation by (we explicitly do not "
            "support OptiX with CUDA 11.5, which roughly corresponds to driver "
            "versions >= 495 and < 510). Please install a newer driver version "
            "(or an older one is fine, too).");

    jitc_optix_init_attempted = true;
    jitc_optix_handle = nullptr;

#if defined(_WIN32)
    const char* optix_fname = "nvoptix.dll";
#elif defined(__linux__)
    const char *optix_fname  = "libnvoptix.so.1";
#else
    const char *optix_fname  = "libnvoptix.dylib";
#endif

#if !defined(_WIN32)
    // Don't dlopen OptiX if it was loaded by another library
    if (dlsym(RTLD_NEXT, "optixLaunch"))
        jitc_optix_handle = RTLD_NEXT;
#endif

    if (!jitc_optix_handle) {
        jitc_optix_handle = jitc_find_library(optix_fname, optix_fname, "DRJIT_LIBOPTIX_PATH");

#if defined(_WIN32)
        if (!jitc_optix_handle)
            jitc_optix_handle = jitc_optix_win32_load_alternative();
#endif

        if (!jitc_optix_handle) {
            jitc_log(Error, "jit_optix_init(): %s could not be loaded -- "
                          "disabling OptiX backend! Set the DRJIT_LIBOPTIX_PATH "
                          "environment variable to specify its path.", optix_fname);
            return false;
        }
    }

    // Load optixQueryFunctionTable from library
    optixQueryFunctionTable = decltype(optixQueryFunctionTable)(
        dlsym(jitc_optix_handle, "optixQueryFunctionTable"));

    if (!optixQueryFunctionTable) {
        jitc_log(Error, "jit_optix_init(): could not find symbol optixQueryFunctionTable");
        return false;
    }

    int rv = optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0,
                                     &jitc_optix_table, sizeof(jitc_optix_table));
    if (rv) {
        jitc_log(Error,
                "jit_optix_init(): Failed to load OptiX library! Very likely, "
                "your NVIDIA graphics driver is too old and not compatible "
                "with the version of OptiX that is being used. In particular, "
                "OptiX 7.4 requires driver revision R495.89 or newer.");

        return false;
    }

    #define LOOKUP(name) name = (decltype(name)) jitc_optix_lookup(#name)

    LOOKUP(optixGetErrorName);
    LOOKUP(optixGetErrorString);
    LOOKUP(optixDeviceContextCreate);
    LOOKUP(optixDeviceContextDestroy);
    LOOKUP(optixDeviceContextSetCacheEnabled);
    LOOKUP(optixDeviceContextSetCacheLocation);
    LOOKUP(optixModuleCreateFromPTX);
    LOOKUP(optixModuleCreateFromPTXWithTasks);
    LOOKUP(optixModuleGetCompilationState);
    LOOKUP(optixModuleDestroy);
    LOOKUP(optixTaskExecute);
    LOOKUP(optixProgramGroupCreate);
    LOOKUP(optixProgramGroupDestroy);
    LOOKUP(optixPipelineCreate);
    LOOKUP(optixPipelineDestroy);
    LOOKUP(optixLaunch);
    LOOKUP(optixSbtRecordPackHeader);
    LOOKUP(optixPipelineSetStackSize);
    LOOKUP(optixProgramGroupGetStackSize);

    #undef LOOKUP

    jitc_log(Info, "jit_optix_init(): loaded OptiX (via 7.4 ABI).");

    jitc_optix_init_success = true;
    return true;
}

void jitc_optix_log(unsigned int level, const char *tag, const char *message, void *) {
    size_t len = strlen(message);
    if (level <= (uint32_t) state.log_level_stderr)
        fprintf(stderr, "jit_optix_log(): [%s] %s%s", tag, message,
                (len > 0 && message[len - 1] == '\n') ? "" : "\n");

    if (strcmp(tag, "DISKCACHE") == 0 &&
        strncmp(message, "Cache miss for key", 18) == 0)
        jitc_optix_cache_hit = false;

    if (strcmp(tag, "DISK CACHE") == 0 &&
        strncmp(message, "OPTIX_CACHE_MAXSIZE is set to 0", 31) == 0)
        jitc_optix_cache_global_disable = true;
}

static OptixPipelineCompileOptions jitc_optix_default_compile_options() {
    OptixPipelineCompileOptions pco { };
    pco.numAttributeValues = 2;
    pco.pipelineLaunchParamsVariableName = "params";

#if defined(NDEBUG)
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif
    return pco;
}


OptixDeviceContext jitc_optix_context() {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixDeviceContext &ctx = state.devices[ts->device].optix_context;

    if (!ctx) {
        if (!jitc_optix_init())
            jitc_raise("Could not initialize OptiX!");

        OptixDeviceContextOptions ctx_opts {
            jitc_optix_log, nullptr, 4,
#if defined(NDEBUG) || !ENABLE_OPTIX_VALIDATION_MODE
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
#else
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
#endif
        };

        jitc_optix_check(optixDeviceContextCreate(ts->context, &ctx_opts, &ctx));

#if !defined(_WIN32)
        jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, jitc_temp_path));
#else
        size_t len = wcstombs(nullptr, jitc_temp_path, 0) + 1;
        std::unique_ptr<char[]> temp(new char[len]);
        wcstombs(temp.get(), jitc_temp_path, len);
        jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, temp.get()));
#endif
        jitc_optix_check(optixDeviceContextSetCacheEnabled(ctx, 1));
    }

    // =====================================================
    // Create default OptiX pipeline for testcases, etc.
    // =====================================================

    if (!state.optix_default_sbt_index) {
        OptixPipelineCompileOptions pco = jitc_optix_default_compile_options();
        OptixModuleCompileOptions mco { };
#if 1
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

        const char *minimal = ".version 6.0 .target sm_50 .address_size 64 "
                              ".entry __miss__dr() { ret; }";

        char log[128];
        size_t log_size = sizeof(log);

        OptixModule mod;
        jitc_optix_check(optixModuleCreateFromPTX(
            ctx, &mco, &pco, minimal, strlen(minimal), log, &log_size, &mod));

        OptixProgramGroupDesc pgd { };
        pgd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgd.miss.module = mod;
        pgd.miss.entryFunctionName = "__miss__dr";

        OptixProgramGroupOptions pgo { };
        OptixProgramGroup pg;
        log_size = sizeof(log);
        jitc_optix_check(optixProgramGroupCreate(ctx, &pgd, 1, &pgo, log, &log_size, &pg));

        OptixShaderBindingTable sbt {};
        sbt.missRecordBase = jitc_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
        jitc_optix_check(optixSbtRecordPackHeader(pg, sbt.missRecordBase));
        sbt.missRecordBase = jitc_malloc_migrate(sbt.missRecordBase, AllocType::Device, 1);
        sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.missRecordCount = 1;

        uint32_t pipeline_index = jitc_optix_configure_pipeline(&pco, mod, &pg, 1);
        auto it2 = state.extra.find(pipeline_index);
        if (it2 == state.extra.end())
            jitc_fail("jitc_optix_context(): 'extra' entry not found!");
        state.optix_default_pipeline = (OptixPipelineData*) it2->second.callback_data;

        uint32_t sbt_index = jitc_optix_configure_sbt(&sbt, pipeline_index);
        auto it = state.extra.find(sbt_index);
        if (it == state.extra.end())
            jitc_fail("jitc_optix_context(): 'extra' entry not found!");
        state.optix_default_sbt
            = (OptixShaderBindingTable*) it->second.callback_data;

        state.optix_default_sbt_index = sbt_index;
        jitc_var_dec_ref_ext(pipeline_index);
    }

    return ctx;
}

void jitc_optix_context_destroy(Device &d) {
    if (d.optix_context) {
        jitc_optix_check(optixDeviceContextDestroy(d.optix_context));
        d.optix_context = nullptr;
    }
}

void *jitc_optix_lookup(const char *name) {
    for (size_t i = 0; i < OPTIX_FUNCTION_TABLE_SIZE; ++i) {
        if (strcmp(name, jitc_optix_table_names[i]) == 0)
            return jitc_optix_table[i];
    }
    jitc_raise("jit_optix_lookup(): function \"%s\" not found!", name);
}

uint32_t jitc_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                       OptixModule module,
                                       const OptixProgramGroup *pg,
                                       uint32_t pg_count) {
    jitc_log(InfoSym, "jitc_optix_configure_pipeline(pg_count=%u)", pg_count);

    if (!pco || !module || !pg || pg_count == 0)
        jitc_raise("jitc_optix_configure_pipeline(): invalid input arguments!");

    uint32_t index = jitc_var_new_stmt(JitBackend::CUDA, VarType::Void, "", 1, 0, 0);

    Extra &extra = state.extra[index];
    Variable *v = jitc_var(index);
    v->extra = 1;
    v->optix = 1;
    v->size = 1;

    OptixPipelineData *p = new OptixPipelineData();
    p->module = module;
    p->program_groups = std::vector<OptixProgramGroup>();
    memcpy(&p->compile_options, pco, sizeof(OptixPipelineCompileOptions));
    for (uint32_t i = 0; i < pg_count; ++i)
        p->program_groups.push_back(pg[i]);
    extra.callback_data = p;

    // Set the OptiX pipeline to use when assembling the kernel
    extra.assemble = [](const Variable * /*v*/, const Extra &extra) {
        ThreadState *ts = thread_state(JitBackend::CUDA);
        OptixPipelineData *p = (OptixPipelineData*) extra.callback_data;
        if (ts->optix_pipeline == state.optix_default_pipeline) {
            jitc_log(InfoSym, "jit_assemble_cuda(): set OptiX pipeline for next kernel launch");
            ts->optix_pipeline = p;
        } else {
            if (ts->optix_pipeline != p)
                jitc_raise("jit_assemble_cuda(): a single OptiX pipeline can "
                           "be used per kernel!");
        }
    };

    // Free pipeline resources when this variable is destroyed
    extra.callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (free) {
            jitc_log(InfoSym, "jit_optix_configure(): free optix pipeline");
            OptixPipelineData *p = (OptixPipelineData*) ptr;
            for (size_t i = 0; i < p->program_groups.size(); i++)
                jitc_optix_check(optixProgramGroupDestroy(p->program_groups[i]));
            jitc_optix_check(optixModuleDestroy(p->module));
            delete p;
        }
    };

    return index;
}

uint32_t jitc_optix_configure_sbt(const OptixShaderBindingTable *sbt,
                                  uint32_t pipeline) {
    jitc_log(InfoSym, "jitc_optix_configure_sbt()");

    if (!sbt || !pipeline)
        jitc_raise("jitc_optix_configure_sbt(): invalid input arguments!");

    if (jitc_var_type(pipeline) != VarType::Void)
        jitc_raise("jitc_optix_configure_sbt(): type mismatch for pipeline argument!");

    uint32_t dep[1] = { pipeline };
    uint32_t index = jitc_var_new_stmt(JitBackend::CUDA, VarType::Void, "", 1, 1, dep);

    Extra &extra = state.extra[index];
    Variable *v = jitc_var(index);
    v->extra = 1;
    v->optix = 1;
    v->size = 1;

    extra.callback_data = new OptixShaderBindingTable();
    memcpy(extra.callback_data, sbt, sizeof(OptixShaderBindingTable));

    // Set the OptiX SBT to use when assembling the kernel
    extra.assemble = [](const Variable */*v*/, const Extra &extra) {
        ThreadState *ts = thread_state(JitBackend::CUDA);
        OptixShaderBindingTable *sbt = (OptixShaderBindingTable*) extra.callback_data;
        if (ts->optix_sbt == state.optix_default_sbt) {
            jitc_log(InfoSym, "jit_assemble_cuda(): set OptiX shader binding table "
                              "for next kernel launch");
            ts->optix_sbt = sbt;
        } else {
            if (ts->optix_sbt != sbt)
                jitc_raise("jit_assemble_cuda(): a single OptiX shader binding table "
                           "can be used per kernel!");
        }
    };

    // Free SBT resources when this variable is destroyed
    extra.callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (free) {
            jitc_log(InfoSym, "jit_optix_configure(): free optix shader binding table");
            OptixShaderBindingTable *sbt = (OptixShaderBindingTable*) ptr;
            if (sbt->hitgroupRecordBase)
                jitc_free(sbt->hitgroupRecordBase);
            if (sbt->missRecordBase)
                jitc_free(sbt->missRecordBase);
            delete sbt;
        }
    };

    return index;
}

void jitc_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt) {
    Extra &extra = state.extra[index];
    memcpy(extra.callback_data, sbt, sizeof(OptixShaderBindingTable));
}

bool jitc_optix_compile(ThreadState *ts, const char *buf, size_t buf_size,
                        const char *kern_name, Kernel &kernel) {
    char error_log[16384];

    if (!jitc_optix_init_success)
        jitc_fail("jit_optix_compile(): OptiX not initialized, make sure "
                  "evaluation happens before Optix shutdown!");

    // =====================================================
    // 2. Compile an OptiX module
    // =====================================================

    OptixModuleCompileOptions mco { };
#if 1
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

    jitc_optix_cache_hit = !jitc_optix_cache_global_disable;
    size_t log_size = sizeof(error_log);
    OptixDeviceContext &optix_context = state.devices[ts->device].optix_context;
    OptixPipelineData &pipeline = *ts->optix_pipeline;

    OptixTask task;
    int rv = optixModuleCreateFromPTXWithTasks(
        optix_context, &mco, &pipeline.compile_options, buf, buf_size,
        error_log, &log_size, &kernel.optix.mod, &task);

    if (rv) {
        jitc_log(Error, "jit_optix_compile(): "
                 "optixModuleCreateFromPTXWithTasks() failed. Please see the "
                 "PTX assembly listing and error message below:\n\n%s\n\n%s",
                 buf, error_log);
        jitc_optix_check(rv);
    }

    std::function<void(OptixTask)> execute_task = [&](OptixTask task) {
        unsigned int max_new_tasks = std::max(pool_size(), 1u);

        std::unique_ptr<OptixTask[]> new_tasks =
            std::make_unique<OptixTask[]>(max_new_tasks);
        unsigned int new_task_count = 0;
        optixTaskExecute(task, new_tasks.get(), max_new_tasks, &new_task_count);

        parallel_for(
            drjit::blocked_range<size_t>(0, new_task_count, 1),
            [&](const drjit::blocked_range<size_t> &range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    OptixTask new_task = new_tasks[i];
                    execute_task(new_task);
                }
            }
        );
    };
    execute_task(task);

    int compilation_state = 0;
    jitc_optix_check(
        optixModuleGetCompilationState(kernel.optix.mod, &compilation_state));
    if (compilation_state != OPTIX_MODULE_COMPILE_STATE_COMPLETED)
        jitc_fail("jit_optix_compile(): optixModuleGetCompilationState() "
                  "indicates that the compilation did not complete "
                  "succesfully. The module's compilation state is: %#06x",
                  compilation_state);

    // =====================================================
    // 3. Create an OptiX program group
    // =====================================================

    size_t n_programs = 1 + callables.size();

    OptixProgramGroupOptions pgo { };
    std::unique_ptr<OptixProgramGroupDesc[]> pgd(
        new OptixProgramGroupDesc[n_programs]);
    memset(pgd.get(), 0, n_programs * sizeof(OptixProgramGroupDesc));

    pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd[0].raygen.module = kernel.optix.mod;
    pgd[0].raygen.entryFunctionName = strdup(kern_name);

    for (size_t i = 0; i < callables.size(); ++i) {
        const char *s = strstr(callables[i].c_str(), "__direct_callable__");
        char tmp[52];
        memcpy(tmp, s, 51);
        tmp[51] = '\0';
        pgd[i + 1].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgd[i + 1].callables.moduleDC = kernel.optix.mod;
        pgd[i + 1].callables.entryFunctionNameDC = strdup(tmp);
    }

    kernel.optix.pg = new OptixProgramGroup[n_programs];
    kernel.optix.pg_count = (uint32_t) n_programs;

    log_size = sizeof(error_log);
    rv = optixProgramGroupCreate(optix_context, pgd.get(),
                                 (unsigned int) n_programs, &pgo, error_log,
                                 &log_size, kernel.optix.pg);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixProgramGroupCreate() "
                 "failed. Please see the PTX assembly listing and error "
                 "message below:\n\n%s\n\n%s", buf, error_log);
        jitc_optix_check(rv);
    }

    const size_t stride = OPTIX_SBT_RECORD_HEADER_SIZE;
    uint8_t *sbt_record = (uint8_t *)
        jitc_malloc(AllocType::HostPinned, n_programs * stride);

    for (size_t i = 0; i <= callables.size(); ++i)
        jitc_optix_check(optixSbtRecordPackHeader(
            kernel.optix.pg[i], sbt_record + stride * i));

    kernel.optix.sbt_record = (uint8_t *)
        jitc_malloc_migrate(sbt_record, AllocType::Device, 1);

    // =====================================================
    // 4. Create an OptiX pipeline
    // =====================================================

    OptixPipelineLinkOptions link_options {};
    link_options.maxTraceDepth = 1;
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    size_t size_before = pipeline.program_groups.size();
#if 0
    OptixStackSizes stack_sizes;
    unsigned int cssRG = 0, dssDC = 0;
#endif

    for (uint32_t i = 0; i < n_programs; ++i) {
        if (i == 0)
            free((char *) pgd[0].raygen.entryFunctionName);
        else
            free((char *) pgd[i].callables.entryFunctionNameDC);
        pipeline.program_groups.push_back(kernel.optix.pg[i]);
#if 0
        jitc_optix_check(optixProgramGroupGetStackSize(
            pipeline.program_groups[i], &stack_sizes));
        if (i == 0)
            cssRG = stack_sizes.cssRG;
        else
            dssDC = std::max(dssDC, stack_sizes.dssDC);
#endif
    }

    log_size = sizeof(error_log);
    rv = optixPipelineCreate(optix_context, &pipeline.compile_options,
                             &link_options, pipeline.program_groups.data(),
                             (unsigned int) pipeline.program_groups.size(),
                             error_log, &log_size, &kernel.optix.pipeline);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineCreate() failed. "
                 "Please see the PTX assembly listing and error message "
                 "below:\n\n%s\n\n%s", buf, error_log);
        jitc_optix_check(rv);
    }

    // Setup the direct stack and continuation stack size.
    // See OptiX documentation for more detail:
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#7235

    OptixStackSizes ssp = {};
    for (size_t i = 0; i < pipeline.program_groups.size(); ++i) {
        OptixStackSizes ss;
        rv = optixProgramGroupGetStackSize(pipeline.program_groups[i], &ss);
        if (rv) {
            jitc_log(Error, "jit_optix_compile(): optixProgramGroupGetStackSize() "
                            "failed:\n\n%s", error_log);
            jitc_optix_check(rv);
        }
        ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
        ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
        ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
        ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
        ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
        ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
        ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
    }

    if (ssp.cssCC > 0)
        jitc_log(Error, "jit_optix_compile(): an OptiX program is using "
                        "continuous callables which is not supported by Dr.Jit!");

    unsigned int max_dc_depth = 2; // Support nested VCalls
    unsigned int dc_stack_size_from_traversal = 0; // DC is not invoked from IS or AH.
    unsigned int dc_stack_size_from_state = max_dc_depth * ssp.dssDC; // DC is invoked from RG, MS, or CH.
    unsigned int continuation_stack_size = ssp.cssRG + std::max(std::max(ssp.cssCH, ssp.cssMS), ssp.cssAH + ssp.cssIS);

    unsigned int max_traversable_graph_depth = 2; // Support instancing
    if (pipeline.compile_options.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
        max_traversable_graph_depth = 1;

    rv = optixPipelineSetStackSize(kernel.optix.pipeline,
                                   dc_stack_size_from_traversal,
                                   dc_stack_size_from_state,
                                   continuation_stack_size,
                                   max_traversable_graph_depth);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineSetStackSize() "
                        "failed:\n\n%s", error_log);
        jitc_optix_check(rv);
    }

    kernel.size = 0;
    kernel.data = nullptr;
    pipeline.program_groups.resize(size_before);
    return jitc_optix_cache_hit;
}

void jitc_optix_free(const Kernel &kernel) {
    jitc_optix_check(optixPipelineDestroy(kernel.optix.pipeline));
    for (uint32_t i = 0; i < kernel.optix.pg_count; ++i)
        jitc_optix_check(optixProgramGroupDestroy(kernel.optix.pg[i]));
    delete[] kernel.optix.pg;
    jitc_optix_check(optixModuleDestroy(kernel.optix.mod));
    jitc_free(kernel.optix.sbt_record);
}

void jitc_optix_launch(ThreadState *ts, const Kernel &kernel,
                       uint32_t launch_size, const void *args,
                       uint32_t n_args) {
    OptixShaderBindingTable &sbt = *ts->optix_sbt;
    sbt.raygenRecord = kernel.optix.sbt_record;

    if (kernel.optix.pg_count > 1) {
        sbt.callablesRecordBase = kernel.optix.sbt_record + OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordCount = kernel.optix.pg_count - 1;
    }

    uint32_t offset = 0;

    /* On Linux and Windows (when using a TCM-style driver), accept kernel
       launches up to the maximum value of 2^30 threads. On Windows, when using
       a more common WDDM-style driver, such a long-running kernel may freeze
       the OS graphics and eventually cause a device reset. That's not good, so
       we submit smaller batches that correspond roughly to 1 sample/pixel for a
       full HD resolution frame. */
    uint32_t limit = state.devices[ts->device].wddm_driver ? (1 << 21) : (1 << 30);

    while (launch_size > 0) {
        uint32_t sub_launch_size = launch_size < limit ? launch_size : limit;

        // Bytes 4..8 used to store optional offset parameter
        if (offset != 0)
            cuMemsetD32Async(
                (CUdeviceptr) ((uint8_t *) args + sizeof(uint32_t)),
                offset, 1, ts->stream);

        jitc_optix_check(
            optixLaunch(kernel.optix.pipeline, ts->stream, (CUdeviceptr) args,
                        n_args * sizeof(void *), &sbt,
                        sub_launch_size, 1, 1));

        launch_size -= sub_launch_size;
        offset += sub_launch_size;
    }
}

void jitc_optix_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask,
                          uint32_t pipeline, uint32_t sbt) {
    VarType types[]{ VarType::UInt64,  VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32};

    if (n_args < 15)
        jitc_raise("jit_optix_ray_trace(): too few arguments (got %u < 15)", n_args);

    uint32_t np = n_args - 15, size = 0;
    if (np > 32)
        jitc_raise("jit_optix_ray_trace(): too many payloads (got %u > 32)", np);

    if (jitc_var_type(pipeline) != VarType::Void)
        jitc_raise("jit_optix_ray_trace(): type mismatch for pipeline argument!");

    if (jitc_var_type(sbt) != VarType::Void)
        jitc_raise("jit_optix_ray_trace(): type mismatch for pipeline argument!");

    // Validate input types, determine size of the operation
    bool placeholder = false, dirty = false;
    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = i < n_args ? args[i] : mask;
        VarType ref = i < n_args ? types[i] : VarType::Bool;
        const Variable *v = jitc_var(index);
        if ((VarType) v->type != ref)
            jitc_raise("jit_optix_ray_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) ref]);
        size = std::max(size, v->size);
        placeholder |= (bool) v->placeholder;
        dirty |= (bool) v->ref_count_se;
    }

    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = (i < n_args) ? args[i] : mask;
        const Variable *v = jitc_var(index);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_optix_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::CUDA));
        dirty = false;

        for (uint32_t i = 0; i <= n_args; ++i) {
            uint32_t index = (i < n_args) ? args[i] : mask;
            dirty |= (bool) jitc_var(index)->ref_count_se;
        }

        if (dirty)
            jitc_raise("jit_optix_ray_trace(): inputs remain dirty after evaluation!");
    }

    // Potentially apply any masks on the mask stack
    Ref valid = steal(jitc_var_mask_apply(mask, size));

    jitc_log(InfoSym, "jit_optix_ray_trace(): tracing %u ray%s, %u payload value%s%s.",
             size, size != 1 ? "s" : "", np, np == 1 ? "" : "s",
             placeholder ? " (part of a recorded computation)" : "");

    uint32_t dep[3] = { valid, pipeline, sbt };
    uint32_t special =
        jitc_var_new_stmt(JitBackend::CUDA, VarType::Void, "", 1, 3, dep);

    // Associate extra record with this variable
    Extra &extra = state.extra[special];
    Variable *v = jitc_var(special);
    v->extra = 1;
    v->optix = 1;
    v->size = size;

    // Register dependencies
    extra.n_dep = n_args;
    extra.dep = (uint32_t *) malloc(sizeof(uint32_t) * extra.n_dep);
    memcpy(extra.dep, args, n_args * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_args; ++i)
        jitc_var_inc_ref_int(args[i]);

    extra.assemble = [](const Variable *v2, const Extra &extra) {
        uint32_t payload_count = extra.n_dep - 15;
        buffer.fmt("    .reg.u32 %%u%u_result_<32>;\n", v2->reg_index);

        const Variable *mask_v = jitc_var(v2->dep[0]);
        bool masked = !mask_v->literal || mask_v->value != 1;
        if (masked)
            buffer.fmt("    @!%s%u bra l_masked_%u;\n", type_prefix[mask_v->type],
                       mask_v->reg_index, v2->reg_index);

        buffer.fmt("    .reg.u32 %%u%u_payload_type;\n", v2->reg_index);
        buffer.fmt("    mov.u32 %%u%u_payload_type, 0;\n", v2->reg_index);
        buffer.fmt("    .reg.u32 %%u%u_payload_count;\n", v2->reg_index);
        buffer.fmt("    mov.u32 %%u%u_payload_count, %u;\n", v2->reg_index,
                   payload_count);

        buffer.put("    call (");

        for (uint32_t i = 0; i < 32; ++i)
            buffer.fmt("%%u%u_result_%u%s", v2->reg_index, i,
                       i + 1 < 32 ? ", " : "");
        buffer.put("), _optix_trace_typed_32, (");

        buffer.fmt("%%u%u_payload_type, ", v2->reg_index);
        for (uint32_t i = 0; i < 15; ++i) {
            const Variable *v3 = jitc_var(extra.dep[i]);
            buffer.fmt("%s%u, ", type_prefix[v3->type], v3->reg_index);
        }

        buffer.fmt("%%u%u_payload_count, ", v2->reg_index);
        for (uint32_t i = 15; i < extra.n_dep; ++i) {
            const Variable *v3 = jitc_var(extra.dep[i]);
            buffer.fmt("%s%u%s", type_prefix[v3->type], v3->reg_index,
                       (i - 15 < 32) ? ", " : "");
        }
        for (uint32_t i = payload_count; i < 32; ++i)
            buffer.fmt("%%u%u_result_%u%s", v2->reg_index, i,
                       (i + 1 < 32) ? ", " : "");

        buffer.put(");\n");

        if (masked)
            buffer.fmt("\nl_masked_%u:\n", v2->reg_index);
    };

    for (uint32_t i = 0; i < np; ++i) {
        char tmp[80];
        snprintf(tmp, sizeof(tmp), "mov.u32 $r0, $r1_result_%u", i);
        args[15 + i] = jitc_var_new_stmt(JitBackend::CUDA, VarType::UInt32, tmp,
                                         0, 1, &special);
        uint32_t index = args[15] + i;
        Variable *v2 = jitc_var(index);
        jitc_cse_drop(index, v2);
        v2->placeholder = placeholder;
        jitc_cse_put(index, v2);
    }

    jitc_var_dec_ref_ext(special);
}

void jitc_optix_mark(uint32_t index) {
    jitc_var(index)->optix = true;
}

void jitc_optix_shutdown() {
    if (!jitc_optix_init_success)
        return;

    jitc_log(Info, "jit_optix_shutdown()");

    #define Z(x) x = nullptr

    #if !defined(_WIN32)
        if (jitc_optix_handle != RTLD_NEXT)
            dlclose(jitc_optix_handle);
    #else
        FreeLibrary((HMODULE) jitc_optix_handle);
    #endif

    memset(jitc_optix_table, 0, sizeof(jitc_optix_table));

    Z(optixGetErrorName);
    Z(optixGetErrorString);
    Z(optixDeviceContextCreate);
    Z(optixDeviceContextDestroy);
    Z(optixDeviceContextSetCacheEnabled);
    Z(optixDeviceContextSetCacheLocation);
    Z(optixModuleCreateFromPTX);
    Z(optixModuleDestroy);
    Z(optixProgramGroupCreate);
    Z(optixProgramGroupDestroy);
    Z(optixPipelineCreate);
    Z(optixPipelineDestroy);
    Z(optixLaunch);
    Z(optixSbtRecordPackHeader);
    Z(optixPipelineSetStackSize);
    Z(optixProgramGroupGetStackSize);

    jitc_optix_handle = nullptr;

    #undef Z

    jitc_optix_init_success = false;
    jitc_optix_init_attempted = false;
}

void jitc_optix_check_impl(OptixResult errval, const char *file,
                          const int line) {
    if (unlikely(errval != 0)) {
        const char *name = optixGetErrorName(errval),
                   *msg  = optixGetErrorString(errval);
        jitc_fail("jit_optix_check(): API error %04i (%s): \"%s\" in "
                 "%s:%i.", (int) errval, name, msg, file, line);
    }
}

void jit_optix_check_impl(int errval, const char *file, const int line) {
    if (errval) {
        lock_guard guard(state.lock);
        jitc_optix_check_impl(errval, file, line);
    }
}

#if defined(_WIN32)
/**
 * Alternative way of finding OptiX based on the official API: nvoptix.dll
 * may not be on the path. Since it is co-located with the OpenGL drivers,
 * we should also enumerate all of them and double-check there.
 */
void *jitc_optix_win32_load_alternative() {
    const char *guid        = "{4d36e968-e325-11ce-bfc1-08002be10318}",
               *suffix      = "nvoptix.dll",
               *driver_name = "OpenGLDriverName";

    unsigned long size  = 0,
                  flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT,
                  suffix_len = (unsigned long) strlen(suffix);

    if (CM_Get_Device_ID_List_SizeA(&size, guid, flags))
        return nullptr;

    std::unique_ptr<char[]> dev_names(new char[size]);
    if (CM_Get_Device_ID_ListA(guid, dev_names.get(), size, flags))
        return nullptr;

    for (char *p = dev_names.get(); *p != '\0'; p += strlen(p) + 1) {
        unsigned long node_handle = 0;
        if (CM_Locate_DevNodeA(&node_handle, p, CM_LOCATE_DEVNODE_NORMAL))
            continue;

        HKEY reg_key = 0;
        if (CM_Open_DevNode_Key(node_handle, KEY_QUERY_VALUE, 0,
                                RegDisposition_OpenExisting, &reg_key,
                                CM_REGISTRY_SOFTWARE))
            continue;

        auto guard = scope_guard([reg_key]{ RegCloseKey(reg_key); });

        if (RegQueryValueExA(reg_key, driver_name, 0, 0, 0, &size))
            continue;

        std::unique_ptr<char[]> path(new char[size + suffix_len]);
        if (RegQueryValueExA(reg_key, driver_name, 0, 0, (LPBYTE) path.get(), &size))
            continue;

        for (int i = (int) size - 1; i >= 0 && path[i] != '\\'; --i)
            path[i] = '\0';

        strncat(path.get(), suffix, suffix_len);
        void* handle = (void *) LoadLibraryA(path.get());

        if (handle)
            return handle;
    }
    return nullptr;
}

#endif
