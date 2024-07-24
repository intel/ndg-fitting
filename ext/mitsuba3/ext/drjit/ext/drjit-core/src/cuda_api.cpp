/*
    src/cuda_api.cpp -- Low-level interface to CUDA driver API

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "util.h"
#include "io.h"
#include "../resources/kernels.h"
#include "cuda_tex.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <lz4.h>

#if defined(DRJIT_DYNAMIC_CUDA)
// Driver API
CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int) = nullptr;
CUresult (*cuCtxSynchronize)() = nullptr;
CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice) = nullptr;
CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
CUresult (*cuDeviceGetCount)(int *) = nullptr;
CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
CUresult (*cuDeviceTotalMem)(size_t *, CUdevice) = nullptr;
CUresult (*cuDriverGetVersion)(int *) = nullptr;
CUresult (*cuEventCreate)(CUevent *, unsigned int) = nullptr;
CUresult (*cuEventDestroy)(CUevent) = nullptr;
CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
CUresult (*cuEventSynchronize)(CUevent) = nullptr;
CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent) = nullptr;
CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
CUresult (*cuInit)(unsigned int) = nullptr;
CUresult (*cuLaunchHostFunc)(CUstream, void (*)(void *), void *) = nullptr;
CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, CUstream, void **,
                           void **) = nullptr;
CUresult (*cuLinkAddData)(CUlinkState, int, void *, size_t,
                          const char *, unsigned int, int *,
                          void **) = nullptr;
CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *) = nullptr;
CUresult (*cuLinkCreate)(unsigned int, int *, void **,
                         CUlinkState *) = nullptr;
CUresult (*cuLinkDestroy)(CUlinkState) = nullptr;
CUresult (*cuPointerGetAttribute)(void* data, int, void*) = nullptr;
CUresult (*cuMemAdvise)(void *, size_t, int, CUdevice) = nullptr;
CUresult (*cuMemAlloc)(void **, size_t) = nullptr;
CUresult (*cuMemAllocHost)(void **, size_t) = nullptr;
CUresult (*cuMemAllocManaged)(void **, size_t, unsigned int) = nullptr;
CUresult (*cuMemFree)(void *) = nullptr;
CUresult (*cuMemFreeHost)(void *) = nullptr;
CUresult (*cuMemPrefetchAsync)(const void *, size_t, CUdevice, CUstream) = nullptr;
CUresult (*cuMemcpy)(void *, const void *, size_t) = nullptr;
CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream) = nullptr;
CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
CUresult (*cuModuleUnload)(CUmodule) = nullptr;
CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction, void *,
                                             size_t, int) = nullptr;
CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;
CUresult (*cuStreamCreate)(CUstream *, unsigned int) = nullptr;
CUresult (*cuStreamDestroy)(CUstream) = nullptr;
CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int) = nullptr;
CUresult (*cuArrayCreate)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *) = nullptr;
CUresult (*cuArray3DCreate)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *) = nullptr;
CUresult (*cuArray3DGetDescriptor)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray) = nullptr;
CUresult (*cuArrayDestroy)(CUarray) = nullptr;
CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                              const CUDA_TEXTURE_DESC *,
                              const CUDA_RESOURCE_VIEW_DESC *) = nullptr;
CUresult (*cuTexObjectGetResourceDesc)(CUDA_RESOURCE_DESC *,
                                       CUtexObject) = nullptr;
CUresult (*cuTexObjectDestroy)(CUtexObject) = nullptr;
CUresult (*cuMemcpy3DAsync)(const CUDA_MEMCPY3D *, CUstream) = nullptr;
CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *, CUstream) = nullptr;
CUresult (*cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream) = nullptr;
CUresult (*cuMemFreeAsync)(CUdeviceptr, CUstream) = nullptr;

static void *jitc_cuda_handle = nullptr;
#endif

// Dr.Jit API
static CUmodule *jitc_cuda_module = nullptr;

CUfunction *jitc_cuda_fill_64 = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_tiny = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_small = nullptr;
CUfunction *jitc_cuda_mkperm_phase_1_large = nullptr;
CUfunction *jitc_cuda_mkperm_phase_3 = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_tiny = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_small = nullptr;
CUfunction *jitc_cuda_mkperm_phase_4_large = nullptr;
CUfunction *jitc_cuda_transpose = nullptr;
CUfunction *jitc_cuda_scan_small_u32 = nullptr;
CUfunction *jitc_cuda_scan_large_u32 = nullptr;
CUfunction *jitc_cuda_scan_large_u32_init = nullptr;
CUfunction *jitc_cuda_compress_small = nullptr;
CUfunction *jitc_cuda_compress_large = nullptr;
CUfunction *jitc_cuda_poke[(int)VarType::Count] { };
CUfunction *jitc_cuda_block_copy[(int)VarType::Count] { };
CUfunction *jitc_cuda_block_sum [(int)VarType::Count] { };
CUfunction *jitc_cuda_reductions[(int) ReduceOp::Count]
                               [(int) VarType::Count] = { };
CUfunction *jitc_cuda_vcall_prepare = nullptr;
int jitc_cuda_devices = 0;
int jitc_cuda_version_major = 0;
int jitc_cuda_version_minor = 0;


static bool jitc_cuda_init_attempted = false;
static bool jitc_cuda_init_success = false;
CUresult jitc_cuda_cuinit_result = CUDA_ERROR_NOT_INITIALIZED;

bool jitc_cuda_init() {
    if (jitc_cuda_init_attempted)
        return jitc_cuda_init_success;
    jitc_cuda_init_attempted = true;

    // We have our own caching scheme, disable CUDA's JIT cache
#if 0
    // On hindsight, this is potentially dangerous because it also disables
    // important caching functionality in PyTorch/Tensorflow/etc.

    // #if !defined(_WIN32)
    //     putenv((char*)"CUDA_CACHE_DISABLE=1");
    // #else
    //     (void) _wputenv(L"CUDA_CACHE_DISABLE=1");
    // #endif
#endif

#if defined(DRJIT_DYNAMIC_CUDA)
    jitc_cuda_handle = nullptr;
#  if defined(_WIN32)
    const char* cuda_fname = "nvcuda.dll",
              * cuda_glob = nullptr;
#  elif defined(__linux__)
    const char *cuda_fname  = "libcuda.so",
               *cuda_glob   = "/usr/lib/{x86_64-linux-gnu,aarch64-linux-gnu}/libcuda.so.*";
#  else
    const char *cuda_fname  = "libcuda.dylib",
               *cuda_glob   = cuda_fname;
#  endif

#  if !defined(_WIN32)
    // Don't dlopen libcuda.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "cuInit"))
        jitc_cuda_handle = RTLD_NEXT;
#  endif

    if (!jitc_cuda_handle) {
        jitc_cuda_handle = jitc_find_library(cuda_fname, cuda_glob, "DRJIT_LIBCUDA_PATH");

        if (!jitc_cuda_handle) // CUDA library cannot be loaded, give up
            return false;
    }

    const char *symbol = nullptr;

    do {
        #define LOAD(name, ...)                                      \
            symbol = strlen(__VA_ARGS__ "") > 0                      \
                ? (#name "_" __VA_ARGS__) : #name;                   \
            name = decltype(name)(dlsym(jitc_cuda_handle, symbol));  \
            if (!name)                                               \
                break;                                               \
            symbol = nullptr

        LOAD(cuCtxEnablePeerAccess);
        LOAD(cuCtxSynchronize);
        LOAD(cuDeviceCanAccessPeer);
        LOAD(cuDeviceGet);
        LOAD(cuDeviceGetAttribute);
        LOAD(cuDeviceGetCount);
        LOAD(cuDeviceGetName);
        LOAD(cuDevicePrimaryCtxRelease, "v2");
        LOAD(cuDevicePrimaryCtxRetain);
        LOAD(cuDeviceTotalMem, "v2");
        LOAD(cuDriverGetVersion);
        LOAD(cuEventCreate);
        LOAD(cuEventDestroy, "v2");
        LOAD(cuEventRecord, "ptsz");
        LOAD(cuEventSynchronize);
        LOAD(cuEventElapsedTime);
        LOAD(cuFuncSetAttribute);
        LOAD(cuGetErrorName);
        LOAD(cuGetErrorString);
        LOAD(cuInit);
        LOAD(cuLaunchHostFunc, "ptsz");
        LOAD(cuLaunchKernel, "ptsz");
        LOAD(cuLinkAddData, "v2");
        LOAD(cuLinkComplete);
        LOAD(cuLinkCreate, "v2");
        LOAD(cuLinkDestroy);
        LOAD(cuMemAdvise);
        LOAD(cuMemAlloc, "v2");
        LOAD(cuMemAllocHost, "v2");
        LOAD(cuMemAllocManaged);
        LOAD(cuMemFree, "v2");
        LOAD(cuMemFreeHost);
        LOAD(cuMemPrefetchAsync, "ptsz");

        LOAD(cuMemcpy, "ptds");
        LOAD(cuMemcpyAsync, "ptsz");
        LOAD(cuMemsetD16Async, "ptsz");
        LOAD(cuMemsetD32Async, "ptsz");
        LOAD(cuMemsetD8Async, "ptsz");
        LOAD(cuModuleGetFunction);
        LOAD(cuModuleLoadData);
        LOAD(cuModuleUnload);
        LOAD(cuOccupancyMaxPotentialBlockSize);
        LOAD(cuCtxPushCurrent, "v2");
        LOAD(cuCtxPopCurrent, "v2");
        LOAD(cuStreamCreate);
        LOAD(cuStreamDestroy, "v2");
        LOAD(cuStreamSynchronize, "ptsz");
        LOAD(cuStreamWaitEvent, "ptsz");
        LOAD(cuPointerGetAttribute);
        LOAD(cuArrayCreate, "v2");
        LOAD(cuArray3DCreate, "v2");
        LOAD(cuArray3DGetDescriptor, "v2");
        LOAD(cuArrayDestroy);
        LOAD(cuTexObjectCreate);
        LOAD(cuTexObjectGetResourceDesc);
        LOAD(cuTexObjectDestroy);
        LOAD(cuMemcpy2DAsync, "v2_ptsz");
        LOAD(cuMemcpy3DAsync, "v2_ptsz");
        #undef LOAD
    } while (false);

    if (symbol) {
        jitc_log(LogLevel::Warn,
                "jit_cuda_init(): could not find symbol \"%s\" -- disabling "
                "CUDA backend!", symbol);
        return false;
    }

    // These two functions are optional
    cuMemAllocAsync = decltype(cuMemAllocAsync)(dlsym(jitc_cuda_handle, "cuMemAllocAsync_ptsz"));
    cuMemFreeAsync = decltype(cuMemFreeAsync)(dlsym(jitc_cuda_handle, "cuMemFreeAsync_ptsz"));
#endif

    jitc_cuda_cuinit_result = cuInit(0);
    if (jitc_cuda_cuinit_result != CUDA_SUCCESS)
        return false;

    cuda_check(cuDeviceGetCount(&jitc_cuda_devices));

    if (jitc_cuda_devices == 0) {
        jitc_log(
            LogLevel::Warn,
            "jit_cuda_init(): No devices found -- disabling CUDA backend!");
        return false;
    }

    int cuda_version;
    cuda_check(cuDriverGetVersion(&cuda_version));

    jitc_cuda_version_major = cuda_version / 1000;
    jitc_cuda_version_minor = (cuda_version % 1000) / 10;

    if (jitc_cuda_version_major < 10) {
        jitc_log(LogLevel::Warn,
                "jit_cuda_init(): your version of CUDA is too old (found %i.%i, "
                "at least 10.x is required) -- disabling CUDA backend!",
                jitc_cuda_version_major, jitc_cuda_version_minor);
        return false;
    }

    jitc_log(LogLevel::Info,
            "jit_cuda_init(): enabling CUDA backend (version %i.%i)",
            jitc_cuda_version_major, jitc_cuda_version_minor);

    for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
        for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++) {
            jitc_cuda_reductions[j][k] =
                (CUfunction *) malloc_check(sizeof(CUfunction) * jitc_cuda_devices);
        }
        jitc_cuda_poke[k] = (CUfunction *) malloc_check(
            sizeof(CUfunction) * jitc_cuda_devices);
        jitc_cuda_block_copy[k] = (CUfunction *) malloc_check(
            sizeof(CUfunction) * jitc_cuda_devices);
        jitc_cuda_block_sum[k] = (CUfunction *) malloc_check(
            sizeof(CUfunction) * jitc_cuda_devices);
    }

    jitc_cuda_module = (CUmodule *) malloc_check(sizeof(CUmodule) * jitc_cuda_devices);

    jitc_lz4_init();

    for (int i = 0; i < jitc_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        scoped_set_context guard(context);

        // Determine the device compute capability
        int cc_minor, cc_major;
        cuda_check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
        cuda_check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
        int cc = cc_major * 10 + cc_minor;

        // Choose an appropriate set of builtin kernels
        const char *kernels           = cc >= 70 ? kernels_70 : kernels_50;
        int kernels_size_uncompressed = cc >= 70 ? kernels_70_size_uncompressed
                                                 : kernels_50_size_uncompressed;
        int kernels_size_compressed   = cc >= 70 ? kernels_70_size_compressed
                                                 : kernels_50_size_compressed;
        XXH128_hash_t kernels_hash;
        kernels_hash.low64  = cc >= 70 ? kernels_70_hash_low64  : kernels_50_hash_low64;
        kernels_hash.high64 = cc >= 70 ? kernels_70_hash_high64 : kernels_50_hash_high64;

        // Decompress the supplemental PTX content
        char *uncompressed =
            (char *) malloc_check(size_t(kernels_size_uncompressed) + jitc_lz4_dict_size + 1);
        memcpy(uncompressed, jitc_lz4_dict, jitc_lz4_dict_size);
        char *uncompressed_ptx = uncompressed + jitc_lz4_dict_size;

        if (LZ4_decompress_safe_usingDict(
                kernels, uncompressed_ptx,
                kernels_size_compressed,
                kernels_size_uncompressed,
                uncompressed,
                jitc_lz4_dict_size) != kernels_size_uncompressed)
            jitc_fail("jit_cuda_init(): decompression of builtin kernels failed!");

        uncompressed_ptx[kernels_size_uncompressed] = '\0';

        hash_combine((size_t &) kernels_hash.low64, (size_t) cc);
        hash_combine((size_t &) kernels_hash.high64, (size_t) cc);

        Kernel kernel;
        if (!jitc_kernel_load(uncompressed_ptx, kernels_size_uncompressed,
                              JitBackend::CUDA, kernels_hash, kernel)) {
            jitc_cuda_compile(uncompressed_ptx, kernels_size_uncompressed,
                              kernel);
            jitc_kernel_write(uncompressed_ptx, kernels_size_uncompressed,
                              JitBackend::CUDA, kernels_hash, kernel);
        }

        free(uncompressed);

        // .. and register it with CUDA
        CUmodule m;
        cuda_check(cuModuleLoadData(&m, kernel.data));
        free(kernel.data);
        jitc_cuda_module[i] = m;

        #define LOAD(name)                                                       \
            if (i == 0)                                                          \
                jitc_cuda_##name = (CUfunction *) malloc_check(                  \
                    sizeof(CUfunction) * jitc_cuda_devices);                     \
            cuda_check(cuModuleGetFunction(&jitc_cuda_##name[i], m, #name))

        LOAD(fill_64);
        LOAD(mkperm_phase_1_tiny);
        LOAD(mkperm_phase_1_small);
        LOAD(mkperm_phase_1_large);
        LOAD(mkperm_phase_3);
        LOAD(mkperm_phase_4_tiny);
        LOAD(mkperm_phase_4_small);
        LOAD(mkperm_phase_4_large);
        LOAD(transpose);
        LOAD(scan_small_u32);
        LOAD(scan_large_u32);
        LOAD(scan_large_u32_init);
        LOAD(compress_small);
        LOAD(compress_large);
        LOAD(vcall_prepare);

        #undef LOAD

        int shared_memory_bytes;
        cuda_check(cuDeviceGetAttribute(
            &shared_memory_bytes,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, i));

        #define MAXIMIZE_SHARED(name)                                            \
            cuda_check(cuFuncSetAttribute(                                       \
                jitc_cuda_##name[i],                                             \
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,                 \
                shared_memory_bytes))

        // Max out the amount of shared memory available to the following kernels
        MAXIMIZE_SHARED(mkperm_phase_1_tiny);
        MAXIMIZE_SHARED(mkperm_phase_1_small);
        MAXIMIZE_SHARED(mkperm_phase_4_tiny);
        MAXIMIZE_SHARED(mkperm_phase_4_small);

        #undef MAXIMIZE_SHARED

        char name[16];
        CUfunction func;
        for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
            snprintf(name, sizeof(name), "poke_%s", type_name_short[k]);
            if (strstr(kernels_list, name)) {
                cuda_check(cuModuleGetFunction(&func, m, name));
                jitc_cuda_poke[k][i] = func;
            } else {
                jitc_cuda_poke[k][i] = nullptr;
            }

            snprintf(name, sizeof(name), "block_copy_%s", type_name_short[k]);
            if (strstr(kernels_list, name)) {
                cuda_check(cuModuleGetFunction(&func, m, name));
                jitc_cuda_block_copy[k][i] = func;
            } else {
                jitc_cuda_block_copy[k][i] = nullptr;
            }

            snprintf(name, sizeof(name), "block_sum_%s", type_name_short[k]);
            if (strstr(kernels_list, name)) {
                cuda_check(cuModuleGetFunction(&func, m, name));
                jitc_cuda_block_sum[k][i] = func;
            } else {
                jitc_cuda_block_sum[k][i] = nullptr;
            }

            for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++) {
                snprintf(name, sizeof(name), "reduce_%s_%s", reduction_name[j],
                         type_name_short[k]);
                if (strstr(kernels_list, name)) {
                    cuda_check(cuModuleGetFunction(&func, m, name));
                    jitc_cuda_reductions[j][k][i] = func;
                } else {
                    jitc_cuda_reductions[j][k][i] = nullptr;
                }
            }
        }
    }

    jitc_cuda_init_success = true;

    return true;
}

void jitc_cuda_compile(const char *buf, size_t buf_size, Kernel &kernel) {
    const uintptr_t log_size = 16384;
    char error_log[log_size], info_log[log_size];

    CUjit_option arg[] = {
        CU_JIT_OPTIMIZATION_LEVEL,
        CU_JIT_LOG_VERBOSE,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_LINE_INFO,
        CU_JIT_GENERATE_DEBUG_INFO
    };

    void *argv[] = {
        (void *) 4,
        (void *) 1,
        (void *) info_log,
        (void *) log_size,
        (void *) error_log,
        (void *) log_size,
        (void *) 0,
        (void *) 0
    };

    CUlinkState link_state;
    cuda_check(cuLinkCreate(sizeof(argv) / sizeof(void *), arg, argv, &link_state));

    int rt = cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void *) buf,
                           buf_size, nullptr, 0, nullptr, nullptr);
    if (rt != CUDA_SUCCESS)
        jitc_fail("jit_cuda_compile(): compilation failed. Please see the PTX "
                  "assembly listing and error message below:\n\n%s\n\n%s",
                  buf, error_log);

    void *link_output = nullptr;
    size_t link_output_size = 0;
    cuda_check(cuLinkComplete(link_state, &link_output, &link_output_size));
    if (rt != CUDA_SUCCESS)
        jitc_fail("jit_cuda_compile(): compilation failed. Please see the PTX "
                  "assembly listing and error message below:\n\n%s\n\n%s",
                  buf, error_log);

    jitc_trace("Detailed linker output:\n%s", info_log);

    kernel.data = malloc_check(link_output_size);
    kernel.size = (uint32_t) link_output_size;
    memcpy(kernel.data, link_output, link_output_size);

    // Destroy the linker invocation
    cuda_check(cuLinkDestroy(link_state));
}

void jitc_cuda_shutdown() {
    if (!jitc_cuda_init_success)
        return;

    jitc_log(Info, "jit_cuda_shutdown()");

    for (int i = 0; i < jitc_cuda_devices; ++i) {
        CUcontext context = nullptr;
        cuda_check(cuDevicePrimaryCtxRetain(&context, i));
        cuda_check(cuModuleUnload(jitc_cuda_module[i]));
        cuda_check(cuDevicePrimaryCtxRelease(i));
        cuda_check(cuDevicePrimaryCtxRelease(i));
    }

    jitc_cuda_devices = 0;

    #define Z(x) do { free(x); x = nullptr; } while (0)

    Z(jitc_cuda_fill_64);
    Z(jitc_cuda_mkperm_phase_1_tiny);
    Z(jitc_cuda_mkperm_phase_1_small);
    Z(jitc_cuda_mkperm_phase_1_large);
    Z(jitc_cuda_mkperm_phase_3);
    Z(jitc_cuda_mkperm_phase_4_tiny);
    Z(jitc_cuda_mkperm_phase_4_small);
    Z(jitc_cuda_mkperm_phase_4_large);
    Z(jitc_cuda_transpose);
    Z(jitc_cuda_scan_small_u32);
    Z(jitc_cuda_scan_large_u32);
    Z(jitc_cuda_scan_large_u32_init);
    Z(jitc_cuda_compress_small);
    Z(jitc_cuda_compress_large);
    Z(jitc_cuda_module);

    for (uint32_t k = 0; k < (uint32_t) VarType::Count; k++) {
        Z(jitc_cuda_poke[k]);
        Z(jitc_cuda_block_copy[k]);
        Z(jitc_cuda_block_sum[k]);
        for (uint32_t j = 0; j < (uint32_t) ReduceOp::Count; j++)
            Z(jitc_cuda_reductions[j][k]);
    }

    #undef Z

#if defined(DRJIT_DYNAMIC_CUDA)
    #define Z(x) x = nullptr

    Z(cuCtxEnablePeerAccess); Z(cuCtxSynchronize); Z(cuDeviceCanAccessPeer);
    Z(cuDeviceGet); Z(cuDeviceGetAttribute); Z(cuDeviceGetCount);
    Z(cuDeviceGetName); Z(cuDevicePrimaryCtxRelease);
    Z(cuDevicePrimaryCtxRetain); Z(cuDeviceTotalMem); Z(cuDriverGetVersion);
    Z(cuEventCreate); Z(cuEventDestroy); Z(cuEventRecord);
    Z(cuEventSynchronize); Z(cuEventElapsedTime); Z(cuFuncSetAttribute);
    Z(cuGetErrorName); Z(cuGetErrorString); Z(cuInit); Z(cuLaunchHostFunc);
    Z(cuLaunchKernel); Z(cuLinkAddData); Z(cuLinkComplete); Z(cuLinkCreate);
    Z(cuLinkDestroy); Z(cuMemAdvise); Z(cuMemAlloc); Z(cuMemAllocHost);
    Z(cuMemAllocManaged); Z(cuMemFree); Z(cuMemFreeHost); Z(cuMemPrefetchAsync);
    Z(cuMemcpy); Z(cuMemcpyAsync); Z(cuMemsetD16Async); Z(cuMemsetD32Async);
    Z(cuMemsetD8Async); Z(cuModuleGetFunction); Z(cuModuleLoadData);
    Z(cuModuleUnload); Z(cuOccupancyMaxPotentialBlockSize);
    Z(cuCtxPushCurrent); Z(cuCtxPopCurrent); Z(cuStreamCreate);
    Z(cuStreamDestroy); Z(cuStreamSynchronize); Z(cuStreamWaitEvent);
    Z(cuPointerGetAttribute); Z(cuArrayCreate); Z(cuArray3DCreate);
    Z(cuArray3DGetDescriptor); Z(cuArrayDestroy); Z(cuTexObjectCreate);
    Z(cuTexObjectGetResourceDesc); Z(cuTexObjectDestroy); Z(cuMemcpy2DAsync);
    Z(cuMemcpy3DAsync); Z(cuMemAllocAsync); Z(cuMemFreeAsync);

#if !defined(_WIN32)
    if (jitc_cuda_handle != RTLD_NEXT)
        dlclose(jitc_cuda_handle);
#else
    FreeLibrary((HMODULE) jitc_cuda_handle);
#endif

    jitc_cuda_handle = nullptr;

    #undef Z
#endif

    jitc_cuda_init_success = false;
    jitc_cuda_init_attempted = false;
}

void *jitc_cuda_lookup(const char *name) {
#if defined(_WIN32) && !defined(DRJIT_DYNAMIC_CUDA)
    jitc_raise("jit_cuda_lookup(): currently unsupported on Windows when the "
               "DRJIT_DYNAMIC_CUDA flag is disabled.");
#else
#  if defined(DRJIT_DYNAMIC_CUDA)
    void *handle = jitc_cuda_handle;
#  else
    void *handle = RTLD_DEFAULT;
#  endif
    void *ptr = dlsym(handle, name);
    if (!ptr)
        jitc_raise("jit_cuda_lookup(): function \"%s\" not found!", name);
    return ptr;
#endif
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (unlikely(errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED)) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        jitc_fail("cuda_check(): API error %04i (%s): \"%s\" in "
                  "%s:%i.", (int) errval, name, msg, file, line);
    }
}
