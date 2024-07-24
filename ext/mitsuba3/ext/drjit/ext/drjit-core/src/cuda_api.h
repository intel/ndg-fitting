/*
    src/cuda_api.h -- Low-level interface to CUDA driver API

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

#if !defined(DRJIT_DYNAMIC_CUDA)
#  include <cuda.h>
#else
#  define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#  define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#  define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS 89
#  define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY 83
#  define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#  define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED 115
#  define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#  define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#  define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#  define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#  define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
#  define CU_DEVICE_ATTRIBUTE_TCC_DRIVER 35

#  define CU_DEVICE_CPU -1

#  define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#  define CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 9
#  define CU_FUNC_CACHE_PREFER_L1 2

#  define CU_JIT_INPUT_PTX 1
#  define CU_JIT_INFO_LOG_BUFFER 3
#  define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#  define CU_JIT_ERROR_LOG_BUFFER 5
#  define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#  define CU_JIT_OPTIMIZATION_LEVEL 7
#  define CU_JIT_GENERATE_DEBUG_INFO 11
#  define CU_JIT_LOG_VERBOSE 12
#  define CU_JIT_GENERATE_LINE_INFO 13

#  define CU_LAUNCH_PARAM_BUFFER_POINTER (void *) 1
#  define CU_LAUNCH_PARAM_BUFFER_SIZE (void *) 2
#  define CU_LAUNCH_PARAM_END (void *) 0

#  define CU_MEM_ATTACH_GLOBAL 1
#  define CU_MEM_ADVISE_SET_READ_MOSTLY 1
#  define CU_SHAREDMEM_CARVEOUT_MAX_L1 0

#  define CU_STREAM_DEFAULT 0
#  define CU_STREAM_NON_BLOCKING 1
#  define CU_EVENT_DEFAULT 0
#  define CU_EVENT_DISABLE_TIMING 2
#  define CU_MEMORYTYPE_HOST 1
#  define CU_POINTER_ATTRIBUTE_MEMORY_TYPE 2

#  define CUDA_ERROR_INVALID_VALUE 1
#  define CUDA_ERROR_NOT_INITIALIZED 3
#  define CUDA_ERROR_DEINITIALIZED 4
#  define CUDA_ERROR_NOT_FOUND 500
#  define CUDA_ERROR_OUT_OF_MEMORY 2
#  define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED 704
#  define CUDA_SUCCESS 0

using CUcontext    = struct CUctx_st *;
using CUmodule     = struct CUmod_st *;
using CUfunction   = struct CUfunc_st *;
using CUlinkState  = struct CUlinkState_st *;
using CUstream     = struct CUstream_st *;
using CUevent      = struct CUevent_st *;
using CUresult     = int;
using CUdevice     = int;
using CUdeviceptr  = void *;
using CUjit_option = int;

// Driver API
extern CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int);
extern CUresult (*cuCtxSynchronize)();
extern CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice);
extern CUresult (*cuDeviceGet)(CUdevice *, int);
extern CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
extern CUresult (*cuDeviceGetCount)(int *);
extern CUresult (*cuDeviceGetName)(char *, int, CUdevice);
extern CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
extern CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
extern CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
extern CUresult (*cuDriverGetVersion)(int *);
extern CUresult (*cuEventCreate)(CUevent *, unsigned int);
extern CUresult (*cuEventDestroy)(CUevent);
extern CUresult (*cuEventRecord)(CUevent, CUstream);
extern CUresult (*cuEventSynchronize)(CUevent);
extern CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent);
extern CUresult (*cuFuncSetAttribute)(CUfunction, int, int);
extern CUresult (*cuGetErrorName)(CUresult, const char **);
extern CUresult (*cuGetErrorString)(CUresult, const char **);
extern CUresult (*cuInit)(unsigned int);
extern CUresult (*cuLaunchHostFunc)(CUstream, void (*)(void *), void *);
extern CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **,
                                  void **);
extern CUresult (*cuLinkAddData)(CUlinkState, int, void *, size_t, const char *,
                                 unsigned int, int *, void **);
extern CUresult (*cuLinkComplete)(CUlinkState, void **, size_t *);
extern CUresult (*cuLinkCreate)(unsigned int, int *, void **, CUlinkState *);
extern CUresult (*cuLinkDestroy)(CUlinkState);
extern CUresult (*cuPointerGetAttribute)(void* data, int, void*);
extern CUresult (*cuMemAdvise)(void *, size_t, int, CUdevice);
extern CUresult (*cuMemAlloc)(void **, size_t);
extern CUresult (*cuMemAllocHost)(void **, size_t);
extern CUresult (*cuMemAllocManaged)(void **, size_t, unsigned int);
extern CUresult (*cuMemFree)(void *);
extern CUresult (*cuMemFreeHost)(void *);
extern CUresult (*cuMemPrefetchAsync)(const void *, size_t, CUdevice, CUstream);
extern CUresult (*cuMemcpy)(void *, const void *, size_t);
extern CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream);
extern CUresult (*cuMemsetD16Async)(void *, unsigned short, size_t, CUstream);
extern CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream);
extern CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream);
extern CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
extern CUresult (*cuModuleLoadData)(CUmodule *, const void *);
extern CUresult (*cuModuleUnload)(CUmodule);
extern CUresult (*cuOccupancyMaxPotentialBlockSize)(int *, int *, CUfunction,
                                                    void *, size_t, int);
extern CUresult (*cuCtxPushCurrent)(CUcontext);
extern CUresult (*cuCtxPopCurrent)(CUcontext*);
extern CUresult (*cuStreamCreate)(CUstream *, unsigned int);
extern CUresult (*cuStreamDestroy)(CUstream);
extern CUresult (*cuStreamSynchronize)(CUstream);
extern CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
extern CUresult (*cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream);
extern CUresult (*cuMemFreeAsync)(CUdeviceptr, CUstream);
#endif

// Dr.Jit API
extern CUfunction *jitc_cuda_fill_64;
extern CUfunction *jitc_cuda_mkperm_phase_1_tiny;
extern CUfunction *jitc_cuda_mkperm_phase_1_small;
extern CUfunction *jitc_cuda_mkperm_phase_1_large;
extern CUfunction *jitc_cuda_mkperm_phase_3;
extern CUfunction *jitc_cuda_mkperm_phase_4_tiny;
extern CUfunction *jitc_cuda_mkperm_phase_4_small;
extern CUfunction *jitc_cuda_mkperm_phase_4_large;
extern CUfunction *jitc_cuda_transpose;
extern CUfunction *jitc_cuda_scan_small_u32;
extern CUfunction *jitc_cuda_scan_large_u32;
extern CUfunction *jitc_cuda_scan_large_u32_init;
extern CUfunction *jitc_cuda_compress_small;
extern CUfunction *jitc_cuda_compress_large;
extern CUfunction *jitc_cuda_poke[(int) VarType::Count];
extern CUfunction *jitc_cuda_block_copy[(int) VarType::Count];
extern CUfunction *jitc_cuda_block_sum [(int) VarType::Count];
extern CUfunction *jitc_cuda_reductions[(int) ReduceOp::Count]
                                      [(int) VarType::Count];
extern CUfunction *jitc_cuda_vcall_prepare;

extern int jitc_cuda_devices;

extern int jitc_cuda_version_major;
extern int jitc_cuda_version_minor;
extern CUresult jitc_cuda_cuinit_result;

/// Try to load the CUDA backend
extern bool jitc_cuda_init();

struct Kernel;

/// Compile an IR string
extern void jitc_cuda_compile(const char *str, size_t size, Kernel &kernel);

/// Fully unload CUDA
extern void jitc_cuda_shutdown();

/// Look up a device driver function
extern void *jitc_cuda_lookup(const char *name);

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)
extern void cuda_check_impl(CUresult errval, const char *file, const int line);

struct scoped_set_context {
    scoped_set_context(CUcontext ctx) {
        cuda_check(cuCtxPushCurrent(ctx));
    }
    ~scoped_set_context() {
        cuda_check(cuCtxPopCurrent(nullptr));
    }
};

struct scoped_set_context_maybe {
    scoped_set_context_maybe(CUcontext ctx) : active(ctx != nullptr) {
        if (active)
            cuda_check(cuCtxPushCurrent(ctx));
    }
    ~scoped_set_context_maybe() {
        if (active)
            cuda_check(cuCtxPopCurrent(nullptr));
    }
    bool active;
};
