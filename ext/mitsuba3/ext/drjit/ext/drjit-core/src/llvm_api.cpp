/*
    src/llvm_api.cpp -- Low-level interface to LLVM driver API

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "llvm_api.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "profiler.h"
#include "eval.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <sys/mman.h>
#endif

#include <lz4.h>

#if !defined(DRJIT_DYNAMIC_LLVM)
#  include <llvm-c/Core.h>
#  include <llvm-c/ExecutionEngine.h>
#  include <llvm-c/Disassembler.h>
#  include <llvm-c/IRReader.h>
#  include <llvm-c/Transforms/IPO.h>
#else

#if defined(__aarch64__)
    #define LLVMInitializeDrJitTarget       LLVMInitializeAArch64Target
    #define LLVMInitializeDrJitTargetInfo   LLVMInitializeAArch64TargetInfo
    #define LLVMInitializeDrJitTargetMC     LLVMInitializeAArch64TargetMC
    #define LLVMInitializeDrJitAsmPrinter   LLVMInitializeAArch64AsmPrinter
    #define LLVMInitializeDrJitDisassembler LLVMInitializeAArch64Disassembler
#else
    #define LLVMInitializeDrJitTarget       LLVMInitializeX86Target
    #define LLVMInitializeDrJitTargetInfo   LLVMInitializeX86TargetInfo
    #define LLVMInitializeDrJitTargetMC     LLVMInitializeX86TargetMC
    #define LLVMInitializeDrJitAsmPrinter   LLVMInitializeX86AsmPrinter
    #define LLVMInitializeDrJitDisassembler LLVMInitializeX86Disassembler
#endif

/// LLVM API
using LLVMBool = int;
using LLVMDisasmContextRef = void *;
using LLVMExecutionEngineRef = void *;
using LLVMModuleRef = void *;
using LLVMMemoryBufferRef = void *;
using LLVMContextRef = void *;
using LLVMPassManagerRef = void *;
using LLVMPassManagerBuilderRef = void *;
using LLVMMCJITMemoryManagerRef = void *;
using LLVMTargetMachineRef = void *;
using LLVMCodeModel = int;

using LLVMMemoryManagerAllocateCodeSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *);

using LLVMMemoryManagerAllocateDataSectionCallback =
    uint8_t *(*) (void *, uintptr_t, unsigned, unsigned, const char *,
                  LLVMBool);

using LLVMMemoryManagerFinalizeMemoryCallback = LLVMBool (*)(void *, char **);

using LLVMMemoryManagerDestroyCallback = void (*)(void *Opaque);

struct LLVMMCJITCompilerOptions {
  unsigned OptLevel;
  LLVMCodeModel CodeModel;
  LLVMBool NoFramePointerElim;
  LLVMBool EnableFastISel;
  void *MCJMM;
};

static void (*LLVMLinkInMCJIT)() = nullptr;
static void (*LLVMInitializeDrJitAsmPrinter)() = nullptr;
static void (*LLVMInitializeDrJitDisassembler)() = nullptr;
static void (*LLVMInitializeDrJitTarget)() = nullptr;
static void (*LLVMInitializeDrJitTargetInfo)() = nullptr;
static void (*LLVMInitializeDrJitTargetMC)() = nullptr;
static char *(*LLVMCreateMessage)(const char *) = nullptr;
static void (*LLVMDisposeMessage)(char *) = nullptr;
static char *(*LLVMGetDefaultTargetTriple)() = nullptr;
static char *(*LLVMGetHostCPUName)() = nullptr;
static char *(*LLVMGetHostCPUFeatures)() = nullptr;
static LLVMContextRef (*LLVMGetGlobalContext)() = nullptr;
static LLVMDisasmContextRef (*LLVMCreateDisasm)(const char *, void *, int,
                                                void *, void *) = nullptr;
static void (*LLVMDisasmDispose)(LLVMDisasmContextRef) = nullptr;
static int (*LLVMSetDisasmOptions)(LLVMDisasmContextRef, uint64_t) = nullptr;
static LLVMModuleRef (*LLVMModuleCreateWithName)(const char *) = nullptr;
static LLVMBool (*LLVMCreateMCJITCompilerForModule)(LLVMExecutionEngineRef *,
                                                    LLVMModuleRef,
                                                    LLVMMCJITCompilerOptions *,
                                                    size_t, char **) = nullptr;
static LLVMMCJITMemoryManagerRef (*LLVMCreateSimpleMCJITMemoryManager)(
    void *, LLVMMemoryManagerAllocateCodeSectionCallback,
    LLVMMemoryManagerAllocateDataSectionCallback,
    LLVMMemoryManagerFinalizeMemoryCallback,
    LLVMMemoryManagerDestroyCallback) = nullptr;
static void (*LLVMDisposeExecutionEngine)(LLVMExecutionEngineRef) = nullptr;
static void (*LLVMAddModule)(LLVMExecutionEngineRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposeModule)(LLVMModuleRef) = nullptr;
static LLVMMemoryBufferRef (*LLVMCreateMemoryBufferWithMemoryRange)(
    const char *, size_t, const char *, LLVMBool) = nullptr;
static LLVMBool (*LLVMParseIRInContext)(LLVMContextRef, LLVMMemoryBufferRef,
                                        LLVMModuleRef *, char **) = nullptr;
static char *(*LLVMPrintModuleToString)(LLVMModuleRef) = nullptr;
static uint64_t (*LLVMGetFunctionAddress)(LLVMExecutionEngineRef, const char *);
static uint64_t (*LLVMGetGlobalValueAddress)(LLVMExecutionEngineRef, const char *);
static LLVMBool (*LLVMRemoveModule)(LLVMExecutionEngineRef, LLVMModuleRef,
                                    LLVMModuleRef *, char **) = nullptr;
static size_t (*LLVMDisasmInstruction)(LLVMDisasmContextRef, uint8_t *,
                                       uint64_t, uint64_t, char *,
                                       size_t) = nullptr;
static LLVMPassManagerRef (*LLVMCreatePassManager)() = nullptr;
static void (*LLVMRunPassManager)(LLVMPassManagerRef, LLVMModuleRef) = nullptr;
static void (*LLVMDisposePassManager)(LLVMPassManagerRef) = nullptr;
static void (*LLVMAddLICMPass)(LLVMPassManagerRef) = nullptr;

static LLVMPassManagerBuilderRef (*LLVMPassManagerBuilderCreate)() = nullptr;
static void (*LLVMPassManagerBuilderSetOptLevel)(LLVMPassManagerBuilderRef,
                                                 unsigned) = nullptr;
static void (*LLVMPassManagerBuilderPopulateModulePassManager)(
    LLVMPassManagerBuilderRef, LLVMPassManagerRef) = nullptr;
static void (*LLVMPassManagerBuilderDispose)(LLVMPassManagerBuilderRef) = nullptr;

static LLVMTargetMachineRef (*LLVMGetExecutionEngineTargetMachine)(
    LLVMExecutionEngineRef) = nullptr;
static bool (*LLVMVerifyModule)(LLVMModuleRef, int action, char **msg);

#define LLVMDisassembler_Option_PrintImmHex       2
#define LLVMDisassembler_Option_AsmPrinterVariant 4

static void *jitc_llvm_handle                    = nullptr;
#endif

/// DrJit API
static LLVMDisasmContextRef jitc_llvm_disasm_ctx = nullptr;
static LLVMContextRef jitc_llvm_context          = nullptr;
static LLVMPassManagerRef jitc_llvm_pass_manager = nullptr;

char    *jitc_llvm_triple          = nullptr;
char    *jitc_llvm_target_cpu      = nullptr;
char    *jitc_llvm_target_features = nullptr;
uint32_t jitc_llvm_vector_width    = 0;
uint32_t jitc_llvm_version_major   = 0;
uint32_t jitc_llvm_version_minor   = 0;
uint32_t jitc_llvm_version_patch   = 0;
uint32_t jitc_llvm_patch_loc       = 0;

static bool     jitc_llvm_init_attempted = false;
static bool     jitc_llvm_init_success   = false;

static uint8_t *jitc_llvm_mem           = nullptr;
static size_t   jitc_llvm_mem_size      = 0;
static size_t   jitc_llvm_mem_offset    = 0;
static bool     jitc_llvm_got           = false;

char *jitc_llvm_vector_width_str = nullptr;
char **jitc_llvm_ones_str = nullptr;
char *jitc_llvm_counter_str = nullptr;

extern "C" {

static uint8_t *jitc_llvm_mem_allocate(void * /* opaque */, uintptr_t size,
                                       unsigned align, unsigned /* id */,
                                       const char *name) {
    if (align == 0)
        align = 16;

    jitc_trace("jit_llvm_mem_allocate(section=%s, size=%zu, align=%u)", name,
              size, (uint32_t) align);

    /* It's bad news if LLVM decides to create a global offset table entry.
       This usually means that a compiler intrinsic didn't resolve to a machine
       instruction, and a function call to an external library was generated
       along with a relocation, which we don't support. */
    if (strncmp(name, ".got", 4) == 0)
        jitc_llvm_got = true;

    size_t offset_align = (jitc_llvm_mem_offset + (align - 1)) / align * align;

    // Zero-fill including padding region
    memset(jitc_llvm_mem + jitc_llvm_mem_offset, 0,
           offset_align - jitc_llvm_mem_offset);

    jitc_llvm_mem_offset = offset_align + size;

    if (jitc_llvm_mem_offset > jitc_llvm_mem_size)
        return nullptr;

    return jitc_llvm_mem + offset_align;
}

static uint8_t *jitc_llvm_mem_allocate_data(void *opaque, uintptr_t size,
                                           unsigned align, unsigned id,
                                           const char *name,
                                           LLVMBool /* read_only */) {
    return jitc_llvm_mem_allocate(opaque, size, align, id, name);
}

static LLVMBool jitc_llvm_mem_finalize(void * /* opaque */, char ** /* err */) {
    return 0;
}

static void jitc_llvm_mem_destroy(void * /* opaque */) { }

} /* extern "C" */ ;

/// Dump assembly representation
void jitc_llvm_disasm(const Kernel &kernel) {
    if (std::max(state.log_level_stderr, state.log_level_callback) <
        LogLevel::Debug)
        return;

    for (uint32_t i = 0; i < kernel.llvm.n_reloc; ++i) {
        uint8_t *func_base = (uint8_t *) kernel.llvm.reloc[i],
                *ptr = func_base;
        if (i == 1)
            continue;
        char ins_buf[256];
        bool last_nop = false;
        jitc_log(Debug, "jit_llvm_disasm(): ========== %u ==========", i);
        do {
            size_t offset      = ptr - (uint8_t *) kernel.data,
                   func_offset = ptr - func_base;
            if (offset >= kernel.size)
                break;
            size_t size =
                LLVMDisasmInstruction(jitc_llvm_disasm_ctx, ptr, kernel.size - offset,
                                      (uintptr_t) ptr, ins_buf, sizeof(ins_buf));
            if (size == 0)
                break;
            char *start = ins_buf;
            while (*start == ' ' || *start == '\t')
                ++start;
            if (strcmp(start, "nop") == 0) {
                if (!last_nop)
                    jitc_log(Debug, "jit_llvm_disasm(): ...");
                last_nop = true;
                ptr += size;
                continue;
            }
            last_nop = false;
            jitc_log(Debug, "jit_llvm_disasm(): 0x%08x   %s", (uint32_t) func_offset, start);
            if (strncmp(start, "ret", 3) == 0)
                break;
            ptr += size;
        } while (true);
    }
}

LLVMExecutionEngineRef jitc_llvm_engine_create(LLVMModuleRef mod_) {
    LLVMMCJITCompilerOptions options;
    options.OptLevel = 3;
    options.CodeModel =
        (LLVMCodeModel)(jitc_llvm_version_major == 7 ? 2 : 3); /* Small */
    options.NoFramePointerElim = false;
    options.EnableFastISel = false;
    options.MCJMM = LLVMCreateSimpleMCJITMemoryManager(
        nullptr,
        jitc_llvm_mem_allocate,
        jitc_llvm_mem_allocate_data,
        jitc_llvm_mem_finalize,
        jitc_llvm_mem_destroy);

    LLVMModuleRef mod = mod_;
    if (mod == nullptr)
        mod = LLVMModuleCreateWithName("drjit");

    LLVMExecutionEngineRef engine = nullptr;
    char *error = nullptr;
    if (LLVMCreateMCJITCompilerForModule(&engine, mod, &options,
                                         sizeof(options), &error)) {
        jitc_log(Warn, "jit_llvm_engine_create(): could not create MCJIT: %s", error);
        return nullptr;
    }

    if (jitc_llvm_patch_loc) {
        uint32_t *base = (uint32_t *) LLVMGetExecutionEngineTargetMachine(engine);
        base[jitc_llvm_patch_loc] = 1;
    }

    return engine;
}


static ProfilerRegion profiler_region_llvm_compile("jit_llvm_compile");

void jitc_llvm_compile(const char *buf, size_t buf_size,
                       const char *kern_name, Kernel &kernel) {
    ProfilerPhase phase(profiler_region_llvm_compile);

    size_t target_size = buf_size * 10;

    if (jitc_llvm_mem_size <= target_size) {
        // Central assumption: LLVM text IR is much larger than the resulting generated code.
#if !defined(_WIN32)
        free(jitc_llvm_mem);
        if (posix_memalign((void **) &jitc_llvm_mem, 4096, target_size))
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#else
        _aligned_free(jitc_llvm_mem);
        jitc_llvm_mem = (uint8_t *) _aligned_malloc(target_size, 4096);
        if (!jitc_llvm_mem)
            jitc_raise("jit_llvm_compile(): could not allocate %zu bytes of memory!", target_size);
#endif
        jitc_llvm_mem_size = target_size;
    }
    jitc_llvm_mem_offset = 0;

    LLVMMemoryBufferRef llvm_buf = LLVMCreateMemoryBufferWithMemoryRange(
        buf, buf_size, kern_name, 0);
    if (unlikely(!llvm_buf))
        jitc_fail("jit_run_compile(): could not create memory buf!");

    // 'buf' is consumed by this function.
    LLVMModuleRef llvm_module = nullptr;
    char *error = nullptr;
    LLVMParseIRInContext(jitc_llvm_context, llvm_buf, &llvm_module, &error);
    if (unlikely(error))
        jitc_fail("jit_llvm_compile(): parsing failed. Please see the LLVM "
                  "IR and error message below:\n\n%s\n\n%s", buf, error);

    if (false) {
        char *llvm_ir = LLVMPrintModuleToString(llvm_module);
        fprintf(stderr, "jit_llvm_compile(): Parsed LLVM IR:\n%s", llvm_ir);
        LLVMDisposeMessage(llvm_ir);
    }

#if !defined(NDEBUG)
    bool status = LLVMVerifyModule(llvm_module, 2, &error);
    if (unlikely(status))
        jitc_fail("jit_llvm_compile(): module could not be verified! Please "
                  "see the LLVM IR and error message below:\n\n%s\n\n%s",
                  buf, error);
#endif

    LLVMRunPassManager(jitc_llvm_pass_manager, llvm_module);

    if (false) {
        char *llvm_ir = LLVMPrintModuleToString(llvm_module);
        fprintf(stderr, "jit_llvm_compile(): Optimized LLVM IR:\n%s", llvm_ir);
        LLVMDisposeMessage(llvm_ir);
    }

    LLVMExecutionEngineRef engine = jitc_llvm_engine_create(llvm_module);

    /// Resolve the kernel entry point
    uint8_t *func = (uint8_t *) LLVMGetFunctionAddress(engine, kern_name);
    if (unlikely(!func))
        jitc_fail("jit_llvm_compile(): internal error: could not fetch function "
                  "address of kernel \"%s\"!\n", kern_name);
    else if (unlikely(func < jitc_llvm_mem))
        jitc_fail("jit_llvm_compile(): internal error: invalid address (1): "
                  "%p < %p!\n", func, jitc_llvm_mem);

    if (jitc_llvm_got)
        jitc_fail(
            "jit_llvm_compile(): a global offset table was generated by LLVM, "
            "which typically means that a compiler intrinsic was not supported "
            "by the target architecture. DrJit cannot handle this case "
            "and will terminate the application now. For reference, the "
            "following kernel code was responsible for this problem:\n\n%s",
            buf);

    std::vector<uint8_t *> reloc;
    reloc.push_back(func);

    /// Does the kernel perform virtual function calls via @callables?
    uint8_t *callables_global =
        (uint8_t *) LLVMGetFunctionAddress(engine, "callables");

    if (callables_global) {
        reloc.push_back(callables_global);

        if (unlikely(callables_global < jitc_llvm_mem))
            jitc_fail("jit_llvm_compile(): internal error: invalid address (2): "
                      "%p < %p!\n", callables_global, jitc_llvm_mem);

        std::vector<void *> global_ptrs(callables.size(), nullptr);

        for (size_t i = 0; i < callables.size(); ++i) {
            const char *s = callables[i].c_str();
            const char *offset = strstr(s, "func_");
            char name_buf[38];
            memcpy(name_buf, offset, 37);
            name_buf[37] = '\0';

            uint8_t *func_2 =
                (uint8_t *) LLVMGetFunctionAddress(engine, name_buf);

            if (unlikely(!func_2))
                jitc_fail("jit_llvm_compile(): internal error: could not fetch function "
                          "address of kernel \"%s\"!\n", name_buf);
            else if (unlikely(func_2 < jitc_llvm_mem))
                jitc_fail("jit_llvm_compile(): internal error: invalid address (3): "
                          "%p < %p!\n", func_2, jitc_llvm_mem);
            reloc.push_back(func_2);
        }
    }

#if !defined(_WIN32)
    void *ptr_result =
        mmap(nullptr, jitc_llvm_mem_offset, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr_result == MAP_FAILED)
        jitc_fail("jit_llvm_compile(): could not mmap() memory: %s",
                  strerror(errno));
#else
    void *ptr_result = VirtualAlloc(nullptr, jitc_llvm_mem_offset,
                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr_result)
        jitc_fail("jit_llvm_compile(): could not VirtualAlloc() memory: %u", GetLastError());
#endif
    memcpy(ptr_result, jitc_llvm_mem, jitc_llvm_mem_offset);

    kernel.data = ptr_result;
    kernel.size = (uint32_t) jitc_llvm_mem_offset;
    kernel.llvm.n_reloc = (uint32_t) reloc.size();
    kernel.llvm.reloc = (void **) malloc_check(sizeof(void *) * reloc.size());

    // Relocate function pointers
    for (size_t i = 0; i < reloc.size(); ++i)
        kernel.llvm.reloc[i] = (uint8_t *) ptr_result + (reloc[i] - jitc_llvm_mem);

    // Write address of @callables
    if (kernel.llvm.n_reloc > 1)
        *((void **) kernel.llvm.reloc[1]) = kernel.llvm.reloc + 1;

#if defined(DRJIT_ENABLE_ITTNOTIFY)
    kernel.llvm.itt = __itt_string_handle_create(kern_name);
#endif

#if !defined(_WIN32)
    if (mprotect(ptr_result, jitc_llvm_mem_offset, PROT_READ | PROT_EXEC) == -1)
        jitc_fail("jit_llvm_compile(): mprotect() failed: %s", strerror(errno));
#else
    DWORD unused;
    if (VirtualProtect(ptr_result, jitc_llvm_mem_offset, PAGE_EXECUTE_READ, &unused) == 0)
        jitc_fail("jit_llvm_compile(): VirtualProtect() failed: %u", GetLastError());
#endif
    LLVMDisposeExecutionEngine(engine);
}

void jitc_llvm_update_strings() {
    Buffer buf {100};
    uint32_t width = jitc_llvm_vector_width;

    // Vector width
    buf.fmt("%u", width);
    free(jitc_llvm_vector_width_str);
    jitc_llvm_vector_width_str = strdup(buf.get());

    buf.clear();
    buf.fmt(
        "$r0_0 = trunc i64 %%index to i32$n"
        "$r0_1 = insertelement <%u x i32> undef, i32 $r0_0, i32 0$n"
        "$r0_2 = shufflevector <%u x i32> $r0_1, <%u x i32> undef, <%u x i32> zeroinitializer$n"
        "$r0 = add <%u x i32> $r0_2, <",
        width, width, width, width, width
    );

    for (uint32_t i = 0; i < width; ++i)
        buf.fmt("i32 %u%s", i, i + 1 < width ? ", " : ">");

    free(jitc_llvm_counter_str);
    jitc_llvm_counter_str = strdup(buf.get());

    if (jitc_llvm_ones_str) {
        for (uint32_t i = 0; i < (uint32_t) VarType::Count; ++i)
            free(jitc_llvm_ones_str[i]);
        free(jitc_llvm_ones_str);
    }

    jitc_llvm_ones_str =
        (char **) malloc(sizeof(char *) * (uint32_t) VarType::Count);

    for (uint32_t i = 0; i < (uint32_t) VarType::Count; ++i) {
        VarType vt = (VarType) i;
        Buffer value {18};
        if (vt == VarType::Bool)
            value.putc('1');
        else if (vt == VarType::Float16 || vt == VarType::Float32 ||
                 vt == VarType::Float64){
            value.put("0x");
            for (uint32_t j = 0; j < 16; ++j)
                value.putc(j < 2 * type_size[i] ? 'F' : '0');
        } else {
            value.put("-1");
        }

        buf.clear();
        buf.putc('<');
        for (uint32_t j = 0; j < width; ++j)
            buf.fmt("%s %s%s", type_name_llvm[i],
                    value.get(),
                    j + 1 < width ? ", " : ">");
        jitc_llvm_ones_str[i] = strdup(buf.get());
    }
}

void jitc_llvm_set_target(const char *target_cpu,
                          const char *target_features,
                          uint32_t vector_width) {
    if (!jitc_llvm_init_success)
        return;

    if (jitc_llvm_target_cpu)
        LLVMDisposeMessage(jitc_llvm_target_cpu);

    if (jitc_llvm_target_features) {
        LLVMDisposeMessage(jitc_llvm_target_features);
        jitc_llvm_target_features = nullptr;
    }

    jitc_llvm_vector_width = vector_width;
    jitc_llvm_target_cpu = LLVMCreateMessage((char *) target_cpu);
    if (target_features)
        jitc_llvm_target_features = LLVMCreateMessage((char *) target_features);

    jitc_llvm_update_strings();
}


/// Convenience function for intrinsic function selection
int jitc_llvm_if_at_least(uint32_t vector_width, const char *feature) {
    return jitc_llvm_vector_width >= vector_width &&
           jitc_llvm_target_features != nullptr &&
           strstr(jitc_llvm_target_features, feature) != nullptr;
}

bool jitc_llvm_init() {
    if (jitc_llvm_init_attempted)
        return jitc_llvm_init_success;
    jitc_llvm_init_attempted = true;

#if defined(DRJIT_DYNAMIC_LLVM)
    jitc_llvm_handle = nullptr;
#  if defined(_WIN32)
    const char *llvm_fname = "LLVM-C.dll",
               *llvm_glob  = nullptr;
#  elif defined(__linux__)
    const char *llvm_fname  = "libLLVM.so",
               *llvm_glob   = "/usr/lib/x86_64-linux-gnu/libLLVM*.so.*";
#  elif defined(__APPLE__) && defined(__x86_64__)
    const char *llvm_fname  = "libLLVM.dylib",
               *llvm_glob   = "/usr/local/Cellar/llvm/*/lib/libLLVM.dylib";
#  elif defined(__APPLE__) && defined(__aarch64__)
    const char *llvm_fname  = "libLLVM.dylib",
               *llvm_glob   = "/opt/homebrew/Cellar/llvm/*/lib/libLLVM.dylib";
#  endif

#  if !defined(_WIN32)
    // Don't dlopen libLLVM.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "LLVMLinkInMCJIT"))
        jitc_llvm_handle = RTLD_NEXT;
#  endif

    if (!jitc_llvm_handle) {
        jitc_llvm_handle = jitc_find_library(llvm_fname, llvm_glob, "DRJIT_LIBLLVM_PATH");

        if (!jitc_llvm_handle) // LLVM library cannot be loaded, give up
            return false;
    }

    #define LOAD2(name)                                                        \
        symbol = #name;                                                        \
        name = decltype(name)(dlsym(jitc_llvm_handle, symbol));                \
        if (!name)                                                             \
            break;                                                             \
        symbol = nullptr

    #define EVAL(x) x
    #define LOAD(name) EVAL(LOAD2(name))

    const char *symbol = nullptr;
    do {
        LOAD(LLVMLinkInMCJIT);
        LOAD(LLVMInitializeDrJitTarget);
        LOAD(LLVMInitializeDrJitTargetInfo);
        LOAD(LLVMInitializeDrJitTargetMC);
        LOAD(LLVMInitializeDrJitAsmPrinter);
        LOAD(LLVMInitializeDrJitDisassembler);
        LOAD(LLVMGetGlobalContext);
        LOAD(LLVMGetDefaultTargetTriple);
        LOAD(LLVMGetHostCPUName);
        LOAD(LLVMGetHostCPUFeatures);
        LOAD(LLVMCreateMessage);
        LOAD(LLVMDisposeMessage);
        LOAD(LLVMCreateDisasm);
        LOAD(LLVMDisasmDispose);
        LOAD(LLVMSetDisasmOptions);
        LOAD(LLVMModuleCreateWithName);
        LOAD(LLVMCreateMCJITCompilerForModule);
        LOAD(LLVMCreateSimpleMCJITMemoryManager);
        LOAD(LLVMDisposeExecutionEngine);
        LOAD(LLVMAddModule);
        LOAD(LLVMDisposeModule);
        LOAD(LLVMCreateMemoryBufferWithMemoryRange);
        LOAD(LLVMParseIRInContext);
        LOAD(LLVMPrintModuleToString);
        LOAD(LLVMGetFunctionAddress);
        LOAD(LLVMGetGlobalValueAddress);
        LOAD(LLVMRemoveModule);
        LOAD(LLVMDisasmInstruction);
        LOAD(LLVMCreatePassManager);
        LOAD(LLVMRunPassManager);
        LOAD(LLVMDisposePassManager);
        LOAD(LLVMAddLICMPass);
        LOAD(LLVMGetExecutionEngineTargetMachine);
        LOAD(LLVMVerifyModule);
        LOAD(LLVMPassManagerBuilderCreate);
        LOAD(LLVMPassManagerBuilderSetOptLevel);
        LOAD(LLVMPassManagerBuilderPopulateModulePassManager);
        LOAD(LLVMPassManagerBuilderDispose);
    } while (false);

    if (symbol) {
        jitc_log(Warn,
                "jit_llvm_init(): could not find symbol \"%s\" -- disabling "
                "LLVM backend!", symbol);
        return false;
    }
#endif

#if defined(DRJIT_DYNAMIC_LLVM)
    void *dlsym_src = jitc_llvm_handle;
#else
    void *dlsym_src = RTLD_NEXT;
#endif

    auto get_version_string = (const char * (*) ()) dlsym(
        dlsym_src, "_ZN4llvm16LTOCodeGenerator16getVersionStringEv");

    if (get_version_string) {
        const char* version_string = get_version_string();
        if (sscanf(version_string, "LLVM version %u.%u.%u", &jitc_llvm_version_major,
                   &jitc_llvm_version_minor, &jitc_llvm_version_patch) != 3) {
            jitc_log(Warn,
                    "jit_llvm_init(): could not parse LLVM version string \"%s\".",
                    version_string);
            return false;
        }

        if (jitc_llvm_version_major < 7) {
            jitc_log(Warn,
                    "jit_llvm_init(): LLVM version 7 or later must be used. (found "
                    "%s). You may want to define the 'DRJIT_LIBLLVM_PATH' "
                    "environment variable to specify the path to "
                    "libLLVM.so/dylib/dll of a particular LLVM version.",
                    version_string);
        }
    } else {
        jitc_llvm_version_major = 10;
        jitc_llvm_version_minor = 0;
        jitc_llvm_version_patch = 0;
    }

    LLVMLinkInMCJIT();
    LLVMInitializeDrJitTargetInfo();
    LLVMInitializeDrJitTarget();
    LLVMInitializeDrJitTargetMC();
    LLVMInitializeDrJitAsmPrinter();
    LLVMInitializeDrJitDisassembler();

    jitc_llvm_context = LLVMGetGlobalContext();
    if (!jitc_llvm_context) {
        jitc_log(Warn, "jit_llvm_init(): could not obtain context!");
        return false;
    }

    jitc_llvm_triple = LLVMGetDefaultTargetTriple();
    jitc_llvm_target_cpu = LLVMGetHostCPUName();
    jitc_llvm_target_features = LLVMGetHostCPUFeatures();

    jitc_llvm_disasm_ctx =
        LLVMCreateDisasm(jitc_llvm_triple, nullptr, 0, nullptr, nullptr);

    if (!jitc_llvm_disasm_ctx) {
        jitc_log(Warn, "jit_llvm_init(): could not create a disassembler!");
        LLVMDisposeMessage(jitc_llvm_triple);
        LLVMDisposeMessage(jitc_llvm_target_cpu);
        LLVMDisposeMessage(jitc_llvm_target_features);
        return false;
    }

    if (LLVMSetDisasmOptions(jitc_llvm_disasm_ctx,
                             LLVMDisassembler_Option_PrintImmHex |
                             LLVMDisassembler_Option_AsmPrinterVariant) == 0) {
        jitc_log(Warn, "jit_llvm_init(): could not configure disassembler!");
        LLVMDisasmDispose(jitc_llvm_disasm_ctx);
        LLVMDisposeMessage(jitc_llvm_triple);
        LLVMDisposeMessage(jitc_llvm_target_cpu);
        LLVMDisposeMessage(jitc_llvm_target_features);
        return false;
    }

#if !defined(__aarch64__)
    LLVMExecutionEngineRef engine = jitc_llvm_engine_create(nullptr);
    if (!engine) {
        LLVMDisasmDispose(jitc_llvm_disasm_ctx);
        LLVMDisposeMessage(jitc_llvm_triple);
        LLVMDisposeMessage(jitc_llvm_target_cpu);
        LLVMDisposeMessage(jitc_llvm_target_features);
        return false;
    }

    /**
       The following is horrible, but it works and was without alternative.

       LLVM MCJIT uses a 'static' relocation model by default. But that's not
       good for us: we want a 'pic' relocation model that allows for *relocatable
       code*. This is important because generated kernels can be reloaded from
       the cache, at which point they may end up anywhere in memory.

       The LLVM C API is quite nice, but it's missing a function to adjust this
       crucial aspect. While we could in principle use C++ the API, this is totally
       infeasible without a hard dependency on the LLVM headers and being being
       permanently tied to a specific version.

       The code below simply hot-patches the TargetMachine* instance, which
       contains the following 3 fields (around byte offset 568 on my machine)

          Reloc::Model RM = Reloc::Static;
          CodeModel::Model CMModel = CodeModel::Small;
          CodeGenOpt::Level OptLevel = CodeGenOpt::Default;

       Which are all 32 bit integers. This interface has been stable since
       ancient times (LLVM 4), and the good thing is that we know the values of
       these fields and can therefore use them as a 12-byte key to find the
       precise byte offset and them overwrite the 'RM' field.
    */

    uint32_t *base = (uint32_t *) LLVMGetExecutionEngineTargetMachine(engine);
    jitc_llvm_patch_loc = 142 - 16;

    int key[3] = { 0, jitc_llvm_version_major == 7 ? 0 : 1, 3 };
    bool found = false;
    for (int i = 0; i < 30; ++i) {
        if (memcmp(base + jitc_llvm_patch_loc, key, sizeof(uint32_t) * 3) == 0) {
            found = true;
            break;
        }
        jitc_llvm_patch_loc += 1;
    }

    LLVMDisposeExecutionEngine(engine);

    if (!found) {
        jitc_log(Warn, "jit_llvm_init(): could not hot-patch TargetMachine relocation model!");
        LLVMDisasmDispose(jitc_llvm_disasm_ctx);
        LLVMDisposeMessage(jitc_llvm_triple);
        LLVMDisposeMessage(jitc_llvm_target_cpu);
        LLVMDisposeMessage(jitc_llvm_target_features);
        return false;
    }
#endif

    jitc_llvm_pass_manager = LLVMCreatePassManager();

#if 0
    LLVMPassManagerBuilderRef pm_builder = LLVMPassManagerBuilderCreate();
    LLVMPassManagerBuilderSetOptLevel(pm_builder, 3);
    LLVMPassManagerBuilderPopulateModulePassManager(pm_builder, jitc_llvm_pass_manager);
    LLVMPassManagerBuilderDispose(pm_builder);
#else
    LLVMAddLICMPass(jitc_llvm_pass_manager);
#endif

    jitc_llvm_vector_width = 1;

    if (strstr(jitc_llvm_target_features, "+sse4.2"))
        jitc_llvm_vector_width = 4;
    if (strstr(jitc_llvm_target_features, "+avx"))
        jitc_llvm_vector_width = 8;
    if (strstr(jitc_llvm_target_features, "+avx512vl"))
        jitc_llvm_vector_width = 16;

#if defined(__APPLE__) && defined(__aarch64__)
    jitc_llvm_vector_width = 4;
    LLVMDisposeMessage(jitc_llvm_target_cpu);
    jitc_llvm_target_cpu = LLVMCreateMessage("apple-a14");
#endif

    jitc_log(Info,
            "jit_llvm_init(): found LLVM %u.%u.%u, target=%s, cpu=%s, vector width=%u.",
            jitc_llvm_version_major, jitc_llvm_version_minor, jitc_llvm_version_patch, jitc_llvm_triple,
            jitc_llvm_target_cpu, jitc_llvm_vector_width);

    jitc_llvm_init_success = jitc_llvm_vector_width > 1;

    if (!jitc_llvm_init_success) {
        jitc_log(Warn,
                 "jit_llvm_init(): no suitable vector ISA found, shutting "
                 "down LLVM backend..");
        jitc_llvm_shutdown();
    }

    jitc_llvm_update_strings();

    return jitc_llvm_init_success;
}

void jitc_llvm_shutdown() {
    if (!jitc_llvm_init_success)
        return;

    jitc_log(Info, "jit_llvm_shutdown()");

    LLVMDisasmDispose(jitc_llvm_disasm_ctx);
    LLVMDisposeMessage(jitc_llvm_triple);
    LLVMDisposeMessage(jitc_llvm_target_cpu);
    LLVMDisposeMessage(jitc_llvm_target_features);
    LLVMDisposePassManager(jitc_llvm_pass_manager);

    jitc_llvm_disasm_ctx = nullptr;
    jitc_llvm_context = nullptr;
    jitc_llvm_pass_manager = nullptr;
    jitc_llvm_target_cpu = nullptr;
    jitc_llvm_target_features = nullptr;
    jitc_llvm_vector_width = 0;

    free(jitc_llvm_vector_width_str);
    jitc_llvm_vector_width_str = nullptr;
    free(jitc_llvm_counter_str);
    jitc_llvm_counter_str = nullptr;
    if (jitc_llvm_ones_str) {
        for (uint32_t i = 0; i < (uint32_t) VarType::Count; ++i)
            free(jitc_llvm_ones_str[i]);
        free(jitc_llvm_ones_str);
    }
    jitc_llvm_ones_str = nullptr;

#if !defined(_WIN32)
    free(jitc_llvm_mem);
#else
    _aligned_free(jitc_llvm_mem);
#endif

    jitc_llvm_mem        = nullptr;
    jitc_llvm_mem_size   = 0;
    jitc_llvm_mem_offset = 0;
    jitc_llvm_got        = false;

#if defined(DRJIT_DYNAMIC_LLVM)
    #define Z(x) x = nullptr

    Z(LLVMLinkInMCJIT); Z(LLVMInitializeDrJitTarget);
    Z(LLVMInitializeDrJitTargetInfo); Z(LLVMInitializeDrJitTargetMC);
    Z(LLVMInitializeDrJitAsmPrinter); Z(LLVMInitializeDrJitDisassembler);
    Z(LLVMGetGlobalContext); Z(LLVMGetDefaultTargetTriple);
    Z(LLVMGetHostCPUName); Z(LLVMGetHostCPUFeatures); Z(LLVMCreateMessage);
    Z(LLVMDisposeMessage); Z(LLVMCreateDisasm); Z(LLVMDisasmDispose);
    Z(LLVMSetDisasmOptions); Z(LLVMModuleCreateWithName);
    Z(LLVMCreateMCJITCompilerForModule); Z(LLVMCreateSimpleMCJITMemoryManager);
    Z(LLVMDisposeExecutionEngine); Z(LLVMAddModule); Z(LLVMDisposeModule);
    Z(LLVMCreateMemoryBufferWithMemoryRange); Z(LLVMParseIRInContext);
    Z(LLVMPrintModuleToString); Z(LLVMGetFunctionAddress);
    Z(LLVMGetGlobalValueAddress); Z(LLVMRemoveModule);
    Z(LLVMDisasmInstruction); Z(LLVMCreatePassManager); Z(LLVMRunPassManager);
    Z(LLVMDisposePassManager); Z(LLVMAddLICMPass);
    Z(LLVMGetExecutionEngineTargetMachine); Z(LLVMVerifyModule);
    Z(LLVMPassManagerBuilderCreate); Z(LLVMPassManagerBuilderSetOptLevel);
    Z(LLVMPassManagerBuilderPopulateModulePassManager);
    Z(LLVMPassManagerBuilderDispose);

#  if !defined(_WIN32)
    if (jitc_llvm_handle != RTLD_NEXT)
        dlclose(jitc_llvm_handle);
#  else
    FreeLibrary((HMODULE) jitc_llvm_handle);
#  endif

    jitc_llvm_handle = nullptr;
#endif

    jitc_llvm_init_success = false;
    jitc_llvm_init_attempted = false;
}
