/*
    src/eval.cpp -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "profiler.h"
#include "util.h"
#include "optix_api.h"
#include "loop.h"

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jitc_eval() calls.
// ====================================================================

/// Ordered list of variables that should be computed
std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
std::vector<ScheduledGroup> schedule_groups;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Kernel parameter buffer and device copy
static std::vector<void *> kernel_params;
static uint8_t *kernel_params_global = nullptr;
static uint32_t kernel_param_count = 0;

/// List of global declarations (intrinsics, constant arrays)
std::vector<std::string> globals;

/// List of device functions or direct callables (OptiX)
std::vector<std::string> callables;

/// Ensure uniqueness of globals/callables arrays
GlobalsMap globals_map;

/// Temporary scratch space for scheduled tasks (LLVM only)
static std::vector<Task *> scheduled_tasks;

/// Hash code of the last generated kernel
XXH128_hash_t kernel_hash { 0, 0 };

/// Name of the last generated kernel
char kernel_name[52 /* strlen("__direct_callable__") + 32 + 1 */] { };

// Total number of operations used across the entire kernel (including functions)
static uint32_t n_ops_total = 0;

/// Are we recording an OptiX kernel?
bool uses_optix = false;

/// Are we currently compiling a virtual function call
bool assemble_func = false;

/// Size and alignment of auxiliary buffer needed by virtual function calls
int32_t alloca_size = -1;
int32_t alloca_align = -1;

/// Specifies the nesting level of virtual calls being compiled
uint32_t vcall_depth = 0;
uint32_t vcall_counter = 0;

/// Information about the kernel launch to go in the kernel launch history
KernelHistoryEntry kernel_history_entry;

// ====================================================================

/// Recursively traverse the computation graph to find variables needed by a computation
static void jitc_var_traverse(uint32_t size, uint32_t index) {
    if (!visited.emplace(size, index).second)
        return;

    Variable *v = jitc_var(index);
    for (int i = 0; i < 4; ++i) {
        uint32_t index2 = v->dep[i];
        if (index2 == 0)
            break;
        jitc_var_traverse(size, index2);
    }

    if (unlikely(v->extra)) {
        auto it = state.extra.find(index);
        if (it == state.extra.end())
            jitc_fail("jit_var_traverse(): could not find matching 'extra' record!");

        const Extra &extra = it->second;
        for (uint32_t i = 0; i < extra.n_dep; ++i) {
            uint32_t index2 = extra.dep[i];
            if (index2 == 0)
                continue;
            jitc_var_traverse(size, index2);
        }
    }

    // If we're really visiting this variable the first time, no matter its size
    if (visited.emplace(0, index).second)
        v->output_flag = false;

    schedule.emplace_back(size, index);
}

void jitc_assemble(ThreadState *ts, ScheduledGroup group) {
    JitBackend backend = ts->backend;

    kernel_params.clear();
    globals.clear();
    callables.clear();
    globals_map.clear();
    alloca_size = alloca_align = -1;
    kernel_history_entry = {};

#if defined(DRJIT_ENABLE_OPTIX)
    uses_optix = ts->backend == JitBackend::CUDA && jit_flag(JitFlag::ForceOptiX);
#endif

    uint32_t n_params_in    = 0,
             n_params_out   = 0,
             n_side_effects = 0,
             n_regs         = 0;

    if (backend == JitBackend::CUDA) {
        uintptr_t size = 0;
        memcpy(&size, &group.size, sizeof(uint32_t));
        kernel_params.push_back((void *) size);

        // The first 3 variables are reserved on the CUDA backend
        n_regs = 4;
    } else {
        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
        for (int i = 0; i < 3; ++i)
            kernel_params.push_back(nullptr);
        n_regs = 1;
    }

    (void) timer();

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        uint32_t index = schedule[group_index].index;
        Variable *v = jitc_var(index);

        // Some sanity checks
        if (unlikely((JitBackend) v->backend != backend))
            jitc_raise("jit_assemble(): variable r%u scheduled in wrong ThreadState", index);
        if (unlikely(v->ref_count_int == 0 && v->ref_count_ext == 0))
            jitc_fail("jit_assemble(): schedule contains unreferenced variable r%u!", index);
        if (unlikely(v->size != 1 && v->size != group.size))
            jitc_fail("jit_assemble(): schedule contains variable r%u with incompatible size "
                     "(%u and %u)!", index, v->size, group.size);
        if (unlikely(!v->data && !v->literal && !v->stmt))
            jitc_fail("jit_assemble(): variable r%u has no statement!", index);
        if (unlikely(v->literal && v->data))
            jitc_fail("jit_assemble(): variable r%u is simultaneously literal and evaluated!", index);
        if (unlikely(v->ref_count_se))
            jitc_fail("jit_assemble(): dirty variable r%u encountered!", index);

        v->param_offset = (uint32_t) kernel_params.size() * sizeof(void *);
        if (v->data) {
            v->param_type = ParamType::Input;
            kernel_params.push_back(v->data);
            n_params_in++;
        } else if (v->output_flag && v->size == group.size) {
            size_t isize = (size_t) type_size[v->type],
                   dsize = (size_t) group.size * isize;

            // Padding to support out-of-bounds accesses in LLVM gather operations
            if (backend == JitBackend::LLVM && isize < 4)
                dsize += 4 - isize;

            void *data =
                jitc_malloc(backend == JitBackend::CUDA ? AllocType::Device
                                                        : AllocType::HostAsync,
                            dsize);

            // The addr. of 'v' can change (jitc_malloc() may release the lock)
            v = jitc_var(index);

            v->data = data;
            v->param_type = ParamType::Output;
            kernel_params.push_back(data);
            n_params_out++;
        } else if (v->literal && (VarType) v->type == VarType::Pointer) {
            v->param_type = ParamType::Input;
            kernel_params.push_back((void *) v->value);
            n_params_in++;
        } else {
            v->param_type = ParamType::Register;
            v->param_offset = 0xFFFF;
            n_side_effects += v->side_effect;
            #if defined(DRJIT_ENABLE_OPTIX)
                uses_optix |= v->optix;
            #endif
        }

        v->reg_index = n_regs++;
    }

    if (unlikely(n_regs > 0xFFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel uses a more than 1 million "
                 "variables (%u) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", n_regs);

    if (unlikely(kernel_params.size() > 0xFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel accesses more than 8192 "
                 "arrays (%zu) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", kernel_params.size());

    kernel_param_count = (uint32_t) kernel_params.size();
    n_ops_total = n_regs;

    // Pass parameters through global memory if too large or using OptiX
    if (backend == JitBackend::CUDA &&
        (uses_optix || kernel_param_count > DRJIT_CUDA_ARG_LIMIT)) {
        size_t size = kernel_param_count * sizeof(void *);
        uint8_t *tmp = (uint8_t *) jitc_malloc(AllocType::HostPinned, size);
        kernel_params_global = (uint8_t *) jitc_malloc(AllocType::Device, size);
        memcpy(tmp, kernel_params.data(), size);
        jitc_memcpy_async(backend, kernel_params_global, tmp, size);
        jitc_free(tmp);
        kernel_params.clear();
        kernel_params.push_back(kernel_params_global);
    }

    bool trace = std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Trace;

    if (unlikely(trace)) {
        buffer.clear();
        for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
            uint32_t index = schedule[group_index].index;
            Variable *v = jitc_var(index);

            buffer.fmt("   - %s%u -> r%u: ", type_prefix[v->type],
                       v->reg_index, index);

            const char *label = jitc_var_label(index);
            if (label)
                buffer.fmt("label=\"%s\", ", label);
            if (v->param_type == ParamType::Input)
                buffer.fmt("in, offset=%u, ", v->param_offset);
            if (v->param_type == ParamType::Output)
                buffer.fmt("out, offset=%u, ", v->param_offset);
            if (v->literal)
                buffer.put("literal, ");
            if (v->size == 1 && v->param_type != ParamType::Output)
                buffer.put("scalar, ");
            if (v->side_effect)
                buffer.put("side effects, ");
            buffer.rewind(2);
            buffer.putc('\n');
        }
        jitc_trace("jit_assemble(size=%u): register map:\n%s",
                  group.size, buffer.get());
    }

    buffer.clear();
    if (backend == JitBackend::CUDA)
        jitc_assemble_cuda(ts, group, n_regs, kernel_param_count);
    else
        jitc_assemble_llvm(ts, group);

    // Replace '^'s in 'drjit_^^^^^^^^' by a hash code
    kernel_hash = hash_kernel(buffer.get());
    snprintf(kernel_name, sizeof(kernel_name), "%s%016llx%016llx",
             uses_optix ? "__raygen__" : "drjit_",
             (unsigned long long) kernel_hash.high64,
             (unsigned long long) kernel_hash.low64);
    const char *name_start = strchr(buffer.get(), '^');
    if (unlikely(!name_start))
        jitc_fail("jit_eval(): could not find kernel name!");
    memcpy((char *) name_start - (uses_optix ? 10 : 6), kernel_name,
           strlen(kernel_name));

    if (unlikely(trace || (jitc_flags() & (uint32_t) JitFlag::PrintIR))) {
        LogLevel level = std::max(state.log_level_stderr, state.log_level_callback);
        jitc_log(level, "%s", buffer.get());
    }

    float codegen_time = timer();

    if (n_side_effects)
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, se=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_side_effects, n_ops_total, jitc_time_string(codegen_time));
    else
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_ops_total, jitc_time_string(codegen_time));

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        kernel_history_entry.backend = backend;
        kernel_history_entry.type = KernelType::JIT;
        kernel_history_entry.hash[0] = kernel_hash.low64;
        kernel_history_entry.hash[1] = kernel_hash.high64;
        kernel_history_entry.ir = (char *) malloc(buffer.size() + 1);
        memcpy(kernel_history_entry.ir, buffer.get(), buffer.size() + 1);
        kernel_history_entry.uses_optix = uses_optix;
        kernel_history_entry.size = group.size;
        kernel_history_entry.input_count = n_params_in;
        kernel_history_entry.output_count = n_params_out + n_side_effects;
        kernel_history_entry.operation_count = n_ops_total;
        kernel_history_entry.codegen_time = codegen_time * 1e-3f;
    }
}

static ProfilerRegion profiler_region_backend_compile("jit_eval: compiling");
static ProfilerRegion profiler_region_backend_load("jit_eval: loading");

Task *jitc_run(ThreadState *ts, ScheduledGroup group) {
    uint64_t flags = 0;

#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_optix) {
        const OptixPipelineCompileOptions &pco = ts->optix_pipeline->compile_options;
        flags =
            ((uint64_t) pco.numAttributeValues << 0)      + // 4 bit
            ((uint64_t) pco.numPayloadValues   << 4)      + // 4 bit
            ((uint64_t) pco.usesMotionBlur     << 8)      + // 1 bit
            ((uint64_t) pco.traversableGraphFlags  << 9)  + // 16 bit
            ((uint64_t) pco.usesPrimitiveTypeFlags << 25);  // 32 bit
    }
#endif

    KernelKey kernel_key((char *) buffer.get(), ts->device, flags);
    auto it = state.kernel_cache.find(
        kernel_key,
        KernelHash::compute_hash(kernel_hash.high64, ts->device, flags));
    Kernel kernel;
    memset(&kernel, 0, sizeof(Kernel)); // quench uninitialized variable warning on MSVC

    if (it == state.kernel_cache.end()) {
        bool cache_hit = false;

        if (!uses_optix)
            cache_hit = jitc_kernel_load(buffer.get(), (uint32_t) buffer.size(),
                                         ts->backend, kernel_hash, kernel);

        if (!cache_hit) {
            ProfilerPhase profiler(profiler_region_backend_compile);
            if (ts->backend == JitBackend::CUDA) {
                if (!uses_optix) {
                    jitc_cuda_compile(buffer.get(), buffer.size(), kernel);
                } else {
#if defined(DRJIT_ENABLE_OPTIX)
                    cache_hit = jitc_optix_compile(
                        ts, buffer.get(), buffer.size(), kernel_name, kernel);
#else
                    jitc_fail("jit_run(): OptiX support was not enabled in DrJit.");
#endif
                }
            } else {
                jitc_llvm_compile(buffer.get(), buffer.size(), kernel_name,
                                  kernel);
            }

            if (kernel.data)
                jitc_kernel_write(buffer.get(), (uint32_t) buffer.size(),
                                  ts->backend, kernel_hash, kernel);
        }

        ProfilerPhase profiler(profiler_region_backend_load);

        if (ts->backend == JitBackend::LLVM) {
            jitc_llvm_disasm(kernel);
        } else if (!uses_optix) {
            CUresult ret = (CUresult) 0;
            /* Unlock while synchronizing */ {
                unlock_guard guard(state.lock);
                ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
            }
            if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
                jitc_flush_malloc_cache(true, true);
                /* Unlock while synchronizing */ {
                    unlock_guard guard(state.lock);
                    ret = cuModuleLoadData(&kernel.cuda.mod, kernel.data);
                }
            }
            cuda_check(ret);

            // Locate the kernel entry point
            char kernel_name_tmp[39];
            snprintf(kernel_name_tmp, sizeof(kernel_name_tmp),
                     "drjit_%016llx%016llx",
                     (unsigned long long) kernel_hash.high64,
                     (unsigned long long) kernel_hash.low64);

            cuda_check(cuModuleGetFunction(&kernel.cuda.func, kernel.cuda.mod,
                                           kernel_name_tmp));

            // Determine a suitable thread count to maximize occupancy
            int unused, block_size;
            cuda_check(cuOccupancyMaxPotentialBlockSize(
                &unused, &block_size,
                kernel.cuda.func, nullptr, 0, 0));
            kernel.cuda.block_size = (uint32_t) block_size;

            // DrJit doesn't use shared memory at all, prefer to have more L1 cache.
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CU_SHAREDMEM_CARVEOUT_MAX_L1));

            free(kernel.data);
            kernel.data = nullptr;
        }

        float link_time = timer();
        jitc_log(Info, "     cache %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                cache_hit ? "load" : "build",
                std::string(jitc_time_string(link_time)).c_str(),
                std::string(jitc_mem_string(kernel.size)).c_str());

        kernel_key.str = (char *) malloc_check(buffer.size() + 1);
        memcpy(kernel_key.str, buffer.get(), buffer.size() + 1);
        state.kernel_cache.emplace(kernel_key, kernel);

        if (cache_hit)
            state.kernel_soft_misses++;
        else
            state.kernel_hard_misses++;

        if (unlikely(jit_flag(JitFlag::KernelHistory))) {
            kernel_history_entry.cache_disk = cache_hit;
            kernel_history_entry.cache_hit = cache_hit;
            if (!cache_hit)
                kernel_history_entry.backend_time = link_time * 1e-3f;
        }
    } else {
        kernel_history_entry.cache_hit = true;
        kernel = it.value();
        state.kernel_hits++;
    }
    state.kernel_launches++;

    if (unlikely(jit_flag(JitFlag::KernelHistory) &&
                 ts->backend == JitBackend::CUDA)) {
        auto &e = kernel_history_entry;
        cuda_check(cuEventCreate((CUevent *) &e.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &e.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) e.event_start, ts->stream));
    }

    Task* ret_task = nullptr;
    if (ts->backend == JitBackend::CUDA) {
#if defined(DRJIT_ENABLE_OPTIX)
        if (unlikely(uses_optix)) {
            jitc_optix_launch(ts, kernel, group.size, kernel_params_global,
                              kernel_param_count);
        }
#endif

        if (!uses_optix) {
            size_t buffer_size = kernel_params.size() * sizeof(void *);

            void *config[] = {
                CU_LAUNCH_PARAM_BUFFER_POINTER,
                kernel_params.data(),
                CU_LAUNCH_PARAM_BUFFER_SIZE,
                &buffer_size,
                CU_LAUNCH_PARAM_END
            };

            uint32_t block_count, thread_count, size = group.size;
            const Device &device = state.devices[ts->device];
            device.get_launch_config(&block_count, &thread_count, size,
                                     (uint32_t) kernel.cuda.block_size);

            cuda_check(cuLaunchKernel(kernel.cuda.func, block_count, 1, 1,
                                      thread_count, 1, 1, 0, ts->stream,
                                      nullptr, config));
        }

        if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
            cuda_check(cuStreamSynchronize(ts->stream));
    } else {
        uint32_t packets =
            (group.size + jitc_llvm_vector_width - 1) / jitc_llvm_vector_width;

        auto callback = [](uint32_t index, void *ptr) {
            void **params = (void **) ptr;
            LLVMKernelFunction kernel = (LLVMKernelFunction) params[0];
            uint32_t size       = (uint32_t) (uintptr_t) params[1],
                     block_size = (uint32_t) ((uintptr_t) params[1] >> 32),
                     start      = index * block_size,
                     end        = std::min(start + block_size, size);

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            // Signal start of kernel
            __itt_task_begin(drjit_domain, __itt_null, __itt_null,
                             (__itt_string_handle *) params[2]);
#endif
            // Perform the main computation
            kernel(start, end, params);

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            // Signal termination of kernel
            __itt_task_end(drjit_domain);
#endif
        };

        uint32_t block_size = DRJIT_POOL_BLOCK_SIZE,
                 blocks = (group.size + block_size - 1) / block_size;

        kernel_params[0] = (void *) kernel.llvm.reloc[0];
        kernel_params[1] = (void *) ((((uintptr_t) block_size) << 32) +
                                     (uintptr_t) group.size);

#if defined(DRJIT_ENABLE_ITTNOTIFY)
        kernel_params[2] = kernel.llvm.itt;
#endif

        jitc_trace("jit_run(): scheduling %u packet%s in %u block%s ..",
                   packets, packets == 1 ? "" : "s", blocks,
                   blocks == 1 ? "" : "s");
        (void) packets; // jitc_trace may be disabled

        ret_task = task_submit_dep(
            nullptr, &ts->task, 1, blocks,
            callback,
            kernel_params.data(),
            (uint32_t) (kernel_params.size() * sizeof(void *)),
            nullptr
        );

        if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
            task_wait(ret_task);
    }

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        if (ts->backend == JitBackend::CUDA) {
            cuda_check(cuEventRecord((CUevent) kernel_history_entry.event_end,
                                     ts->stream));
        } else {
            task_retain(ret_task);
            kernel_history_entry.task = ret_task;
        }

        state.kernel_history.append(kernel_history_entry);
    }

    return ret_task;
}

static ProfilerRegion profiler_region_eval("jit_eval");

/// Evaluate all computation that is queued on the given ThreadState
void jitc_eval(ThreadState *ts) {
    if (!ts || (ts->scheduled.empty() && ts->side_effects.empty()))
        return;

    ProfilerPhase profiler(profiler_region_eval);

    /* The function 'jitc_eval()' modifies several global data structures
       and should never be executed concurrently. However, there are a few
       places where it needs to temporarily release the main lock as part of
       its work, which is dangerous because another thread could use that
       opportunity to enter 'jitc_eval()' and cause corruption. The following
       therefore temporarily unlocks 'state.lock' and then locks a separate
       lock 'state.eval_lock' specifically guarding these data structures */

    lock_release(state.lock);
    lock_guard guard(state.eval_lock);
    lock_acquire(state.lock);

    jitc_var_loop_simplify();

    visited.clear();
    schedule.clear();
    vcall_counter = 0;

    // Collect variables that must be computed along with their dependencies
    for (int j = 0; j < 2; ++j) {
        auto &source = j == 0 ? ts->scheduled : ts->side_effects;
        if (j == 1 && (jitc_flags() & (uint32_t) JitFlag::Recording))
            break;

        for (size_t i = 0; i < source.size(); ++i) {
            uint32_t index = source[i];
            auto it = state.variables.find(index);
            if (it == state.variables.end())
                continue;

            Variable *v = &it.value();

            // Skip variables that aren't externally referenced or already evaluated
            if (v->ref_count_ext == 0 || v->data)
                continue;

            jitc_var_traverse(v->size, index);
            v->output_flag = (VarType) v->type != VarType::Void;
        }

        source.clear();
    }

    if (schedule.empty())
        return;

    // Order variables into groups of matching size
    std::sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            if (a.size > b.size)
                return true;
            else if (a.size < b.size)
                return false;
            else if (a.index > b.index)
                return false;
            else if (a.index < b.index)
                return true;
            else
                return false;
        });

    // Partition into groups of matching size
    schedule_groups.clear();
    if (schedule[0].size == schedule[schedule.size() - 1].size) {
        schedule_groups.emplace_back(schedule[0].size, 0,
                                     (uint32_t) schedule.size());
    } else {
        uint32_t cur = 0;
        for (uint32_t i = 1; i < (uint32_t) schedule.size(); ++i) {
            if (schedule[i - 1].size != schedule[i].size) {
                schedule_groups.emplace_back(schedule[cur].size, cur, i);
                cur = i;
            }
        }

        schedule_groups.emplace_back(schedule[cur].size,
                                     cur, (uint32_t) schedule.size());
    }

    jitc_log(Info, "jit_eval(): launching %zu kernel%s.",
            schedule_groups.size(),
            schedule_groups.size() == 1 ? "" : "s");

    scoped_set_context_maybe guard2(ts->context);
    scheduled_tasks.clear();

    const char *dr_raise_id = getenv("DRJIT_KERNEL_RAISE");
    bool dr_raise = false;

    for (ScheduledGroup &group : schedule_groups) {
        jitc_assemble(ts, group);

        if (dr_raise_id) {
            char tmp[17];
            snprintf(tmp, sizeof(tmp), "%016llx",
                     (unsigned long long) kernel_hash.high64);
            if (strncmp(dr_raise_id, tmp, 16) == 0)
                dr_raise = true;
        }

        scheduled_tasks.push_back(jitc_run(ts, group));

        if (ts->backend == JitBackend::CUDA) {
            jitc_free(kernel_params_global);
            kernel_params_global = nullptr;
        }
    }

    if (ts->backend == JitBackend::LLVM) {
        if (scheduled_tasks.size() == 1) {
            task_release(ts->task);
            ts->task = scheduled_tasks[0];
        } else {
            if (unlikely(scheduled_tasks.empty()))
                jitc_fail("jit_eval(): no tasks generated!");

            // Insert a barrier task
            Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                             (uint32_t) scheduled_tasks.size());
            task_release(ts->task);
            for (Task *t : scheduled_tasks)
                task_release(t);
            ts->task = new_task;
        }
    }

    /* At this point, all variables and their dependencies are computed, which
       means that we can remove internal edges between them. This in turn will
       cause many of the variables to be garbage-collected. */
    jitc_log(Debug, "jit_eval(): cleaning up..");

    for (ScheduledVariable sv : schedule) {
        uint32_t index = sv.index;

        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        v->reg_index = 0;
        if (!(v->output_flag || v->side_effect))
            continue;

        if (unlikely(v->extra)) {
            auto it2 = state.extra.find(index);
            if (it2 == state.extra.end())
                jitc_fail("jit_eval(): could not find 'extra' record of variable %u", index);
            const Extra &extra = it2->second;

            if (extra.callback) {
                if (extra.callback_internal) {
                    extra.callback(index, 0, extra.callback_data);
                } else {
                    unlock_guard guard_2(state.lock);
                    extra.callback(index, 0, extra.callback_data);
                }
                v = jitc_var(index);
            }
        }

        jitc_cse_drop(index, v);

        if (unlikely(v->literal))
            jitc_fail("jit_eval(): internal error: did not expect a literal "
                      "variable here!");
        if (v->free_stmt)
            free(v->stmt);

        uint32_t dep[4], side_effect = v->side_effect;
        memcpy(dep, v->dep, sizeof(uint32_t) * 4);
        memset(v->dep, 0, sizeof(uint32_t) * 4);
        v->stmt = nullptr;
        v->output_flag = false;
        v->side_effect = false;

        if (side_effect)
            jitc_var_dec_ref_ext(index);
        for (int j = 0; j < 4; ++j)
            jitc_var_dec_ref_int(dep[j]);
    }

    jitc_free_flush(ts);
    jitc_log(Info, "jit_eval(): done.");

    if (dr_raise) {
        jitc_log(Warn, "jit_eval(): raising an exception as requested.");
        jitc_raise("jit_eval(): raising an exception as requested.");
    }
}

static ProfilerRegion profiler_region_assemble_func("jit_assemble_func");

std::pair<XXH128_hash_t, uint32_t>
jitc_assemble_func(ThreadState *ts, const char *name, uint32_t inst_id,
                   uint32_t in_size, uint32_t in_align, uint32_t out_size,
                   uint32_t out_align, uint32_t data_offset,
                   const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                   uint32_t n_in, const uint32_t *in, uint32_t n_out,
                   const uint32_t *out_nested, uint32_t n_se,
                   const uint32_t *se, bool use_self) {
    ProfilerPhase profiler(profiler_region_assemble_func);

    visited.clear();
    schedule.clear();

    for (uint32_t i = 0; i < n_in; ++i) {
        if (in[i] == 0)
            continue;

        const Variable *v = jitc_var(in[i]);
        if (!v->literal)
            visited.emplace(1, in[i]);
    }

    auto traverse = [](uint32_t index) {
        if (!index)
            return;
        jitc_var_traverse(1, index);
        Variable *v = jitc_var(index);
        v->output_flag = (VarType) v->type != VarType::Void;
    };

    for (uint32_t i = 0; i < n_out; ++i)
        traverse(out_nested[i]);

    for (uint32_t i = 0; i < n_se; ++i)
        traverse(se[i]);

    std::sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.index < b.index;
        });

    uint32_t n_regs = ts->backend == JitBackend::CUDA ? 4 : 1;

    for (auto &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        v->reg_index = n_regs++;
    }

    size_t offset = buffer.size();

    bool assemble_func_prev = assemble_func;
    assemble_func = true;

    vcall_depth++;
    if (ts->backend == JitBackend::CUDA)
        jitc_assemble_cuda_func(name, inst_id, n_regs, in_size, in_align,
                                out_size, out_align, data_offset, data_map,
                                n_out, out_nested, use_self);
    else
        jitc_assemble_llvm_func(name, inst_id, in_size, data_offset, data_map,
                                n_out, out_nested, use_self);
    vcall_depth--;

    assemble_func = assemble_func_prev;

    buffer.putc('\n');

    size_t kernel_length = buffer.size() - offset;
    char *kernel_str = (char *) buffer.get() + offset;

    if (jit_flag(JitFlag::VCallDeduplicate)) {
        kernel_hash = XXH128(kernel_str, kernel_length, 0);
    } else {
        kernel_hash.low64 = (uint64_t) vcall_counter++;
        kernel_hash.high64 = 0;
    }

    auto result = globals_map.emplace(kernel_hash, (uint32_t) callables.size());
    if (result.second) {
        // Replace '^'s in 'func_^^^..' or '__direct_callable__^^^..' with hash
        char *id = strchr(kernel_str, '^');
        char tmp[33];
        snprintf(tmp, sizeof(tmp), "%016llx%016llx",
                 (unsigned long long) kernel_hash.high64,
                 (unsigned long long) kernel_hash.low64);
        memcpy(id, tmp, 32);
        callables.push_back(std::string(kernel_str, kernel_length));
        n_ops_total += n_regs;
    }
    buffer.rewind(kernel_length);

    return { kernel_hash, result.first->second };
}

void jitc_register_global(const char *str) {
    auto global_hash = XXH128(str, strlen(str), 0);
    if (globals_map.emplace(global_hash, (uint32_t) globals_map.size()).second)
        globals.push_back(str);
}
