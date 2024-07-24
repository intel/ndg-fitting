/*
    src/api.cpp -- C -> C++ API locking wrappers

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "util.h"
#include "registry.h"
#include "llvm_api.h"
#include "cuda_tex.h"
#include "op.h"
#include "vcall.h"
#include "loop.h"
#include "printf.h"
#include <thread>
#include <condition_variable>
#include <drjit-core/texture.h>

#if defined(DRJIT_ENABLE_OPTIX)
#include <drjit-core/optix.h>
#include "optix_api.h"
#endif

#include <nanothread/nanothread.h>

void jit_init(uint32_t backends) {
    lock_guard guard(state.lock);
    jitc_init(backends);
}

void jit_init_async(uint32_t backends) {
    /// Probably overkill for a simple wait flag..
    struct Sync {
        bool flag = false;
        std::mutex lock;
        std::condition_variable cv;
    };

    std::shared_ptr<Sync> sync = std::make_shared<Sync>();
    std::unique_lock<std::mutex> guard(sync->lock);

    std::thread([backends, sync]() {
        lock_guard guard2(state.lock);
        {
            std::unique_lock<std::mutex> guard3(sync->lock);
            sync->flag = true;
            sync->cv.notify_one();
        }
        jitc_init(backends);
    }).detach();

    while (!sync->flag)
        sync->cv.wait(guard);
}

int jit_has_backend(JitBackend backend) {
    lock_guard guard(state.lock);

    bool result;
    switch (backend) {
        case JitBackend::LLVM:
            result = state.backends & (uint32_t) JitBackend::LLVM;
            break;

        case JitBackend::CUDA:
            result = (state.backends & (uint32_t) JitBackend::CUDA)
                && !state.devices.empty();
            break;

        default:
            jitc_raise("jit_has_backend(): invalid input!");
    }

    return (int) result;
}

void jit_shutdown(int light) {
    lock_guard guard(state.lock);
    jitc_shutdown(light);
}

uint32_t jit_cse_scope(JitBackend backend) {
    lock_guard guard(state.lock);
    return thread_state(backend)->cse_scope;
}

void jit_set_cse_scope(JitBackend backend, uint32_t scope_index) {
    lock_guard guard(state.lock);
    if (unlikely(scope_index >= (1 << 24)))
        jitc_raise("jit_set_cse_scope(): overflow (scope index exceeds the 24 "
                   "bit counter of the Variable data structure)!");
    jitc_trace("jit_set_cse_scope(%u)", scope_index);
    thread_state(backend)->cse_scope = scope_index;
}

void jit_new_cse_scope(JitBackend backend) {
    lock_guard guard(state.lock);
    uint32_t scope_index = ++state.cse_scope_ctr;
    if (unlikely(scope_index >= (1 << 24)))
        jitc_raise("jit_new_cse_scope(): overflow (scope index exceeds the 24 "
                   "bit counter of the Variable data structure)!");
    jitc_trace("jit_new_cse_scope(%u)", scope_index);
    thread_state(backend)->cse_scope = scope_index;
}

void jit_set_log_level_stderr(LogLevel level) {
    /// Allow changing this variable without acquiring a lock
    state.log_level_stderr = level;
}

LogLevel jit_log_level_stderr() {
    /// Allow reading this variable without acquiring a lock
    return state.log_level_stderr;
}

void jit_set_log_level_callback(LogLevel level, LogCallback callback) {
    lock_guard guard(state.lock);
    state.log_level_callback = callback ? level : Disable;
    state.log_callback = callback;
}

LogLevel jit_log_level_callback() {
    lock_guard guard(state.lock);
    return state.log_level_callback;
}

void jit_log(LogLevel level, const char* fmt, ...) {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vlog(level, fmt, args);
    va_end(args);
}

void jit_raise(const char* fmt, ...) {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vraise(fmt, args);
    // va_end(args); (dead code)
}

void jit_fail(const char* fmt, ...) {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vfail(fmt, args);
    // va_end(args); (dead code)
}

void jit_set_flags(uint32_t flags) {
    jitc_set_flags(flags);
}

uint32_t jit_flags() {
    return jitc_flags();
}

void jit_set_flag(JitFlag flag, int enable) {
    uint32_t flags = jitc_flags();

    if (enable)
        flags |= (uint32_t) flag;
    else
        flags &= ~(uint32_t) flag;

    jitc_set_flags(flags);
}

int jit_flag(JitFlag flag) {
    return (jitc_flags() & (uint32_t) flag) ? 1 : 0;
}

uint32_t jit_record_checkpoint(JitBackend backend) {
    lock_guard guard(state.lock);
    uint32_t result =
        (uint32_t) thread_state(backend)->side_effects_recorded.size();
    if (jit_flag(JitFlag::Recording))
        result |= 0x80000000u;
    return result;
}

uint32_t jit_record_begin(JitBackend backend, uint32_t *vcall_bound_index) {
    uint32_t result = jit_record_checkpoint(backend);
    jit_set_flag(JitFlag::Recording, true);
    if (vcall_bound_index) {
        ThreadState *ts = thread_state(backend);
        *vcall_bound_index = ts->vcall_bound_index;
        ts->vcall_bound_index = state.variable_index;
    }
    return result;
}

void jit_record_end(JitBackend backend, uint32_t value, uint32_t *vcall_bound_index) {
    lock_guard guard(state.lock);

    // Set recording flag to previous value
    jit_set_flag(JitFlag::Recording, value & 0x80000000u);
    value &= 0x7fffffff;

    ThreadState *ts = thread_state(backend);

    auto &se = ts->side_effects_recorded;
    if (value > se.size())
        jitc_raise("jit_record_end(): position lies beyond the end of the queue!");

    while (value != se.size()) {
        jitc_var_dec_ref_ext(se.back());
        se.pop_back();
    }

    if (vcall_bound_index)
        ts->vcall_bound_index = *vcall_bound_index;
}

void* jit_cuda_stream() {
    lock_guard guard(state.lock);
    return jitc_cuda_stream();
}

void* jit_cuda_context() {
    lock_guard guard(state.lock);
    return jitc_cuda_context();
}

void jit_cuda_push_context(void* ctx) {
    lock_guard guard(state.lock);
    jitc_cuda_push_context(ctx);
}

void* jit_cuda_pop_context() {
    lock_guard guard(state.lock);
    return jitc_cuda_pop_context();
}

int jit_cuda_device_count() {
    lock_guard guard(state.lock);
    return (int) state.devices.size();
}

void jit_cuda_set_device(int device) {
    lock_guard guard(state.lock);
    jitc_cuda_set_device(device);
}

int jit_cuda_device() {
    lock_guard guard(state.lock);
    return thread_state(JitBackend::CUDA)->device;
}

int jit_cuda_device_raw() {
    lock_guard guard(state.lock);
    return state.devices[thread_state(JitBackend::CUDA)->device].id;
}

int jit_cuda_compute_capability() {
    lock_guard guard(state.lock);
    return state.devices[thread_state(JitBackend::CUDA)->device].compute_capability;
}

void jit_cuda_set_target(uint32_t ptx_version, uint32_t compute_capability) {
    lock_guard guard(state.lock);
    ThreadState *ts = thread_state(JitBackend::CUDA);
    ts->ptx_version = ptx_version;
    ts->compute_capability = compute_capability;
}

void *jit_cuda_lookup(const char *name) {
    lock_guard guard(state.lock);
    return jitc_cuda_lookup(name);
}

void jit_llvm_set_thread_count(uint32_t size) {
    pool_set_size(nullptr, size);
}

void jit_llvm_set_target(const char *target_cpu,
                         const char *target_features,
                         uint32_t vector_width) {
    lock_guard guard(state.lock);
    jitc_llvm_set_target(target_cpu, target_features, vector_width);
}

const char *jit_llvm_target_cpu() {
    lock_guard guard(state.lock);
    return jitc_llvm_target_cpu;
}

const char *jit_llvm_target_features() {
    lock_guard guard(state.lock);
    return jitc_llvm_target_features;
}

void jit_llvm_version(int *major, int *minor, int *patch) {
    lock_guard guard(state.lock);
    if (major)
        *major = jitc_llvm_version_major;
    if (minor)
        *minor = jitc_llvm_version_minor;
    if (patch)
        *patch = jitc_llvm_version_patch;
}

int jit_llvm_if_at_least(uint32_t vector_width, const char *feature) {
    lock_guard guard(state.lock);
    return jitc_llvm_if_at_least(vector_width, feature);
}

uint32_t jit_llvm_vector_width() {
    return jitc_llvm_vector_width;
}

void jit_sync_thread() {
    lock_guard guard(state.lock);
    jitc_sync_thread();
}

void jit_sync_device() {
    lock_guard guard(state.lock);
    jitc_sync_device();
}

void jit_sync_all_devices() {
    lock_guard guard(state.lock);
    jitc_sync_all_devices();
}

void jit_flush_kernel_cache() {
    lock_guard guard(state.lock);
    jitc_flush_kernel_cache();
}

void *jit_malloc(AllocType type, size_t size) {
    lock_guard guard(state.lock);
    return jitc_malloc(type, size);
}

void jit_free(void *ptr) {
    lock_guard guard(state.lock);
    jitc_free(ptr);
}

void jit_flush_malloc_cache() {
    lock_guard guard(state.lock);
    jitc_flush_malloc_cache(true, false);
}

void jit_malloc_clear_statistics() {
    lock_guard guard(state.lock);
    jitc_malloc_clear_statistics();
}

void jit_malloc_prefetch(void *ptr, int device) {
    lock_guard guard(state.lock);
    jitc_malloc_prefetch(ptr, device);
}

enum AllocType jit_malloc_type(void *ptr) {
    lock_guard guard(state.lock);
    return jitc_malloc_type(ptr);
}

int jit_malloc_device(void *ptr) {
    lock_guard guard(state.lock);
    return jitc_malloc_device(ptr);
}

void *jit_malloc_migrate(void *ptr, AllocType type, int move) {
    lock_guard guard(state.lock);
    return jitc_malloc_migrate(ptr, type, move);
}

enum AllocType jit_var_alloc_type(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_alloc_type(index);
}

int jit_var_device(uint32_t index) {
    if (index == 0)
        return -1;
    lock_guard guard(state.lock);
    return jitc_var_device(index);
}

uint32_t jit_var_new_stmt(JitBackend backend, JIT_ENUM VarType vt,
                          const char *stmt, int stmt_static, uint32_t n_dep,
                          const uint32_t *dep) {

    lock_guard guard(state.lock);
    return jitc_var_new_stmt(backend, vt, stmt, stmt_static, n_dep, dep);
}

uint32_t jit_var_new_literal(JitBackend backend, VarType type, const void *value,
                             size_t size, int eval, int is_class) {
    lock_guard guard(state.lock);
    return jitc_var_new_literal(backend, type, value, size, eval, is_class);
}

uint32_t jit_var_new_counter(JitBackend backend, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_new_counter(backend, size, true);
}

uint32_t jit_var_new_op(JitOp op, uint32_t n_dep, const uint32_t *dep) {
    lock_guard guard(state.lock);
    return jitc_var_new_op(op, n_dep, dep);
}

uint32_t jit_var_new_cast(uint32_t index, VarType target_type,
                          int reinterpret) {
    lock_guard guard(state.lock);
    return jitc_var_new_cast(index, target_type, reinterpret);
}

uint32_t jit_var_new_gather(uint32_t source, uint32_t index,
                            uint32_t mask) {
    lock_guard guard(state.lock);
    return jitc_var_new_gather(source, index, mask);
}

uint32_t jit_var_new_scatter(uint32_t target, uint32_t value,
                             uint32_t index, uint32_t mask,
                             ReduceOp reduce_op) {
    lock_guard guard(state.lock);
    return jitc_var_new_scatter(target, value, index, mask, reduce_op);
}

uint32_t jit_var_new_pointer(JitBackend backend, const void *value,
                             uint32_t dep, int write) {
    lock_guard guard(state.lock);
    return jitc_var_new_pointer(backend, value, dep, write);
}

uint32_t jit_var_wrap_vcall(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_wrap_vcall(index);
}

void jit_var_inc_ref_ext_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    lock_guard guard(state.lock);
    jitc_var_inc_ref_ext(index);
}

void jit_var_dec_ref_ext_impl(uint32_t index) noexcept(true) {
    if (index == 0)
        return;
    lock_guard guard(state.lock);
    jitc_var_dec_ref_ext(index);
}

int jit_var_exists(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return state.variables.find(index) != state.variables.end();
}

uint32_t jit_var_ref_int(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return (uint32_t) jitc_var(index)->ref_count_int;
}

uint32_t jit_var_ref_ext(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return (uint32_t) jitc_var(index)->ref_count_ext;
}

void *jit_var_ptr(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_ptr(index);
}

size_t jit_var_size(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return (size_t) jitc_var(index)->size;
}

const char *jit_var_stmt(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    Variable *v = jitc_var(index);
    return v->literal ? nullptr : v->stmt;
}

int jit_var_is_literal(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return (int) jitc_var(index)->literal;
}

int jit_var_is_evaluated(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return (int) (jitc_var(index)->data != nullptr);
}

int jit_var_is_placeholder(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return (int) jitc_var(index)->placeholder;
}

uint32_t jit_var_resize(uint32_t index, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_resize(index, size);
}

VarType jit_var_type(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_type(index);
}

const char *jit_var_label(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_label(index);
}

uint32_t jit_var_set_label(uint32_t index, const char *label) {
    if (unlikely(index == 0))
        return 0;

    lock_guard guard(state.lock);

    Variable *v = jitc_var(index);

    // Replicate literals when being labeled
    uint32_t result;
    if (v->literal && (v->ref_count_int != 0 || v->ref_count_ext != 1)) {
        Variable v2;
        memcpy(&v2.value, &v->value, sizeof(uint64_t));
        v2.size = v->size;
        v2.type = v->type;
        v2.literal = 1;
        v2.backend = v->backend;
        result = jitc_var_new(v2, true);
    } else {
        jitc_var_inc_ref_ext(index, v);
        result = index;
    }

    jitc_var_set_label(result, label);

    return result;
}

void jit_var_set_callback(uint32_t index,
                          void (*callback)(uint32_t, int, void *),
                          void *payload) {
    lock_guard guard(state.lock);
    jitc_var_set_callback(index, callback, payload);
}

uint32_t jit_var_mem_map(JitBackend backend, VarType type, void *ptr, size_t size, int free) {
    lock_guard guard(state.lock);
    return jitc_var_mem_map(backend, type, ptr, size, free);
}

uint32_t jit_var_mem_copy(JitBackend backend, AllocType atype, VarType vtype,
                          const void *value, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mem_copy(backend, atype, vtype, value, size);
}

uint32_t jit_var_copy(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_copy(index);
}

uint32_t jit_var_migrate(uint32_t index, AllocType type) {
    lock_guard guard(state.lock);
    return jitc_var_migrate(index, type);
}

void jit_var_mark_side_effect(uint32_t index) {
    lock_guard guard(state.lock);
    jitc_var_mark_side_effect(index);
}

uint32_t jit_var_mask_peek(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_var_mask_peek(backend);
}

uint32_t jit_var_mask_apply(uint32_t index, uint32_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mask_apply(index, size);
}

void jit_var_mask_push(JitBackend backend, uint32_t index) {
    lock_guard guard(state.lock);
    jitc_var_mask_push(backend, index);
}

void jit_var_mask_pop(JitBackend backend) {
    lock_guard guard(state.lock);
    jitc_var_mask_pop(backend);
}

uint32_t jit_var_mask_default(JitBackend backend, uint32_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mask_default(backend, size);
}

int jit_var_any(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_any(index);
}

int jit_var_all(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_all(index);
}

uint32_t jit_var_reduce(uint32_t index, ReduceOp reduce_op) {
    lock_guard guard(state.lock);
    return jitc_var_reduce(index, reduce_op);
}

const char *jit_var_whos() {
    lock_guard guard(state.lock);
    return jitc_var_whos();
}

const char *jit_var_graphviz() {
    lock_guard guard(state.lock);
    return jitc_var_graphviz();
}

const char *jit_var_str(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_str(index);
}

void jit_var_read(uint32_t index, size_t offset, void *dst) {
    lock_guard guard(state.lock);
    jitc_var_read(index, offset, dst);
}

uint32_t jit_var_write(uint32_t index, size_t offset, const void *src) {
    lock_guard guard(state.lock);
    return jitc_var_write(index, offset, src);
}

void jit_var_printf(JitBackend backend, uint32_t mask, const char *fmt,
                    uint32_t narg, const uint32_t *arg) {
    lock_guard guard(state.lock);
    jitc_var_printf(backend, mask, fmt, narg, arg);
}

void jit_eval() {
    lock_guard guard(state.lock);
    jitc_eval(thread_state_cuda);
    jitc_eval(thread_state_llvm);
}

int jit_var_eval(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return jitc_var_eval(index);
}

int jit_var_schedule(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return jitc_var_schedule(index);
}

void jit_prefix_push(JitBackend backend, const char *value) {
    lock_guard guard(state.lock);
    jitc_prefix_push(backend, value);
}

void jit_prefix_pop(JitBackend backend) {
    lock_guard guard(state.lock);
    jitc_prefix_pop(backend);
}

void jit_memset_async(JitBackend backend, void *ptr, uint32_t size, uint32_t isize,
                      const void *src) {
    lock_guard guard(state.lock);
    jitc_memset_async(backend, ptr, size, isize, src);
}

void jit_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    lock_guard guard(state.lock);
    jitc_memcpy(backend, dst, src, size);
}

void jit_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    lock_guard guard(state.lock);
    jitc_memcpy_async(backend, dst, src, size);
}

void jit_reduce(JitBackend backend, VarType type, ReduceOp rtype, const void *ptr,
                uint32_t size, void *out) {
    lock_guard guard(state.lock);
    jitc_reduce(backend, type, rtype, ptr, size, out);
}

void jit_scan_u32(JitBackend backend, const uint32_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_scan_u32(backend, in, size, out);
}

uint32_t jit_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.lock);
    return jitc_compress(backend, in, size, out);
}

uint32_t jit_mkperm(JitBackend backend, const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    lock_guard guard(state.lock);
    return jitc_mkperm(backend, values, size, bucket_count, perm, offsets);
}

void jit_block_copy(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    lock_guard guard(state.lock);
    jitc_block_copy(backend, type, in, out, size, block_size);
}

void jit_block_sum(JitBackend backend, enum VarType type, const void *in, void *out,
                   uint32_t size, uint32_t block_size) {
    lock_guard guard(state.lock);
    jitc_block_sum(backend, type, in, out, size, block_size);
}

uint32_t jit_registry_put(JitBackend backend, const char *domain, void *ptr) {
    lock_guard guard(state.lock);
    return jitc_registry_put(backend, domain, ptr);
}

void jit_registry_remove(JitBackend backend, void *ptr) {
    lock_guard guard(state.lock);
    jitc_registry_remove(backend, ptr);
}

uint32_t jit_registry_get_id(JitBackend backend, const void *ptr) {
    lock_guard guard(state.lock);
    return jitc_registry_get_id(backend, ptr);
}

const char *jit_registry_get_domain(JitBackend backend, const void *ptr) {
    lock_guard guard(state.lock);
    return jitc_registry_get_domain(backend, ptr);
}

void *jit_registry_get_ptr(JitBackend backend, const char *domain, uint32_t id) {
    lock_guard guard(state.lock);
    return jitc_registry_get_ptr(backend, domain, id);
}

uint32_t jit_registry_get_max(JitBackend backend, const char *domain) {
    lock_guard guard(state.lock);
    return jitc_registry_get_max(backend, domain);
}

void jit_registry_trim() {
    lock_guard guard(state.lock);
    jitc_registry_trim();
}

void jit_registry_clear() {
    lock_guard guard(state.lock);
    jitc_registry_clean();
}

void jit_registry_set_attr(JitBackend backend, void *self, const char *name,
                           const void *value, size_t size) {
    lock_guard guard(state.lock);
    jitc_registry_set_attr(backend, self, name, value, size);
}

const void *jit_registry_attr_data(JitBackend backend, const char *domain,
                                   const char *name) {
    lock_guard guard(state.lock);
    return jitc_registry_attr_data(backend, domain, name);
}

uint32_t jit_var_registry_attr(JitBackend backend, VarType type,
                               const char *domain, const char *name) {
    lock_guard guard(state.lock);
    return jitc_var_registry_attr(backend, type, domain, name);
}

void jit_vcall_set_self(JitBackend backend, uint32_t value, uint32_t index) {
    lock_guard guard(state.lock);
    jitc_vcall_set_self(backend, value, index);
}

void jit_vcall_self(JitBackend backend, uint32_t *value, uint32_t *index) {
    lock_guard guard(state.lock);
    jitc_vcall_self(backend, value, index);
}

uint32_t jit_var_vcall(const char *name, uint32_t self, uint32_t mask,
                       uint32_t n_inst, const uint32_t *inst_id, uint32_t n_in,
                       const uint32_t *in, uint32_t n_out_nested,
                       const uint32_t *out_nested, const uint32_t *se_offset,
                       uint32_t *out) {
    lock_guard guard(state.lock);
    return jitc_var_vcall(name, self, mask, n_inst, inst_id, n_in, in,
                          n_out_nested, out_nested, se_offset, out);
}

uint32_t jit_var_loop_init(size_t n_indices, uint32_t **indices) {
    lock_guard guard(state.lock);
    return jitc_var_loop_init(n_indices, indices);
}

uint32_t jit_var_loop_cond(uint32_t loop_init, uint32_t cond, size_t n_indices,
                           uint32_t **indices) {
    lock_guard guard(state.lock);
    return jitc_var_loop_cond(loop_init, cond, n_indices, indices);
}

uint32_t jit_var_loop(const char *name, uint32_t loop_init, uint32_t loop_cond,
                      size_t n_indices, uint32_t *indices_in,
                      uint32_t **indices, uint32_t checkpoint, int first_round) {
    lock_guard guard(state.lock);
    return jitc_var_loop(name, loop_init, loop_cond, n_indices, indices_in,
                         indices, checkpoint, first_round);
}

struct VCallBucket *
jit_var_vcall_reduce(JitBackend backend, const char *domain, uint32_t index,
                     uint32_t *bucket_count_out) {
    lock_guard guard(state.lock);
    return jitc_var_vcall_reduce(backend, domain, index, bucket_count_out);
}

void jit_kernel_history_clear() {
    lock_guard guard(state.lock);
    state.kernel_history.clear();
}

struct KernelHistoryEntry *jit_kernel_history() {
    lock_guard guard(state.lock);
    jitc_sync_thread();
    return state.kernel_history.get();
}

#if defined(DRJIT_ENABLE_OPTIX)
OptixDeviceContext jit_optix_context() {
    lock_guard guard(state.lock);
    return jitc_optix_context();
}

void *jit_optix_lookup(const char *name) {
    lock_guard guard(state.lock);
    return jitc_optix_lookup(name);
}

uint32_t jit_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                      OptixModule module,
                                      const OptixProgramGroup *pg,
                                      uint32_t pg_count) {
    lock_guard guard(state.lock);
    return jitc_optix_configure_pipeline(pco, module, pg, pg_count);
}

uint32_t jit_optix_configure_sbt(const OptixShaderBindingTable *sbt, uint32_t pipeline) {
    lock_guard guard(state.lock);
    return jitc_optix_configure_sbt(sbt, pipeline);
}

void jit_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt) {
    lock_guard guard(state.lock);
    jitc_optix_update_sbt(index, sbt);
}

void jit_optix_ray_trace(uint32_t nargs, uint32_t *args, uint32_t mask,
                         uint32_t pipeline, uint32_t sbt) {
    lock_guard guard(state.lock);
    jitc_optix_ray_trace(nargs, args, mask, pipeline, sbt);
}

void jit_optix_mark(uint32_t index) {
    lock_guard guard(state.lock);
    jitc_optix_mark(index);
}

#endif

void jit_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                        const uint32_t *in, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_llvm_ray_trace(func, scene, shadow_ray, in, out);
}

void *jit_cuda_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                          int filter_mode, int wrap_mode) {
    lock_guard guard(state.lock);
    return jitc_cuda_tex_create(ndim, shape, n_channels, filter_mode, wrap_mode);
}

void jit_cuda_tex_get_shape(size_t ndim, const void *texture_handle,
                            size_t *shape) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_get_shape(ndim, texture_handle, shape);
}

void jit_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                             const void *src_ptr, void *dst_texture) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_memcpy_d2t(ndim, shape, src_ptr, dst_texture);
}

void jit_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                             const void *src_texture, void *dst_ptr) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_memcpy_t2d(ndim, shape, src_texture, dst_ptr);
}

void jit_cuda_tex_lookup(size_t ndim, const void *texture_handle,
                         const uint32_t *pos, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_lookup(ndim, texture_handle, pos, out);
}

void jit_cuda_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                               const uint32_t *pos, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_bilerp_fetch(ndim, texture_handle, pos, out);
}

void jit_cuda_tex_destroy(void *texture) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_destroy(texture);
}
