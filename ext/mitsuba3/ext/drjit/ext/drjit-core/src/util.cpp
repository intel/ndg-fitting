/*
    src/util.cpp -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <condition_variable>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "vcall.h"
#include "profiler.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

const char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                       "min", "max", "and", "or" };

/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
void jitc_submit_cpu(KernelType type, ThreadState *ts, Func &&func,
                     uint32_t width, uint32_t size = 1, bool release_prev = true) {

    struct Payload { Func f; };
    Payload payload{ std::forward<Func>(func) };

    static_assert(std::is_trivially_copyable<Payload>::value &&
                  std::is_trivially_destructible<Payload>::value, "Internal error!");

    Task *new_task = task_submit_dep(
        nullptr, &ts->task, 1, size,
        [](uint32_t index, void *payload) { ((Payload *) payload)->f(index); },
        &payload, sizeof(Payload));

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        task_wait(new_task);

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        KernelHistoryEntry entry = {};
        entry.backend = JitBackend::LLVM;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        task_retain(new_task);
        entry.task = new_task;
        state.kernel_history.append(entry);
    }

    if (release_prev)
        task_release(ts->task);

    ts->task = new_task;
}

void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
                     uint32_t thread_count, uint32_t shared_mem_bytes,
                     CUstream stream, void **args, void **extra,
                     uint32_t width) {

    KernelHistoryEntry entry = {};

    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        cuda_check(cuEventCreate((CUevent *) &entry.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &entry.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) entry.event_start, stream));
    }

    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1, thread_count, 1, 1,
                              shared_mem_bytes, stream, args, extra));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        cuda_check(cuStreamSynchronize(stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::CUDA;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        cuda_check(cuEventRecord((CUevent) entry.event_end, stream));

        state.kernel_history.append(entry);
    }
}

/// Fill a device memory region with constants of a given type
void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size_,
                       uint32_t isize, const void *src) {
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
              (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    size_t size = size_;

    // Try to convert into ordinary memset if possible
    uint64_t zero = 0;
    if (memcmp(src, &zero, isize) == 0) {
        size *= isize;
        isize = 1;
    }

    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        switch (isize) {
            case 1:
                cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                           ((uint8_t *) src)[0], size,
                                           ts->stream));
                break;

            case 2:
                cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                            ((uint16_t *) src)[0], size,
                                            ts->stream));
                break;

            case 4:
                cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                            ((uint32_t *) src)[0], size,
                                            ts->stream));
                break;

            case 8: {
                    const Device &device = state.devices[ts->device];
                    uint32_t block_count, thread_count;
                    device.get_launch_config(&block_count, &thread_count, size_);
                    void *args[] = { &ptr, &size_, (void *) src };
                    CUfunction kernel = jitc_cuda_fill_64[device.id];
                    jitc_submit_gpu(KernelType::Reduce, kernel, block_count,
                                    thread_count, 0, ts->stream, args, nullptr,
                                    size_);
                }
                break;
        }
    } else {
        uint8_t src8[8] { };
        memcpy(&src8, src, isize);

        jitc_submit_cpu(KernelType::Other, ts,
            [ptr, src8, size, isize](uint32_t) {
                switch (isize) {
                    case 1:
                        memset(ptr, src8[0], size);
                        break;

                    case 2: {
                            uint16_t value = ((uint16_t *) src8)[0],
                                    *p = (uint16_t *) ptr;
                            for (uint32_t i = 0; i < size; ++i)
                                p[i] = value;
                        }
                        break;

                    case 4: {
                            uint32_t value = ((uint32_t *) src8)[0],
                                    *p = (uint32_t *) ptr;
                            for (uint32_t i = 0; i < size; ++i)
                                p[i] = value;
                        }
                        break;

                    case 8: {
                            uint64_t value = ((uint64_t *) src8)[0],
                                    *p = (uint64_t *) ptr;
                            for (uint32_t i = 0; i < size; ++i)
                                p[i] = value;
                        }
                        break;
                }
            },

            (uint32_t) size
        );
    }
}

/// Perform a synchronous copy operation
void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    // Temporarily release the lock while copying
    unlock_guard guard(state.lock);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard_2(ts->context);
        cuda_check(cuStreamSynchronize(ts->stream));
        cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
    } else {
        jitc_sync_thread(ts);
        memcpy(dst, src, size);
    }
}

/// Perform an asynchronous copy operation
void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                                 ts->stream));
    } else {
        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [dst, src, size](uint32_t) {
                memcpy(dst, src, size);
            },

            (uint32_t) size
        );
    }
}

using Reduction = void (*) (const void *ptr, uint32_t start, uint32_t end, void *out);

template <typename Value>
static Reduction jitc_reduce_create(ReduceOp rtype) {
    using UInt = uint_with_size_t<Value>;

    switch (rtype) {
        case ReduceOp::Add:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result += ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Mul:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = 1;
                for (uint32_t i = start; i != end; ++i)
                    result *= ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Max:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::min()
                                   : -std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Min:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::max()
                                   :  std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::min(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Or:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result |= ptr[i];
                *((UInt *) out) = result;
            };

        case ReduceOp::And:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = (UInt) -1;
                for (uint32_t i = start; i != end; ++i)
                    result &= ptr[i];
                *((UInt *) out) = result;
            };

        default: jitc_raise("jit_reduce_create(): unsupported reduction type!");
    }
}

static Reduction jitc_reduce_create(VarType type, ReduceOp rtype) {
    switch (type) {
        case VarType::Int8:    return jitc_reduce_create<int8_t  >(rtype);
        case VarType::UInt8:   return jitc_reduce_create<uint8_t >(rtype);
        case VarType::Int16:   return jitc_reduce_create<int16_t >(rtype);
        case VarType::UInt16:  return jitc_reduce_create<uint16_t>(rtype);
        case VarType::Int32:   return jitc_reduce_create<int32_t >(rtype);
        case VarType::UInt32:  return jitc_reduce_create<uint32_t>(rtype);
        case VarType::Int64:   return jitc_reduce_create<int64_t >(rtype);
        case VarType::UInt64:  return jitc_reduce_create<uint64_t>(rtype);
        case VarType::Float32: return jitc_reduce_create<float   >(rtype);
        case VarType::Float64: return jitc_reduce_create<double  >(rtype);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

void jitc_reduce(JitBackend backend, VarType type, ReduceOp rtype, const void *ptr,
                uint32_t size, void *out) {
    ThreadState *ts = thread_state(backend);

    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, rtype=%s, size=%u)",
            (uintptr_t) ptr, type_name[(int) type],
            reduction_name[(int) rtype], size);

    uint32_t tsize = type_size[(int) type];

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_reductions[(int) rtype][(int) type][device.id];
        if (!func)
            jitc_raise("jit_reduce(): no existing kernel for type=%s, rtype=%s!",
                      type_name[(int) type], reduction_name[(int) rtype]);

        uint32_t thread_count = 1024,
                 shared_size = thread_count * tsize,
                 block_count;

        device.get_launch_config(&block_count, nullptr, size, thread_count);

        if (size <= 1024) {
            // This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };

            jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                            shared_size, ts->stream, args, nullptr, size);
        } else {
            void *temp = jitc_malloc(AllocType::Device, size_t(block_count) * tsize);

            // First reduction
            void *args_1[] = { &ptr, &size, &temp };

            jitc_submit_gpu(KernelType::Reduce, func, block_count, thread_count,
                            shared_size, ts->stream, args_1, nullptr, size);

            // Second reduction
            void *args_2[] = { &temp, &block_count, &out };

            jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                            shared_size, ts->stream, args_2, nullptr, size);

            jitc_free(temp);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = DRJIT_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        void *target = out;
        if (blocks > 1)
            target = jitc_malloc(AllocType::HostAsync, blocks * tsize);

        Reduction reduction = jitc_reduce_create(type, rtype);
        jitc_submit_cpu(
            KernelType::Reduce,
            ts,
            [block_size, size, tsize, ptr, reduction, target](uint32_t index) {
                reduction(ptr, index * block_size,
                          std::min((index + 1) * block_size, size),
                          (uint8_t *) target + index * tsize);
            },

            size,
            std::max(1u, blocks));

        if (blocks > 1) {
            jitc_reduce(backend, type, rtype, target, blocks, out);
            jitc_free(target);
        }
    }
}

/// 'All' reduction for boolean arrays
bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        jitc_memset_async(backend, values + size, trailing, sizeof(bool), &filler);
    }

    bool result;
    if (backend == JitBackend::CUDA) {
        uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
        jitc_reduce(backend, VarType::UInt32, ReduceOp::And, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
        jitc_free(out);
    } else {
        uint8_t out[4];
        jitc_reduce(backend, VarType::UInt32, ReduceOp::And, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
    }

    return result;
}

/// 'Any' reduction for boolean arrays
bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        jitc_memset_async(backend, values + size, trailing, sizeof(bool), &filler);
    }

    bool result;
    if (backend == JitBackend::CUDA) {
        uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
        jitc_reduce(backend, VarType::UInt32, ReduceOp::Or, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
        jitc_free(out);
    } else {
        uint8_t out[4];
        jitc_reduce(backend, VarType::UInt32, ReduceOp::Or, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
    }

    return result;
}

/// Exclusive prefix sum
void jitc_scan_u32(JitBackend backend, const uint32_t *in, uint32_t size, uint32_t *out) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        const Device &device = state.devices[ts->device];
        scoped_set_context guard(ts->context);

        if (size == 0) {
            return;
        } else if (size == 1) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, sizeof(uint32_t),
                                       ts->stream));
        } else if (size <= 4096) {
            /// Kernel for small arrays
            uint32_t items_per_thread = 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * sizeof(uint32_t);

            jitc_log(Debug,
                    "jit_scan(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=small, threads=%u, shared=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, thread_count,
                    shared_size);

            void *args[] = { &in, &out, &size };
            jitc_submit_gpu(
                KernelType::Other, jitc_cuda_scan_small_u32[device.id], 1,
                thread_count, shared_size, ts->stream, args, nullptr, size);
        } else {
            /// Kernel for large arrays
            uint32_t items_per_thread = 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * sizeof(uint32_t),
                     scratch_items    = block_count + 32;

            jitc_log(Debug,
                    "jit_scan(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                    "scratch=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, block_count,
                    thread_count, shared_size, scratch_items * 4);

            uint64_t *scratch = (uint64_t *) jitc_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            /// Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_scan_large_u32_init[device.id],
                            block_count_init, thread_count_init, 0, ts->stream,
                            args, nullptr, scratch_items);

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &scratch };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_scan_large_u32[device.id], block_count,
                            thread_count, shared_size, ts->stream, args_2,
                            nullptr, scratch_items);
            scratch -= 32;

            jitc_free(scratch);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = DRJIT_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        jitc_log(Debug,
                "jit_scan(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        uint32_t *scratch = nullptr;

        if (blocks > 1) {
            scratch = (uint32_t *) jitc_malloc(AllocType::HostAsync,
                                              blocks * sizeof(uint32_t));

            jitc_submit_cpu(
                KernelType::Other,
                ts,
                [block_size, size, in, scratch](uint32_t index) {
                    uint32_t start = index * block_size,
                             end = std::min(start + block_size, size);

                    uint32_t accum = 0;
                    for (uint32_t i = start; i != end; ++i)
                        accum += in[i];

                    scratch[index] = accum;
                },

                size, blocks
            );

            jitc_scan_u32(backend, scratch, blocks, scratch);
        }

        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [block_size, size, in, out, scratch](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t accum = 0;
                if (scratch)
                    accum = scratch[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t value = in[i];
                    out[i] = accum;
                    accum += value;
                }
            },

            size, blocks
        );

        jitc_free(scratch);
    }
}

/// Mask compression
uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    if (size == 0)
        return 0;

    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        const Device &device = state.devices[ts->device];
        scoped_set_context guard(ts->context);

        uint32_t *count_out = (uint32_t *) jitc_malloc(
            AllocType::HostPinned, sizeof(uint32_t));

        if (size <= 4096) {
            // Kernel for small arrays
            uint32_t items_per_thread = 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * sizeof(uint32_t),
                     trailer          = thread_count * items_per_thread - size;

            jitc_log(Debug,
                    "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=small, threads=%u, shared=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, thread_count,
                    shared_size);

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           ts->stream));

            void *args[] = { &in, &out, &size, &count_out };
            jitc_submit_gpu(
                KernelType::Other, jitc_cuda_compress_small[device.id], 1,
                thread_count, shared_size, ts->stream, args, nullptr, size);
        } else {
            // Kernel for large arrays
            uint32_t items_per_thread = 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * sizeof(uint32_t),
                     scratch_items    = block_count + 32,
                     trailer          = items_per_block * block_count - size;

            jitc_log(Debug,
                    "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                    "scratch=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, block_count,
                    thread_count, shared_size, scratch_items * 4);

            uint64_t *scratch = (uint64_t *) jitc_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            // Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_scan_large_u32_init[device.id],
                            block_count_init, thread_count_init, 0, ts->stream,
                            args, nullptr, scratch_items);

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           ts->stream));

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &scratch, &count_out };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_compress_large[device.id], block_count,
                            thread_count, shared_size, ts->stream, args_2,
                            nullptr, scratch_items);
            scratch -= 32;

            jitc_free(scratch);
        }
        jitc_sync_thread();
        uint32_t count_out_v = *count_out;
        jitc_free(count_out);
        return count_out_v;
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = DRJIT_POOL_BLOCK_SIZE;
            blocks     = (size + block_size - 1) / block_size;
        }

        uint32_t count_out = 0;

        jitc_log(Debug,
                "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        uint32_t *scratch = nullptr;

        if (blocks > 1) {
            scratch = (uint32_t *) jitc_malloc(AllocType::HostAsync,
                                              blocks * sizeof(uint32_t));

            jitc_submit_cpu(
                KernelType::Other,
                ts,
                [block_size, size, in, scratch](uint32_t index) {
                    uint32_t start = index * block_size,
                             end = std::min(start + block_size, size);

                    uint32_t accum = 0;
                    for (uint32_t i = start; i != end; ++i)
                        accum += (uint32_t) in[i];

                    scratch[index] = accum;
                },

                size, blocks
            );

            jitc_scan_u32(backend, scratch, blocks, scratch);
        }

        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [block_size, size, scratch, in, out, &count_out](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t accum = 0;
                if (scratch)
                    accum = scratch[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t value = (uint32_t) in[i];
                    if (value)
                        out[accum] = i;
                    accum += value;
                }

                if (end == size)
                    count_out = accum;
            },

            size, blocks
        );

        jitc_free(scratch);
        jitc_sync_thread();

        return count_out;
    }
}

static void cuda_transpose(ThreadState *ts, const uint32_t *in, uint32_t *out,
                           uint32_t rows, uint32_t cols) {
    const Device &device = state.devices[ts->device];

    uint16_t blocks_x = (uint16_t) ((cols + 15u) / 16u),
             blocks_y = (uint16_t) ((rows + 15u) / 16u);

    scoped_set_context guard(ts->context);
    jitc_log(Debug,
            "jit_transpose(" DRJIT_PTR " -> " DRJIT_PTR
            ", rows=%u, cols=%u, blocks=%ux%u)",
            (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y);

    void *args[] = { &in, &out, &rows, &cols };

    cuda_check(cuLaunchKernel(
        jitc_cuda_transpose[device.id], blocks_x, blocks_y, 1, 16, 16, 1,
        16 * 17 * sizeof(uint32_t), ts->stream, args, nullptr));
}

static ProfilerRegion profiler_region_mkperm("jit_mkperm");
static ProfilerRegion profiler_region_mkperm_phase_1("jit_mkperm_phase_1");
static ProfilerRegion profiler_region_mkperm_phase_2("jit_mkperm_phase_2");

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jitc_mkperm(JitBackend backend, const uint32_t *ptr, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_mkperm(): bucket_count cannot be zero!");

    ProfilerPhase profiler(profiler_region_mkperm);
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];

        // Don't use more than 1 block/SM due to shared memory requirement
        const uint32_t warp_size = 32;
        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size, 1024, 1);

        // Always launch full warps (the kernel impl. assumes that this is the case)
        uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
        thread_count = warp_count * warp_size;

        uint32_t bucket_size_1   = bucket_count * sizeof(uint32_t),
                 bucket_size_all = bucket_size_1 * block_count;

        /* If there is a sufficient amount of shared memory, atomically accumulate into a
           shared memory buffer. Otherwise, use global memory, which is much slower. */
        uint32_t shared_size = 0;
        const char *variant = nullptr;
        CUfunction phase_1 = nullptr, phase_4 = nullptr;
        bool initialize_buckets = false;

        if (bucket_size_1 * warp_count <= device.shared_memory_bytes) {
            /* "Tiny" variant, which uses shared memory atomics to produce a stable
               permutation. Handles up to 512 buckets with 64KiB of shared memory. */

            phase_1 = jitc_cuda_mkperm_phase_1_tiny[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_tiny[device.id];
            shared_size = bucket_size_1 * warp_count;
            bucket_size_all *= warp_count;
            variant = "tiny";
        } else if (bucket_size_1 <= device.shared_memory_bytes) {
            /* "Small" variant, which uses shared memory atomics and handles up to
               16K buckets with 64KiB of shared memory. The permutation can be
               somewhat unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.) */

            phase_1 = jitc_cuda_mkperm_phase_1_small[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_small[device.id];
            shared_size = bucket_size_1;
            variant = "small";
        } else {
            /* "Large" variant, which uses global memory atomics and handles
               arbitrarily many elements (though this is somewhat slower than the
               previous two shared memory variants). The permutation can be somewhat
               unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.)
               Buckets must be zero-initialized explicitly. */

            phase_1 = jitc_cuda_mkperm_phase_1_large[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_large[device.id];
            variant = "large";
            initialize_buckets = true;
        }

        bool needs_transpose = bucket_size_1 != bucket_size_all;
        uint32_t *buckets_1, *buckets_2, *counter = nullptr;
        buckets_1 = buckets_2 =
            (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

        // Scratch space for matrix transpose operation
        if (needs_transpose)
            buckets_2 = (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

        if (offsets) {
            counter = (uint32_t *) jitc_malloc(AllocType::Device, sizeof(uint32_t)),
            cuda_check(cuMemsetD8Async((CUdeviceptr) counter, 0, sizeof(uint32_t),
                                       ts->stream));
        }

        if (initialize_buckets)
            cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                       bucket_size_all, ts->stream));

        /* Determine the amount of work to be done per block, and ensure that it is
           divisible by the warp size (the kernel implementation assumes this.) */
        uint32_t size_per_block = (size + block_count - 1) / block_count;
        size_per_block = (size_per_block + warp_size - 1) / warp_size * warp_size;

        jitc_log(Debug,
                "jit_mkperm(" DRJIT_PTR
                ", size=%u, bucket_count=%u, block_count=%u, thread_count=%u, "
                "size_per_block=%u, variant=%s, shared_size=%u)",
                (uintptr_t) ptr, size, bucket_count, block_count, thread_count,
                size_per_block, variant, shared_size);

        // Phase 1: Count the number of occurrences per block
        void *args_1[] = { &ptr, &buckets_1, &size, &size_per_block,
                           &bucket_count };

        jitc_submit_gpu(KernelType::VCallReduce, phase_1, block_count,
                        thread_count, shared_size, ts->stream, args_1, nullptr,
                        size);

        // Phase 2: exclusive prefix sum over transposed buckets
        if (needs_transpose)
            cuda_transpose(ts, buckets_1, buckets_2,
                           bucket_size_all / bucket_size_1, bucket_count);

        jitc_scan_u32(backend, buckets_2, bucket_size_all / sizeof(uint32_t), buckets_2);

        if (needs_transpose)
            cuda_transpose(ts, buckets_2, buckets_1, bucket_count,
                           bucket_size_all / bucket_size_1);

        // Phase 3: collect non-empty buckets (optional)
        if (likely(offsets)) {
            uint32_t block_count_3, thread_count_3;
            device.get_launch_config(&block_count_3, &thread_count_3,
                                     bucket_count * block_count);

            // Round up to a multiple of the thread count
            uint32_t bucket_count_rounded =
                (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

            void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                               &size,      &counter,      &offsets };

            jitc_submit_gpu(KernelType::VCallReduce,
                            jitc_cuda_mkperm_phase_3[device.id], block_count_3,
                            thread_count_3, sizeof(uint32_t) * thread_count_3,
                            ts->stream, args_3, nullptr, size);

            cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * size_t(bucket_count)),
                                     (CUdeviceptr) counter, sizeof(uint32_t),
                                     ts->stream));

            cuda_check(cuEventRecord(ts->event, ts->stream));
        }

        // Phase 4: write out permutation based on bucket counts
        void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                           &bucket_count };

        jitc_submit_gpu(KernelType::VCallReduce, phase_4, block_count,
                        thread_count, shared_size, ts->stream, args_4, nullptr,
                        size);

        if (likely(offsets)) {
            unlock_guard guard_2(state.lock);
            cuda_check(cuEventSynchronize(ts->event));
        }

        jitc_free(buckets_1);
        if (needs_transpose)
            jitc_free(buckets_2);
        jitc_free(counter);

        return offsets ? offsets[4 * bucket_count] : 0u;
    } else { // if (!ts->cuda)
        uint32_t blocks = 1, block_size = size, pool_size = ::pool_size();

        if (pool_size > 1) {
            // Try to spread out uniformly over cores
            blocks = pool_size * 4;
            block_size = (size + blocks - 1) / blocks;

            // But don't make the blocks too small
            block_size = std::max((uint32_t) DRJIT_POOL_BLOCK_SIZE, block_size);

            // Finally re-adjust block count given the selected block size
            blocks = (size + block_size - 1) / block_size;
        }

        jitc_log(Debug,
                "jit_mkperm(" DRJIT_PTR
                ", size=%u, bucket_count=%u, block_size=%u, blocks=%u)",
                (uintptr_t) ptr, size, bucket_count, block_size, blocks);

        uint32_t **buckets =
            (uint32_t **) jitc_malloc(AllocType::HostAsync, sizeof(uint32_t *) * blocks);

        uint32_t unique_count = 0;

        // Phase 1
        jitc_submit_cpu(
            KernelType::VCallReduce,
            ts,
            [block_size, size, buckets, bucket_count, ptr](uint32_t index) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_1);

                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                size_t bsize = sizeof(uint32_t) * (size_t) bucket_count;
                uint32_t *buckets_local = (uint32_t *) malloc_check(bsize);
                memset(buckets_local, 0, bsize);

                 for (uint32_t i = start; i != end; ++i)
                     buckets_local[ptr[i]]++;

                 buckets[index] = buckets_local;
            },

            size, blocks
        );

        // Local accumulation step
        jitc_submit_cpu(
            KernelType::VCallReduce,
            ts,
            [bucket_count, blocks, buckets, offsets, &unique_count](uint32_t) {
                uint32_t sum = 0, unique_count_local = 0;
                for (uint32_t i = 0; i < bucket_count; ++i) {
                    uint32_t sum_local = 0;
                    for (uint32_t j = 0; j < blocks; ++j) {
                        uint32_t value = buckets[j][i];
                        buckets[j][i] = sum + sum_local;
                        sum_local += value;
                    }
                    if (sum_local > 0) {
                        if (offsets) {
                            offsets[unique_count_local*4] = i;
                            offsets[unique_count_local*4 + 1] = sum;
                            offsets[unique_count_local*4 + 2] = sum_local;
                            offsets[unique_count_local*4 + 3] = 0;
                        }
                        unique_count_local++;
                        sum += sum_local;
                    }
                }

                unique_count = unique_count_local;
            },

            size
        );

        Task *local_task = ts->task;

        // Phase 2
        jitc_submit_cpu(
            KernelType::VCallReduce,
            ts,
            [block_size, size, buckets, perm, ptr](uint32_t index) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_2);

                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t *buckets_local = buckets[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t idx = buckets_local[ptr[i]]++;
                    perm[idx] = i;
                }
                free(buckets_local);
            },

            size, blocks, false
        );

        // Free memory (happens asynchronously after the above stmt.)
        jitc_free(buckets);

        task_wait_and_release(local_task);

        return unique_count;
    }
}

using BlockOp = void (*) (const void *ptr, void *out, uint32_t start, uint32_t end, uint32_t block_size);

template <typename Value> static BlockOp jitc_block_copy_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start;
        Value *out = (Value *) out_ + start * block_size;
        for (uint32_t i = start; i != end; ++i) {
            Value value = *in++;
            for (uint32_t j = 0; j != block_size; ++j)
                *out++ = value;
        }
    };
}

template <typename Value> static BlockOp jitc_block_sum_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start * block_size;
        Value *out = (Value *) out_ + start;
        for (uint32_t i = start; i != end; ++i) {
            Value sum = 0;
            for (uint32_t j = 0; j != block_size; ++j)
                sum += *in++;
            *out++ = sum;
        }
    };
}

static BlockOp jitc_block_copy_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_copy_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_copy_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_copy_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_copy_create<uint64_t>();
        case VarType::Float32: return jitc_block_copy_create<float   >();
        case VarType::Float64: return jitc_block_copy_create<double  >();
        default: jitc_raise("jit_block_copy_create(): unsupported data type!");
    }
}

static BlockOp jitc_block_sum_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_sum_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_sum_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_sum_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_sum_create<uint64_t>();
        case VarType::Float32: return jitc_block_sum_create<float   >();
        case VarType::Float64: return jitc_block_sum_create<double  >();
        default: jitc_raise("jit_block_sum_create(): unsupported data type!");
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

/// Replicate individual input elements to larger blocks
void jitc_block_copy(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_copy(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_copy(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    if (block_size == 1) {
        uint32_t tsize = type_size[(int) type];
        jitc_memcpy_async(backend, out, in, size * tsize);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        size *= block_size;

        CUfunction func = jitc_cuda_block_copy[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_copy(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = DRJIT_POOL_BLOCK_SIZE;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_copy_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Sum over elements within blocks
void jitc_block_sum(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_sum(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_sum(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    uint32_t tsize = type_size[(int) type];
    size_t out_size = size * tsize;

    if (block_size == 1) {
        jitc_memcpy_async(backend, out, in, out_size);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];

        size *= block_size;

        CUfunction func = jitc_cuda_block_sum[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_sum(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, out_size, ts->stream));
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = DRJIT_POOL_BLOCK_SIZE;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_sum_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Asynchronously update a single element in memory
void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_poke[(int) type][device.id];
        void *args[] = { &dst, (void *) src };
        jitc_submit_gpu(KernelType::Other, func, 1, 1, 0,
                        ts->stream, args, nullptr, 1);
    } else {
        uint8_t src8[8] { };
        memcpy(&src8, src, size);

        jitc_submit_cpu(
            KernelType::Other,
            ts,
            [src8, size, dst](uint32_t) {
                memcpy(dst, &src8, size);
            },

            size
        );
    }
}


void jitc_vcall_prepare(JitBackend backend, void *dst_, VCallDataRecord *rec_, uint32_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_vcall_prepare[device.id];
        void *args[] = { &dst_, &rec_, &size };

        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size);

        jitc_log(InfoSym,
                 "jit_vcall_prepare(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, blocks=%u, threads=%u)",
                 (uintptr_t) rec_, (uintptr_t) dst_, size, block_count,
                 thread_count);

        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, 1);

        jitc_free(rec_);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = DRJIT_POOL_BLOCK_SIZE;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        jitc_log(InfoSym,
                 "jit_vcall_prepare(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, work_units=%u)",
                 (uintptr_t) rec_, (uintptr_t) dst_, size, work_units);

        jitc_submit_cpu(
            KernelType::Other, ts,
            [dst_, rec_, size, work_unit_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                for (uint32_t i = start; i != end; ++i) {
                    VCallDataRecord rec = rec_[i];

                    bool literal = rec.literal;
                    uintptr_t value = rec.value;
                    void *dst = (uint8_t *) dst_ + rec.offset;

                    switch (rec.size) {
                        case 1: *(uint8_t *)  dst = literal ? (uint8_t)  value : *(uint8_t *)  value; break;
                        case 2: *(uint16_t *) dst = literal ? (uint16_t) value : *(uint16_t *) value; break;
                        case 4: *(uint32_t *) dst = literal ? (uint32_t) value : *(uint32_t *) value; break;
                        case 8: *(uint64_t *) dst = literal ? (uint64_t) value : *(uint64_t *) value; break;
                    }
                }
            },
            size, work_units);

        jitc_submit_cpu(
            KernelType::Other, ts, [rec_](uint32_t) { jitc_free(rec_); }, 1);
    }
}
