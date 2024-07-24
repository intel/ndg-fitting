/*
    nanothread/nanothread.h -- Simple thread pool with a task-based API

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#if defined(_MSC_VER)
#  if defined(NANOTHREAD_BUILD)
#    define NANOTHREAD_EXPORT    __declspec(dllexport)
#  else
#    define NANOTHREAD_EXPORT    __declspec(dllimport)
#  endif
#else
#  define NANOTHREAD_EXPORT      __attribute__ ((visibility("default")))
#endif

#if defined(__cplusplus)
#  define NANOTHREAD_DEF(x) = x
#else
#  define NANOTHREAD_DEF(x)
#endif

#define NANOTHREAD_AUTO ((uint32_t) -1)

typedef struct Pool Pool;
typedef struct Task Task;

#if defined(__cplusplus)
#define NANOTHREAD_THROW     noexcept(false)
extern "C" {
#else
#define NANOTHREAD_THROW
#endif

/**
 * \brief Create a new thread pool
 *
 * \param size
 *     Specifies the desired number of threads. The default value of
 *     \c NANOTHREAD_AUTO choses a thread count equal to the number of
 *     available cores.
 *
 * \param ftz
 *     Should denormalized floating point numbers be flushed to zero?
 *     The pool workers will initialize their floating point control
 *     registers accordingly.
 */
extern NANOTHREAD_EXPORT Pool *
pool_create(uint32_t size NANOTHREAD_DEF(NANOTHREAD_AUTO),
            int ftz NANOTHREAD_DEF(1));

/**
 * \brief Destroy the thread pool and discard remaining unfinished work.
 *
 * It is undefined behavior to destroy the thread pool while other threads
 * are waiting for the completion of scheduled work via \ref task_wait().
 *
 * \param pool
 *     The thread pool to destroy. \c nullptr refers to the default pool.
 */
extern NANOTHREAD_EXPORT void pool_destroy(Pool *pool NANOTHREAD_DEF(0));

/**
 * \brief Return the number of threads that are part of the pool
 *
 * \param pool
 *     The thread pool to query. \c nullptr refers to the default pool.
 */
extern NANOTHREAD_EXPORT uint32_t pool_size(Pool *pool NANOTHREAD_DEF(0));

/**
 * \brief Resize the thread pool to the given number of threads
 *
 * \param pool
 *     The thread pool to resize. \c nullptr refers to the default pool.
 */
extern NANOTHREAD_EXPORT void pool_set_size(Pool *pool, uint32_t size);

/**
 * \brief Enable/disable time profiling
 *
 * Profiling must be enabled to use the \ref task_time() function.
 *
 * \param value
 *     A nonzero value indicates that profiling should be enabled.
 */
extern NANOTHREAD_EXPORT void pool_set_profile(int value);

/// Check whether time profiling is enabled (global setting)
extern NANOTHREAD_EXPORT int pool_profile();

/**
 * \brief Return a unique number identifying the current worker thread
 *
 * When called from a thread pool worker (e.g. while executing a parallel
 * task), this function returns a unique identifying number between 1 and the
 * pool's total thread count.
 *
 * The IDs of separate thread pools overlap. When the current thread is not a
 * thread pool worker, the function returns zero.
 */
extern NANOTHREAD_EXPORT uint32_t pool_thread_id();

/*
 * \brief Submit a new task to a thread pool
 *
 * This function submits a new task consisting of \c size work units to the
 * thread pool \c pool.
 *
 * <b>Callback</b>: The task callback \c func will be invoked \c size times by
 * the various thread pool workers. Its first argument will range from
 * <tt>0</tt> to \c <tt>size - 1</tt>, and the second argument refers to a
 * payload memory region specified via the \c payload parameter.
 *
 * <b>Parents</bb>: The \c parent and \c parent_count parameters can be used to
 * specify parent tasks that must be completed before execution of this task
 * can commence. If the task does not depend on any other tasks (e.g.
 * <tt>parent_count == 0</tt> and <tt>parent == nullptr</tt>), or when all of
 * those other tasks have already finished executing, then it will be
 * immediately appended to the end of the task queue. Otherwise, the task will
 * be scheduled once all parent tasks have finished executing.
 *
 * <b>Payload storage</b>: The callback payload is handled using one of two
 * possible modes:
 *
 * <ol>
 *    <li>When <tt>size == 0</tt> or <tt>payload_deleter != nullptr</tt>, the
 *    value of the \c payload parameter is simply forwarded to the callback \c
 *    func. In the latter case, <tt>payload_deleter(payload)</tt> is invoked
 *    following completion of the task, which can carry out additional cleanup
 *    operations if needed. In both cases, the memory region targeted by \c
 *    payload may be accessed asynchronously and must remain valid until the
 *    task is done.</li>
 *
 *    <li>Otherwise, the function will internally create a copy of the payload
 *    and free it following completion of the task. In this case, it is fine to
 *    delete the the memory region targeted by \c payload right after the
 *    function call.</li>
 * </ol>
 *
 * The function returns a task handle that can be used to schedule other
 * dependent tasks, and to wait for task completion if desired. This handle
 * must eventually be released using \ref task_release() or \ref
 * task_release_and_wait(). A failure to do so will result in memory leaks.
 *
 * <b>Small task optimization</b>: If desired, small tasks can be executed
 * right away without using the thread pool. This happens under the following
 * conditions:
 *
 * <ol>
 *   <li>The task is "small" (\c size == 1).</li>
 *   <li>The task does not depend on any parent tasks.</li>
 *   <li>The \c always_async parameter is set to 0</li>
 * </ol>
 *
 * \remark
 *     Barriers and similar dependency relations can be encoded by via
 *     artificial tasks using <tt>size == 0</tt> and <tt>func == nullptr<tt>
 *     along with a set of parent tasks.
 *
 * \param pool
 *     The thread pool that should execute the specified task. \c nullptr
 *     refers to the default pool.
 *
 * \param parent
 *     List of parents of size \c parent_count. \c nullptr-valued elements
 *     are ignored
 *
 * \param parent_count
 *     Number of parent tasks
 *
 * \param size
 *     Total number of work units; the callback \c func will be called this
 *     many times if provided.
 *
 * \param func
 *     Callback function that will be invoked to perform the actual computation.
 *     If set to \c nullptr, the callback is ignored. This can be used to create
 *     artificial tasks that only encode dependencies.
 *
 * \param payload
 *     Optional payload that is passed to the function \c func
 *
 * \param payload_size
 *     When \c payload_deleter is equal to \c nullptr and when \c size is
 *     nonzero, a temporary copy of the payload will be made. This parameter is
 *     necessary to specify the payload size in that case.
 *
 * \param payload_deleter
 *     Optional callback that will be invoked to free the payload
 *
 * \param always_async
 *     If set to a nonzero value, execution will always happen asynchronously,
 *     even in cases where the task being scheduled has no parents, and
 *     when only encodes a small amount of work (\c size == 1). Otherwise
 *     it will be executed synchronously, and the function will return \c nullptr.
 *
 * \return
 *     A task handle that must eventually be released via \ref task_release()
 *     or \ref task_wait_and_release(). The function returns \c nullptr when
 *     no task was generated (e.g. when there are no parent tasks, and either
 *     <tt>size==0</tt>, or when <tt>size==1</tt> and the task was executed
 *     synchronously.)
 */
extern NANOTHREAD_EXPORT
Task *task_submit_dep(Pool *pool,
                      const Task * const *parent,
                      uint32_t parent_count,
                      uint32_t size NANOTHREAD_DEF(1),
                      void (*func)(uint32_t, void *) NANOTHREAD_DEF(0),
                      void *payload NANOTHREAD_DEF(0),
                      uint32_t payload_size NANOTHREAD_DEF(0),
                      void (*payload_deleter)(void *) NANOTHREAD_DEF(0),
                      int always_async NANOTHREAD_DEF(0));

/*
 * \brief Release a task handle so that it can eventually be reused
 *
 * Releasing a task handle does not impact the tasks's execution, which could
 * be in one of three states: waiting, running, or complete. This operation is
 * important because it frees internal resources that would otherwise leak.
 *
 * Following a call to \ref task_release(), the associated task can no
 * longer be used as a direct parent of other tasks, and it is no longer
 * possible to wait for its completion using an operation like \ref
 * task_wait().
 *
 * \param pool
 *     The thread pool containing the task. \c nullptr refers to the default pool.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern NANOTHREAD_EXPORT void task_release(Task *task);

/*
 * \brief Wait for the completion of the specified task
 *
 * This function causes the calling thread to sleep until all work units of
 * 'task' have been completed.
 *
 * If an exception was caught during parallel excecution of 'task', the
 * function \ref task_wait() will re-raise this exception in the context of the
 * caller. Note that if a parallel task raises many exceptions, only a single
 * one of them will be be captured in this way.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern NANOTHREAD_EXPORT void task_wait(Task *task) NANOTHREAD_THROW;

/*
 * \brief Wait for the completion of the specified task and release its handle
 *
 * This function is equivalent to calling \ref task_wait() followed by \ref
 * task_release().
 *
 * If an exception was caught during parallel excecution of 'task', the
 * function \ref task_wait_and_release() will perform the release step and then
 * re-raise this exception in the context of the caller. Note that if a
 * parallel task raises many exceptions, only a single one of them will be be
 * captured in this way.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern NANOTHREAD_EXPORT void task_wait_and_release(Task *task) NANOTHREAD_THROW;

/**
 * \brief Return the time consumed by the task in milliseconds
 *
 * To use this function, you must first enable time profiling via \ref
 * pool_set_profile() before launching tasks.
 */
extern NANOTHREAD_EXPORT float task_time(Task *task) NANOTHREAD_THROW;

/*
 * \brief Increase the reference count of a task
 *
 * In advanced use case, it may be helpful if multiple parts of the system can
 * hold references to a task (and e.g. query timing information or
 * completeness). The \c task_retain operation enables this by increasing an
 * internal reference counter so that \ref task_release() must be called
 * multiple times before the task is actually released.
 *
 * \param task
 *     The task in question. When equal to \c nullptr, the operation is a no-op.
 */
extern NANOTHREAD_EXPORT void task_retain(Task *task);

/// Convenience wrapper around task_submit_dep(), but without dependencies
static inline
Task *task_submit(Pool *pool,
                  uint32_t size NANOTHREAD_DEF(1),
                  void (*func)(uint32_t, void *) NANOTHREAD_DEF(0),
                  void *payload NANOTHREAD_DEF(0),
                  uint32_t payload_size NANOTHREAD_DEF(0),
                  void (*payload_deleter)(void *) NANOTHREAD_DEF(0),
                  int always_async NANOTHREAD_DEF(0)) {

    return task_submit_dep(pool, 0, 0, size, func, payload, payload_size,
                           payload_deleter, always_async);
}

/// Convenience wrapper around task_submit(), but fully synchronous
static inline
void task_submit_and_wait(Pool *pool,
                          uint32_t size NANOTHREAD_DEF(1),
                          void (*func)(uint32_t, void *) NANOTHREAD_DEF(0),
                          void *payload NANOTHREAD_DEF(0)) {

    Task *task = task_submit(pool, size, func, payload, 0, 0, 0);
    task_wait_and_release(task);
}

#if defined(__cplusplus)
}

#include <utility>

namespace drjit {
    template <typename Int> struct blocked_range {
    public:
        blocked_range(Int begin, Int end, Int block_size = 1)
            : m_begin(begin), m_end(end), m_block_size(block_size) { }

        struct iterator {
            Int value;

            iterator(Int value) : value(value) { }

            Int operator*() const { return value; }
            operator Int() const { return value;}

            void operator++() { value++; }
            bool operator==(const iterator &it) { return value == it.value; }
            bool operator!=(const iterator &it) { return value != it.value; }
        };

        uint32_t blocks() const {
            return (uint32_t) ((m_end - m_begin + m_block_size - 1) / m_block_size);
        }

        iterator begin() const { return iterator(m_begin); }
        iterator end() const { return iterator(m_end); }
        Int block_size() const { return m_block_size; }

    private:
        Int m_begin;
        Int m_end;
        Int m_block_size;
    };

    template <typename Int, typename Func>
    void parallel_for(const blocked_range<Int> &range, Func &&func,
                      Pool *pool = nullptr) {

        struct Payload {
            Func *f;
            Int begin, end, block_size;
        };

        Payload payload{ &func, range.begin(), range.end(),
                         range.block_size() };

        auto callback = [](uint32_t index, void *payload) {
            Payload *p = (Payload *) payload;
            Int begin = p->begin + p->block_size * (Int) index,
                end = begin + p->block_size;

            if (end > p->end)
                end = p->end;

            (*p->f)(blocked_range<Int>(begin, end));
        };

        task_submit_and_wait(pool, range.blocks(), callback, &payload);
    }

    template <typename Int, typename Func>
    Task *parallel_for_async(const blocked_range<Int> &range, Func &&func,
                             const Task * const *parents,
                             size_t parent_count,
                             Pool *pool = nullptr) {
        using BaseFunc = typename std::decay<Func>::type;

        struct Payload {
            BaseFunc f;
            Int begin, end, block_size;
        };

        auto callback = [](uint32_t index, void *payload) {
            Payload *p = (Payload *) payload;
            Int begin = p->begin + p->block_size * (Int) index,
                end = begin + p->block_size;

            if (end > p->end)
                end = p->end;

            p->f(blocked_range<Int>(begin, end));
        };

        if (std::is_trivially_copyable<BaseFunc>::value &&
            std::is_trivially_destructible<BaseFunc>::value) {
            Payload payload{ std::forward<Func>(func), range.begin(),
                             range.end(), range.block_size() };

            return task_submit_dep(pool, parents,
                                   (uint32_t) parent_count, range.blocks(),
                                   callback, &payload, sizeof(Payload), nullptr, 1);
        } else {
            Payload *payload = new Payload{ std::forward<Func>(func), range.begin(),
                                            range.end(), range.block_size() };

            auto deleter = [](void *payload) {
                delete (Payload *) payload;
            };

            return task_submit_dep(pool, parents,
                                   (uint32_t) parent_count, range.blocks(),
                                   callback, payload, 0, deleter, 1);
        }
    }

    template <typename Int, typename Func>
    Task *parallel_for_async(const blocked_range<Int> &range, Func &&func,
                             std::initializer_list<const Task *> parents = { },
                             Pool *pool = nullptr) {
        return parallel_for_async(range, func, parents.begin(), parents.size(),
                                  pool);
    }

    template <typename Func>
    Task *do_async(Func &&func, const Task * const *parents, size_t parent_count,
                   Pool *pool = nullptr) {
        using BaseFunc = typename std::decay<Func>::type;

        struct Payload {
            BaseFunc f;
        };

        auto callback = [](uint32_t /* unused */, void *payload) {
            ((Payload *) payload)->f();
        };

        if (std::is_trivially_copyable<BaseFunc>::value &&
            std::is_trivially_destructible<BaseFunc>::value) {
            Payload payload {std::forward<Func>(func) };

            return task_submit_dep(pool, parents,
                                   (uint32_t) parent_count, 1, callback,
                                   &payload, sizeof(Payload), nullptr, 1);
        } else {
            Payload *payload = new Payload{ std::forward<Func>(func) };

            auto deleter = [](void *payload) { delete (Payload *) payload; };

            return task_submit_dep(pool, parents,
                                   (uint32_t) parent_count, 1, callback,
                                   payload, 0, deleter, 1);
        }
    }

    template <typename Func>
    Task *do_async(Func &&func, std::initializer_list<const Task *> parents = {},
                   Pool *pool = nullptr) {
        return do_async(func, parents.begin(), parents.size(), pool);
    }
}
#endif
