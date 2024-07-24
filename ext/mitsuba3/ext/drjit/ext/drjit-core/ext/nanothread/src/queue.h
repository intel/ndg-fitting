/*
    src/queue.h -- Lock-free task queue implementation used by nanothread

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>

#if defined(_WIN32)
#  include <windows.h>
#endif

struct Pool;

constexpr uint64_t high_bit  = (uint64_t) 0x0000000100000000ull;
constexpr uint64_t high_mask = (uint64_t) 0xFFFFFFFF00000000ull;
constexpr uint64_t low_mask  = (uint64_t) 0x00000000FFFFFFFFull;

inline uint64_t shift(uint32_t value) { return ((uint64_t) value) << 32; }

struct Task {
    /**
     * \brief Wide 16 byte pointer to a task in the worker pool. In addition to the
     * pointer itself, it encapsulates two more pieces of information:
     *
     * 1. The lower 32 bit of the \c value field store how many remaining work
     *    units the task contains
     *
     * 2. The upper 32 bit of the \c value field contain a counter to prevent
     *    the ABA problem during atomic updates.
     */
    struct alignas(16) Ptr {
        Task *task;
        uint64_t value;

        Ptr(Task *task = nullptr, uint64_t value = 0) : task(task), value(value) { }

        Task::Ptr update_remain(uint32_t remain = 0) const {
            return Ptr{ task, remain | ((value & high_mask) + high_bit) };
        }

        Task::Ptr update_task(Task *new_task) const {
            return Ptr{ new_task, (value & high_mask) + high_bit };
        }

        operator bool() const { return task != nullptr; }

        uint32_t remain() const { return (uint32_t) value; }

        bool operator==(const Task::Ptr &other) const {
            return task == other.task && value == other.value;
        }
    };

    /// Singly linked list, points to the next element
    Task::Ptr next;

    /**
     * \brief Reference count of this instance
     *
     * The reference count is arranged as a 2-tuple of 32 bit counters. When
     * submitting a work unit, its reference count is initially set to <tt>(3,
     * size)</tt>, where \c size is the number of associated work units. The
     * number '3' indicates three special references
     *
     *  - 1. A reference by the user code, which may e.g. wait for task completion
     *  - 2. A reference as part of the queue data structure
     *  - 3. A reference because the lower part is nonzero
     *
     * The function <tt>TaskQueue::release(task, high=true/false)</tt> can be
     * used to reduce the high and low parts separately.
     *
     * When the low part reaches zero, it assumed that all associated work
     * units have been completed, at which point child tasks are scheduled
     * and the task's payload is cleared. When both high and low parts reach
     * zero, it is assumed that no part of the system holds a reference to the
     * task, and it can be recycled.
     */
    std::atomic<uint64_t> refcount;

    /// Number of parent tasks that this task is waiting for
    std::atomic<uint32_t> wait_parents;

    /// Number of threads that are waiting for this task in task_wait()
    std::atomic<uint32_t> wait_count;

    /// Total number of work units in this task
    uint32_t size;

    /// Callback of the work unit
    void (*func)(uint32_t, void *);

    /// Pool that this tasks belongs to
    Pool *pool;

    /// Payload to be delivered to 'func'
    void *payload;

    /// Custom deleter used to free 'payload'
    void (*payload_deleter)(void *);

    /// Successor tasks that depend on this task
    std::vector<Task *> children;

    /// Atomic flag stating whether the 'exception' field is already used
    std::atomic<bool> exception_used;

    /// Pointer to an exception in case the task failed
    std::exception_ptr exception;

#if !defined(_WIN32)
    timespec time_start, time_end;
#else
    LARGE_INTEGER time_start, time_end;
#endif

    /// Fixed-size payload storage region
    alignas(8) uint8_t payload_storage[256];

    void clear() {
        if (payload_deleter)
            payload_deleter(payload);
        payload_deleter = nullptr;
        payload = nullptr;
        children.clear();
#if !defined(NDEBUG)
        memset(payload_storage, 0xFF, sizeof(payload_storage));
#endif
    }
};

/**
 * Modified implementation of the lock-free queue presented in the paper
 *
 * "Simple, fast and practical non-blocking and blocking concurrent queue algorithms"
 * by Maged Michael and Michael Scott.
 *
 * The main difference compared to a Michael-Scott queue is that each queue
 * item also has a *size* \c N that effectively creates \c N adjacent copies of
 * the item (but using a counter, which is more efficient than naive
 * replication). The \ref pop() operation returns the a pointer to the item and
 * a number in the range <tt>[0, N-1]</tt> indicating the item's index.
 *
 * Tasks can also have children. Following termination of a task, the queue
 * will push any children that don't depend on other unfinished work.
 *
 * The implementation here is designed to work on standard weakly ordered
 * memory architecture (e.g. AArch64), but likely would not not work an
 * completely weakly ordered architecture like the DEC Alpha.
 */
struct TaskQueue {
public:
    /// Create an empty task queue
    TaskQueue();

    /// Free the queue and delete any remaining tasks
    ~TaskQueue();

    /**
     * \brief Allocate a new task record consisting of \c size work units
     *
     * The implementation tries to fetch an available task instance from a
     * pool of completed tasks, if possible. Otherwise, a new task is created.
     *
     * It is assumed that the caller will populate the remaining fields of the
     * returned task and then invoke \ref push() to submit the task to the
     * queue. The reference count of the returned task is initially set to
     * <tt>(2, size)</tt>, where \c size is the number of associated work
     * units. The number '2' indicates two special references by user code and
     * by the queue itself, which don't correspond to outstanding work.
     *
     * Initializes the Tasks' \c wait \c size, \c refcount, and \c next fields.
     */
    Task *alloc(uint32_t size);

    /**
     * \brief Decrease the reference count of a task.
     *
     * The implementation moves the task into a pool of completed tasks once
     * the task is no longer referenced by any thread or data structure.
     */
    void release(Task *task, bool high = false);

    /// Increase the reference count of a task.
    void retain(Task *task);

    /// Append a task at the end of the queue
    void push(Task *task);

    /// Register an inter-task dependency
    void add_dependency(Task *task, Task *child);

    /**
     * \brief Pop a task from the queue
     *
     * When the queue is nonempty, this function returns a task instance and a
     * number in the range <tt>[0, size - 1]</tt>, where \c size is the number
     * of work units in the task. Otherwise, it returns \c nullptr and 0.
     */
    std::pair<Task *, uint32_t> pop();

    /**
     * \breif Fetch a task from the queue, or sleep
     *
     * This function repeatedly tries to fetch work from the queue and sleeps
     * if no work is available for an extended amount of time (~50 ms).
     *
     * The function stops trying to acquire work and returns <tt>(nullptr,
     * 0)</tt> when the supplied function <tt>stopping_criterion(payload)</tt>
     * evaluates to true.
     */
    std::pair<Task *, uint32_t> pop_or_sleep(bool (*stopping_criterion)(void *), void *payload);

    /// Wake sleeping threads
    void wakeup();

private:
    /// Head and tail of a lock-free list data structure
    Task::Ptr head, tail;

    /// Head of a lock-free stack storing unused tasks
    Task::Ptr recycle;

    /// Number of task instances created (for debugging)
    std::atomic<uint32_t> tasks_created;

    /// Upper 32 bit: sleep phase, lower 32 bit: number of sleepers
    std::atomic<uint64_t> sleep_state;

    /// Mutex protecting the fields below
    std::mutex sleep_mutex;

    /// Condition variable used to manage workers that are asleep
    std::condition_variable sleep_cv;
};


extern "C" uint32_t pool_thread_id();

extern int profile_tasks;

#define DJT_STR_2(x) #x
#define DJT_STR(x)   DJT_STR_2(x)

// #define DJT_DEBUG
#if defined(DJT_DEBUG)
#  define DJT_TRACE(fmt, ...)                                                  \
      fprintf(stderr, "%03u: " fmt "\n", pool_thread_id(), ##__VA_ARGS__)
#else
#  define DJT_TRACE(fmt, ...) do { } while (0)
#endif

#define DJT_ASSERT(x)                                                          \
    if (!(x)) {                                                                \
        fprintf(stderr, "Assertion failed in " __FILE__                        \
                        ":" DJT_STR(__LINE__) ": " #x "\n");                   \
        abort();                                                               \
    }

