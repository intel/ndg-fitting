/*
    src/queue.cpp -- Lock-free task queue implementation used by nanothread

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "queue.h"
#include <cstdio>
#include <ctime>

#if defined(_WIN32)
#  include <windows.h>
#endif

#if defined(_MSC_VER)
#  include <intrin.h>
#elif defined(__SSE2__)
#  include <emmintrin.h>
#endif

/// Put worker threads to sleep after 500K attempts to get work
#define NANOTHREAD_MAX_ATTEMPTS 500000

/// Reduce power usage in busy-wait CAS loops
static void pause() {
#if defined(_M_X64) || defined(__SSE2__)
    _mm_pause();
#endif
}

/// Atomic 16 byte compare-and-swap & release barrier on ARM
static bool cas(Task::Ptr &ptr, Task::Ptr &expected, Task::Ptr desired) {
#if defined(_MSC_VER)
    #if defined(_M_ARM64)
        return _InterlockedCompareExchange128_rel(
            (__int64 volatile *) &ptr, (__int64) desired.value,
            (__int64) desired.task, (__int64 *) &expected);
    #else
        return _InterlockedCompareExchange128(
            (__int64 volatile *) &ptr, (__int64) desired.value,
            (__int64) desired.task, (__int64 *) &expected);
    #endif
#else
    return __atomic_compare_exchange(&ptr, &expected, &desired, true,
                                     __ATOMIC_RELEASE, __ATOMIC_RELAXED);
#endif
}

// *Non-atomic* 16 byte load, acquire barrier on ARM
static Task::Ptr ldar(Task::Ptr &source) {
#if defined(_MSC_VER)
    using P = unsigned __int64 volatile *;
    #if defined(_M_ARM64)
        uint64_t value_1 = __ldar64((P) &source);
        uint64_t value_2 = __ldar64(((P) &source) + 1);
    #else
        uint64_t value_1 = *((P) &source);
        uint64_t value_2 = *(((P) &source) + 1);
    #endif
    return Task::Ptr{ (Task *) value_1, (uint64_t) value_2 };
#else
    uint64_t value_1 = __atomic_load_n((uint64_t *) &source, __ATOMIC_ACQUIRE);
    uint64_t value_2 = __atomic_load_n((((uint64_t *) &source) + 1), __ATOMIC_ACQUIRE);
    return Task::Ptr{ (Task *) value_1, value_2 };
#endif
}

TaskQueue::TaskQueue() : tasks_created(0), sleep_state(0) {
    head = Task::Ptr(alloc(0));
    tail = head;
}

TaskQueue::~TaskQueue() {
    uint32_t created = tasks_created.load(),
             deleted = 0, incomplete = 0,
             incomplete_size = 0;

    // Free jobs that are still in the queue
    Task::Ptr ptr = head;
    while (ptr.task) {
        Task *task = ptr.task;

        if (ptr.remain() != 0) {
            incomplete_size += ptr.remain();
            incomplete++;
        }

        for (Task *child : task->children) {
            uint32_t wait = child->wait_parents.fetch_sub(1);
            DJT_ASSERT(wait != 0);
            if (wait == 1)
                push(child);
        }

        task->clear();
        deleted++;
        ptr = task->next;
        delete task;
    }

    // Free jobs on the free-job stack
    ptr = recycle;
    while (ptr.task) {
        Task *task = ptr.task;
        DJT_ASSERT(task->payload == nullptr && task->children.empty());
        deleted++;
        ptr = task->next;
        delete task;
    }

    if (created != deleted)
        fprintf(stderr,
                "nanothread: %u/%u tasks were leaked! Did you forget to call "
                "task_release()?\n", created - deleted, created);

    if (incomplete > 0)
        fprintf(stderr, "nanothread: %u tasks with %u work units were not "
                "completed!\n", incomplete, incomplete_size);
}

Task *TaskQueue::alloc(uint32_t size) {
    Task::Ptr node = ldar(recycle);

    while (true) {
        // Stop if stack is empty
        if (!node)
            break;

        // Load the next node
        Task::Ptr next = ldar(node.task->next);

        // Next, try to move it to the stack head
        if (cas(recycle, node, node.update_task(next.task)))
            break;

        pause();
    }

    Task *task;

    if (node.task) {
        task = node.task;
    } else {
        task = new Task();
        tasks_created++;
    }

    task->next = Task::Ptr();
    task->refcount.store(size + (size == 0 ? high_bit : (3 * high_bit)),
                         std::memory_order_relaxed);
    task->wait_parents.store(0, std::memory_order_relaxed);
    task->wait_count.store(0, std::memory_order_relaxed);
    task->size = size;
    memset(&task->time_start, 0, sizeof(task->time_start));
    memset(&task->time_end, 0, sizeof(task->time_end));

    DJT_TRACE("created new task %p with size=%u", task, size);

    return task;
}

void TaskQueue::release(Task *task, bool high) {
    uint64_t result = task->refcount.fetch_sub(high ? high_bit : 1);
    uint32_t ref_lo = (uint32_t) result,
             ref_hi = (uint32_t) (result >> 32);

    DJT_ASSERT((!high || ref_hi > 0) && (high || ref_lo > 0));
    ref_hi -= (uint32_t) high;
    ref_lo -= (uint32_t) !high;

    DJT_TRACE("dec_ref(%p, (%i, %i)) -> ref = (%u, %u)", task, (int) high,
              (int) !high, ref_hi, ref_lo);

    // If all work has completed: schedule children and free payload
    if (!high && ref_lo == 0) {
        DJT_TRACE("all work associated with task %p has completed.", task);

        if (profile_tasks) {
            #if defined(_WIN32)
                QueryPerformanceCounter(&task->time_end);
            #else
                clock_gettime(CLOCK_MONOTONIC, &task->time_end);
            #endif
        }

        for (Task *child : task->children) {
            uint32_t wait = child->wait_parents.fetch_sub(1);

            DJT_TRACE("notifying child %p of task %p: wait=%u", child, task,
                      wait - 1);

            DJT_ASSERT(wait > 0);

            if (task->exception_used.load()) {
                bool expected = false;
                if (child->exception_used.compare_exchange_strong(expected, true)) {
                    DJT_TRACE("propagating exception to child %p of task %p.",
                              child, task);
                    child->exception = task->exception;
                } else {
                    DJT_TRACE("not propagating exception to child %p of "
                              "task %p (already stored).", child, task);
                }
            }

            if (wait == 1) {
                DJT_TRACE("Child %p of task %p is ready for execution.", child,
                          task);
                push(child);
            }
        }

        task->clear();

        // Possible that waiting threads were put to sleep
        if (task->wait_count.load() > 0)
            wakeup();

        release(task, true);
    } else if (high && ref_hi == 0) {
        // Nobody holds any references at this point, recycle task

        DJT_ASSERT(ref_lo == 0);
        DJT_TRACE("all usage of task %p is done, recycling.", task);

        Task::Ptr node = ldar(recycle);
        while (true) {
            task->next = node;

            if (cas(recycle, node, node.update_task(task)))
                break;

            pause();
        }
    }
}

void TaskQueue::add_dependency(Task *parent, Task *child) {
    if (!parent)
        return;

    uint64_t refcount =
        parent->refcount.load(std::memory_order_relaxed);

    /* Increase the parent task's reference count to prevent the cleanup
       handler in release() from starting while the following executes. */
    while (true) {
        if ((uint32_t) refcount == 0) {
            // Parent task has already completed
            if (parent->exception_used.load()) {
                bool expected = false;
                if (child->exception_used.compare_exchange_strong(expected, true)) {
                    DJT_TRACE("propagating exception to child %p of task %p.",
                              child, parent);
                    child->exception = parent->exception;
                } else {
                    DJT_TRACE("not propagating exception to child %p of "
                              "task %p (already stored).", child, parent);
                }
            }
            return;
        }

        if (parent->refcount.compare_exchange_weak(refcount, refcount + 1,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed))
            break;

        pause();
    }

    // Otherwise, register the child task with the parent
    parent->children.push_back(child);
    uint32_t wait = ++child->wait_parents;
    (void) wait;

    DJT_TRACE("registering dependency: parent=%p, child=%p, child->wait=%u",
              parent, child, wait);

    /* Undo the parent->refcount change. If the task completed in the
       meantime, child->wait_parents will also be decremented by
       this call. */
    release(parent);
}

void TaskQueue::retain(Task *task) {
    DJT_TRACE("retain(task=%p)", task);
    task->refcount.fetch_add(high_bit);
}

void TaskQueue::push(Task *task) {
    uint32_t size = task->size;

    DJT_TRACE("push(task=%p, size=%u)", task, size);

    while (true) {
        // Lead tail and tail->next, and double-check, in this order
        Task::Ptr tail_c = ldar(tail);
        Task::Ptr &next = tail_c.task->next;
        Task::Ptr next_c = ldar(next);
        Task::Ptr tail_c_2 = ldar(tail);

        // Detect inconsistencies due to contention
        if (tail_c == tail_c_2) {
            if (!next_c.task) {
                // Tail was pointing to last node, try to insert here
                if (cas(next, next_c, Task::Ptr(task, size))) {
                    // Best-effort attempt to redirect tail to the added element
                    cas(tail, tail_c, tail_c.update_task(task));
                    break;
                }
            } else {
                // Tail wasn't pointing to the last node, try to update
                cas(tail, tail_c, tail_c.update_task(next_c.task));
            }
        }

        pause();
    }

    // Wake sleeping threads, if any
    if (sleep_state.load(std::memory_order_acquire) & low_mask)
        wakeup();
}

std::pair<Task *, uint32_t> TaskQueue::pop() {
    uint32_t index;
    Task *task;

    while (true) {
        // Lead head, tail, and next element, and double-check, in this order
        Task::Ptr head_c = ldar(head);
        Task::Ptr tail_c = ldar(tail);
        Task::Ptr &next = head_c.task->next;
        Task::Ptr next_c = ldar(next);
        Task::Ptr head_c_2 = ldar(head);

        // Detect inconsistencies due to contention
        if (head_c == head_c_2) {
            if (head_c.task != tail_c.task) {
                uint32_t remain = next_c.remain();

                if (remain > 1) {
                    // More than 1 remaining work units, update work counter
                    if (cas(next, next_c, next_c.update_remain(remain - 1))) {
                        task = next_c.task;
                        index = task->size - remain;
                        break;
                    }
                } else {
                    DJT_ASSERT(remain == 1);
                    // Head node is removed from the queue, reduce refcount
                    if (cas(head, head_c, head_c.update_task(next_c.task))) {
                        task = next_c.task;
                        index = task->size - 1;
                        release(head_c.task, true);
                        break;
                    }
                }
            } else {
                // Task queue was empty
                if (!next_c.task) {
                    task = nullptr;
                    index = 0;
                    pause();
                    break;
                } else {
                    // Advance the tail, it's falling behind
                    cas(tail, tail_c, tail_c.update_task(next_c.task));
                }
            }
        }

        pause();
    }

    if (task) {
        DJT_TRACE("pop(task=%p, index=%u)", task, index);

        if (index == 0 && profile_tasks) {
            #if defined(_WIN32)
                QueryPerformanceCounter(&task->time_start);
            #else
                clock_gettime(CLOCK_MONOTONIC, &task->time_start);
            #endif
        }
    }

    return { task, index };
}

void TaskQueue::wakeup() {
    std::unique_lock<std::mutex> guard(sleep_mutex);
    uint64_t value = sleep_state.load();
    DJT_TRACE("wakeup(): sleep_state := (%u, 0)", (uint32_t) (sleep_state >> 32) + 1);
    sleep_state = (value + high_bit) & high_mask;
    sleep_cv.notify_all();
}

#if defined(DJT_DEBUG)
double time_milliseconds() {
    #if defined(_WIN32)
        LARGE_INTEGER ticks, ticks_per_sec;
        QueryPerformanceCounter(&ticks);
        QueryPerformanceFrequency(&ticks_per_sec);
        return (double) (ticks.QuadPart * 1000) / (double) ticks_per_sec.QuadPart;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    #endif
}
#endif

std::pair<Task *, uint32_t> TaskQueue::pop_or_sleep(bool (*stopping_criterion)(void *), void *payload) {
    std::pair<Task *, uint32_t> result(nullptr, 0);
    uint32_t attempts = 0;

#if defined(DJT_DEBUG)
    double start = time_milliseconds();
#endif

    while (true) {
        result = pop();

        if (result.first || stopping_criterion(payload))
            break;

        attempts++;

        if (attempts >= NANOTHREAD_MAX_ATTEMPTS) {
            std::unique_lock<std::mutex> guard(sleep_mutex);

            uint64_t value = ++sleep_state, phase = value & high_mask;
            DJT_TRACE("pop_or_sleep(): falling asleep after %.2f milliseconds, "
                      "sleep_state := (%u, %u)!",
                      time_milliseconds() - start, (uint32_t)(value >> 32),
                      (uint32_t) value);

            // Try once more to fetch a job
            result = pop();

            /* If the following is true, somebody added work, or the stopping
               became active while this thread was about to go to sleep. */
            if (result.first || stopping_criterion(payload)) {
                // Reduce sleep_state if we're still in the same phase.
                DJT_TRACE("sleep aborted.");
                while (true) {
                    if (sleep_state.compare_exchange_strong(value, value - 1))
                        break;
                    if ((value & high_mask) != phase)
                        break;
                    pause();
                }
                break;
            }

            /* The push() code above has the structure

                - A1. Enqueue work
                - A2. Check sleep_state, and wake threads if nonzero

               While the code here has the structure

                - B1. Increase sleep_state
                - B2. Try to dequeue work
                - B3. Wait for wakeup signal

               This ordering excludes the possibility that the thread sleeps
               erroneously while work is available or added later on.
            */

            while ((sleep_state & high_mask) == phase)
                sleep_cv.wait(guard);

            value = sleep_state.load();
            DJT_TRACE("pop_or_sleep(): woke up -- sleep_state=(%u, %u)",
                      (uint32_t)(value >> 32), (uint32_t) value);
        }
    }

    return result;
}

