#include <nanothread/nanothread.h>
#include <cstdlib>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif


int main(int, char**) {
    uint32_t temp[10000];

    memset(temp, 0, sizeof(int) * 1000);

    // Create a worker per CPU thread
    Pool *pool = pool_create(100);

    Task *task = drjit::parallel_for_async(
        drjit::blocked_range<uint32_t>(0, 1000, 1),

        // Task callback function. Will be called with index = 0..999
        [&](drjit::blocked_range<uint32_t> range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                printf("Worker thread %u: work unit %u\n", pool_thread_id(), i);

                // Sleep for a bit
                #if defined(_WIN32)
                    Sleep(500);
                #else
                    usleep(500000);
                #endif

                // Use payload to communicate some data to the caller
                temp[i] = i;
            }
        },

        {}, pool
    );

    // Synchronous interface: submit a task and wait for it to complete
    task_wait(task);

    // .. contents of 'temp' are now ready ..
    for (uint32_t i = 0; i < 1000; ++i) {
        if (temp[i] != i) {
            fprintf(stderr, "Test failed!\n");
            abort();
        }
    }

    // Clean up used resources
    pool_destroy(pool);
}
