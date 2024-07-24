#include <nanothread/nanothread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

// Task callback function. Will be called with index = 0..999
void my_task(uint32_t index, void *payload) {
    printf("Worker thread %u: work unit %u\n", pool_thread_id(), index);

    // Sleep for a bit
#if defined(_WIN32)
    Sleep(500);
#else
    usleep(500000);
#endif

    // Use payload to communicate some data to the caller
    ((uint32_t *) payload)[index] = index;
}

int main(int argc, char** argv) {
    (void) argc; (void) argv; // Command line arguments unused

    uint32_t temp[10000];

    memset(temp, 0, sizeof(int) * 1000);

    // Create a worker per CPU thread
    Pool *pool = pool_create(100, 0);

    // Synchronous interface: submit a task and wait for it to complete
    task_submit_and_wait(
        pool,
        1000,     // How many work units does this task contain?
        my_task, // Function to be executed
        temp     // Optional payload, will be passed to function
    );

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
