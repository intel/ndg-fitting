#include <nanothread/nanothread.h>
#include <stdlib.h>

Task *tetranacci(Pool *pool, uint32_t i, uint32_t *out) {
    if (i < 4) {
        *out = (i == 3) ? 1 : 0;
        return nullptr;
    }

    uint32_t *tmp = new uint32_t[4];

    Task *task[4] = {
        tetranacci(pool, i - 1, tmp),
        tetranacci(pool, i - 2, tmp + 1),
        tetranacci(pool, i - 3, tmp + 2),
        tetranacci(pool, i - 4, tmp + 3)
    };

    Task *rv = drjit::do_async(
        [tmp, out]() {
            *out = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            delete[] tmp;
        }, { task[0], task[1], task[2], task[3] },
        pool
    );

    task_release(task[0]);
    task_release(task[1]);
    task_release(task[2]);
    task_release(task[3]);

    return rv;
}

Task * tetranacci_2(Pool *pool, uint32_t i, uint32_t *out) {
    if (i < 4) {
        *out = (i == 3) ? 1 : 0;
        return nullptr;
    }

    return drjit::do_async(
        [pool, i, out]() {
            uint32_t tmp[4];
            Task *task[4];

            for (int k = 0; k < 4; ++k)
                task[k] = tetranacci_2(pool, i - k - 1, tmp + k);
            for (int k = 0; k < 4; ++k)
                task_wait_and_release(task[k]);

            *out = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }, {}, pool
    );
}

int main(int, char**) {
    // Create a worker per CPU thread
    for (int i = 0; i< 100; ++i) {
        printf("Testing with %i threads..\n", i);
        Pool *pool = pool_create(i);

        uint32_t out = 0;
        Task *task = tetranacci(pool, 16, &out);
        task_wait_and_release(task);
        if (out != 2872)
            abort();

        task = tetranacci_2(pool, 16, &out);
        task_wait_and_release(task);
        if (out != 2872)
            abort();

        // Clean up used resources
        pool_destroy(pool);
    }
}
