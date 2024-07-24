#include "test.h"
#include "traits.h"
#include "ekloop.h"

TEST_BOTH(01_record_loop) {
    // Tests a simple loop evaluated at once, or in parts
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            Float z = 1;

            Loop<Mask> loop("MyLoop", x, y, z);
            while (loop(x < 5)) {
                y += Float(x);
                x += 1;
                z += 1;
            }

            if (j == 0) {
                jit_var_schedule(z.index());
                jit_var_schedule(y.index());
                jit_var_schedule(x.index());
            }

            bool match_1 = strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0;
            bool match_2 = strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0;
            bool match_3 = strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0;

            if (!(match_1 && match_2 && match_3))
                fprintf(stderr, "Mismatch: %s\n%s\n%s\n", x.str(), y.str(), z.str());

            jit_assert(match_1);
            jit_assert(match_2);
            jit_assert(match_3);
        }
    }
}

TEST_BOTH(02_side_effect) {
    // Tests that side effects happen (and only once, even if the loop is re-evaluated)
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_BOTH(03_side_effect_2) {
    // Tests that side effects work even if they don't reference any loop variables
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(2), UInt32(2));
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_BOTH(04_side_effect_masking) {
    // Tests that side effects are correctly masked by the loop condition
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(1000000);
            UInt32 target = zero<UInt32>(10);

            Loop<Mask> loop("MyLoop", x);
            while (loop(x < 9)) {
                // This is sure to segfault if not masked correctly
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                x += 1;
            }

            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]") == 0);
        }
    }
}

TEST_BOTH(05_optimize_invariant) {
    /* Test to check that variables which stay unchanged or constant and
       equal-valued are optimized out of the loop */
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        UInt32 j = 0,
               v1 = 123,
               v1_orig = v1,
               v2 = opaque<UInt32>(123),
               v2_orig = v2,
               v3 = 124, v3_orig = v3,
               v4 = 125, v4_orig = v4,
               v5 = 1,   v5_orig = v5,
               v6 = 0,   v6_orig = v6;

        Loop<Mask> loop("MyLoop", j, v1, v2, v3, v4, v5, v6);
        int count = 0;
        while (loop(j < 10)) {
            j += 1;
            (void) v1; // v1 stays unchanged
            (void) v2; // v2 stays unchanged
            v3 = 124;  // v3 is overwritten with same value
            v4 = 100;  // v4 is overwritten with different value
            (void) v5; // v5 stays unchanged
            v6 += v5;  // v6 is modified by a loop-invariant variable
            ++count;
        }

        if (i == 0)
            jit_assert(count == 10);
        else if (i == 1)
            jit_assert(count == 1);
        else if (i == 2)
            jit_assert(count == 2);

        if (i == 2) {
            jit_assert( jit_var_is_literal(v1.index()) && v1.index() == v1_orig.index());
            jit_assert(!jit_var_is_literal(v2.index()) && v2.index() == v2_orig.index());
            jit_assert( jit_var_is_literal(v3.index()) && v3.index() == v3_orig.index());
            jit_assert(!jit_var_is_literal(v4.index()) && v4.index() != v4_orig.index());
            jit_assert( jit_var_is_literal(v5.index()) && v5.index() == v5_orig.index());
            jit_assert(!jit_var_is_literal(v6.index()) && v6.index() != v6_orig.index());
        }

        jit_var_schedule(v1.index());
        jit_var_schedule(v2.index());
        jit_var_schedule(v3.index());
        jit_var_schedule(v4.index());
        jit_var_schedule(v5.index());
        jit_var_schedule(v6.index());

        jit_assert(v1 == 123 && v2 == 123 && v3 == 124 && v4 == 100 && v5 == 1 && v6 == 10);
    }
}

TEST_BOTH(06_garbage_collection) {
    // Checks that unused loop variables are optimized away

    jit_set_flag(JitFlag::LoopRecord, 1);
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);

        UInt32 j = 0;
        UInt32 v1 = opaque<UInt32>(1);
        UInt32 v2 = opaque<UInt32>(2);
        UInt32 v3 = opaque<UInt32>(3);
        UInt32 v4 = opaque<UInt32>(4);
        uint32_t v1i = v1.index(), v2i = v2.index(),
                 v3i = v3.index(), v4i = v4.index();
        // jit_var_set_label(v1i, "v1");
        // jit_var_set_label(v2i, "v2");
        // jit_var_set_label(v3i, "v3");
        // jit_var_set_label(v4i, "v4");

        Loop<Mask> loop("MyLoop", j, v1, v2, v3, v4);
        while (loop(j < 4)) {
            UInt32 tmp = v4;
            v4 = v1;
            v1 = v2;
            v2 = v3;
            v3 = tmp;
            j += 1;
        }

        v1 = UInt32();
        v2 = UInt32();
        v3 = UInt32();

        /// evaluate some unrelated calculation to trigger loop simplification
        {
            UInt32 tmp = UInt32(1, 2) + 1;
            tmp.str();
        }


        jit_assert(jit_var_exists(v1i) && jit_var_exists(v2i) && jit_var_exists(v3i) && jit_var_exists(v4i));

        v4 = UInt32();
        {
            UInt32 tmp = UInt32(1, 2) + 1;
            tmp.str();
        }
        if (i == 0)
            jit_assert(jit_var_exists(v1i) && jit_var_exists(v2i) && jit_var_exists(v3i) && jit_var_exists(v4i));
        else
            jit_assert(!jit_var_exists(v1i) && !jit_var_exists(v2i) && !jit_var_exists(v3i) && !jit_var_exists(v4i));

        j = UInt32();
        jit_assert(!jit_var_exists(v1i) && !jit_var_exists(v2i) && !jit_var_exists(v3i) && !jit_var_exists(v4i));
    }
}

TEST_BOTH(07_collatz) {
    // A more interesting nested loop
    auto collatz = [](const char *name, UInt32 value) -> UInt32 {
        UInt32 counter = 0;
        // jit_var_set_label(value.index(), "value");
        // jit_var_set_label(counter.index(), "counter");

        Loop<Mask> loop(name, value, counter);
        while (loop(neq(value, 1))) {
            Mask is_even = eq(value & UInt32(1), 0);
            value = select(is_even, value / 2, value*3 + 1);
            counter += 1;
        }

        return counter;
    };

    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);
        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 buf = full<UInt32>(1000, 11);
            if (j == 1) {
                UInt32 k = 1;
                Loop<Mask> loop_1("Outer", k);
                while (loop_1(k <= 10)) {
                    scatter(buf, collatz("Inner", k), k - 1);
                    k += 1;
                }
            } else {
                for (uint32_t k = 1; k <= 10; ++k) {
                    char tmpname[20];
                    snprintf(tmpname, sizeof(tmpname), "Inner [%u]", k);
                    scatter(buf, collatz(tmpname, k), UInt32(k - 1));
                }
            }
            jit_assert(strcmp(buf.str(), "[0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000]") == 0);
        }
    }
}

TEST_BOTH(08_nested_write) {
    // Nested loop where both loops write to the same loop variable
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        UInt32 k = arange<UInt32>(10)*12;
        Loop<Mask> loop_1("Outer", k);

        while (loop_1(neq(k % 7, 0))) {
            Loop<Mask> loop_2("Inner", k);

            while (loop_2(neq(k % 3, 0))) {
                k += 1;
            }
            k += 1;
        }
        jit_assert(strcmp(k.str(), "[0, 28, 28, 49, 49, 70, 91, 84, 112, 112]") == 0);
    }
}

TEST_BOTH(09_optim_cond) {
    // Loop condition depends on variables that are optimized away (loop-invariants)

    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        UInt32 k = arange<UInt32>(3), l = 10;
        Loop<Mask> loop("Outer", k, l);

        while (loop(k + l < 30)) {
            k += 1;
        }

        jit_assert(strcmp(k.str(), "[20, 20, 20]") == 0);
        if (i == 2)
            jit_assert(jit_var_is_literal(l.index()));
    }
}
