#include "test.h"
#include <cstring>

TEST_BOTH(01_gather) {
    Int32 r = arange<Int32>(100) + 100;
    UInt32 index = UInt32(34, 62, 75, 2);
    Int32 ref = Int32(134, 162, 175, 102);
    Int32 value = gather<Int32>(r, index);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(02_gather_mask) {
    Mask r = eq(arange<Int32>(100) & Int32(1), 1);
    UInt32 index = UInt32(33, 62, 75, 2);
    Mask ref = UInt32(1, 0, 1, 0);
    Mask value = gather<Mask>(r, index);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(03_gather_masked) {
    Int32 r = arange<Int32>(100) + 100;
    UInt32 index = UInt32(34, 62, 75, 2);
    Mask mask = index > 50;
    UInt32 ref = UInt32(0, 162, 175, 0);
    UInt32 value = gather<UInt32>(r, index, mask);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(04_gather_mask_masked) {
    Mask r = eq(arange<Int32>(100) & Int32(1), 1);
    UInt32 index = UInt32(33, 62, 75, 2);
    Mask ref = UInt32(0, 0, 1, 0);
    Mask mask = index > 50;
    Mask value = gather<Mask>(r, index, mask);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(05_gather_scalar) {
    /* unmasked, doesn't launch any kernels */ {
        Int32 r = 124;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        UInt32 ref = 124;
        UInt32 value = gather<UInt32>(r, index);
        jit_assert(all(eq(ref, value)));
    }
    /* masked */ {
        Int32 r = 124;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask mask = index > 50;
        UInt32 ref = UInt32(0, 124, 124, 0);
        UInt32 value = gather<UInt32>(r, index, mask);
        jit_assert(all(eq(ref, value)));
    }
}

TEST_BOTH(06_gather_scalar_mask) {
    /* unmasked, doesn't launch any kernels */ {
        Mask r = true;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask ref = true;
        Mask value = gather<Mask>(r, index);
        jit_assert(all(eq(ref, value)));
    }
    /* masked */ {
        Mask r = true;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask mask = index > 50;
        Mask ref = Mask(false, true, true, false);
        Mask value = gather<Mask>(r, index, mask);
        jit_assert(all(eq(ref, value)));
    }
}

TEST_BOTH(07_scatter) {
    UInt32 r = arange<UInt32>(10);
    UInt32 index = UInt32(1, 7, 5);
    UInt32 value = UInt32(8, 2, 3);
    UInt32 ref = UInt32(0, 8, 2, 3, 4, 3, 6, 2, 8, 9);
    scatter(r, value, index);
    jit_assert(all(eq(ref, r)));
}

TEST_BOTH(08_scatter_mask) {
    UInt32 r = arange<UInt32>(10);
    UInt32 index = UInt32(1, 7, 5);
    UInt32 value = UInt32(8, 2, 3);
    Mask mask = Mask(true, false, true);
    UInt32 ref = UInt32(0, 8, 2, 3, 4, 3, 6, 7, 8, 9);
    scatter(r, value, index, mask);
    jit_assert(all(eq(ref, r)));
}

TEST_BOTH(09_safety) {
    /* Collapse adjacent scatters */ {
        Float a = arange<Float>(5);
        a.eval();
        uint32_t index = a.index();
        scatter(a, Float(0), UInt32(0));
        jit_assert(index == a.index());
        scatter(a, Float(1), UInt32(0));
        jit_assert(index == a.index());
        scatter(a, Float(2), UInt32(0));
        jit_assert(index == a.index());
    }

    /* Make safety copies with multiple ext. refs */ {
        Float a = arange<Float>(5), b = a;
        jit_assert(a.index() == b.index());
        a.eval();
        uint32_t index = a.index();
        jit_assert(index == b.index());
        scatter(a, Float(0), UInt32(0));
        jit_assert(index != a.index() && a.index() != b.index());
        index = a.index();
        scatter(a, Float(1), UInt32(0));
        jit_assert(index == a.index() && a.index() != b.index());
    }

    /* Make safety copies in the presence of gathers */ {
        Float a = arange<Float>(5);
        a.eval();
        uint32_t index = a.index();

        Float b = gather(a, UInt32(0));
        jit_assert(index == a.index());

        scatter(a, Float(0), UInt32(0));
        jit_assert(index != a.index());
    }
}

TEST_BOTH(10_scatter_atomic_rmw) {
    /* scatter 16 values */ {
        Float target = zero<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_reduce(ReduceOp::Add, target, Float(1), index);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]") == 0);
    }

    /* scatter 17 values, tests LLVM masking */ {
        Float target = zero<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);

        scatter_reduce(ReduceOp::Add, target, Float(1), index);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0]") == 0);
    }

    /* masked scatter */ {
        Float target = zero<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);
        Mask mask = neq(index, 7);

        scatter_reduce(ReduceOp::Add, target, Float(1), index, mask);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0]") == 0);
    }
}
