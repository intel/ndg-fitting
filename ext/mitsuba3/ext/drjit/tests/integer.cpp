/*
    tests/basic.cpp -- tests basic operators involving different types

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

DRJIT_TEST_INT(test01_or) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a | b; },
        [](Value a, Value b) -> Value { return a | b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a || b; },
        [](Value a, Value b) -> Value { return a | b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x |= b; return x; },
        [](Value a, Value b) -> Value { return a | b; }
    );
}

DRJIT_TEST_INT(test02_and) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a & b; },
        [](Value a, Value b) -> Value { return a & b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a && b; },
        [](Value a, Value b) -> Value { return a & b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x &= b; return x; },
        [](Value a, Value b) -> Value { return a & b; }
    );
}

DRJIT_TEST_INT(test03_xor) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a ^ b; },
        [](Value a, Value b) -> Value { return a ^ b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x ^= b; return x; },
        [](Value a, Value b) -> Value { return a ^ b; }
    );
}

DRJIT_TEST_INT(test04_not) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return ~a; },
        [](Value a) -> Value { return ~a; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return !a; },
        [](Value a) -> Value { return ~a; }
    );
}

DRJIT_TEST_INT(test05_sign) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return drjit::sign(a); },
        [](Value a) -> Value { return a >= 0 ? Value(1) : Value(-1); }
    );
}

DRJIT_TEST_TYPE(test06_shift, uint32_t) {
    assert((T(0xDEADBEEFu) >> 4u) == T(0x0DEADBEEu));
    assert((T(0xDEADBEEFu) << 4u) == T(0xEADBEEF0u));
    assert((sr<4>(T(0xDEADBEEFu)) == T(0x0DEADBEEu)));
    assert((sl<4>(T(0xDEADBEEFu)) == T(0xEADBEEF0u)));
}

DRJIT_TEST_TYPE(test06_shift, int32_t) {
    assert((T((int32_t) 0xDEADBEEF) >> 4) == T((int32_t) 0xFDEADBEE));
    assert((T((int32_t) 0xDEADBEEF) << 4) == T((int32_t) 0xEADBEEF0));
    assert((sr<4>(T((int32_t) 0xDEADBEEF)) == T((int32_t) 0xFDEADBEE)));
    assert((sl<4>(T((int32_t) 0xDEADBEEF)) == T((int32_t) 0xEADBEEF0)));
}

DRJIT_TEST_TYPE(test06_shift, uint64_t) {
    assert((T(0xCAFEBABEDEADBEEFull) >> 4u) == T(0x0CAFEBABEDEADBEEull));
    assert((T(0xCAFEBABEDEADBEEFull) << 4u) == T(0xAFEBABEDEADBEEF0ull));
    assert((sr<4>(T(0xCAFEBABEDEADBEEFull)) == T(0x0CAFEBABEDEADBEEull)));
    assert((sl<4>(T(0xCAFEBABEDEADBEEFull)) == T(0xAFEBABEDEADBEEF0ull)));
}

DRJIT_TEST_TYPE(test06_shift, int64_t) {
    assert((T((int64_t) 0xDEADBEEFCAFEBABEll) >> 4) == T((int64_t) 0xFDEADBEEFCAFEBABll));
    assert((T((int64_t) 0xDEADBEEFCAFEBABEll) << 4) == T((int64_t) 0xEADBEEFCAFEBABE0ll));
    assert((sr<4>(T((int64_t) 0xDEADBEEFCAFEBABEll)) == T((int64_t) 0xFDEADBEEFCAFEBABll)));
    assert((sl<4>(T((int64_t) 0xDEADBEEFCAFEBABEll)) == T((int64_t) 0xEADBEEFCAFEBABE0ll)));
}

DRJIT_TEST_INT(test07_bmi) {
    if (std::is_signed<Value>::value)
        return;
    Value lzcnt_ref[] = { 32, 31, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28, 28 };
    Value tzcnt_ref[] = { 32, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2 };
    Value popcnt_ref[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2 };

    for (int i = 0; i < 13; ++i) {
        Value v = (Value) i;
        assert(lzcnt(T(v))[0] == lzcnt_ref[i] + (sizeof(Value) == 8 ? 32 : 0));
        assert(lzcnt(v) == lzcnt_ref[i] + (sizeof(Value) == 8 ? 32 : 0));
        assert(tzcnt(T(v))[0] == tzcnt_ref[i] + ((sizeof(Value) == 8 && i == 0) ? 32 : 0));
        assert(tzcnt(v) == tzcnt_ref[i] + ((sizeof(Value) == 8 && i == 0) ? 32 : 0));
        assert(popcnt(T(v))[0] == popcnt_ref[i]);
        assert(popcnt(v) == popcnt_ref[i]);
    }

    Value v = (Value) -1;
    assert(lzcnt(T(v))[0] == 0);
    assert(lzcnt(v) == 0);
    assert(tzcnt(T(v))[0] == 0);
    assert(tzcnt(v) == 0);
    assert(popcnt(T(v))[0] == (sizeof(Value) == 8 ? 64 : 32));
    assert(popcnt(v) == (sizeof(Value) == 8 ? 64 : 32));
}
