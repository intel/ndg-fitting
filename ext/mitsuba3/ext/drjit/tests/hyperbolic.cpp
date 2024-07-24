#include "test.h"

DRJIT_TEST_FLOAT(test01_sinh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sinh(a); },
        [](double a) { return std::sinh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test02_cosh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return cosh(a); },
        [](double a) { return std::cosh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test03_sincosh_sin) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincosh(a).first; },
        [](double a) { return std::sinh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test04_sincosh_cos) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sincosh(a).second; },
        [](double a) { return std::cosh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test05_tanh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return tanh(a); },
        [](double a) { return std::tanh(a); },
        Value(-10), Value(10),
        7
    );
}

DRJIT_TEST_FLOAT(test06_csch) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return csch(a); },
        [](double a) { return 1/std::sinh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test07_sech) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return sech(a); },
        [](double a) { return 1/std::cosh(a); },
        Value(-10), Value(10),
        9
    );
}

DRJIT_TEST_FLOAT(test08_coth) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return coth(a); },
        [](double a) { return 1/std::tanh(a); },
        Value(-10), Value(10),
        8
    );
}

DRJIT_TEST_FLOAT(test09_asinh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return asinh(a); },
        [](double a) { return std::asinh(a); },
        Value(-30), Value(30),
        3
    );
}

DRJIT_TEST_FLOAT(test11_acosh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return acosh(a); },
        [](double a) { return std::acosh(a); },
        Value(1), Value(10),
        5
    );
}

DRJIT_TEST_FLOAT(test12_atanh) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return atanh(a); },
        [](double a) { return std::atanh(a); },
        Value(-1 + 0.001), Value(1 - 0.001),
        3
    );
}

