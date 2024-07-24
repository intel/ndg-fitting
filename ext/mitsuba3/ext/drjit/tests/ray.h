/*
    tests/ray.h -- showcases how to extend Dr.Jit vectorization to custom
    data types

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/array.h>

using namespace drjit;

/**
 * Generic 3D ray class: can represent either a single ray, a static ray
 * bundle, or a dynamic heap-allocated bundle of rays
 */
template <typename Vector_> struct Ray {
    using Vector = Vector_;
    using Value = value_t<Vector>;

    Vector o;
    Vector d;

    /// Compute a position along a ray
    Vector operator()(const Value &t) const { return o + t*d; }

    DRJIT_STRUCT(Ray, o, d)
};

DRJIT_STRUCT_SUPPORT(Ray, o, d)

