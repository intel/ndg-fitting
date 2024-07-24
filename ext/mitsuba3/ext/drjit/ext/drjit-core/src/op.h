/*
    src/op.h -- Conversion of standard operations into PTX and LLVM IR

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

/// Create a variable representing the result of a standard operation
extern uint32_t jitc_var_new_op(JitOp ot, uint32_t n_dep, const uint32_t *dep);

/// Perform an ordinary or reinterpreting cast of the variable 'index'
extern uint32_t jitc_var_new_cast(uint32_t index, VarType target_type,
                                  int reinterpret);

/// Create a variable that reads from another variable
extern uint32_t jitc_var_new_gather(uint32_t source, uint32_t index,
                                    uint32_t mask);

/// Schedule a scatter opartion that writes to an array
extern uint32_t jitc_var_new_scatter(uint32_t target, uint32_t value,
                                     uint32_t index, uint32_t mask,
                                     ReduceOp reduce_op);

template <typename... Ts>
uint32_t jitc_var_new_op_n(JitOp op, const Ts &... indices_) {
    uint32_t indices[] = { indices_... };
    return jitc_var_new_op(op, (uint32_t) sizeof...(Ts), indices);
}

