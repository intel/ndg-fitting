/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include <tsl/robin_set.h>

/// A single variable that is scheduled to execute for a launch with 'size' entries
struct ScheduledVariable {
    uint32_t size;
    uint32_t index;

    ScheduledVariable(uint32_t size, uint32_t index)
        : size(size), index(index) { }
};

/// Start and end index of a group of variables that will be merged into the same kernel
struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

/// Hashing helper for GlobalMap
struct XXH128Hash {
    size_t operator()(const XXH128_hash_t &h) const { return h.low64; }
};
struct XXH128Eq {
    size_t operator()(const XXH128_hash_t &h1, const XXH128_hash_t &h2) const {
        return h1.low64 == h2.low64 && h1.high64 == h2.high64;
    }
};

/// Cache data structure for global declarations
using GlobalsMap = tsl::robin_map<XXH128_hash_t, uint32_t, XXH128Hash, XXH128Eq>;

/// Name of the last generated kernel
extern char kernel_name[52];

/// List of global declarations (intrinsics, constant arrays)
extern std::vector<std::string> globals;

/// List of device functions or direct callables (OptiX)
extern std::vector<std::string> callables;

/// Ensure uniqueness of globals/callables arrays
extern GlobalsMap globals_map;

/// Are we recording an OptiX kernel?
extern bool uses_optix;

/// Are we currently compiling a virtual function call
extern bool assemble_func;

/// Size and alignment of auxiliary buffer needed by virtual function calls
extern int32_t alloca_size;
extern int32_t alloca_align;

/// Specifies the nesting level of virtual calls being compiled
extern uint32_t vcall_depth;

/// Ordered list of variables that should be computed
extern std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
extern std::vector<ScheduledGroup> schedule_groups;

/// Evaluate all computation that is queued on the current thread
extern void jitc_eval(ThreadState *ts);

/// Used by jitc_eval() to generate PTX source code
extern void jitc_assemble_cuda(ThreadState *ts, ScheduledGroup group,
                               uint32_t n_regs, uint32_t n_params);

/// Used by jitc_eval() to generate LLVM IR source code
extern void jitc_assemble_llvm(ThreadState *ts, ScheduledGroup group);

/// Used by jitc_vcall() to generate source code for vcalls
extern std::pair<XXH128_hash_t, uint32_t>
jitc_assemble_func(ThreadState *ts, const char *name, uint32_t inst_id,
                   uint32_t in_size, uint32_t in_align, uint32_t out_size,
                   uint32_t out_align, uint32_t data_offset,
                   const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                   uint32_t n_in, const uint32_t *in, uint32_t n_out,
                   const uint32_t *out_nested, uint32_t n_se,
                   const uint32_t *se, bool use_self);

/// Used by jitc_vcall() to generate PTX source code for vcalls
extern void
jitc_assemble_cuda_func(const char *name, uint32_t inst_id, uint32_t n_regs,
                        uint32_t in_size, uint32_t in_align, uint32_t out_size,
                        uint32_t out_align, uint32_t data_offset,
                        const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                        uint32_t n_out, const uint32_t *out_nested,
                        bool use_self);

/// Used by jitc_vcall() to generate LLVM IR source code for vcalls
extern void
jitc_assemble_llvm_func(const char *name, uint32_t inst_id,
                        uint32_t in_size, uint32_t data_offset,
                        const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                        uint32_t n_out, const uint32_t *out_nested,
                        bool use_self);

/// Register a global declaration that will be included in the final program
extern void jitc_register_global(const char *str);
