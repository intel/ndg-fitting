/*
    src/vcall.cpp -- Code generation for virtual function calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "registry.h"
#include "util.h"
#include "op.h"
#include "profiler.h"
#include "vcall.h"

/// Encodes information about a virtual function call
struct VCall {
    JitBackend backend;

    /// A descriptive name
    char *name = nullptr;

    /// ID of call variable
    uint32_t id = 0;

    /// Number of instances
    uint32_t n_inst = 0;

    /// Mapping from instance ID -> vcall branch
    std::vector<uint32_t> inst_id;

    /// Input variables at call site
    std::vector<uint32_t> in;
    /// Input placeholder variables
    std::vector<uint32_t> in_nested;

    /// Output variables at call site
    std::vector<uint32_t> out;
    /// Output variables *per instance*
    std::vector<uint32_t> out_nested;

    /// Per-instance offsets into side effects list
    std::vector<uint32_t> checkpoints;
    /// Compressed side effect index list
    std::vector<uint32_t> side_effects;

    /// Mapping from variable index to offset into call data
    tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> data_map;
    std::vector<uint32_t> data_offset;

    uint64_t *offset_d = nullptr, *offset_h = nullptr;
    size_t offset_d_size = 0;

    /// Storage in bytes for inputs/outputs before simplifications
    uint32_t in_count_initial = 0;
    uint32_t in_size_initial = 0;
    uint32_t out_size_initial = 0;

    /// Does this vcall need self as argument
    bool use_self = false;

    ~VCall() {
        for (uint32_t index : out_nested)
            jitc_var_dec_ref_ext(index);
        clear_side_effects();
        free(name);
    }

    void clear_side_effects() {
        if (checkpoints.empty() || checkpoints.back() == checkpoints.front())
            return;
        for (uint32_t index : side_effects)
            jitc_var_dec_ref_ext(index);
        side_effects.clear();
        std::fill(checkpoints.begin(), checkpoints.end(), 0);
    }
};

// Forward declarations
static void jitc_var_vcall_assemble(VCall *vcall, uint32_t self_reg,
                                    uint32_t mask_reg, uint32_t offset_reg,
                                    uint32_t data_reg);

static void jitc_var_vcall_assemble_cuda(
    VCall *vcall, const std::vector<XXH128_hash_t> &callable_hash,
    uint32_t vcall_reg, uint32_t self_reg, uint32_t mask_reg,
    uint32_t offset_reg, uint32_t data_reg, uint32_t n_out, uint32_t in_size,
    uint32_t in_align, uint32_t out_size, uint32_t out_align);

static void jitc_var_vcall_assemble_llvm(
    VCall *vcall, const std::vector<XXH128_hash_t> &callable_hash,
    uint32_t vcall_reg, uint32_t self_reg, uint32_t mask_reg,
    uint32_t offset_reg, uint32_t data_reg, uint32_t n_out, uint32_t in_size,
    uint32_t in_align, uint32_t out_size, uint32_t out_align);

static void jitc_var_vcall_collect_data(
    tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
    uint32_t &data_offset, uint32_t inst_id, uint32_t index, bool &use_self,
    bool &optix);

void jitc_vcall_set_self(JitBackend backend, uint32_t value, uint32_t index) {
    ThreadState *ts = thread_state(backend);

    if (ts->vcall_self_index) {
        jitc_var_dec_ref_ext(ts->vcall_self_index);
        ts->vcall_self_index = 0;
    }

    ts->vcall_self_value = value;

    if (value) {
        if (index) {
            jitc_var_inc_ref_ext(index);
            ts->vcall_self_index = index;
        } else {
            Variable v;
            if (backend == JitBackend::CUDA)
                v.stmt = (char *) "mov.u32 $r0, self";
            else
                v.stmt = (char *) "$r0 = bitcast <$w x i32> %self to <$w x i32>";
            v.size = 1u;
            v.type = (uint32_t) VarType::UInt32;
            v.backend = (uint32_t) backend;
            v.placeholder = true;
            ts->vcall_self_index = jitc_var_new(v, true);
        }
    }
}

void jitc_vcall_self(JitBackend backend, uint32_t *value, uint32_t *index) {
    ThreadState *ts = thread_state(backend);
    *value = ts->vcall_self_value;
    *index = ts->vcall_self_index;
}

/// Weave a virtual function call into the computation graph
uint32_t jitc_var_vcall(const char *name, uint32_t self, uint32_t mask_,
                        uint32_t n_inst, const uint32_t *inst_id, uint32_t n_in,
                        const uint32_t *in, uint32_t n_out_nested,
                        const uint32_t *out_nested, const uint32_t *checkpoints,
                        uint32_t *out) {

    const uint32_t checkpoint_mask = 0x7fffffff;

#if defined(DRJIT_ENABLE_NVTX) || defined(DRJIT_ENABLE_ITTNOTIFY)
    std::string profile_name = std::string("jit_var_vcall: ") + name;
    ProfilerRegion profiler_region(profile_name.c_str());
    ProfilerPhase profiler(profiler_region);
#endif

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_vcall(): must have at least one instance!");

    if (n_out_nested % n_inst != 0)
        jitc_raise("jit_var_vcall(): list of all output indices must be a "
                   "multiple of the instance count!");

    uint32_t n_out = n_out_nested / n_inst, size = 0,
             in_size_initial = 0, out_size_initial = 0;

    bool placeholder = false, optix = false, dirty = false;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        placeholder |= (bool) self_v->placeholder;
        dirty |= (bool) self_v->ref_count_se;
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_vcall(): 'self' argument must be of type "
                       "UInt32 (was: %s)", type_name[self_v->type]);
    }

    size = std::max(size, jitc_var(mask_)->size);

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->vcall_iface && !v->literal) {
            if (!v->dep[0])
                jitc_raise("jit_var_vcall(): placeholder variable r%u does not "
                           "reference another input!", in[i]);
            Variable *v2 = jitc_var(v->dep[0]);
            placeholder |= (bool) v2->placeholder;
            optix |= (bool) v2->optix;
            dirty |= (bool) v2->ref_count_se;
            size = std::max(size, v2->size);
        } else if (!v->literal) {
            jitc_raise("jit_var_vcall(): input variable r%u must either be a "
                       "literal or placeholder wrapping another variable!", in[i]);
        }
        if (v->size != 1)
            jitc_raise("jit_var_vcall(): size of input variable r%u must be 1!", in[i]);
        in_size_initial += type_size[v->type];
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v  = jitc_var(out_nested[i]),
                       *v0 = jitc_var(out_nested[i % n_out]);
        size = std::max(size, v->size);
        if (v->type != v0->type)
            jitc_raise(
                "jit_var_vcall(): output types don't match between instances!");

        if (i < n_out)
            out_size_initial += type_size[v->type];
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v = jitc_var(out_nested[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    ThreadState *ts = thread_state(backend);
    if (ts->side_effects_recorded.size() !=
        (checkpoints[n_inst] & checkpoint_mask))
        jitc_raise("jitc_var_vcall(): side effect queue doesn't have the "
                   "expected size!");

    if (dirty) {
        jitc_eval(ts);

        dirty = jitc_var(self)->ref_count_se;
        for (uint32_t i = 0; i < n_in; ++i) {
            const Variable *v = jitc_var(in[i]);
            if (v->vcall_iface && !v->literal)
                dirty |= (bool) jitc_var(v->dep[0])->ref_count_se;
        }

        if (unlikely(dirty))
            jitc_raise("jit_var_vcall(): inputs remain dirty after evaluation!");
    }

    // =====================================================
    // 3. Apply any masks on the stack, ignore NULL args
    // =====================================================

    Ref mask;
    {
        uint32_t zero = 0;
        Ref null_instance = steal(jitc_var_new_literal(backend, VarType::UInt32, &zero, 1, 0)),
            is_non_null   = steal(jitc_var_new_op_n(JitOp::Neq, self, null_instance)),
            mask_2        = steal(jitc_var_new_op_n(JitOp::And, mask_, is_non_null));
        mask = steal(jitc_var_mask_apply(mask_2, size));
    }

    // =====================================================
    // 3. Stash information about inputs and outputs
    // =====================================================

    std::unique_ptr<VCall> vcall(new VCall());
    vcall->backend = backend;
    vcall->name = strdup(name);
    vcall->n_inst = n_inst;
    vcall->inst_id = std::vector<uint32_t>(inst_id, inst_id + n_inst);
    vcall->in.reserve(n_in);
    vcall->in_nested.reserve(n_in);
    vcall->out.reserve(n_out);
    vcall->out_nested.reserve(n_out_nested);
    vcall->in_size_initial = in_size_initial;
    vcall->out_size_initial = out_size_initial;
    vcall->in_count_initial = n_in;
    vcall->checkpoints.reserve(n_inst + 1);
    vcall->side_effects = std::vector<uint32_t>(
        ts->side_effects_recorded.begin() + (checkpoints[0] & checkpoint_mask),
        ts->side_effects_recorded.begin() + (checkpoints[n_inst] & checkpoint_mask));
    ts->side_effects_recorded.resize(checkpoints[0] & checkpoint_mask);

    // =====================================================
    // 4. Collect evaluated data accessed by the instances
    // =====================================================

    vcall->data_offset.reserve(n_inst);

    // Collect accesses to evaluated variables/pointers
    uint32_t data_size = 0, inst_id_max = 0;
    for (uint32_t i = 0; i < n_inst; ++i) {
        if (unlikely(checkpoints[i] > checkpoints[i + 1]))
            jitc_raise("jitc_var_vcall(): checkpoints parameter is not "
                       "monotonically increasing!");

        uint32_t id = inst_id[i];
        vcall->data_offset.push_back(data_size);

        for (uint32_t j = 0; j < n_out; ++j)
            jitc_var_vcall_collect_data(vcall->data_map, data_size, i,
                                        out_nested[j + i * n_out],
                                        vcall->use_self, optix);

        for (uint32_t j = checkpoints[i]; j != checkpoints[i + 1]; ++j)
            jitc_var_vcall_collect_data(vcall->data_map, data_size, i,
                                        vcall->side_effects[j - checkpoints[0]],
                                        vcall->use_self, optix);

        // Restore to full alignment
        data_size = (data_size + 7) / 8 * 8;
        inst_id_max = std::max(id, inst_id_max);
    }

    // Allocate memory + wrapper variables for call offset and data arrays
    vcall->offset_d_size = (inst_id_max + 1) * sizeof(uint64_t);

    AllocType at =
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync;
    vcall->offset_d = (uint64_t *) jitc_malloc(at, vcall->offset_d_size);
    uint8_t *data_d = (uint8_t *) jitc_malloc(at, data_size);

    Ref data_buf, data_v,
        offset_buf = steal(jitc_var_mem_map(
            backend, VarType::UInt64, vcall->offset_d, inst_id_max + 1, 1)),
        offset_v =
            steal(jitc_var_new_pointer(backend, vcall->offset_d, offset_buf, 0));

    char temp[128];
    snprintf(temp, sizeof(temp), "VCall: %s [call offsets]", name);
    jitc_var_set_label(offset_buf, temp);

    if (data_size) {
        data_buf = steal(
            jitc_var_mem_map(backend, VarType::UInt8, data_d, data_size, 1));
        snprintf(temp, sizeof(temp), "VCall: %s [call data]", name);
        jitc_var_set_label(data_buf, temp);

        data_v = steal(jitc_var_new_pointer(backend, data_d, data_buf, 0));

        VCallDataRecord *rec = (VCallDataRecord *)
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::HostPinned
                                                    : AllocType::Host,
                        sizeof(VCallDataRecord) * vcall->data_map.size());

        VCallDataRecord *p = rec;

        for (auto kv : vcall->data_map) {
            uint32_t index = (uint32_t) kv.first, offset = kv.second;
            if (offset == (uint32_t) -1)
                continue;

            const Variable *v = jitc_var(index);
            p->offset = offset;
            p->literal = v->literal;
            p->size = (uint8_t) type_size[v->type];
            p->value = v->literal ? (uintptr_t) v->value : (uintptr_t) v->data;
            p++;
        }

        std::sort(rec, p,
                  [](const VCallDataRecord &a, const VCallDataRecord &b) {
                      return a.offset < b.offset;
                  });

        jitc_vcall_prepare(backend, data_d, rec, (uint32_t)(p - rec));
    } else {
        vcall->data_map.clear();
    }

#if 0
    uint32_t range_check = 0;
    {
        uint32_t one = 1;
        Ref range_min = steal(jitc_var_new_literal(backend, VarType::UInt32, &one, 1, 0));
        Ref range_max = steal(jitc_var_new_literal(backend, VarType::UInt32, &n_inst, 1, 0));
        Ref mask1 = steal(jitc_var_new_op_n(JitOp::Lt, self, range_min));
        Ref mask2 = steal(jitc_var_new_op_n(JitOp::Gt, self, range_max));
        Ref mask3 = steal(jitc_var_new_op_n(JitOp::Or, mask1, mask2));
        Ref mask4 = steal(jitc_var_new_op_n(JitOp::And, mask3, mask));
        char fmt[256];
        snprintf(fmt, sizeof(fmt), "Device assertion failure: VCall (\"%s\") instance ID out of range [1..%u], got %%u\n",
                 name, n_inst);
        range_check = jitc_var_printf(backend, mask4, fmt, 1, &self);
    }
#endif

    // =====================================================
    // 5. Create special variable encoding the function call
    // =====================================================

    uint32_t deps_special[4] = { self, mask, offset_v, data_v };
    Ref vcall_v = steal(jitc_var_new_stmt(backend, VarType::Void, "", 0,
                                          data_size ? 4 : 3, deps_special));

    vcall->id = vcall_v;

    {
        Variable *v = jitc_var(vcall_v);
        v->placeholder = placeholder;
        v->size = size;
        v->optix = optix;
    }

    uint32_t n_devirt = 0, flags = jitc_flags();

    bool vcall_optimize = flags & (uint32_t) JitFlag::VCallOptimize,
         vcall_inline   = flags & (uint32_t) JitFlag::VCallInline;

    std::vector<bool> uniform(n_out, true);
    for (uint32_t i = 0; i < n_inst; ++i) {
        for (uint32_t j = 0; j < n_out; ++j) {
            uint32_t index = out_nested[i * n_out + j];
            uniform[j] = uniform[j] && index == out_nested[j];

            /* Hold a reference to the nested computation until the cleanup
               callback later below is invoked. */
            jitc_var_inc_ref_ext(index);
            vcall->out_nested.push_back(index);
        }

        vcall->checkpoints.push_back(checkpoints[i] - checkpoints[0]);
    }
    vcall->checkpoints.push_back(checkpoints[n_inst] - checkpoints[0]);


    if (vcall_optimize) {
        for (uint32_t j = 0; j < n_out; ++j) {
            if (!uniform[j])
                continue;

            /* Should this output value be de-virtualized? We want to avoid
               completely removing the virtual function call.. */
            if (n_inst == 1 && !vcall_inline && !jitc_var(out_nested[j])->literal)
                continue;

            uint32_t dep[2] = { out_nested[j], mask };
            Ref result_v = steal(jitc_var_new_op(JitOp::And, 2, dep));
            Variable *v = jitc_var(result_v);

            if ((bool) v->placeholder != placeholder || v->size != size ||
                (bool) v->optix != optix) {
                if (v->ref_count_ext != 1 || v->ref_count_int != 0) {
                    result_v = steal(jitc_var_copy(result_v));
                    v = jitc_var(result_v);
                }
                jitc_cse_drop(result_v, v);
                v->placeholder = placeholder;
                v->optix = optix;
                v->size = size;
                jitc_cse_put(result_v, v);
            }

            out[j] = result_v.release();
            n_devirt++;

            for (uint32_t i = 0; i < n_inst; ++i) {
                uint32_t &index_2 = vcall->out_nested[i * n_out + j];
                jitc_var_dec_ref_ext(index_2);
                index_2 = 0;
            }
        }
    }

    uint32_t se_count = checkpoints[n_inst] - checkpoints[0];

    jitc_log(InfoSym,
             "jit_var_vcall(r%u, self=r%u): call (\"%s\") with %u instance%s, %u "
             "input%s, %u output%s (%u devirtualized), %u side effect%s, %u "
             "byte%s of call data, %u elements%s%s", (uint32_t) vcall_v, self, name, n_inst,
             n_inst == 1 ? "" : "s", n_in, n_in == 1 ? "" : "s", n_out,
             n_out == 1 ? "" : "s", n_devirt, se_count, se_count == 1 ? "" : "s",
             data_size, data_size == 1 ? "" : "s", size,
             (n_devirt == n_out && se_count == 0) ? " (optimized away)" : "",
             placeholder ? " (part of a recorded computation)" : "");

    // =====================================================
    // 6. Create output variables
    // =====================================================

    auto var_callback = [](uint32_t index, int free, void *ptr) {
        if (!ptr)
            return;

        VCall *vcall_2 = (VCall *) ptr;
        if (!free) {
            // Disable callback
            state.extra[index].callback_data = nullptr;
        }

        // An output variable is no longer referenced. Find out which one.
        uint32_t n_out_2 = (uint32_t) vcall_2->out.size(),
                 offset  = (uint32_t) -1;

        for (uint32_t i = 0; i < n_out_2; ++i) {
            if (vcall_2->out[i] == index)
                offset = i;
        }

        if (offset == (uint32_t) -1)
            jitc_fail("jit_var_vcall(): expired output variable %u could not "
                      "be located!", index);

        // Inform recursive computation graphs via reference counting
        for (uint32_t j = 0; j < vcall_2->n_inst; ++j) {
            uint32_t &index_2 = vcall_2->out_nested[n_out_2 * j + offset];
            jitc_var_dec_ref_ext(index_2);
            index_2 = 0;
        }
        vcall_2->out[offset] = 0;

        // Check if any input parameters became irrelevant
        for (uint32_t i = 0; i < vcall_2->in.size(); ++i) {
            uint32_t index_2 = vcall_2->in_nested[i];
            if (index_2 &&
                state.variables.find(index_2) == state.variables.end()) {
                Extra *e = &state.extra[vcall_2->id];
                if (unlikely(e->dep[i] != vcall_2->in[i]))
                    jitc_fail("jit_var_vcall(): internal error! (1)");
                jitc_var_dec_ref_int(vcall_2->in[i]);
                if (state.extra.find(vcall_2->id) == state.extra.end())
                    jitc_fail("jit_var_vcall(): internal error! (2)");
                e = &state.extra[vcall_2->id]; // may have changed
                e->dep[i] = 0;
                vcall_2->in[i] = 0;
                vcall_2->in_nested[i] = 0;
            }
        }
    };

    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i];
        if (!index) {
            vcall->out.push_back(0);
            continue;
        }

        const Variable *v = jitc_var(index);
        Variable v2;
        v2.stmt = (char *) "";
        v2.size = size;
        v2.placeholder = placeholder;
        v2.optix = optix;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.dep[0] = vcall_v;
        v2.extra = 1;
        jitc_var_inc_ref_int(vcall_v);
        uint32_t index_2 = jitc_var_new(v2, true);
        Extra &extra = state.extra[index_2];

        if (vcall_optimize) {
            extra.callback = var_callback;
            extra.callback_data = vcall.get();
            extra.callback_internal = true;
        }


        vcall->out.push_back(index_2);
        snprintf(temp, sizeof(temp), "VCall: %s [out %u]", name, i);
        jitc_var_set_label(index_2, temp);
        out[i] = index_2;
    }

    // =====================================================
    // 7. Optimize calling conventions by reordering args
    // =====================================================

    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        if (!v->vcall_iface || v->literal)
            continue;

        // Ignore unreferenced inputs
        if (vcall_optimize && v->ref_count_int == 0) {
            auto& on = vcall->out_nested;
            auto it  = std::find(on.begin(), on.end(), index);
            // Only skip if this variable isn't also an output
            if (it == on.end())
                continue;
        }

        vcall->in_nested.push_back(index);

        uint32_t &index_2 = v->dep[0];
        vcall->in.push_back(index_2);
        jitc_var_inc_ref_int(index_2);
    }

    auto comp = [](uint32_t i0, uint32_t i1) {
        int s0 = i0 ? type_size[jitc_var(i0)->type] : 0;
        int s1 = i1 ? type_size[jitc_var(i1)->type] : 0;
        return s0 > s1;
    };

    std::sort(vcall->in.begin(), vcall->in.end(), comp);
    std::sort(vcall->in_nested.begin(), vcall->in_nested.end(), comp);

    std::sort(vcall->out.begin(), vcall->out.end(), comp);
    for (uint32_t i = 0; i < n_inst; ++i)
        std::sort(vcall->out_nested.begin() + (i + 0) * n_out,
                  vcall->out_nested.begin() + (i + 1) * n_out, comp);

    // =====================================================
    // 8. Install code generation and deallocation callbacks
    // =====================================================

    size_t dep_size = vcall->in.size() * sizeof(uint32_t);

    {
        Variable *v_special = jitc_var(vcall_v);
        v_special->extra = 1;
        v_special->size = size;
    }

    Extra *e_special = &state.extra[vcall_v];
    e_special->n_dep = (uint32_t) vcall->in.size();
    e_special->dep = (uint32_t *) malloc_check(dep_size);

    // Steal input dependencies from placeholder arguments
    memcpy(e_special->dep, vcall->in.data(), dep_size);

    e_special->callback = [](uint32_t, int free, void *ptr) {
        if (free)
            delete (VCall *) ptr;
    };
    e_special->callback_internal = true;
    e_special->callback_data = vcall.release();

    e_special->assemble = [](const Variable *v, const Extra &extra) {
        const Variable *self_2 = jitc_var(v->dep[0]),
                       *valid_2 = jitc_var(v->dep[1]);
        uint32_t self_reg = self_2->reg_index,
                 mask_reg = valid_2->reg_index,
                 offset_reg = 0, data_reg = 0;

        offset_reg = jitc_var(v->dep[2])->reg_index;
        if (v->dep[3])
            data_reg = jitc_var(v->dep[3])->reg_index;

        jitc_var_vcall_assemble((VCall *) extra.callback_data, self_reg,
                                mask_reg, offset_reg, data_reg);
    };

    snprintf(temp, sizeof(temp), "VCall: %s", name);
    jitc_var_set_label(vcall_v, temp);

    Ref se_v;
    if (se_count) {
        /* The call has side effects. Create a dummy variable to ensure that
         * they are evaluated */
        uint32_t vcall_id = vcall_v;
        se_v = steal(
            jitc_var_new_stmt(backend, VarType::Void, "", 1, 1, &vcall_id));
        snprintf(temp, sizeof(temp), "VCall: %s [side effects]", name);
        jitc_var(se_v)->placeholder = placeholder;
        jitc_var_set_label(se_v, temp);
    }

    vcall_v.reset();

    return se_v.release();
}

static ProfilerRegion profiler_region_vcall_assemble("jit_var_vcall_assemble");

/// Called by the JIT compiler when compiling a virtual function call
static void jitc_var_vcall_assemble(VCall *vcall,
                                    uint32_t self_reg,
                                    uint32_t mask_reg,
                                    uint32_t offset_reg,
                                    uint32_t data_reg) {
    uint32_t vcall_reg = jitc_var(vcall->id)->reg_index;

    ProfilerPhase profiler(profiler_region_vcall_assemble);

    // =====================================================
    // 1. Need to backup state before we can JIT recursively
    // =====================================================

    struct JitBackupRecord {
        ScheduledVariable sv;
        uint32_t param_type : 2;
        uint32_t output_flag : 1;
        uint32_t reg_index;
        uint32_t param_offset;
    };

    std::vector<JitBackupRecord> backup;
    backup.reserve(schedule.size());

    for (const ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        backup.push_back(JitBackupRecord{ sv, v->param_type, v->output_flag,
                                          v->reg_index, v->param_offset });
    }

    int32_t alloca_size_backup = alloca_size;
    int32_t alloca_align_backup = alloca_align;
    alloca_size = alloca_align = -1;

    // =====================================================
    // 2. Determine calling conventions (input/output size,
    //    and alignment). Less state may need to be passed
    //    compared to when the vcall was first created.
    // =====================================================

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1,
             n_in = (uint32_t) vcall->in.size(),
             n_out = (uint32_t) vcall->out.size(),
             n_in_active = 0,
             n_out_active = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
        auto it = state.variables.find(vcall->in[i]);
        if (it == state.variables.end())
            continue;

        uint32_t size = type_size[it.value().type],
                 offset = in_size;
        Variable *v = &it.value();
        in_size += size;
        in_align = std::max(size, in_align);
        n_in_active++;

        // Transfer parameter offset to instances
        auto it2 = state.variables.find(vcall->in_nested[i]);
        if (it2 == state.variables.end())
            continue;

        Variable *v2 = &it2.value();
        v2->param_offset = offset;
        v2->reg_index = v->reg_index;
    }

    for (uint32_t i = 0; i < n_out; ++i) {
        auto it = state.variables.find(vcall->out_nested[i]);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        uint32_t size = type_size[v->type];
        out_size += size;
        out_align = std::max(size, out_align);
        n_out_active++;
    }

    // Input and output are packed into the same array in LLVM mode
    if (vcall->backend == JitBackend::LLVM)
        in_size = (in_size + out_align - 1) / out_align * out_align;

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    std::vector<XXH128_hash_t> callable_hash(vcall->n_inst);

    AllocType at = vcall->backend == JitBackend::CUDA ? AllocType::HostPinned
                                                      : AllocType::Host;
    vcall->offset_h = (uint64_t *) jitc_malloc(at, vcall->offset_d_size);
    memset(vcall->offset_h, 0, vcall->offset_d_size);

    ThreadState *ts = thread_state(vcall->backend);
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        uint32_t data_offset = vcall->data_offset[i];

        auto result = jitc_assemble_func(
            ts, vcall->name, i, in_size, in_align, out_size, out_align,
            data_offset, vcall->data_map, n_in, vcall->in.data(),
            n_out, vcall->out_nested.data() + n_out * i,
            vcall->checkpoints[i + 1] - vcall->checkpoints[i],
            vcall->side_effects.data() + vcall->checkpoints[i],
            vcall->use_self);

        if (vcall->backend == JitBackend::LLVM)
            result.second += 1;

        // high part: instance data offset, low part: callable index
        vcall->offset_h[vcall->inst_id[i]] =
            (((uint64_t) data_offset) << 32) | result.second;
        callable_hash[i] = result.first;
    }

    jitc_memcpy_async(vcall->backend, vcall->offset_d, vcall->offset_h,
                      vcall->offset_d_size);

    size_t se_count = vcall->side_effects.size();
    vcall->clear_side_effects();

    // =====================================================
    // 4. Restore previously backed-up JIT state
    // =====================================================

    schedule.clear();
    for (const JitBackupRecord &b : backup) {
        Variable *v = jitc_var(b.sv.index);
        v->param_type = b.param_type;
        v->output_flag = b.output_flag;
        v->reg_index = b.reg_index;
        v->param_offset = b.param_offset;
        schedule.push_back(b.sv);
    }

    alloca_size = alloca_size_backup;
    alloca_align = alloca_align_backup;

    if (vcall->backend == JitBackend::CUDA)
        jitc_var_vcall_assemble_cuda(vcall, callable_hash, vcall_reg, self_reg,
                                     mask_reg, offset_reg, data_reg, n_out,
                                     in_size, in_align, out_size, out_align);
    else
        jitc_var_vcall_assemble_llvm(vcall, callable_hash, vcall_reg, self_reg,
                                     mask_reg, offset_reg, data_reg, n_out,
                                     in_size, in_align, out_size, out_align);

    std::sort(
        callable_hash.begin(), callable_hash.end(), [](const auto &a, const auto &b) {
            return std::tie(a.high64, a.low64) < std::tie(b.high64, b.low64);
        });

    size_t n_unique = std::unique(
        callable_hash.begin(), callable_hash.end(), [](const auto &a, const auto &b) {
            return std::tie(a.high64, a.low64) == std::tie(b.high64, b.low64);
        }) - callable_hash.begin();

    // Free call offset table asynchronously
    if (vcall->backend == JitBackend::CUDA) {
        jitc_free(vcall->offset_h);
    } else {
        Task *new_task = task_submit_dep(
            nullptr, &ts->task, 1, 1,
            [](uint32_t, void *payload) { jitc_free(*((void **) payload)); },
            &vcall->offset_h, sizeof(void *));
        task_release(ts->task);
        ts->task = new_task;
    }
    vcall->offset_h = nullptr;

    jitc_log(
        InfoSym,
        "jit_var_vcall_assemble(): indirect call (\"%s\") to %zu/%zu instances, "
        "passing %u/%u inputs (%u/%u bytes), %u/%u outputs (%u/%u bytes), %zu side effects",
        vcall->name, n_unique, callable_hash.size(), n_in_active,
        vcall->in_count_initial, in_size, vcall->in_size_initial, n_out_active,
        n_out, out_size, vcall->out_size_initial, se_count);
}

/// Virtual function call code generation -- CUDA/PTX-specific bits
static void jitc_var_vcall_assemble_cuda(
    VCall *vcall, const std::vector<XXH128_hash_t> &callable_hash,
    uint32_t vcall_reg, uint32_t self_reg, uint32_t mask_reg, uint32_t offset_reg,
    uint32_t data_reg, uint32_t n_out, uint32_t in_size, uint32_t in_align,
    uint32_t out_size, uint32_t out_align) {

    // =====================================================
    // 1. Conditional branch
    // =====================================================

    buffer.fmt("\n    @!%%p%u bra l_masked_%u;\n\n", mask_reg, vcall_reg);

    // =====================================================
    // 2. Determine unique callable ID
    // =====================================================

    buffer.fmt("    { // VCall: %s\n"
               "        mad.wide.u32 %%rd3, %%r%u, 8, %%rd%u;\n"
               "        ld.global.u64 %%rd3, [%%rd3];\n"
               "        cvt.u32.u64 %%r3, %%rd3;\n",
               vcall->name, self_reg, offset_reg);
    // %r3: callable ID
    // %rd3: (high 32 bit): data offset

    // =====================================================
    // 3. Turn callable ID into a function pointer
    // =====================================================

    if (!uses_optix)
        buffer.fmt("        ld.global.u64 %%rd2, callables[%%r3];\n");
    else
        buffer.put("        call (%rd2), _optix_call_direct_callable, (%r3);\n");

    // =====================================================
    // 4. Obtain pointer to supplemental call data
    // =====================================================

    if (data_reg) {
        buffer.fmt("        shr.u64 %%rd3, %%rd3, 32;\n"
                   "        add.u64 %%rd3, %%rd3, %%rd%u;\n",
                   data_reg);
    }

    // %rd2: function pointer (if applicable)
    // %rd3: call data pointer with offset

    // =====================================================
    // 5. Generate the actual function call
    // =====================================================

    buffer.put("\n");
    tsl::robin_set<uint32_t> seen;
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        uint32_t callable_id = (uint32_t) vcall->offset_h[vcall->inst_id[i]];
        if (!seen.insert(callable_id).second)
            continue;
        buffer.fmt("        // target %zu = %s%016llx%016llx\n",
                   seen.size(),
                   uses_optix ? "__direct_callable__" : "func_",
                   (unsigned long long) callable_hash[i].high64,
                   (unsigned long long) callable_hash[i].low64);
    }
    buffer.put("\n");

    // Special handling for predicates
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;

        if ((VarType) v2->type != VarType::Bool)
            continue;

        buffer.fmt("        selp.u16 %%w%u, 1, 0, %%p%u;\n",
                    v2->reg_index, v2->reg_index);
    }

    buffer.put("        {\n");

    // Call prototype
    buffer.put("            proto: .callprototype");
    if (out_size)
        buffer.fmt(" (.param .align %u .b8 result[%u])", out_align, out_size);
    buffer.put(" _(");
    if (vcall->use_self) {
        buffer.put(".reg .u32 self");
        if (data_reg || in_size)
            buffer.put(", ");
    }
    if (data_reg) {
        buffer.put(".reg .u64 data");
        if (in_size)
            buffer.put(", ");
    }
    if (in_size)
        buffer.fmt(".param .align %u .b8 params[%u]", in_align, in_size);
    buffer.put(");\n");

    // Input/output parameter arrays
    if (out_size)
        buffer.fmt("            .param .align %u .b8 out[%u];\n", out_align, out_size);
    if (in_size)
        buffer.fmt("            .param .align %u .b8 in[%u];\n", in_align, in_size);

    // =====================================================
    // 5.1. Pass the input arguments
    // =====================================================

    uint32_t offset = 0;
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = type_size[v2->type];

        const char *tname = type_name_ptx[v2->type],
                   *prefix = type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        buffer.fmt("            st.param.%s [in+%u], %s%u;\n", tname, offset,
                   prefix, v2->reg_index);

        offset += size;
    }

    if (vcall->use_self) {
        buffer.fmt("            call %s%%rd2, (%%r%u%s%s), proto;\n",
                   out_size ? "(out), " : "", self_reg,
                   data_reg ? ", %rd3" : "",
                   in_size ? ", in" : "");
    } else {
        buffer.fmt("            call %s%%rd2, (%s%s%s), proto;\n",
                    out_size ? "(out), " : "", data_reg ? "%rd3" : "",
                    data_reg && in_size ? ", " : "", in_size ? "in" : "");
    }

    // =====================================================
    // 5.2. Read back the output arguments
    // =====================================================

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i],
                 index_2 = vcall->out[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it->second.type],
                 load_offset = offset;
        offset += size;

        // Skip if outer access expired
        auto it2 = state.variables.find(index_2);
        if (it2 == state.variables.end())
            continue;

        const Variable *v2 = &it2.value();
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        const char *tname = type_name_ptx[v2->type],
                   *prefix = type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        buffer.fmt("            ld.param.%s %s%u, [out+%u];\n",
                   tname, prefix, v2->reg_index, load_offset);
    }

    buffer.put("        }\n");

    // =====================================================
    // 6. Special handling for predicates return value(s)
    // =====================================================

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if ((VarType) v2->type != VarType::Bool)
            continue;
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        // Special handling for predicates
        buffer.fmt("        setp.ne.u16 %%p%u, %%w%u, 0;\n",
                   v2->reg_index, v2->reg_index);
    }


    buffer.fmt("        bra.uni l_done_%u;\n", vcall_reg);
    buffer.put("    }\n");

    // =====================================================
    // 7. Prepare output registers for masked lanes
    // =====================================================

    buffer.fmt("\nl_masked_%u:\n", vcall_reg);
    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        buffer.put("    ");
        buffer.fmt("mov.%s %s%u, 0;\n",
                   type_name_ptx_bin[v2->type],
                   type_prefix[v2->type], v2->reg_index);
    }

    buffer.fmt("\nl_done_%u:\n", vcall_reg);
}

/// Virtual function call code generation -- LLVM IR-specific bits
static void jitc_var_vcall_assemble_llvm(
    VCall *vcall, const std::vector<XXH128_hash_t> &callable_hash,
    uint32_t vcall_reg, uint32_t self_reg, uint32_t mask_reg,
    uint32_t offset_reg, uint32_t data_reg, uint32_t n_out, uint32_t in_size,
    uint32_t in_align, uint32_t out_size, uint32_t out_align) {

    uint32_t width = jitc_llvm_vector_width;
    alloca_size  = std::max(alloca_size, (int32_t) ((in_size + out_size) * width));
    alloca_align = std::max(alloca_align, (int32_t) (std::max(in_align, out_align) * width));

    // =====================================================
    // 1. Declare a few intrinsics that we will use
    // =====================================================

    char tmp[128];
    snprintf(tmp, sizeof(tmp),
             "declare i32 @llvm.experimental.vector.reduce.umax.v%ui32(<%u x i32>)\n\n",
             width, width);
    jitc_register_global(tmp);
    jitc_register_global("@callables = internal local_unnamed_addr global i8** null, align 8\n\n");

    /* How to prevent @callables from being optimized away as a constant, while at the same time
       not turning it an external variable that would require a global offset table (GOT)?
       Let's make a dummy function that writes to it.. */
    jitc_register_global("define void @set_callables(i8** %ptr) local_unnamed_addr #0 {\n"
                         "    store i8** %ptr, i8*** @callables\n"
                         "    ret void\n"
                         "}\n\n");

    snprintf(tmp, sizeof(tmp),
             "declare <%u x i64> @llvm.masked.gather.v%ui64(<%u x i64*>, i32, "
             "<%u x i1>, <%u x i64>)\n\n",
             width, width, width, width, width);
    jitc_register_global(tmp);

    buffer.fmt("    br label %%l%u_start\n"
               "\nl%u_start:\n"
               "    ; VCall: %s\n",
               vcall_reg, vcall_reg, vcall->name);

    tsl::robin_set<uint32_t, UInt32Hasher> seen;
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        uint32_t callable_id = (uint32_t) vcall->offset_h[i + 1];
        if (!seen.insert(callable_id).second)
            continue;
        buffer.fmt("    ;  - target %zu = @func_%016llx%016llx;\n",
                   seen.size(),
                   (unsigned long long) callable_hash[i].high64,
                   (unsigned long long) callable_hash[i].low64);
    }

    if (!assemble_func) {
        buffer.fmt("\n"
                   "    %%u%u_self_ptr_0 = bitcast i8* %%rd%u to i64*\n"
                   "    %%u%u_self_ptr = getelementptr i64, i64* %%u%u_self_ptr_0, <%u x i32> %%r%u\n",
                   vcall_reg, offset_reg,
                   vcall_reg, vcall_reg, width, self_reg);
    } else {
        buffer.fmt("\n"
                   "    %%u%u_self_ptr_0 = bitcast <%u x i8*> %%rd%u to <%u x i64*>\n"
                   "    %%u%u_self_ptr = getelementptr i64, <%u x i64*> %%u%u_self_ptr_0, <%u x i32> %%r%u\n",
                   vcall_reg, width, offset_reg, width,
                   vcall_reg, width, vcall_reg, width, self_reg);

    }

    buffer.fmt("    %%u%u_self_combined = call <%u x i64> @llvm.masked.gather.v%ui64(<%u x i64*> %%u%u_self_ptr, i32 8, <%u x i1> %%p%u, <%u x i64> zeroinitializer)\n"
               "    %%u%u_self_initial = trunc <%u x i64> %%u%u_self_combined to <%u x i32>\n",
               vcall_reg, width, width, width, vcall_reg, width, mask_reg, width,
               vcall_reg, width, vcall_reg, width);

    if (data_reg) {
        buffer.fmt("    %%u%u_offset_1 = lshr <%u x i64> %%u%u_self_combined, <", vcall_reg, width, vcall_reg);
        for (uint32_t i = 0; i < width; ++i)
            buffer.fmt("i64 32%s", i + 1 < width ? ", " : "");
        buffer.put(">\n");
        buffer.fmt("    %%u%u_offset = trunc <%u x i64> %%u%u_offset_1 to <%u x i32>\n",
                   vcall_reg, width, vcall_reg, width);
    }

    // =====================================================
    // 2. Pass the input arguments
    // =====================================================

    uint32_t offset = 0;
    for (uint32_t i = 0; i < (uint32_t) vcall->in.size(); ++i) {
        uint32_t index = vcall->in[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t vti = v2->type;
        const VarType vt = (VarType) vti;
        uint32_t size = type_size[vti];

        const char *prefix = type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : type_name_llvm[vti];

        buffer.fmt(
            "    %%u%u_in_%u_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_in_%u_1 = bitcast i8* %%u%u_in_%u_0 to <%u x %s> *\n",
            vcall_reg, i, offset,
            vcall_reg, i, vcall_reg, i, width, tname
        );

        if (vt != VarType::Bool) {
            buffer.fmt("    store <%u x %s> %s%u, <%u x %s>* %%u%u_in_%u_1, align %u\n",
                       width, tname, prefix, v2->reg_index,
                       width, tname, vcall_reg, i, size * width);
        } else {
            buffer.fmt("    %%u%u_%u_zext = zext <%u x i1> %s%u to <%u x i8>\n"
                       "    store <%u x %s> %%u%u_%u_zext, <%u x %s>* %%u%u_in_%u_1, align %u\n",
                       vcall_reg, i, width, prefix, v2->reg_index, width,
                       width, tname, vcall_reg, i, width,
                       tname, vcall_reg, i, size * width);
        }

        offset += size * width;
    }

    if (out_size)
        buffer.fmt("    %%u%u_out = getelementptr i8, i8* %%buffer, i32 %u\n",
                   vcall_reg, in_size * width);

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t vti = it->second.type;
        uint32_t size = type_size[vti];
        const char *tname = (VarType) vti == VarType::Bool
                            ? "i8" : type_name_llvm[vti];

        buffer.fmt(
            "    %%u%u_tmp_%u_0 = getelementptr inbounds i8, i8* %%u%u_out, i64 %u\n"
            "    %%u%u_tmp_%u_1 = bitcast i8* %%u%u_tmp_%u_0 to <%u x %s> *\n"
            "    store <%u x %s> zeroinitializer, <%u x %s>* %%u%u_tmp_%u_1, align %u\n",
            vcall_reg, i, vcall_reg, offset,
            vcall_reg, i, vcall_reg, i, width, tname,
            width, tname, width, tname, vcall_reg, i, size * width);

        offset += size * width;
    }

    // =====================================================
    // 3. Perform one call to each unique instance
    // =====================================================

    buffer.fmt("    br label %%l%u_check\n", vcall_reg);

    buffer.fmt("\nl%u_check:\n"
               "    %%u%u_self = phi <%u x i32> [ %%u%u_self_initial, %%l%u_start ], [ %%u%u_self_next, %%l%u_call ]\n",
               vcall_reg, vcall_reg, width, vcall_reg, vcall_reg, vcall_reg, vcall_reg);
    buffer.fmt("    %%u%u_next = call i32 @llvm.experimental.vector.reduce.umax.v%ui32(<%u x i32> %%u%u_self)\n", vcall_reg, width, width, vcall_reg);
    buffer.fmt("    %%u%u_valid = icmp ne i32 %%u%u_next, 0\n"
               "    br i1 %%u%u_valid, label %%l%u_call, label %%l%u_end\n",
               vcall_reg, vcall_reg, vcall_reg, vcall_reg, vcall_reg);

    buffer.fmt("\nl%u_call:\n"
               "    %%u%u_bcast_0 = insertelement <%u x i32> undef, i32 %%u%u_next, i32 0\n"
               "    %%u%u_bcast = shufflevector <%u x i32> %%u%u_bcast_0, <%u x i32> undef, <%u x i32> zeroinitializer\n"
               "    %%u%u_active = icmp eq <%u x i32> %%u%u_self, %%u%u_bcast\n"
               "    %%u%u_func_0 = getelementptr inbounds i8*, i8** %%callables, i32 %%u%u_next\n"
               "    %%u%u_func_1 = load i8*, i8** %%u%u_func_0\n",
               vcall_reg,
               vcall_reg, width, vcall_reg, // bcast_0
               vcall_reg, width, vcall_reg, width, width, // bcast
               vcall_reg, width, vcall_reg, vcall_reg, // active
               vcall_reg, vcall_reg, // func_0
               vcall_reg, vcall_reg // func_1
       );

    // Cast into correctly typed function pointer
    buffer.fmt(
        "    %%u%u_func = bitcast i8* %%u%u_func_1 to void (<%u x i1>",
        vcall_reg, vcall_reg, width);

    if (vcall->use_self)
        buffer.fmt(", <%u x i32>", width);

    buffer.put(", i8*");
    if (data_reg) {
        if (vcall_depth == 0)
            buffer.fmt(", i8*, <%u x i32>", width);
        else
            buffer.fmt(", <%u x i8*>, <%u x i32>", width, width);
    }
    buffer.put(")*\n");

    // Perform the actual function call
    buffer.fmt("    call void %%u%u_func(<%u x i1> %%u%u_active",
            vcall_reg, width, vcall_reg);

    if (vcall->use_self)
        buffer.fmt(", <%u x i32> %%r%u", width, self_reg);

    buffer.put(", i8* %buffer");

    if (data_reg) {
        if (vcall_depth == 0)
            buffer.fmt(", i8* %%rd%u, <%u x i32> %%u%u_offset", data_reg,
                       width, vcall_reg);
        else
            buffer.fmt(", <%u x i8*> %%rd%u, <%u x i32> %%u%u_offset", width,
                       data_reg, width, vcall_reg);
    }
    buffer.put(")\n");


    buffer.fmt("    %%u%u_self_next = select <%u x i1> %%u%u_active, <%u x i32> zeroinitializer, <%u x i32> %%u%u_self\n"
               "    br label %%l%u_check\n"
               "\nl%u_end:\n",
               vcall_reg, width, vcall_reg, width, width, vcall_reg, vcall_reg,
               vcall_reg);

    // =====================================================
    // 5. Read back the output arguments
    // =====================================================

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i],
                 index_2 = vcall->out[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it->second.type],
                 load_offset = offset;
        offset += size * width;

        // Skip if outer access expired
        auto it2 = state.variables.find(index_2);
        if (it2 == state.variables.end())
            continue;

        const Variable *v2 = &it2.value();

        uint32_t vti = v2->type;
        const VarType vt = (VarType) vti;
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        const char *prefix = type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : type_name_llvm[vti];

        buffer.fmt(
            "    %%u%u_out_%u_0 = getelementptr inbounds i8, i8* %%u%u_out, i64 %u\n"
            "    %%u%u_out_%u_1 = bitcast i8* %%u%u_out_%u_0 to <%u x %s> *\n"
            "    %s%u%s = load <%u x %s>, <%u x %s>* %%u%u_out_%u_1, align %u\n",
            vcall_reg, i, vcall_reg, load_offset, vcall_reg, i, vcall_reg, i,
            width, tname,
            prefix, v2->reg_index, vt == VarType::Bool ? "_0" : "", width, tname, width, tname,
            vcall_reg, i, size * width);

            if (vt == VarType::Bool)
                buffer.fmt("    %s%u = trunc <%u x i8> %s%u_0 to <%u x i1>\n",
                           prefix, v2->reg_index, width, prefix, v2->reg_index, width);
    }

    buffer.fmt("    br label %%l%u_done\n"
               "\nl%u_done:\n", vcall_reg, vcall_reg);
}

/// Collect scalar / pointer variables referenced by a computation
void jitc_var_vcall_collect_data(tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                                 uint32_t &data_offset, uint32_t inst_id,
                                 uint32_t index, bool &use_self, bool &optix) {
    uint64_t key = (uint64_t) index + (((uint64_t) inst_id) << 32);
    auto it_and_status = data_map.emplace(key, (uint32_t) -1);
    if (!it_and_status.second)
        return;

    Variable *v = jitc_var(index);
    ThreadState *ts = thread_state(v->backend);

    /* Scalar literals created outside of this virtual function call should be
       considered as data to avoid baking them inside of the kernel IR. This rule
       shouldn't be applied to input literal arguments of the virtual function
       call or masks. */
    bool ext_literal = v->literal &&
                       !v->vcall_iface &&
                       (VarType) v->type != VarType::Bool &&
                       index < ts->vcall_bound_index;

    if (!v->literal && v->stmt && strstr(v->stmt, "self"))
        use_self = true;

    if (v->optix)
        optix = true;

    if (v->vcall_iface) {
        return;
    } else if (v->data || ext_literal || (VarType) v->type == VarType::Pointer) {
        uint32_t tsize = type_size[v->type];
        uint32_t offset = (data_offset + tsize - 1) / tsize * tsize;
        it_and_status.first.value() = offset;
        data_offset = offset + tsize;

        if (v->size != 1)
            jitc_raise(
                "jit_var_vcall(): the virtual function call associated with "
                "instance %u accesses an evaluated variable r%u of type "
                "%s and size %u. However, only *scalar* (size == 1) "
                "evaluated variables can be accessed while recording "
                "virtual function calls",
                inst_id, index, type_name[v->type], v->size);
    } else {
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t index_2 = v->dep[i];
            if (!index_2)
                break;

            jitc_var_vcall_collect_data(data_map, data_offset, inst_id, index_2,
                                        use_self, optix);
        }
        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_var_vcall_collect_data(): could not find "
                          "matching 'extra' record!");

            const Extra &extra = it->second;
            for (uint32_t i = 0; i < extra.n_dep; ++i) {
                uint32_t index_2 = extra.dep[i];
                if (index_2 == 0)
                    continue; // not break
                jitc_var_vcall_collect_data(data_map, data_offset, inst_id,
                                            index_2, use_self, optix);
            }
        }
    }
}

// Compute a permutation to reorder an array of registered pointers
VCallBucket *jitc_var_vcall_reduce(JitBackend backend, const char *domain,
                                   uint32_t index, uint32_t *bucket_count_out) {
    auto it = state.extra.find(index);
    if (it != state.extra.end()) {
        auto &v = it.value();
        if (v.vcall_bucket_count) {
            *bucket_count_out = v.vcall_bucket_count;
            return v.vcall_buckets;
        }
    }

    uint32_t bucket_count = jitc_registry_get_max(backend, domain) + 1;
    if (unlikely(bucket_count == 1)) {
        *bucket_count_out = 0;
        return nullptr;
    }

    // Ensure input index array is fully evaluated
    jitc_var_eval(index);

    uint32_t size = jitc_var(index)->size;

    jitc_log(Debug, "jit_vcall(r%u, domain=\"%s\")", index, domain);

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    if (backend == JitBackend::LLVM)
        perm_size += jitc_llvm_vector_width * sizeof(uint32_t);

    uint8_t *offsets = (uint8_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync, perm_size);

    // Compute permutation
    const uint32_t *self = (const uint32_t *) jitc_var_ptr(index);
    uint32_t unique_count = jitc_mkperm(backend, self, size,
                                        bucket_count, perm, (uint32_t *) offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jitc_var_mem_map(backend, VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.type = (uint32_t) VarType::UInt32;
    v2.backend = (uint32_t) backend;
    v2.dep[3] = perm_var;
    v2.retain_data = true;
    v2.unaligned = 1;

    struct InputBucket {
        uint32_t id, offset, size, unused;
    };

    InputBucket *input_buckets = (InputBucket *) offsets;

    std::sort(
        input_buckets,
        input_buckets + unique_count,
        [](const InputBucket &b1, const InputBucket &b2) {
            return b1.size > b2.size;
        }
    );

    for (uint32_t i = 0; i < unique_count; ++i) {
        InputBucket bucket = input_buckets[i];

        // Create variable for permutation subrange
        v2.data = perm + bucket.offset;
        v2.size = bucket.size;

        jitc_var_inc_ref_int(perm_var);

        uint32_t index2 = jitc_var_new(v2);

        VCallBucket bucket_out;
        bucket_out.ptr = jitc_registry_get_ptr(backend, domain, bucket.id);
        bucket_out.index = index2;
        bucket_out.id = bucket.id;

        memcpy(input_buckets + i, &bucket_out, sizeof(VCallBucket));

        jitc_trace("jit_var_vcall_reduce(): registered variable %u: bucket %u "
                   "(" DRJIT_PTR ") of size %u.", index2, bucket_out.id,
                   (uintptr_t) bucket_out.ptr, bucket.size);
    }

    jitc_var_dec_ref_ext(perm_var);

    *bucket_count_out = unique_count_out;

    jitc_var(index)->extra = true;
    Extra &extra = state.extra[index];
    extra.vcall_bucket_count = unique_count_out;
    extra.vcall_buckets = (VCallBucket *) offsets;
    return extra.vcall_buckets;
}
