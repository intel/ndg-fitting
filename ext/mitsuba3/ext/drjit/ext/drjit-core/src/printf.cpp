#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "op.h"

/// Forward declaration
static void jitc_var_printf_assemble_llvm(const Variable *v, const Extra &extra);
static void jitc_var_printf_assemble_cuda(const Variable *v, const Extra &extra);

uint32_t jitc_var_printf(JitBackend backend, uint32_t mask, const char *fmt,
                         uint32_t narg, const uint32_t *arg) {
    ThreadState *ts = thread_state(backend);
    uint32_t size;
    bool dirty;

    {
        Variable *mask_v = jitc_var(mask);
        size = mask_v->size;
        dirty = mask_v->ref_count_se != 0;
    }

    for (uint32_t i = 0; i < narg; ++i) {
        const Variable *v = jitc_var(arg[i]);
        if (unlikely(size != v->size && v->size != 1 && size != 1))
            jitc_raise("jit_var_printf(): arrays have incompatible size!");
        size = std::max(size, v->size);
        dirty |= v->ref_count_se != 0;
    }

    if (dirty) {
        jitc_eval(ts);
        dirty = false;
        if (mask)
            dirty = jitc_var(mask)->ref_count_se != 0;
        for (uint32_t i = 0; i < narg; ++i)
            dirty |= jitc_var(arg[i])->ref_count_se != 0;
        jitc_fail("jit_var_printf(): variable remains dirty after evaluation!");
    }

    Ref mask_combined = steal(jitc_var_mask_apply(mask, size));

    Ref printf_target;
    if (backend == JitBackend::LLVM)
        printf_target =
            steal(jitc_var_new_pointer(backend, (const void *) &printf, 0, 0));

    Ref printf_var =
        steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 0, nullptr));

    Variable *v = jitc_var(printf_var);
    v->extra = 1;
    v->side_effect = 1;
    v->size = size;
    v->dep[0] = mask_combined;
    jitc_var_inc_ref_int(mask_combined);

    if (backend == JitBackend::LLVM) {
        v->dep[1] = printf_target;
        jitc_var_inc_ref_int(printf_target);
    }

    size_t dep_size = narg * sizeof(uint32_t);
    Extra &e = state.extra[printf_var];
    e.n_dep = narg;
    e.dep = (uint32_t *) malloc(dep_size);
    memcpy(e.dep, arg, dep_size);
    for (uint32_t i = 0; i < narg; ++i)
        jitc_var_inc_ref_int(arg[i]);

    e.assemble = backend == JitBackend::CUDA
                     ? jitc_var_printf_assemble_cuda
                     : jitc_var_printf_assemble_llvm;
    e.callback_data = strdup(fmt);
    e.callback = [](uint32_t, int free_var, void *ptr) {
        if (free_var && ptr) {
            free(ptr);
        }
    };
    e.callback_internal = true;
    uint32_t result = printf_var.release();

    jitc_log(Debug, "jit_var_printf(void r%u, fmt=\"%s\")", result, fmt);
    jitc_var_mark_side_effect(result);

    return result;
}

static void jitc_var_printf_assemble_cuda(const Variable *v,
                                          const Extra &extra) {
    size_t buffer_offset = buffer.size();
    const char *fmt = (const char *) extra.callback_data;
    auto hash = XXH128(fmt, strlen(fmt), 0);
    buffer.fmt(".global .align 1 .b8 data_%016llu%016llu[] = { ",
               (unsigned long long) hash.high64,
               (unsigned long long) hash.low64);
    for (uint32_t i = 0; ; ++i) {
        buffer.put_uint32((uint32_t) fmt[i]);
        if (fmt[i] == '\0')
            break;
        buffer.put(", ");
    }
    buffer.put(" };\n\n");

    jitc_register_global(buffer.get() + buffer_offset);
    jitc_register_global(".extern .func (.param .b32 rv) vprintf (.param .b64 fmt, .param .b64 buf);\n\n");
    buffer.rewind(buffer.size() - buffer_offset);
    buffer.put("    {\n");

    uint32_t offset = 0, align = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        uint32_t vti = v2->type;
        if ((VarType) vti == VarType::Void)
            continue;
        uint32_t tsize = type_size[vti];
        if ((VarType) vti == VarType::Float32)
            tsize = 8;

        offset = (offset + tsize - 1) / tsize * tsize;
        offset += tsize;
        align = std::max(align, tsize);
    }

    if (align == 0)
        align = 1;
    if (offset == 0)
        offset = 1;

    buffer.fmt("        .local .align %u .b8 buf[%u];\n", align, offset);

    offset = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        uint32_t vti = v2->type;
        uint32_t tsize = type_size[vti];
        if ((VarType) vti == VarType::Void)
            continue;
        else if ((VarType) vti == VarType::Float32)
            tsize = 8;

        offset = (offset + tsize - 1) / tsize * tsize;

        if ((VarType) vti == VarType::Float32) {
            buffer.fmt("        cvt.f64.f32 %%d3, %%f%u;\n"
                       "        st.local.f64 [buf+%u], %%d3;\n",
                       v2->reg_index, offset);
        } else {
            buffer.fmt("        st.local.%s [buf+%u], %s%u;\n",
                       type_name_ptx[vti], offset, type_prefix[vti],
                       v2->reg_index);
        }

        offset += tsize;
    }
    buffer.fmt("\n"
               "        .reg.b64 %%fmt_generic, %%buf_generic;\n"
               "        cvta.global.u64 %%fmt_generic, data_%016llu%016llu;\n"
               "        cvta.local.u64 %%buf_generic, buf;\n"
               "        {\n"
               "            .param .b64 fmt_p;\n"
               "            .param .b64 buf_p;\n"
               "            .param .b32 rv_p;\n"
               "            st.param.b64 [fmt_p], %%fmt_generic;\n"
               "            st.param.b64 [buf_p], %%buf_generic;\n"
               "            ",
               (unsigned long long) hash.high64,
               (unsigned long long) hash.low64);
    if (v->dep[0]) {
        Variable *v2 = jitc_var(v->dep[0]);
        if (!v2->literal || v2->value != 1)
            buffer.fmt("@%%p%u ", v2->reg_index);
    }
    buffer.put("call (rv_p), vprintf, (fmt_p, buf_p);\n"
               "        }\n"
               "    }\n");
}

static void jitc_var_printf_assemble_llvm(const Variable *v,
                                          const Extra &extra) {
    const uint32_t width = jitc_llvm_vector_width;
    size_t buffer_offset = buffer.size();
    const char *fmt = (const char *) extra.callback_data;
    size_t length = strlen(fmt);

    auto hash = XXH128(fmt, strlen(fmt), 0);

    buffer.fmt("@data_%016llu%016llu = private unnamed_addr constant [%zu x i8] [",
               (unsigned long long) hash.high64,
               (unsigned long long) hash.low64,
               length + 1);

    for (uint32_t i = 0; ; ++i) {
        buffer.put("i8 ");
        buffer.put_uint32((uint32_t) fmt[i]);
        if (fmt[i] == '\0')
            break;
        buffer.put(", ");
    }

    buffer.put("], align 1\n\n");
    jitc_register_global(buffer.get() + buffer_offset);
    buffer.rewind(buffer.size() - buffer_offset);

    const Variable *mask = jitc_var(v->dep[0]),
                   *target = jitc_var(v->dep[1]);

    uint32_t idx = v->reg_index;

    buffer.fmt("    br label %%l_%u_start\n\n"
               "l_%u_start: ; ---- printf_async() ----\n",
               idx, idx);

    if (assemble_func) {
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i8* @llvm.experimental.vector.reduce.umax.v%ui8(<%u x i8*>)\n\n",
            width, width);
        jitc_register_global(global);

        buffer.fmt("    %%r%u_func_0 = call i8* @llvm.experimental.vector.reduce.umax.v%ui8(<%u x i8*> %%rd%u)"
                   "    %%r%u_func = bitcast i8* %%r%u_func_0 to i32 (i8*, ...)*\n",
                   idx, width, width, target->reg_index,
                   idx, idx);
    } else {
        buffer.fmt("    %%r%u_func = bitcast i8* %%rd%u to i32 (i8*, ...)*\n",
                   idx, target->reg_index);
    }

    buffer.fmt("    %%r%u_fmt = getelementptr [%zu x i8], [%zu x i8]* @data_%016llu%016llu, i64 0, i64 0\n"
               "    br label %%l_%u_cond\n\n"
               "l_%u_cond: ; ---- printf_async() ----\n"
               "    %%r%u_idx = phi i32 [ 0, %%l_%u_start ], [ %%r%u_next, %%l_%u_tail ]\n"
               "    %%r%u_cond = extractelement <%u x i1> %%p%u, i32 %%r%u_idx\n"
               "    br i1 %%r%u_cond, label %%l_%u_body, label %%l_%u_tail\n\n"
               "l_%u_body: ; ---- printf_async() ----\n",
               idx, length + 1, length + 1, (unsigned long long) hash.high64, (unsigned long long) hash.low64,
               idx,
               idx,
               idx, idx, idx, idx,
               idx, jitc_llvm_vector_width, mask->reg_index, idx,
               idx, idx, idx,
               idx);

    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        uint32_t vti = v2->type;

        buffer.fmt(
            "    %%r%u_%u%s = extractelement <%u x %s> %s%u, i32 %%r%u_idx\n",
            idx, i, (VarType) vti == VarType::Float32 ? "_0" : "",
            jitc_llvm_vector_width, type_name_llvm[vti], type_prefix[vti],
            v2->reg_index, idx);

        if ((VarType) vti == VarType::Float32)
            buffer.fmt("    %%r%u_%u = fpext float %%r%u_%u_0 to double\n", idx, i, idx, i);
    }

    buffer.fmt("    call i32 (i8*, ...) %%r%u_func (i8* %%r%u_fmt", v->reg_index, v->reg_index);
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        uint32_t vti = v2->type;
        if (vti == (uint32_t) VarType::Float32)
            vti = (uint32_t) VarType::Float64;
        buffer.fmt(", %s %%r%u_%u", type_name_llvm[vti], idx, i);
    }
    buffer.fmt(")\n"
               "    br label %%l_%u_tail\n\n"
               "l_%u_tail: ; ---- printf_async() ----\n"
               "    %%r%u_next = add i32 %%r%u_idx, 1\n"
               "    %%r%u_cond_2 = icmp ult i32 %%r%u_next, %u\n"
               "    br i1 %%r%u_cond_2, label %%l_%u_cond, label %%l_%u_end\n\n"
               "l_%u_end:\n",
               idx, idx,
               idx, idx,
               idx, idx, jitc_llvm_vector_width,
               idx, idx, idx, idx);
}
