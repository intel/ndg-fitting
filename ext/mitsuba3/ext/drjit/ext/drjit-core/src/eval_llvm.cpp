#include "eval.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "op.h"

// Forward declaration
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v, bool in_function);

void jitc_assemble_llvm(ThreadState *, ScheduledGroup group) {
    uint32_t width = jitc_llvm_vector_width;
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

    buffer.put("define void @drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(i64 %start, i64 %end, "
               "i8** noalias %params) #0 {\n"
               "entry:\n"
               "    br label %body\n"
               "\n"
               "body:\n"
               "    %index = phi i64 [ %index_next, %suffix ], [ %start, %entry ]\n");

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        const Variable *v = jitc_var(index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;

        const char *prefix = type_prefix[vti],
                   *tname = vt == VarType::Bool
                            ? "i8" : type_name_llvm[vti];
        uint32_t tsize = type_size[vti],
                 id = v->reg_index,
                 align = v->unaligned ? 1 : (tsize * width),
                 size = v->size;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_llvm(): internal error: 'extra' entry not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(index);
                if (label && label[0])
                    buffer.fmt("    ; %s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (v->param_type == ParamType::Input && size == 1 && vt == VarType::Pointer) {
            buffer.fmt(
                "    %s%u_p1 = getelementptr inbounds i8*, i8** %%params, i32 %u\n"
                "    %s%u = load i8*, i8** %s%u_p1, align 8, !alias.scope !2\n",
                prefix, id, v->param_offset / (uint32_t) sizeof(void *), prefix,
                id, prefix, id);
        } else if (v->param_type != ParamType::Register) {
            buffer.fmt(
                "    %s%u_p1 = getelementptr inbounds i8*, i8** %%params, i32 %u\n"
                "    %s%u_p2 = load i8*, i8** %s%u_p1, align 8, !alias.scope !2\n"
                "    %s%u_p3 = bitcast i8* %s%u_p2 to %s*\n",
                prefix, id, v->param_offset / (uint32_t) sizeof(void *), prefix,
                id, prefix, id, prefix, id, prefix, id, tname);

            // For output parameters, and non-scalar inputs
            if (v->param_type != ParamType::Input || size != 1) {
                buffer.fmt("    %s%u_p4 = getelementptr inbounds %s, %s* %s%u_p3, i64 %%index\n"
                           "    %s%u_p5 = bitcast %s* %s%u_p4 to <%u x %s>*\n",
                           prefix, id, tname, tname, prefix, id, prefix, id, tname,
                           prefix, id, width, tname);
            }
        }

        if (likely(v->param_type == ParamType::Input)) {
            if (v->literal)
                continue;

            if (size != 1) {
                if (vt != VarType::Bool) {
                    buffer.fmt("    %s%u = load <%u x %s>, <%u x %s>* %s%u_p5, align %u, !alias.scope !2, !nontemporal !{i32 1}\n",
                               prefix, id, width, tname, width, tname, prefix, id, align);
                } else {
                    buffer.fmt("    %s%u_0 = load <%u x i8>, <%u x i8>* %s%u_p5, align %u, !alias.scope !2, !nontemporal !{i32 1}\n"
                               "    %s%u = trunc <%u x i8> %s%u_0 to <%u x i1>\n",
                               prefix, id, width, width, prefix, id, align,
                               prefix, id, width, prefix, id, width);
                }
            } else {
                if (vt != VarType::Bool) {
                    buffer.fmt("    %s%u_0 = load %s, %s* %s%u_p3, align %u, !alias.scope !2\n"
                               "    %s%u_1 = insertelement <%u x %s> undef, %s %s%u_0, i32 0\n"
                               "    %s%u = shufflevector <%u x %s> %s%u_1, <%u x %s> undef, <%u x i32> zeroinitializer\n",
                               prefix, id, tname, tname, prefix, id, tsize,
                               prefix, id, width, tname, tname, prefix, id,
                               prefix, id, width, tname, prefix, id, width, tname, width);
                } else {
                    buffer.fmt("    %s%u_0 = load i8, i8* %s%u_p3, align %u, !alias.scope !2\n"
                               "    %s%u_1 = trunc i8 %s%u_0 to i1\n"
                               "    %s%u_2 = insertelement <%u x i1> undef, i1 %s%u_1, i32 0\n"
                               "    %s%u = shufflevector <%u x i1> %s%u_2, <%u x i1> undef, <%u x i32> zeroinitializer\n",
                               prefix, id, prefix, id, tsize,
                               prefix, id,
                               prefix, id, prefix, id, width, prefix, id,
                               prefix, id, width, prefix, id, width, width);
                }
            }
        } else {
            jitc_render_stmt_llvm(index, v, false);
        }

        if (v->param_type == ParamType::Output) {
            if (vt != VarType::Bool) {
                buffer.fmt("    store <%u x %s> %s%u, <%u x %s>* %s%u_p5, align %u, !noalias !2, !nontemporal !{i32 1}\n",
                           width, tname, prefix, id, width, tname, prefix, id, align);
            } else {
                buffer.fmt("    %s%u_e = zext <%u x i1> %s%u to <%u x i8>\n"
                           "    store <%u x i8> %s%u_e, <%u x i8>* %s%u_p5, align %u, !noalias !2, !nontemporal !{i32 1}\n",
                           prefix, id, width, prefix, id, width,
                           width, prefix, id, width, prefix, id, align);
            }
        }
    }

    buffer.put("\n"
               "    br label %suffix\n"
               "\n"
               "suffix:\n");
    buffer.fmt("    %%index_next = add i64 %%index, %u\n", width);
    buffer.put("    %cond = icmp uge i64 %index_next, %end\n"
               "    br i1 %cond, label %done, label %body, !llvm.loop !2\n\n"
               "done:\n"
               "    ret void\n"
               "}\n"
               "\n");

    /* The program requires extra memory or uses callables. Insert
       setup code the top of the function to accomplish this */
    if (!callables.empty() || alloca_size >= 0) {
        // Ultimately we want to insert at this location
        size_t header_offset = (char *) strchr(buffer.get(), ':') - buffer.get() + 2;

        // Append at the end for now
        size_t cur_offset = buffer.size();
        if (!callables.empty())
            buffer.put("    %callables = load i8**, i8*** @callables\n");

        if (alloca_size >= 0)
            buffer.fmt("    %%buffer = alloca i8, i32 %i, align %i\n",
                       alloca_size, alloca_align);
        buffer.put("\n");

        size_t buffer_size = buffer.size(),
               insertion_size = buffer_size - cur_offset;

        // Extra space for moving things around
        buffer.putc('\0', insertion_size);

        // Move the generated source code to make space for the header addition
        memmove((char *) buffer.get() + header_offset + insertion_size,
                buffer.get() + header_offset, buffer_size - header_offset);

        // Finally copy the code to the insertion point
        memcpy((char *) buffer.get() + header_offset,
               buffer.get() + buffer_size, insertion_size);

        buffer.rewind(insertion_size);
    }

    for (const std::string &s : callables)
        buffer.put(s.c_str(), s.length());

    for (const std::string &s : globals)
        buffer.put(s.c_str(), s.length());

    buffer.put("!0 = !{!0}\n"
               "!1 = !{!1, !0}\n"
               "!2 = !{!1}\n"
               "!3 = !{!\"llvm.loop.unroll.disable\", !\"llvm.loop.vectorize.enable\", i1 0}\n\n");

    buffer.fmt("attributes #0 = { norecurse nounwind \"frame-pointer\"=\"none\" \"no-builtins\" \"no-stack-arg-probe\" "
               "\"target-cpu\"=\"%s\" \"target-features\"=\"",
               jitc_llvm_target_cpu);

#if !defined(__aarch64__)
    buffer.put("-vzeroupper");
    if (jitc_llvm_target_features)
        buffer.putc(',');
#endif

    if (jitc_llvm_target_features)
        buffer.put(jitc_llvm_target_features, strlen(jitc_llvm_target_features));

    buffer.put("\" }");
}

void jitc_assemble_llvm_func(const char *name, uint32_t inst_id,
                             uint32_t in_size, uint32_t data_offset,
                             const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                             uint32_t n_out, const uint32_t *out_nested,
                             bool use_self) {
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);
    uint32_t width = jitc_llvm_vector_width;
    if (use_self) {
        buffer.fmt("define void @func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
                   "<%u x i1> %%mask, <%u x i32> %%self, i8* noalias %%params",
                   width, width);
    } else {
        buffer.fmt("define void @func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
                   "<%u x i1> %%mask, i8* noalias %%params",
                   width);
    }
    if (!data_map.empty()) {
        if (vcall_depth == 1)
            buffer.fmt(", i8* noalias %%data, <%u x i32> %%offsets", width);
        else
            buffer.fmt(", <%u x i8*> %%data, <%u x i32> %%offsets", width, width);
    }

    buffer.fmt(") #0 {\n"
               "entry:\n"
               "    ; VCall: %s\n", name);

    alloca_size = alloca_align = -1;

    for (ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;
        uint32_t id = v->reg_index;
        const char *prefix = type_prefix[vti],
                   *tname  = vt == VarType::Bool ? "i8" : type_name_llvm[vti];

        if (unlikely(v->extra)) {
            auto it = state.extra.find(sv.index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_llvm(): internal error: 'extra' entry "
                          "not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(sv.index);
                if (label && label[0])
                    buffer.fmt("    ; %s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (v->vcall_iface && !v->literal) {
            buffer.fmt("    %s%u_i0 = getelementptr inbounds i8, i8* %%params, i64 %u\n"
                       "    %s%u_i1 = bitcast i8* %s%u_i0 to <%u x %s>*\n"
                       "    %s%u%s = load <%u x %s>, <%u x %s>* %s%u_i1, align %u\n",
                       prefix, id, v->param_offset * width,
                       prefix, id, prefix, id,
                       width, tname,
                       prefix, id, vt == VarType::Bool ? "_i2" : "", width,
                       tname, width, tname, prefix, id, width * type_size[vti]);
            if (vt == VarType::Bool)
                buffer.fmt("    %s%u = trunc <%u x i8> %s%u_i2 to <%u x i1>\n",
                           prefix, id, width, prefix, id, width);
        } else if (v->literal || v->data) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst_id) << 32);
            auto it = data_map.find(key);

            if (!v->data && vt != VarType::Pointer) {
                if (unlikely(it == data_map.end() || it->second == (uint32_t) -1)) {
                    jitc_render_stmt_llvm(sv.index, v, true);
                    continue;
                }
            } else {
                if (unlikely(it == data_map.end()))
                    jitc_fail("jitc_assemble_llvm_func(): could not find entry for "
                              "variable r%u in 'data_map'", sv.index);
                if (it->second == (uint32_t) -1)
                    jitc_fail(
                        "jitc_assemble_llvm_func(): variable r%u is referenced by "
                        "a recorded function call. However, it was evaluated "
                        "between the recording step and code generation (which "
                        "is happening now). This is not allowed.", sv.index);
            }

            uint32_t offset = it->second - data_offset;

            size_t intrinsic_offset = buffer.size();
            buffer.fmt("declare <%u x %s> @llvm.masked.gather.v%u%s(<%u x "
                       "%s*>, i32, <%u x i1>, <%u x %s>)\n\n",
                       width, tname, width, type_name_llvm_abbrev[vti], width,
                       tname, width, width, tname);
            jitc_register_global(buffer.get() + intrinsic_offset);
            size_t intrinsic_length = buffer.size() - intrinsic_offset;
            buffer.rewind(intrinsic_length);

            if (vcall_depth == 1) {
                buffer.fmt("    %s%u_p1 = getelementptr inbounds i8, i8* %%data, i32 %u\n"
                           "    %s%u_p2 = getelementptr inbounds i8, i8* %s%u_p1, <%u x i32> %%offsets\n",
                           prefix, id, offset,
                           prefix, id, prefix, id, width);
            } else {
                buffer.fmt("    %s%u_p1 = getelementptr inbounds i8, <%u x i8*> %%data, i32 %u\n"
                           "    %s%u_p2 = getelementptr inbounds i8, <%u x i8*> %s%u_p1, <%u x i32> %%offsets\n",
                           prefix, id, width, offset,
                           prefix, id, width, prefix, id, width);
            }

            buffer.fmt(
                "    %s%u_p3 = bitcast <%u x i8*> %s%u_p2 to <%u x %s*>\n"
                "    %s%u%s = call <%u x %s> @llvm.masked.gather.v%u%s(<%u x %s*> %s%u_p3, i32 %u, <%u x i1> %%mask, <%u x %s> zeroinitializer)\n",
                prefix, id, width, prefix, id, width, tname,
                prefix, id,
                (vt == VarType::Pointer || vt == VarType::Bool) ? "_p4" : "",
                width, tname, width, type_name_llvm_abbrev[vti], width, tname, prefix, id, type_size[vti], width, width, tname
            );

            if (vt == VarType::Pointer) {
                buffer.fmt(
                    "    %s%u = inttoptr <%u x i64> %s%u_p4 to <%u x i8*>\n",
                    prefix, id, width, prefix, id, width);
            } else if (v->literal && vt == VarType::Bool) {
                buffer.fmt("    %s%u = trunc <%u x i8> %s%u_p4 to <%u x i1>\n",
                           prefix, id, width, prefix, id, width);
            }
        } else {
            jitc_render_stmt_llvm(sv.index, v, true);
        }
    }

    uint32_t output_offset = in_size * width;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = out_nested[i];
        if (!index)
            continue;
        const Variable *v = jitc_var(index);
        uint32_t vti = v->type;
        const VarType vt = (VarType) vti;
        const char *tname = vt == VarType::Bool ? "i8" : type_name_llvm[vti],
                   *prefix = type_prefix[vti];
        uint32_t align = type_size[vti] * width, reg_index = v->reg_index;

        buffer.fmt(
            "    %%out_%u_0 = getelementptr inbounds i8, i8* %%params, i64 %u\n"
            "    %%out_%u_1 = bitcast i8* %%out_%u_0 to <%u x %s>*\n"
            "    %%out_%u_2 = load <%u x %s>, <%u x %s>* %%out_%u_1, align %u\n",
            i, output_offset,
            i, i, width, tname,
            i, width, tname, width, tname, i, align);

        if (vt == VarType::Bool)
            buffer.fmt("    %%out_%u_zext = zext <%u x i1> %s%u to <%u x i8>\n"
                       "    %%out_%u_3 = select <%u x i1> %%mask, <%u x i8> %%out_%u_zext, <%u x i8> %%out_%u_2\n",
                       i, width, prefix, reg_index, width,
                       i, width, width, i, width, i);
        else
            buffer.fmt("    %%out_%u_3 = select <%u x i1> %%mask, <%u x %s> %s%u, <%u x %s> %%out_%u_2\n",
                       i, width, width, tname, prefix, reg_index, width, tname, i);

        buffer.fmt(
            "    store <%u x %s> %%out_%u_3, <%u x %s>* %%out_%u_1, align %u\n",
            width, tname, i, width, tname, i, align);
        output_offset += type_size[vti] * width;
    }

    /* The function requires extra memory or uses callables. Insert
       setup code the top of the function to accomplish this */
    if (alloca_size >= 0) {
        // Ultimately we want to insert at this location
        size_t header_offset = (char *) strrchr(buffer.get(), '{') - buffer.get() + 9;

        // Append at the end for now
        size_t cur_offset = buffer.size();
        if (!callables.empty())
            buffer.put("    %callables = load i8**, i8*** @callables\n");

        if (alloca_size >= 0)
            buffer.fmt("    %%buffer = alloca i8, i32 %i, align %i\n",
                       alloca_size, alloca_align);

        size_t buffer_size = buffer.size(),
               insertion_size = buffer_size - cur_offset;

        // Extra space for moving things around
        buffer.putc('\0', insertion_size);
        //
        // Move the generated source code to make space for the header addition
        memmove((char *) buffer.get() + header_offset + insertion_size,
                buffer.get() + header_offset, buffer_size - header_offset);

        // Finally copy the code to the insertion point
        memcpy((char *) buffer.get() + header_offset,
               buffer.get() + buffer_size, insertion_size);

        buffer.rewind(insertion_size);
    }

    buffer.put("    ret void;\n"
               "}\n");
}

/// Convert an IR template with '$' expressions into valid IR
static void jitc_render_stmt_llvm(uint32_t index, const Variable *v, bool in_function) {
    if (v->literal) {
        uint32_t reg = v->reg_index, width = jitc_llvm_vector_width;
        uint32_t vt = v->type;
        const char *prefix = type_prefix[vt],
                   *tname = type_name_llvm[vt];
        uint64_t value = v->value;

        if (vt == (uint32_t) VarType::Float32) {
            float f;
            memcpy(&f, &value, sizeof(float));
            double d = f;
            memcpy(&value, &d, sizeof(uint64_t));
            vt = (uint32_t) VarType::Float64;
        }

#if 0
        /* The commented code here uses snprintf(), which internally relies on
           FILE*-style streams and becomes a performance bottleneck of the
           whole stringification process .. */

        buffer.fmt("    %s%u_1 = insertelement <%u x %s> undef, %s %llu, i32 0\n"
                   "    %s%u = shufflevector <%u x %s> %s%u_1, <%u x %s> undef, <%u x i32> zeroinitializer\n",
                   prefix, reg, width, tname, tname, (unsigned long long) value,
                   prefix, reg, width, tname, prefix, reg, width, tname, width);
#else
        // .. the ridiculous explicit variant below is equivalent and faster.

        size_t tname_len = strlen(tname),
               prefix_len = strlen(prefix);

        buffer.put("    ");
        buffer.put(prefix, strlen(prefix));
        buffer.put_uint32(reg);
        buffer.put("_1 = insertelement <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> undef, ");
        buffer.put(tname, tname_len);

        if (vt == (uint32_t) VarType::Float64) {
            buffer.put(" 0x");
            buffer.put_uint64_hex(value);
        } else {
            buffer.putc(' ');
            buffer.put_uint64(value);
        }

        buffer.put(", i32 0\n    ");
        buffer.put(prefix, prefix_len);
        buffer.put_uint32(reg);
        buffer.put(" = shufflevector <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> ");
        buffer.put(prefix, prefix_len);
        buffer.put_uint32(reg);
        buffer.put("_1, <");
        buffer.put_uint32(width);
        buffer.put(" x ");
        buffer.put(tname, tname_len);
        buffer.put("> undef, <");
        buffer.put_uint32(width);
        buffer.put(" x i32> zeroinitializer\n");
#endif
    } else {
        const char *s = v->stmt;
        if (unlikely(*s == '\0'))
            return;
        buffer.put("    ");
        char c;
        size_t intrinsic_start = 0;
        do {
            const char *start = s;
            while (c = *s, c != '\0' && c != '$')
                s++;
            buffer.put(start, s - start);

            if (c == '$') {
                s++;
                const char **prefix_table = nullptr, tname = *s++;
                switch (tname) {
                    case '[':
                        intrinsic_start = buffer.size();
                        continue;

                    case ']':
                        buffer.put("\n\n");
                        jitc_register_global(buffer.get() + intrinsic_start);
                        buffer.rewind(buffer.size() - intrinsic_start);
                        continue;

                    case 'n': buffer.put("\n    "); continue;
                    case 'w': buffer.put(jitc_llvm_vector_width_str,
                                         strlen(jitc_llvm_vector_width_str)); continue;
                    case 't': prefix_table = type_name_llvm; break;
                    case 'T': prefix_table = type_name_llvm_big; break;
                    case 'b': prefix_table = type_name_llvm_bin; break;
                    case 'a': prefix_table = type_name_llvm_abbrev; break;
                    case 's': prefix_table = type_size_str; break;
                    case 'r': prefix_table = type_prefix; break;
                    case 'i': prefix_table = nullptr; break;
                    case '<': if (in_function) {
                                  buffer.putc('<');
                                  buffer.put(jitc_llvm_vector_width_str,
                                             strlen(jitc_llvm_vector_width_str));
                                  buffer.put(" x ");
                               }
                               continue;
                    case '>': if (in_function)
                                  buffer.putc('>');
                               continue;
                    case 'o': prefix_table = (const char **) jitc_llvm_ones_str; break;
                    default:
                        jitc_fail("jit_render_stmt_llvm(): encountered invalid \"$\" "
                                  "expression (unknown character \"%c\") in \"%s\"!", tname, v->stmt);
                }

                uint32_t arg_id = *s++ - '0';
                if (unlikely(arg_id > 4))
                    jitc_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                              "expression (argument out of bounds)!", v->stmt);

                uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
                if (unlikely(dep_id == 0))
                    jitc_fail("jit_render_stmt_llvm(%s): encountered invalid \"$\" "
                              "expression (referenced variable %u is missing)!", v->stmt, arg_id);

                const Variable *dep = jitc_var(dep_id);
                if (likely(prefix_table)) {
                    const char *prefix = prefix_table[(int) dep->type];
                    buffer.put(prefix, strlen(prefix));
                }

                if (tname == 'r' || tname == 'i')
                    buffer.put_uint32(dep->reg_index);
            }
        } while (c != '\0');

        buffer.putc('\n');
    }
}

static void jitc_llvm_ray_trace_assemble(const Variable *v, const Extra &extra);

void jitc_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                         const uint32_t *in, uint32_t *out) {
    const uint32_t n_args = 14;
    bool double_precision = ((VarType) jitc_var(in[2])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;

    VarType types[]{ VarType::Bool,   VarType::Bool,  float_type,
                     float_type,      float_type,     float_type,
                     float_type,      float_type,     float_type,
                     float_type,      float_type,     VarType::UInt32,
                     VarType::UInt32, VarType::UInt32 };

    bool placeholder = false, dirty = false;
    uint32_t size = 0;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *v = jitc_var(in[i]);
        if ((VarType) v->type != types[i])
            jitc_raise("jitc_llvm_ray_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) types[i]]);
        size = std::max(size, v->size);
        placeholder |= (bool) v->placeholder;
        dirty |= (bool) v->ref_count_se;
    }

    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jitc_llvm_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if ((VarType) jitc_var(func)->type != VarType::Pointer ||
        (VarType) jitc_var(scene)->type != VarType::Pointer)
        jitc_raise("jitc_llvm_ray_trace(): 'func', and 'scene' must be pointer variables!");

    if (dirty) {
        jitc_eval(thread_state(JitBackend::LLVM));
        dirty = false;

        for (uint32_t i = 0; i < n_args; ++i)
            dirty |= (bool) jitc_var(in[i])->ref_count_se;

        if (dirty)
            jitc_raise(
                "jit_llvm_ray_trace(): inputs remain dirty after evaluation!");
    }

    // ----------------------------------------------------------
    Ref valid = steal(jitc_var_mask_apply(in[1], size));
    {
        int32_t minus_one_c = -1, zero_c = 0;
        Ref minus_one = steal(jitc_var_new_literal(
                JitBackend::LLVM, VarType::Int32, &minus_one_c, 1, 0)),
            zero = steal(jitc_var_new_literal(
                JitBackend::LLVM, VarType::Int32, &zero_c, 1, 0));

        uint32_t deps_2[3] = { valid, minus_one, zero };
        valid = steal(jitc_var_new_op(JitOp::Select, 3, deps_2));
    }

    jitc_log(InfoSym, "jitc_llvm_ray_trace(): tracing %u %sray%s%s%s", size,
             shadow_ray ? "shadow " : "", size != 1 ? "s" : "",
             placeholder ? " (part of a recorded computation)" : "",
             double_precision ? " (double precision)" : "");

    Ref op = steal(jitc_var_new_stmt_n(JitBackend::LLVM, VarType::Void,
                               shadow_ray ? "// Ray trace (shadow ray)"
                                          : "// Ray trace",
                               1, func, scene));
    Variable *v_op = jitc_var(op);
    v_op->size = size;
    v_op->extra = 1;

    Extra &e = state.extra[op];
    e.dep = (uint32_t *) malloc_check(sizeof(uint32_t) * n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        uint32_t index = i != 1 ? in[i] : valid;
        jitc_var_inc_ref_int(index);
        e.dep[i] = index;
    }
    e.n_dep = n_args;
    e.assemble = jitc_llvm_ray_trace_assemble;

    char tmp[128];
    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i) {
        snprintf(tmp, sizeof(tmp),
                 "$r0 = bitcast <$w x $t0> $r1_out_%u to <$w x $t0>", i);
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        out[i] = jitc_var_new_stmt_n(JitBackend::LLVM, vt, tmp, 0, op);
    }
}

static void jitc_llvm_ray_trace_assemble(const Variable *v, const Extra &extra) {
    const uint32_t width = jitc_llvm_vector_width;
    const uint32_t id = v->reg_index;
    bool shadow_ray = strstr(v->stmt, "(shadow ray)") != nullptr;
    bool double_precision = ((VarType) jitc_var(extra.dep[2])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;
    uint32_t float_size = double_precision ? 8 : 4;

    uint32_t ctx_size = 6 * 4, alloca_size_rt;

    if (shadow_ray)
        alloca_size_rt = (9 * float_size + 4 * 4) * width;
    else
        alloca_size_rt = (14 * float_size + 7 * 4) * width;

    alloca_size  = std::max(alloca_size, (int32_t) (alloca_size_rt + ctx_size));
    alloca_align = std::max(alloca_align, (int32_t) (float_size * width));

    /* Offsets:
        0  bool coherent
        1  uint32_t valid
        1  float org_x
        2  float org_y
        3  float org_z
        4  float tnear
        5  float dir_x
        6  float dir_y
        7  float dir_z
        8  float time
        9  float tfar
        10 uint32_t mask
        11 uint32_t id
        12 uint32_t flags
        13 float Ng_x
        14 float Ng_y
        15 float Ng_z
        16 float u
        17 float v
        18 uint32_t primID
        19 uint32_t geomID
        20 uint32_t instID[] */
    buffer.fmt("\n    ; -------- Ray %s -------\n", shadow_ray ? "test" : "trace");

    uint32_t offset = 0;
    for (int i = 0; i < 13; ++i) {
        if (jitc_llvm_vector_width == 1 && i == 0)
            continue; // valid flag not needed for 1-lane versions

        const Variable *v2 = jitc_var(extra.dep[i + 1]);
        const char *tname = type_name_llvm[v2->type];
        uint32_t tsize = type_size[v2->type];
        buffer.fmt(
            "    %%u%u_in_%u_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_in_%u_1 = bitcast i8* %%u%u_in_%u_0 to <%u x %s> *\n"
            "    store <%u x %s> %s%u, <%u x %s>* %%u%u_in_%u_1, align %u\n",
            id, i, offset,
            id, i, id, i, width, tname,
            width, tname, type_prefix[v2->type], v2->reg_index, width, tname, id, i,
            float_size * width);
        offset += tsize * width;
    }

    if (!shadow_ray) {
        buffer.fmt(
            "    %%u%u_in_geomid_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_in_geomid_1 = bitcast i8* %%u%u_in_geomid_0 to <%u x i32> *\n"
            "    store <%u x i32> %s, <%u x i32>* %%u%u_in_geomid_1, align %u\n",
            id, (14 * float_size + 5 * 4) * width, id, id, width, width,
            jitc_llvm_ones_str[(int) VarType::Int32], width, id, float_size * width);
    }

    const Variable *coherent = jitc_var(extra.dep[0]);

    buffer.fmt(
        "    %%u%u_in_ctx_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
        "    %%u%u_in_ctx_1 = bitcast i8* %%u%u_in_ctx_0 to <6 x i32> *\n",
        id, alloca_size_rt, id, id);

    if (coherent->literal && coherent->value == 0) {
        buffer.fmt("    store <6 x i32> <i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0>, <6 x i32>* %%u%u_in_ctx_1, align 4\n", id);
    } else if (coherent->literal && coherent->value == 1) {
        buffer.fmt("    store <6 x i32> <i32 1, i32 0, i32 0, i32 0, i32 -1, i32 0>, <6 x i32>* %%u%u_in_ctx_1, align 4\n", id);
    } else {
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i1 @llvm.experimental.vector.reduce.and.v%ui1(<%u x i1>)\n\n",
            width, width);
        jitc_register_global(global);

        buffer.fmt("    %%u%u_coherent = call i1 @llvm.experimental.vector.reduce.and.v%ui1(<%u x i1> %%p%u)\n"
                   "    %%u%u_ctx = select i1 %%u%u_coherent, <6 x i32> <i32 1, i32 0, i32 0, i32 0, i32 -1, i32 0>, <6 x i32> <i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0>\n"
                   "    store <6 x i32> %%u%u_ctx, <6 x i32>* %%u%u_in_ctx_1, align 4\n",
                   id, width, width, coherent->reg_index,
                   id, id,
                   id, id);
    }

    const Variable *func  = jitc_var(v->dep[0]),
                   *scene = jitc_var(v->dep[1]);

    /* When the ray tracing operation occurs inside a recorded function, it's
     * possible that multiple different scenes are being traced simultaneously.
     * In that case, it's necessary to perform one ray tracing call per scene,
     * which is implemented by the following loop. */
    if (!assemble_func) {
        if (jitc_llvm_vector_width > 1) {
            buffer.fmt(
                "    %%u%u_func = bitcast i8* %%rd%u to void (i8*, i8*, i8*, i8*)*\n"
                "    call void %%u%u_func(i8* %%u%u_in_0_0, i8* %%rd%u, i8* %%u%u_in_ctx_0, i8* %%u%u_in_1_0)\n",
                id, func->reg_index,
                id, id, scene->reg_index, id, id
            );
        } else {
            buffer.fmt(
                "    %%u%u_func = bitcast i8* %%rd%u to void (i8*, i8*, i8*)*\n"
                "    call void %%u%u_func(i8* %%rd%u, i8* %%u%u_in_ctx_0, i8* %%u%u_in_1_0)\n",
                id, func->reg_index,
                id, scene->reg_index, id, id
            );
        }
    } else {
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i64 @llvm.experimental.vector.reduce.umax.v%ui64(<%u x i64>)\n\n",
            width, width);
        jitc_register_global(global);

        // =====================================================
        // 1. Prepare the loop for the ray tracing calls
        // =====================================================

        buffer.fmt("    br label %%l%u_start\n"
                   "\nl%u_start:\n"
                   "    ; Ray tracing\n"
                   "    %%u%u_func_i64 = call i64 @llvm.experimental.vector.reduce.umax.v%ui64(<%u x i64> %%rd%u_p4)\n"
                   "    %%u%u_func_ptr = inttoptr i64 %%u%u_func_i64 to i8*",
                   id, id,
                   id, width, width, func->reg_index,
                   id, id);

        if (jitc_llvm_vector_width > 1) {
            buffer.fmt(
                "    %%u%u_func = bitcast i8* %%u%u_func_ptr to void (i8*, i8*, i8*, i8*)*\n",
                id, id
            );
        } else {
            buffer.fmt(
                "    %%u%u_func = bitcast i8* %%u%u_func_ptr to void (i8*, i8*, i8*)*\n",
                id, id
            );
        }

        // Get original mask, to be overwritten at every iteration
        buffer.fmt("    %%u%u_mask_ptr = bitcast i8* %%u%u_in_0_0 to <%u x i1> *\n"
                   "    %%u%u_mask_value = load <%u x i32>, <%u x i32>* %%u%u_in_0_1, align 64\n"
                   "    br label %%l%u_check\n",
                   id, id, width,
                   id, width, width, id,
                   id);

        // =====================================================
        // 2. Move on to the next instance & check if finished
        // =====================================================

        buffer.fmt("\nl%u_check:\n"
                   "    %%u%u_scene = phi <%u x i8*> [ %%rd%u, %%l%u_start ], [ %%u%u_scene_next, %%l%u_call ]\n"
                   "    %%u%u_scene_i64 = ptrtoint <%u x i8*> %%u%u_scene to <%u x i64>\n"
                   "    %%u%u_next_i64 = call i64 @llvm.experimental.vector.reduce.umax.v%ui64(<%u x i64> %%u%u_scene_i64)\n"
                   "    %%u%u_next = inttoptr i64 %%u%u_next_i64 to i8*\n"
                   "    %%u%u_valid = icmp ne i8* %%u%u_next, null\n"
                   "    br i1 %%u%u_valid, label %%l%u_call, label %%l%u_end\n",
                   id,
                   id, width, scene->reg_index, id, id, id,
                   id, width, id, width,
                   id, width, width, id,
                   id, id,
                   id, id,
                   id, id, id);

        // =====================================================
        // 3. Perform ray tracing call to each unique instance
        // =====================================================

        buffer.fmt("\nl%u_call:\n"
                   "    %%u%u_bcast_0 = insertelement <%u x i64> undef, i64 %%u%u_next_i64, i32 0\n"
                   "    %%u%u_bcast_1 = shufflevector <%u x i64> %%u%u_bcast_0, <%u x i64> undef, <%u x i32> zeroinitializer\n"
                   "    %%u%u_bcast_2 = inttoptr <%u x i64> %%u%u_bcast_1 to <%u x i8*>\n"
                   "    %%u%u_active = icmp eq <%u x i8*> %%u%u_scene, %%u%u_bcast_2\n"
                   "    %%u%u_active_2 = select <%u x i1> %%u%u_active, <%u x i32> %%u%u_mask_value, <%u x i32> zeroinitializer\n"
                   "    store <%u x i32> %%u%u_active_2, <%u x i32>* %%u%u_in_0_1, align 64\n",
                   id,
                   id, width, id,
                   id, width, id, width, width,
                   id, width, id, width,
                   id, width, id, id,
                   id, width, id, width, id, width,
                   width, id, width, id);

        if (jitc_llvm_vector_width > 1) {
            buffer.fmt(
                "    call void %%u%u_func(i8* %%u%u_in_0_0, i8* %%u%u_next, i8* %%u%u_in_ctx_0, i8* %%u%u_in_1_0)\n",
                id, id, id, id, id
            );
        } else {
            buffer.fmt(
                "    call void %%u%u_func(i8* %%u%u_next, i8* %%u%u_in_ctx_0, i8* %%u%u_in_1_0)\n",
                id, id, id, id
            );
        }
        buffer.fmt("    %%u%u_scene_next = select <%u x i1> %%u%u_active, <%u x i8*> zeroinitializer, <%u x i8*> %%u%u_scene\n"
                   "    br label %%l%u_check\n"
                   "\nl%u_end:\n",
                   id, width, id, width, width, id,
                   id, id);
    }

    offset = (8 * float_size + 4) * width;

    for (int i = 0; i < (shadow_ray ? 1 : 6); ++i) {
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        const char *tname = type_name_llvm[(int) vt];
        uint32_t tsize = type_size[(int) vt];
        buffer.fmt(
            "    %%u%u_out_%u_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_out_%u_1 = bitcast i8* %%u%u_out_%u_0 to <%u x %s> *\n"
            "    %%u%u_out_%u = load <%u x %s>, <%u x %s>* %%u%u_out_%u_1, align %u\n",
            id, i, offset,
            id, i, id, i, width, tname,
            id, i, width, tname, width, tname, id, i, float_size * width);

        if (i == 0)
            offset += (4 * float_size + 3 * 4) * width;
        else
            offset += tsize * width;
    }

    buffer.put("    ; -------------------\n\n");
}

