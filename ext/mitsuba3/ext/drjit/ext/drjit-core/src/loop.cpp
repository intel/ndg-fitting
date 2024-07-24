#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "op.h"
#include "profiler.h"
#include <tsl/robin_set.h>

struct Loop {
    // A descriptive name
    char *name = nullptr;
    // Backend targeted by this loop
    JitBackend backend;
    // Variable index of loop initialization node
    uint32_t init = 0;
    // Variable index of loop end node
    uint32_t end = 0;
    /// Variable index of loop condition
    uint32_t cond = 0;
    /// Variable index of side effect placeholder
    uint32_t se = 0;
    /// Number of side effects
    uint32_t se_count = 0;
    /// Storage size in bytes for all variables before simplification
    uint32_t storage_size_initial = 0;
    /// Input variables before loop
    std::vector<uint32_t> in;
    /// Input variables before branch condition
    std::vector<uint32_t> in_cond;
    /// Input variables before loop body
    std::vector<uint32_t> in_body;
    /// Output placeholder variables within loop
    std::vector<uint32_t> out_body;
    /// Output variables after loop
    std::vector<uint32_t> out;
    /// Are there unused loop variables that could be stripped away?
    bool simplify = false;
};

static std::vector<Loop *> loops;

// Forward declarations
static void jitc_var_loop_callback(uint32_t index, int free, void *ptr);
static void jitc_var_loop_assemble_init(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_cond(const Variable *v, const Extra &extra);
static void jitc_var_loop_assemble_end(const Variable *v, const Extra &extra);

/// Create a variable that wraps another, optionally with an extra dependency
static void wrap(Variable &v, uint32_t &index, uint32_t dep = 0) {
    Variable *v2 = jitc_var(index);
    v.type = v2->type;
    v.dep[0] = index;
    v.dep[1] = dep;
    jitc_var_inc_ref_int(index, v2);
    jitc_var_dec_ref_ext(index, v2);
    jitc_var_inc_ref_int(dep);
    index = 0; // in case jitc_var_new throws
    index = jitc_var_new(v, true);
}

uint32_t jitc_var_loop_init(size_t n_indices, uint32_t **indices) {
    if (n_indices == 0)
        jitc_raise("jit_var_loop_init(): no loop state variables specified!");

    // Determine the variable size
    JitBackend backend = (JitBackend) jitc_var(*indices[0])->backend;
    uint32_t size = 0;
    bool dirty = false;

    for (size_t i = 0; i < n_indices; ++i) {
        const Variable *v = jitc_var(*indices[i]);
        uint32_t vsize = v->size;
        if (size != 0 && vsize != 1 && size != 1 && vsize != size)
            jitc_raise("jit_var_loop_init(): loop state variables have an "
                       "inconsistent size (%u vs %u)!", vsize, size);
        if (vsize > size)
            size = vsize;

        dirty |= (bool) v->ref_count_se;
    }

    // Ensure side effects are fully processed
    if (dirty) {
        jitc_eval(thread_state(backend));
        dirty = false;
        for (size_t i = 0; i < n_indices; ++i)
            dirty |= (bool) jitc_var(*indices[i])->ref_count_se;
        if (unlikely(dirty))
            jitc_raise(
                "jit_var_loop_init(): inputs remain dirty after evaluation!");
    }

    Variable v;
    v.size = size;
    v.placeholder = 1;
    v.backend = (uint32_t) backend;

    // Copy loop state before entering loop (CUDA)
    if (backend == JitBackend::CUDA) {
        v.stmt = (char *) "mov.$t0 $r0, $r1";
        for (size_t i = 0; i < n_indices; ++i)
            wrap(v, *indices[i]);
    }

    // Create a special node indicating the loop start
    Ref result = steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 0, nullptr));
    jitc_var(result)->size = size;

    // Create Phi nodes (LLVM)
    if (backend == JitBackend::LLVM) {
        v.stmt = (char *) "$r0 = phi <$w x $t0> [ $r0_final, %l_$i2_tail ], "
                          "[ $r1, %l_$i2_start ]";
        for (size_t i = 0; i < n_indices; ++i)
            wrap(v, *indices[i], result);
    }

    jitc_log(Debug, "jit_var_loop_init(r%u, n_indices=%zu, size=%u)",
             (uint32_t) result, n_indices, size);

    return result.release();
}

uint32_t jitc_var_loop_cond(uint32_t loop_init, uint32_t cond,
                            size_t n_indices, uint32_t **indices) {
    uint32_t size;
    JitBackend backend;
    {
        Variable *v = jitc_var(loop_init);
        size = v->size;
        backend = (JitBackend) v->backend;
    }

    uint32_t dep[2] = { cond, loop_init };

    Ref result =
        steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 2, dep));

    Variable v;
    v.size = size;
    v.placeholder = 1;
    v.backend = (uint32_t) backend;

    // Create Phi nodes to represent state at the beginning of the loop body
    v.stmt = (char *) (backend == JitBackend::LLVM
                 ? "$r0 = phi <$w x $t0> [ $r1, %l_$i2_cond ]"
                 : "mov.$t0 $r0, $r1");

    for (size_t i = 0; i < n_indices; ++i)
        wrap(v, *indices[i], loop_init);

    jitc_log(Debug, "jit_var_loop_cond(r%u, loop_init=r%u, cond=r%u)",
             (uint32_t) result, loop_init, cond);

    return result.release();
}

uint32_t jitc_var_loop(const char *name, uint32_t loop_init,
                       uint32_t loop_cond, size_t n_indices,
                       uint32_t *indices_in, uint32_t **indices,
                       uint32_t checkpoint, int first_round) {
    if (n_indices == 0)
        jitc_raise("jit_var_loop(): no loop state variables specified!");

    JitBackend backend = (JitBackend) jitc_var(loop_init)->backend;
    ThreadState *ts = thread_state(backend);

    const uint32_t checkpoint_mask = 0x7fffffff;
    bool optimize = jitc_flags() & (uint32_t) JitFlag::LoopOptimize;
    auto &se = ts->side_effects_recorded;

    checkpoint &= checkpoint_mask;
    if (unlikely(checkpoint > se.size()))
        jitc_raise("jit_var_loop(): 'checkpoint' parameter is out of bounds");

    // Allocate a data structure that will capture all loop-related information
    std::unique_ptr<Loop> loop(new Loop());
    loop->backend = backend;
    loop->in.reserve(n_indices);
    loop->in_body.reserve(n_indices);
    loop->in_cond.reserve(n_indices);
    loop->out_body.reserve(n_indices);
    loop->out.reserve(n_indices);
    loop->name = strdup(name);
    loop->se_count = (uint32_t) se.size() - checkpoint;
    loop->init = loop_init;
    loop->cond = jitc_var(loop_cond)->dep[0];

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    bool placeholder = false;
    uint32_t size = 1, n_invariant = 0;
    {
        const Variable *v = jitc_var(loop->cond);
        if ((VarType) v->type != VarType::Bool)
            jitc_raise("jit_var_loop(): loop condition must be a boolean variable");
        if (!v->placeholder)
            jitc_raise("jit_var_loop(): loop condition does not depend on any of the loop variables");
        size = v->size;
    }

    for (size_t i = 0; i < n_indices; ++i) {
        // ============= 1.1. Input side =============

        uint32_t index_1 = indices_in[i];
        Variable *v1 = jitc_var(index_1);
        loop->storage_size_initial += type_size[v1->type];

        if (optimize && !first_round && v1->dep[1] != loop_init) {
            loop->in_body.push_back(0);
            loop->in_cond.push_back(0);
            loop->in.push_back(0);
            loop->out_body.push_back(0);
            n_invariant++;
            continue;
        }

        if (!v1->placeholder || !v1->dep[0] || v1->dep[1] != loop_init)
            jitc_raise("jit_var_loop(): loop state input variable %zu (r%u) is "
                       "invalid (case 1)!", i, index_1);

        uint32_t index_2 = v1->dep[0];
        Variable *v2 = jitc_var(index_2);

        if (!v2->placeholder || !v2->dep[0])
            jitc_raise("jit_var_loop(): loop state input variable %zu (r%u) is "
                       "invalid (case 2)!", i, index_2);

        uint32_t index_3 = v2->dep[0];
        Variable *v3 = jitc_var(index_3);

        loop->in_body.push_back(index_1);
        loop->in_cond.push_back(index_2);
        loop->in.push_back(index_3);
        placeholder |= (bool) v3->placeholder;

        // ============ 1.2. Output side =============

        uint32_t index_o = *indices[i];
        loop->out_body.push_back(index_o);

        const Variable *vo = jitc_var(index_o);

        if (size != v1->size && size != 1 && v1->size != 1)
            jitc_raise("jit_var_loop(): initial shape of loop state variable %zu (r%u) "
                       "is incompatible with the loop (%u vs %u entries)!",
                       i, index_1, size, v1->size);

        size = std::max(size, v1->size);

        if (size != vo->size && size != 1 && vo->size != 1)
            jitc_raise("jit_var_loop(): final shape of loop state variable %zu (r%u) "
                       "is incompatible with the loop (%u vs %u entries)!",
                       i, index_o, size, vo->size);
        size = std::max(size, vo->size);

        // =========== 1.3. Optimizations ============

        if (optimize && first_round) {
            bool eq_literal =
                     v3->literal && vo->literal && v3->value == vo->value,
                 unchanged = index_o == index_1;

            if (eq_literal || unchanged) {
                if (backend == JitBackend::LLVM) {
                    // Rewrite the previously generated phi expression. Needed
                    // in case the loop condition depends on a loop-invariant
                    // variable (see loop test 09_optim_cond)
                    v2->stmt = (char *) "$r0 = phi <$w x $t0> [ $r0, "
                                        "%l_$i2_tail ], [ $r1, %l_$i2_start ]";
                }
                jitc_var_inc_ref_ext(index_3);
                jitc_var_dec_ref_ext(index_1);
                indices_in[i] = index_3;
                n_invariant++;
            }
        }
    }

    char temp[256];
    temp[0] = '\0';
    if (n_invariant)
        snprintf(temp, sizeof(temp),
                 first_round
                     ? ", %u loop-invariant variables detected, recording "
                       "loop once more to optimize further.."
                     : ", %u loop-invariant variables eliminated",
                 n_invariant);

    jitc_log(InfoSym,
             "jit_var_loop(loop_init=r%u, loop_cond=r%u): loop (\"%s\") with "
             "%zu loop variable%s, %u side effect%s, %u elements%s%s",
             loop_init, loop_cond, name, n_indices, n_indices == 1 ? "" : "s",
             loop->se_count, loop->se_count == 1 ? "" : "s", size, temp,
             placeholder ? " (part of a recorded computation)" : "");

    if (n_invariant && first_round) {
        // Release recorded computation
        for (size_t i = 0; i < n_indices; ++i) {
            jitc_var_inc_ref_ext(indices_in[i]);
            jitc_var_dec_ref_ext(*indices[i]);
            *indices[i] = indices_in[i];
        }

        // Release side effects
        while (checkpoint != se.size()) {
            jitc_var_dec_ref_ext(se.back());
            se.pop_back();
        }

        return (uint32_t) -1; // record loop once more
    }

    // =====================================================
    // 2. Configure & label (GraphViz) variables
    // =====================================================

    for (size_t i = 0; i < n_indices; ++i) {
        if (!loop->in_body[i])
            continue;

        const char *label = jitc_var_label(loop->in[i]);

        snprintf(temp, sizeof(temp), "%s%sLoop (%s) [in %zu, body]",
                 label ? label : "", label ? ", " : "", name, i);
        jitc_var_set_label(loop->in_body[i], temp);

        snprintf(temp, sizeof(temp), "%s%sLoop (%s) [in %zu, cond]",
                 label ? label : "", label ? ", " : "", name, i);

        jitc_var_set_label(loop->in_cond[i], temp);
    }

    {
        // Set a label and custom code generation hook
        Variable *v = jitc_var(loop_init);
        v->extra = 1;
        Extra &e = state.extra[loop_init];
        e.assemble = jitc_var_loop_assemble_init;

        snprintf(temp, sizeof(temp), "Loop (%s) [init]", name);
        jitc_var_set_label(loop_init, temp);
    }

    {
        // Set a label and custom code generation hook
        Variable *v = jitc_var(loop_cond);
        v->extra = 1;
        Extra &e = state.extra[loop_cond];
        e.assemble = jitc_var_loop_assemble_cond;
        e.callback_data = loop.get();

        snprintf(temp, sizeof(temp), "Loop (%s) [cond]", name);
        jitc_var_set_label(loop_cond, temp);
    }

    // =====================================================
    // 3. Create variable representing the end of the loop
    // =====================================================

    uint32_t loop_end_dep[2] = { loop_init, loop_cond };

    Ref loop_end = steal(
        jitc_var_new_stmt(backend, VarType::Void, "", 1, 2, loop_end_dep));
    loop->end = loop_end;

    {
        // Set a label and custom code generation hook
        Variable *v = jitc_var(loop_end);
        v->extra = 1;
        Extra &e = state.extra[loop_end];
        e.assemble = jitc_var_loop_assemble_end;
        e.callback_data = loop.get();

        /* A loop can create complex variable dependencies across iterations
           that are not directly visible in the dependency structure maintained
           by Dr.Jit. To prevent variables from premature garbage collection, the
           loop end variable references the state variables at two points:

           1. before the loop condition
           2. after the loop body

           The jitc_var_loop_simplify() routine analyzes cross-iteration
           dependencies to remove any dead variables/code. */

        e.n_dep = (uint32_t) (2 * n_indices);
        e.dep = (uint32_t *) malloc_check(e.n_dep * sizeof(uint32_t));

        memcpy(e.dep, loop->out_body.data(), n_indices * sizeof(uint32_t));
        memcpy(e.dep + n_indices, loop->in_cond.data(),
               n_indices * sizeof(uint32_t));

        for (size_t i = 0; i < n_indices; ++i) {
            jitc_var_inc_ref_int(loop->out_body[i]);
            jitc_var_inc_ref_int(loop->in_cond[i]);
        }

        snprintf(temp, sizeof(temp), "Loop (%s) [end]", name);
        jitc_var_set_label(loop_end, temp);
    }

    // =====================================================
    // 4. Create variable representing side effects
    // =====================================================

    Ref loop_se;
    if (loop->se_count) {
        uint32_t *dep = (uint32_t *) malloc_check(loop->se_count * sizeof(uint32_t));
        for (uint32_t i = 0; i < loop->se_count; ++i) {
            uint32_t index = se[se.size() - loop->se_count + i];
            Variable *v = jitc_var(index);
            v->side_effect = false;
            jitc_var_inc_ref_int(index, v);
            jitc_var_dec_ref_ext(index, v);
            snprintf(temp, sizeof(temp), "Loop (%s) [side effects]", name);
            dep[i] = index;
        }
        se.resize(checkpoint);

        uint32_t loop_se_dep[1] = { loop_end };
        loop_se = steal(
            jitc_var_new_stmt(backend, VarType::Void, "", 1, 1, loop_se_dep));

        // Set a label and custom code generation hook
        Variable *v = jitc_var(loop_se);
        v->extra = 1;
        v->size = size;
        v->placeholder = placeholder;
        Extra &e = state.extra[loop_se];
        e.n_dep = loop->se_count;
        e.dep = dep;

        snprintf(temp, sizeof(temp), "Loop (%s) [side effects]", name);
        jitc_var_set_label(loop_se, temp);
        loop->se = loop_se;
    }

    // =====================================================
    // 5. Create output variables
    // =====================================================

    {
        Variable v2;
        if (backend == JitBackend::CUDA)
            v2.stmt = (char *) "mov.$t0 $r0, $r1";
        else
            v2.stmt = (char *) "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>";
        v2.size = size;
        v2.placeholder = placeholder;
        v2.backend = (uint32_t) backend;
        v2.dep[1] = loop_end;

        for (size_t i = 0; i < n_indices; ++i) {
            uint32_t index = loop->out_body[i];

            if (!index) { // Optimized away
                loop->out.push_back(0);
                uint32_t index_2 = indices_in[i];
                jitc_var_inc_ref_ext(index_2);
                jitc_var_dec_ref_ext(*indices[i]);
                *indices[i] = index_2;
                continue;
            }

            v2.type = jitc_var(index)->type;
            v2.dep[0] = loop->in_cond[i];
            jitc_var_inc_ref_int(loop->in_cond[i]);
            jitc_var_inc_ref_int(loop_end);

            uint32_t index_2 = jitc_var_new(v2, true);
            loop->out.push_back(index_2);
            jitc_var_dec_ref_ext(*indices[i]);
            *indices[i] = index_2;

            const char *label = jitc_var_label(loop->in[i]);
            snprintf(temp, sizeof(temp), "%s%sLoop (%s) [out %zu]",
                     label ? label : "", label ? ", " : "", name, i);
            jitc_var_set_label(index_2, temp);

            if (optimize) {
                Extra &e = state.extra[index_2];
                e.callback = jitc_var_loop_callback;
                e.callback_data = loop.get();
                e.callback_internal = true;
            }
        }
    }

    // =====================================================
    // 6. Transfer ownership of loop data structure
    // =====================================================

    {
        Extra &e = state.extra[loop_init];
        e.callback = [](uint32_t, int free_var, void *ptr) {
            if (free_var && ptr) {
                Loop *loop_2 = (Loop *) ptr;
                jitc_trace("jit_var_loop(\"%s\"): freeing loop.", loop_2->name);
                free(loop_2->name);
                loops.erase(std::remove(loops.begin(), loops.end(), loop_2), loops.end());
                delete loop_2;
            }
        };
        e.callback_internal = true;
        loops.push_back(loop.get());
        e.callback_data = loop.release();
    }

    return loop_se.release();
}


static void jitc_var_loop_callback(uint32_t index, int free, void *ptr) {
    if (!ptr)
        return;

    Loop *loop = (Loop *) ptr;

    if (!free) // Disable callback
        state.extra[index].callback_data = nullptr;

    // An output variable is no longer referenced. Find out which one.
    uint32_t n2 = (uint32_t) loop->out.size();
    uint32_t offset = (uint32_t) -1;
    for (uint32_t i = 0; i < n2; ++i) {
        if (loop->out[i] == index) {
            offset = i;
            break;
        }
    }

    if (unlikely(offset == (uint32_t) -1))
        jitc_fail("jit_var_loop_callback(): expired output variable %u could "
                  "not be located!", index);

    loop->out[offset] = 0;

    // Mark the loop for eventual simplification
    jitc_trace("jit_var_loop_callback(%s): variable %u collected, marking loop "
               "for simplification.", loop->name, offset);
    loop->simplify = true;
}

static void jitc_var_loop_dfs(tsl::robin_set<uint32_t, UInt32Hasher> &set, uint32_t lowest_index, uint32_t index) {
    if (!set.insert(index).second)
        return;
    // jitc_trace("jitc_var_dfs(r%u)", index);

    const Variable *v = jitc_var(index);
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index_2 = v->dep[i];
        if (index_2 >= lowest_index)
            jitc_var_loop_dfs(set, lowest_index, index_2);
    }

    if (unlikely(v->extra)) {
        auto it = state.extra.find(index);
        if (unlikely(it == state.extra.end()))
            jitc_fail("jit_var_loop_dfs(): could not find matching 'extra' record!");

        const Extra &extra = it->second;
        for (uint32_t i = 0; i < extra.n_dep; ++i) {
            uint32_t index_2 = extra.dep[i];
            if (index_2 >= lowest_index)
                jitc_var_loop_dfs(set, lowest_index, index_2);
        }
    }
}

static size_t jitc_var_loop_simplify(Loop *loop, tsl::robin_set<uint32_t, UInt32Hasher> &visited) {
    loop->simplify = false;

    if (state.variables.find(loop->end) == state.variables.end())
        return 0;

    const uint32_t n = (uint32_t) loop->in.size();

    /// Determine variable range to be traversed
    uint32_t lowest_index = 0xFFFFFFFF,
             highest_index = 0,
             n_freed = 0;

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = loop->in_cond[i];
        if (index)
            lowest_index = std::min(lowest_index, index);
        index = loop->out_body[i];
        if (index)
            highest_index = std::max(highest_index, index);
    }

    visited.clear();
    if (highest_index > lowest_index)
        visited.reserve(highest_index - lowest_index);

    // Find all inputs that are reachable from the outputs that are still alive
    for (uint32_t i = 0; i < n; ++i) {
        if (!loop->out[i] || !loop->out_body[i])
            continue;
        // jitc_trace("jit_var_loop_simplify(): DFS from %u (r%u)", i, loop->out_body[i]);
        visited.insert(loop->in_cond[i]);
        jitc_var_loop_dfs(visited, lowest_index, loop->out_body[i]);
    }

    // Also search from loop condition
    // jitc_trace("jit_var_loop_simplify(): DFS from loop condition (r%u)", loop->cond);
    jitc_var_loop_dfs(visited, lowest_index, loop->cond);

    // Find all inputs that are reachable from the side effects
    if (loop->se) {
        auto it = state.extra.find(loop->se);
        if (it != state.extra.end()) {
            const Extra &e = it->second;
            for (uint32_t i = 0; i < e.n_dep; ++i) {
                // jitc_trace("jit_var_loop_simplify(): DFS from side effect %u (r%u)", i, e.dep[i]);
                jitc_var_loop_dfs(visited, lowest_index, e.dep[i]);
            }
        }
    }

    /// Propagate until no further changes
    bool again;
    do {
        again = false;
        for (uint32_t i = 0; i < n; ++i) {
            if (loop->in_cond[i] &&
                visited.find(loop->in_cond[i]) != visited.end() &&
                visited.find(loop->out_body[i]) == visited.end()) {
                jitc_var_loop_dfs(visited, lowest_index, loop->out_body[i]);
                again = true;
            }
        }
    } while (again);

    /// Remove loop variables that are never referenced
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t index = loop->in_cond[i];
        if (index == 0 || visited.find(index) != visited.end())
            continue;
        n_freed++;

        jitc_trace("jit_var_loop_simplify(): freeing unreferenced loop "
                   "variable %u (r%u -> r%u -> r%u)", i, loop->in[i],
                   loop->in_cond[i], loop->in_body[i]);

        Extra &e_end = state.extra[loop->end];
        if (unlikely(e_end.dep[n + i] != loop->in_cond[i]))
            jitc_fail("jit_var_loop_simplify: internal error (3)");
        if (unlikely(e_end.dep[i] != loop->out_body[i]))
            jitc_fail("jit_var_loop_simplify: internal error (2)");
        e_end.dep[i] = e_end.dep[n + i] = 0;

        jitc_var_dec_ref_int(loop->in_cond[i]);
        jitc_var_dec_ref_int(loop->out_body[i]);

        loop->in[i] = loop->in_cond[i] = loop->in_body[i] = loop->out_body[i] = 0;
    }

    jitc_log(InfoSym,
             "jit_var_loop_simplify(\"%s\"): freed %u loop variables, visited "
             "%zu variables.",
             loop->name, n_freed, visited.size());

    return n_freed;
}

static ProfilerRegion profiler_region_var_loop_simplify("jit_var_loop_simplify");

void jitc_var_loop_simplify() {
    tsl::robin_set<uint32_t, UInt32Hasher> visited;

    ProfilerPhase profiler(profiler_region_var_loop_simplify);
    bool progress;
    do {
        progress = false;
        for (size_t i = 0; i < loops.size(); ++i) {
            Loop *loop = loops[i];
            if (loop->simplify)
                progress |= jitc_var_loop_simplify(loop, visited) > 0;
        }
    } while (progress);
}

static void jitc_var_loop_assemble_init(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->init)->reg_index;

    if (loop->backend == JitBackend::LLVM) {
        buffer.fmt("    br label %%l_%u_start\n", loop_reg);
        buffer.fmt("\nl_%u_start:\n", loop_reg);
        buffer.fmt("    br label %%l_%u_cond\n", loop_reg);
    }

    buffer.fmt("\nl_%u_cond: %s Loop (%s)\n", loop_reg,
               loop->backend == JitBackend::CUDA ? "//" : ";",
               loop->name);
}

static void jitc_var_loop_assemble_cond(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->init)->reg_index,
             mask_reg = jitc_var(loop->cond)->reg_index,
             width = jitc_llvm_vector_width;

    if (loop->backend == JitBackend::CUDA) {
        buffer.fmt("    @!%%p%u bra l_%u_done;\n", mask_reg, loop_reg);
    } else {
        char global[128];
        snprintf(
            global, sizeof(global),
            "declare i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1>)\n\n",
            width, width);
        jitc_register_global(global);

        buffer.fmt("    %%p%u = call i1 @llvm.experimental.vector.reduce.or.v%ui1(<%u x i1> %%p%u)\n"
                   "    br i1 %%p%u, label %%l_%u_body, label %%l_%u_done\n",
                   loop_reg, width, width, mask_reg, loop_reg, loop_reg, loop_reg);
    }

    buffer.fmt("\nl_%u_body:\n", loop_reg);
}

static void jitc_var_loop_assemble_end(const Variable *, const Extra &extra) {
    Loop *loop = (Loop *) extra.callback_data;
    uint32_t loop_reg = jitc_var(loop->init)->reg_index,
             mask_reg = jitc_var(loop->cond)->reg_index;

    if (loop->backend == JitBackend::LLVM)
        buffer.fmt("    br label %%l_%u_tail\n"
                   "\nl_%u_tail:\n", loop_reg, loop_reg);

    uint32_t n_variables = 0,
             storage_size = 0;

    uint32_t width = jitc_llvm_vector_width;
    for (size_t i = 0; i < loop->in_body.size(); ++i) {
        auto it_in = state.variables.find(loop->in_cond[i]),
             it_out = state.variables.find(loop->out_body[i]);

        if (it_in == state.variables.end())
            continue;
        else if (it_out == state.variables.end())
            jitc_fail("jit_var_loop_assemble_end(): internal error!");

        const Variable *v_in = &it_in->second,
                       *v_out = &it_out->second;
        uint32_t vti = it_in->second.type;

        if (loop->backend == JitBackend::LLVM) {
            buffer.fmt("    %s%u_final = select <%u x i1> %%p%u, <%u x %s> %s%u, "
                       "<%u x %s> %s%u\n",
                       type_prefix[vti], v_in->reg_index, width, mask_reg, width,
                       type_name_llvm[vti], type_prefix[vti], v_out->reg_index,
                       width, type_name_llvm[vti], type_prefix[vti],
                       v_in->reg_index);
        } else {
            buffer.fmt("    mov.%s %s%u, %s%u;\n", type_name_ptx[vti],
                       type_prefix[vti], v_in->reg_index, type_prefix[vti],
                       v_out->reg_index);
        }

        n_variables++;
        storage_size += type_size[vti];
    }

    if (loop->backend == JitBackend::CUDA)
        buffer.fmt("    bra l_%u_cond;\n", loop_reg);
    else
        buffer.fmt("    br label %%l_%u_cond;\n", loop_reg);

    buffer.fmt("\nl_%u_done:\n", loop_reg);

    jitc_log(InfoSym,
             "jit_var_loop_assemble(): loop (\"%s\") with %u/%u loop "
             "variable%s (%u/%u bytes), %u side effect%s",
             loop->name, n_variables, (uint32_t) loop->in.size(),
             n_variables == 1 ? "" : "s", storage_size, loop->storage_size_initial,
             loop->se_count, loop->se_count == 1 ? "" : "s");

    if (loop->se) {
        loop->se_count = 0;
        loop->se = 0;
        loop->simplify = true;
    }
}
