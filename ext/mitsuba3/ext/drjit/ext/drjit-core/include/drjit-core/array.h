/*
    drjit-core/array.h -- Simple C++ array class with operator overloading

    This library implements convenient wrapper class around the C API in
    'drjit/jit.h'.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/traits.h>

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_> struct JitArray {
    using Value = Value_;
    using Mask = JitArray<Backend_, bool>;
    static constexpr VarType Type = var_type<Value>::value;
    static constexpr JitBackend Backend = Backend_;
    static constexpr bool IsArray = true;
    static constexpr bool IsClass =
        std::is_pointer<Value_>::value &&
        std::is_class<typename std::remove_pointer<Value_>::type>::value;
	template <typename T> using ReplaceValue = JitArray<Backend_, T>;
    using ActualValue = typename std::conditional<IsClass, uint32_t, Value>::type;

    JitArray() = default;

    ~JitArray() { jit_var_dec_ref_ext(m_index); }

    JitArray(const JitArray &a) : m_index(a.m_index) {
        jit_var_inc_ref_ext(m_index);
    }

    template <typename T> JitArray(const JitArray<Backend_, T> &v) {
        m_index = jit_var_new_cast(v.index(), Type, 0);
    }

    JitArray(JitArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <bool B = IsClass, enable_if_t<B> = 0>
    JitArray(Value value) {
        uint32_t av = jit_registry_get_id(Backend, (void *) (uintptr_t) value);
        m_index = jit_var_new_literal(Backend, Type, &av, 1, 0, IsClass);
    }

    template <bool B = IsClass, enable_if_t<!B> = 0>
    JitArray(Value value) {
        m_index = jit_var_new_literal(Backend, Type, &value, 1, 0, IsClass);
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 && IsClass)> = 0>
    JitArray(Ts&&... ts) {
        uint32_t data[] = { jit_registry_get_id(Backend,  ts)... };
        m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                   sizeof...(Ts));
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 && !IsClass)> = 0>
    JitArray(Ts&&... ts) {
        Value data[] = { (Value) ts... };
        m_index = jit_var_mem_copy(Backend, AllocType::Host, Type, data,
                                   sizeof...(Ts));
    }

    JitArray &operator=(const JitArray &a) {
        jit_var_inc_ref_ext(a.m_index);
        jit_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    JitArray &operator=(JitArray &&a) {
        uint32_t tmp = m_index;
        m_index = a.m_index;
        a.m_index = tmp;
        return *this;
    }

    JitArray operator-() const {
        return steal(jit_var_new_op_1(JitOp::Neg, m_index));
    }

    JitArray operator~() const {
        return steal(jit_var_new_op_1(JitOp::Not, m_index));
    }

    JitArray operator+(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Add, m_index, v.m_index));
    }

    JitArray operator-(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Sub, m_index, v.m_index));
    }

    JitArray operator*(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Mul, m_index, v.m_index));
    }

    JitArray operator/(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Div, m_index, v.m_index));
    }

    JitArray operator%(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Mod, m_index, v.m_index));
    }

    Mask operator>(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(JitOp::Gt, m_index, v.m_index));
    }

    Mask operator>=(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(JitOp::Ge, m_index, v.m_index));
    }

    Mask operator<(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(JitOp::Lt, m_index, v.m_index));
    }

    Mask operator<=(const JitArray &v) const {
        return Mask::steal(jit_var_new_op_2(JitOp::Le, m_index, v.m_index));
    }

    friend Mask eq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_new_op_2(JitOp::Eq, v1.m_index, v2.m_index));
    }

    friend Mask neq(const JitArray &v1, const JitArray &v2) {
        return Mask::steal(jit_var_new_op_2(JitOp::Neq, v1.m_index, v2.m_index));
    }

    bool operator==(const JitArray &v) const {
        return all(eq(*this, v));
    }

    bool operator!=(const JitArray &v) const {
        return any(neq(*this, v));
    }

    template <typename T, enable_if_t<std::is_same<T, Value_>::value ||
                                      std::is_same<T, bool>::value> = 0>
    JitArray operator|(const JitArray<Backend_, T> &v) const {
        return steal(jit_var_new_op_2(JitOp::Or, m_index, v.index()));
    }

    template <typename T, enable_if_t<std::is_same<T, Value_>::value ||
                                      std::is_same<T, bool>::value> = 0>
    JitArray operator&(const JitArray<Backend_, T> &v) const {
        return steal(jit_var_new_op_2(JitOp::And, m_index, v.index()));
    }

    JitArray operator^(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Xor, m_index, v.m_index));
    }

    JitArray operator<<(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Shl, m_index, v.m_index));
    }

    JitArray operator>>(const JitArray &v) const {
        return steal(jit_var_new_op_2(JitOp::Shr, m_index, v.m_index));
    }

    JitArray &operator+=(const JitArray &v) { return operator=(*this + v); }
    JitArray &operator-=(const JitArray &v) { return operator=(*this - v); }
    JitArray &operator*=(const JitArray &v) { return operator=(*this * v); }
    JitArray &operator/=(const JitArray &v) { return operator=(*this / v); }
    template <typename T = Value_, enable_if_t<!std::is_same<T, bool>::value> = 0>
    JitArray &operator|=(const JitArray &v) { return operator=(*this | v); }
    template <typename T = Value_, enable_if_t<!std::is_same<T, bool>::value> = 0>
    JitArray &operator&=(const JitArray &v) { return operator=(*this & v); }
    JitArray &operator|=(const JitArray<Backend, bool> &v) { return operator=(*this | v); }
    JitArray &operator&=(const JitArray<Backend, bool> &v) { return operator=(*this & v); }
    JitArray &operator^=(const JitArray &v) { return operator=(*this ^ v); }
    JitArray& operator<<=(const JitArray &v) { return operator=(*this << v); }
    JitArray& operator>>=(const JitArray &v) { return operator=(*this >> v); }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator&&(const JitArray &v) const {
        return operator&(v);
    }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator||(const JitArray &v) const {
        return operator|(v);
    }

    template <typename V = Value, enable_if_t<std::is_same<V, bool>::value> = 0>
    JitArray operator!() const {
        return operator~();
    }

    JitArray& schedule() {
        jit_var_schedule(m_index);
        return *this;
    }

    const JitArray& schedule() const {
        jit_var_schedule(m_index);
        return *this;
    }

    JitArray& eval() {
        jit_var_eval(m_index);
        return *this;
    }

    const JitArray& eval() const {
        jit_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }

    size_t size() const {
        return jit_var_size(m_index);
    }

	void resize(size_t size) {
        uint32_t index = jit_var_resize(m_index, size);
        jit_var_dec_ref_ext(m_index);
        m_index = index;
    }

    uint32_t index() const {
        return m_index;
    }

    uint32_t* index_ptr() { return &m_index; }

    const char *str() const {
        return jit_var_str(m_index);
    }

    const Value *data() const {
        return (const Value *) jit_var_ptr(m_index);
    }

    Value *data() {
        return (Value *) jit_var_ptr(m_index);
    }

    Value read(size_t offset) const {
        Value out;
        jit_var_read(m_index, offset, &out);
        return out;
    }

    void write(size_t offset, Value value) {
        uint32_t index = jit_var_write(m_index, offset, &value);
        jit_var_dec_ref_ext(m_index);
        m_index = index;
    }

    static JitArray map(void *ptr, size_t size, bool free = false) {
        return steal(
            jit_var_mem_map(Backend, Type, ptr, size, free ? 1 : 0));
    }

    static JitArray copy(const void *ptr, size_t size) {
        return steal(
            jit_var_mem_copy(Backend, AllocType::Host, Type, ptr, size));
    }

    static JitArray steal(uint32_t index) {
        JitArray result;
        result.m_index = index;
        return result;
    }

    static JitArray<Backend_, uint32_t> counter(size_t size) {
        return JitArray<Backend_, uint32_t>::steal(
            jit_var_new_counter(Backend, size));
    }

    // ------------------------------------------------------

    friend JitArray abs(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Abs, v.m_index));
    }

    friend JitArray sqrt(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Sqrt, v.m_index));
    }

    friend JitArray ceil(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Ceil, v.m_index));
    }

    friend JitArray floor(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Floor, v.m_index));
    }

    friend JitArray round(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Round, v.m_index));
    }

    friend JitArray trunc(const JitArray &v) {
        return steal(jit_var_new_op_1(JitOp::Trunc, v.m_index));
    }

    friend JitArray fmadd(const JitArray &a, const JitArray &b,
                          const JitArray &c) {
        return steal(
            jit_var_new_op_3(JitOp::Fmadd, a.m_index, b.m_index, c.m_index));
    }

    friend JitArray select(const Mask &a, const JitArray &b,
                           const JitArray &c) {
        return steal(
            jit_var_new_op_3(JitOp::Select, a.index(), b.m_index, c.m_index));
    }

    friend JitArray min(const JitArray &a, const JitArray &b) {
        return steal(
            jit_var_new_op_2(JitOp::Min, a.m_index, b.m_index));
    }

    friend JitArray max(const JitArray &a, const JitArray &b) {
        return steal(
            jit_var_new_op_2(JitOp::Max, a.m_index, b.m_index));
    }

    friend JitArray hsum(const JitArray &v) {
        return steal(jit_var_reduce(v.m_index, ReduceOp::Add));
    }

    friend JitArray hmul(const JitArray &v) {
        return steal(jit_var_reduce(v.m_index, ReduceOp::Mul));
    }

    friend JitArray hmin(const JitArray &v) {
        return steal(jit_var_reduce(v.m_index, ReduceOp::Min));
    }

    friend JitArray hmax(const JitArray &v) {
        return steal(jit_var_reduce(v.m_index, ReduceOp::Max));
    }

    friend bool all(const JitArray &a) { return jit_var_all(a.m_index); }
    friend bool any(const JitArray &a) { return jit_var_any(a.m_index); }
    friend bool none(const JitArray &a) { return !jit_var_any(a.m_index); }

	friend const char *label(const JitArray &v) {
		return jit_var_label(v.m_index);
	}

	friend void set_label(JitArray &v, const char *label) {
		uint32_t index = jit_var_set_label(v.m_index, label);
		jit_var_dec_ref_ext(v.m_index);
		v.m_index = index;
	}
protected:
    uint32_t m_index = 0;
};

template <typename Array>
Array empty(size_t size) {
    size_t byte_size = size * sizeof(typename Array::Value);
    void *ptr =
        jit_malloc(Array::Backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync,
                   byte_size);
    return Array::steal(
        jit_var_map_mem(Array::Backend, Array::Type, ptr, size, 1));
}

template <typename Array>
Array zero(size_t size = 1) {
    typename Array::Value value = 0;
    return Array::steal(
        jit_var_new_literal(Array::Backend, Array::Type, &value, size));
}

template <typename Array>
Array full(const typename Array::Value &value, size_t size = 1) {
    return Array::steal(
        jit_var_new_literal(Array::Backend, Array::Type, &value, size, false));
}

template <typename Array>
Array opaque(const typename Array::Value &value, size_t size = 1) {
    return Array::steal(
        jit_var_new_literal(Array::Backend, Array::Type, &value, size, true));
}

template <typename Array, typename Index>
Array gather(const Array &source, const JitArray<Array::Backend, Index> index,
             const JitArray<Array::Backend, bool> &mask = true) {
    return Array::steal(
        jit_var_new_gather(source.index(), index.index(), mask.index()));
}

template <typename Array, typename Index>
void scatter(Array &target, const Array &value, const JitArray<Array::Backend, Index> index,
             const JitArray<Array::Backend, bool> &mask = true) {
    target = Array::steal(jit_var_new_scatter(target.index(), value.index(),
                                              index.index(), mask.index(),
                                              ReduceOp::None));
}

template <typename Array, typename Index>
void scatter_reduce(ReduceOp op, Array &target, const Array &value,
                    const JitArray<Array::Backend, Index> index,
                    const JitArray<Array::Backend, bool> &mask = true) {
    target = Array::steal(jit_var_new_scatter(target.index(), value.index(),
                                              index.index(), mask.index(), op));
}

template <typename Array>
Array arange(size_t start, size_t stop, size_t step) {
    using UInt32 = typename Array::template ReplaceValue<uint32_t>;
    using Value = typename Array::Value;

    size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);

    return fmadd(Array(UInt32::counter(size)), Array((Value) step),
                 Array((Value) start));
}

template <typename Array> Array arange(size_t size) {
    return arange<Array>(0, size, 1);
}

template <typename Array>
Array linspace(typename Array::Value min, typename Array::Value max, size_t size) {
    using UInt32 = typename Array::template ReplaceValue<uint32_t>;
    using Value = typename Array::Value;

    Value step = (max - min) / Value(size - 1);
    return fmadd(Array(UInt32::counter(size)), Array(step), Array(min));
}

#if 0
template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
void jit_var_schedule(Args&&... args) {
    bool unused[] = { (jit_var_schedule(args), false)..., false };
    (void) unused;
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 0)> = 0>
void jit_eval(Args&&... args) {
    jit_var_schedule(args...);
    if (sizeof...(Args) > 0)
        ::jit_eval();
}
#endif

template <typename T> using CUDAArray = JitArray<JitBackend::CUDA, T>;
template <typename T> using LLVMArray = JitArray<JitBackend::LLVM, T>;

NAMESPACE_END(drjit)
