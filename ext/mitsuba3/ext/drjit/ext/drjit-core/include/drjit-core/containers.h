/*
    drjit-core/containers.h -- Tiny self-contained unique_ptr/vector/tuple

    unique_ptr/vector/tuple are used by the Dr.Jit parent project and some test
    cases in this repository. Unfortunately, the std::... versions of these
    containers pull in ~800KB / 31K LOC of headers into *every compile unit*,
    which is insane. This file satisifies all needs with < 5KB and 170 LOC.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <utility>

NAMESPACE_BEGIN(drjit)

template <typename T> struct dr_unique_ptr {
    using Type = std::remove_extent_t<T>;

    dr_unique_ptr() = default;
    dr_unique_ptr(const dr_unique_ptr &) = delete;
    dr_unique_ptr &operator=(const dr_unique_ptr &) = delete;
    dr_unique_ptr(Type *data) : m_data(data) { }
    dr_unique_ptr(dr_unique_ptr &&other) : m_data(other.m_data) {
        other.m_data = nullptr;
    }
    ~dr_unique_ptr() { reset(); }

    dr_unique_ptr &operator=(dr_unique_ptr &&other) {
        reset();
        m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    void reset() noexcept(true) {
        if constexpr (std::is_array_v<T>)
            delete[] m_data;
        else
            delete m_data;
        m_data = nullptr;
    }

    Type& operator[](size_t index) { return m_data[index]; }
    const Type& operator[](size_t index) const { return m_data[index]; }

    Type* get() { return m_data; }
    const Type* get() const { return m_data; }
    Type* operator->() { return m_data; }
    const Type* operator->() const { return m_data; }

    Type* release () {
        Type *tmp = m_data;
        m_data = nullptr;
        return tmp;
    }

protected:
    Type *m_data = nullptr;
};

template <typename T> struct dr_vector {
    dr_vector() = default;
    dr_vector(const dr_vector &v)
        : m_data(new T[v.m_size]), m_size(v.m_size), m_capacity(v.m_size) {
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = v.m_data[i];
    }
    dr_vector &operator=(const dr_vector &) = delete;
    dr_vector(dr_vector &&) = default;
    dr_vector &operator=(dr_vector &&) = default;
    dr_vector(size_t size, const T &value)
        : m_data(new T[size]), m_size(size), m_capacity(size) {
        for (size_t i = 0; i < size; ++i)
            m_data[i] = value;
    }
    dr_vector(const T *start, const T *end) {
        m_size = m_capacity = end-start;
        m_data = new T[end - start];
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = start[i];
    }

    void push_back(const T &value) {
        if (m_size >= m_capacity)
            expand();
        m_data[m_size++] = value;
    }

    void clear() { m_size = 0; }
    size_t size() const { return m_size; }
    T *data() { return m_data.get(); }
    const T *data() const { return m_data.get(); }

    void expand() {
        size_t capacity_new = m_capacity == 0 ? 1 : (m_capacity * 2);
        dr_unique_ptr<T[]> data_new(new T[capacity_new]);
        for (size_t i = 0; i < m_size; ++i)
            data_new[i] = m_data[i];
        m_data = std::move(data_new);
        m_capacity = capacity_new;
    }

    T &operator[](size_t i) { return m_data[i]; }
    const T &operator[](size_t i) const { return m_data[i]; }

protected:
    dr_unique_ptr<T[]> m_data;
    size_t m_size = 0;
    size_t m_capacity = 0;
};

struct dr_index_vector : dr_vector<uint32_t> {
    using Base = dr_vector<uint32_t>;
    using Base::Base;
    using Base::operator=;

    dr_index_vector(size_t size) : Base(size, 0) { }
    ~dr_index_vector() { clear(); }

    void push_back(uint32_t value) {
        jit_var_inc_ref_ext_impl(value);
        Base::push_back(value);
    }

    void clear() {
        for (size_t i = 0; i < size(); ++i)
            jit_var_dec_ref_ext_impl(operator[](i));
        Base::clear();
    }
};

// Tiny self-contained tuple to avoid having to import 1000s of LOC from <tuple>
template <typename... Ts> struct dr_tuple;
template <> struct dr_tuple<> {
    template <size_t> using type = void;
};

template <typename T, typename... Ts> struct dr_tuple<T, Ts...> : dr_tuple<Ts...> {
    using Base = dr_tuple<Ts...>;

    dr_tuple() = default;
    dr_tuple(const dr_tuple &) = default;
    dr_tuple(dr_tuple &&) = default;
    dr_tuple& operator=(dr_tuple &&) = default;
    dr_tuple& operator=(const dr_tuple &) = default;

    dr_tuple(const T& value, const Ts&... ts)
        : Base(ts...), value(value) { }

    dr_tuple(T&& value, Ts&&... ts)
        : Base(std::move(ts)...), value(std::move(value)) { }

    template <size_t I> auto& get() {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I> const auto& get() const {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I>
    using type =
        std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

private:
    T value;
};

template <typename... Ts> dr_tuple(Ts &&...) -> dr_tuple<std::decay_t<Ts>...>;

NAMESPACE_END(drjit)
