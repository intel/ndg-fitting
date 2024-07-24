#pragma once

#if defined(_MSC_VER)
#  pragma warning (disable:5033) // 'register' is no longer a supported storage class
#endif

#include <nanogui/nanogui.h>
#include <nanogui/opengl.h>

#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <nanogui/python.h>
#include "py_doc.h"

PYBIND11_MAKE_OPAQUE(nanogui::Color)

#define D(...) DOC(nanogui, __VA_ARGS__ )

#define DECLARE_LAYOUT(Name) \
    class Py##Name : public Name { \
    public: \
        using Name::Name; \
        NANOGUI_LAYOUT_OVERLOADS(Name); \
    }

#define DECLARE_WIDGET(Name) \
    class Py##Name : public Name { \
    public: \
        using Name::Name; \
        NANOGUI_WIDGET_OVERLOADS(Name); \
    }

#define DECLARE_SCREEN(Name) \
    class Py##Name : public Name { \
    public: \
        using Name::Name; \
        NANOGUI_WIDGET_OVERLOADS(Name); \
        NANOGUI_SCREEN_OVERLOADS(Name); \
    }

using namespace nanogui;

namespace py = pybind11;
using namespace py::literals;

/// Make pybind aware of the ref-counted wrapper type
PYBIND11_DECLARE_HOLDER_TYPE(T, ref<T>);
