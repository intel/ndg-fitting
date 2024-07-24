cmake-defaults
==============

This repository establishes a standard build environment for RGL projects
including at least the following:

- [Mitsuba 2](https://github.com/mitsuba-renderer/mitsuba2)
- [Dr.Jit](https://github.com/mitsuba-renderer/drjit)
- [Dr.Jit-Core](https://github.com/mitsuba-renderer/drjit-core)
- [nanothread](https://github.com/mitsuba-renderer/nanothread)
- [Struct-JIT](https://github.com/mitsuba-renderer/struct-jit)

Usage
-----

Include as a Git submodule and ``include()`` after having declared the CMake
minimum version, project name, and build options.

```cmake
cmake_minimum_required(VERSION 3.13...3.18)

project(sjit
  DESCRIPTION "Struct-JIT"
  LANGUAGES CXX
)

option(SJIT_ENABLE_PYTHON "Build Python extension library?" ON)
# Other options here

include(ext/cmake-defaults/CMakeLists.txt)
```
