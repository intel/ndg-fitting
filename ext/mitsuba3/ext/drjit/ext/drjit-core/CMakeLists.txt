# ----------------------------------------------------------
# Top-level Dr.Jit-Core CMake file, requires a recent version
# ----------------------------------------------------------
cmake_minimum_required (VERSION 3.13...3.18)

project(drjit-core
  DESCRIPTION
    "Dr.Jit-Core"
  LANGUAGES
    CXX C
)

# ----------------------------------------------------------
#  Optional features available to users
# ----------------------------------------------------------

option(DRJIT_DYNAMIC_LLVM "Resolve LLVM dynamically at run time?" ON)
option(DRJIT_ENABLE_TESTS "Build Dr.Jit-Core test suite?" OFF)

if (NOT APPLE)
  option(DRJIT_DYNAMIC_CUDA "Resolve CUDA dynamically at run time?" ON)
  option(DRJIT_ENABLE_OPTIX "Allow the use of OptiX ray tracing calls in kernels?" ON)
endif()

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanothread/ext/cmake-defaults")
  message(FATAL_ERROR "The Dr.Jit-Core dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

# ----------------------------------------------------------
#  Build defaults for projects by the Realistic Graphics Lab
# ----------------------------------------------------------

include(ext/nanothread/ext/cmake-defaults/CMakeLists.txt)

# ----------------------------------------------------------
#  Print a few messages explaining what will be compiled
# ----------------------------------------------------------

if (DRJIT_DYNAMIC_LLVM)
  message(STATUS "Dr.Jit-Core: LLVM will be loaded dynamically at runtime.")
else()
  message(STATUS "Dr.Jit-Core: linking against LLVM shared libraries.")
endif()

if (NOT APPLE)
  if (DRJIT_DYNAMIC_CUDA)
    message(STATUS "Dr.Jit-Core: CUDA will be loaded dynamically at runtime.")
  else()
    message(STATUS "Dr.Jit-Core: linking against CUDA shared libraries.")
  endif()

  if (DRJIT_ENABLE_OPTIX)
    message(STATUS "Dr.Jit-Core: OptiX support enabled.")
  else()
    message(STATUS "Dr.Jit-Core: OptiX support disabled.")
  endif()
endif()

# ----------------------------------------------------------
#  Build the nanothread library
# ----------------------------------------------------------

add_subdirectory(ext/nanothread)
mark_as_advanced(NANOTHREAD_ENABLE_TESTS)

# ----------------------------------------------------------
#  Build Dr.Jit-Core
# ----------------------------------------------------------

if (DRJIT_ENABLE_OPTIX)
  set(DRJIT_OPTIX_FILES src/optix_api.h src/optix_api.cpp)
endif()

add_library(
  drjit-core SHARED
  include/drjit-core/jit.h
  include/drjit-core/traits.h
  include/drjit-core/array.h

  src/common.h
  src/internal.h
  src/alloc.h
  src/hash.h
  src/profiler.h
  src/log.h           src/log.cpp
  src/var.h           src/var.cpp
  src/op.h            src/op.cpp
  src/malloc.h        src/malloc.cpp
  src/registry.h      src/registry.cpp
  src/util.h          src/util.cpp
  src/cuda_api.h      src/cuda_api.cpp
  src/cuda_tex.h      src/cuda_tex.cpp
  src/llvm_api.h      src/llvm_api.cpp
  src/io.h            src/io.cpp
  src/eval.h          src/eval.cpp
  src/vcall.h         src/vcall.cpp
  src/loop.h          src/loop.cpp
  src/printf.h        src/printf.cpp
  src/eval_cuda.cpp
  src/eval_llvm.cpp
  src/init.cpp
  src/api.cpp

  ${DRJIT_OPTIX_FILES}

  # LZ4 compression library & XXHash hash function
  ext/lz4/lz4.h ext/lz4/lz4.c
  ext/lz4/xxhash.h ext/lz4/xxh3.h ext/lz4/xxhash.c

  # Precompiled kernels in compressed PTX format
  resources/kernels.h resources/kernels.c
)

target_compile_features(drjit-core PRIVATE cxx_std_14)

target_include_directories(drjit-core PRIVATE
  ext/nanothread/include
  ext/robin_map/include
  ext/lz4
)

if (MSVC)
  # Conditional expression is constant (a few in robin_hash.h)
  target_compile_options(drjit-core PRIVATE /wd4127)

  # Allow throwing exceptions even from "extern C" code
  target_compile_options(drjit-core PUBLIC /EHs)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # GCC static analysis incorrectly warns about free() application to Variable::stmt
  target_compile_options(drjit-core PRIVATE -Wno-free-nonheap-object)
  target_link_options(drjit-core PRIVATE -Wno-free-nonheap-object)
endif()

target_compile_definitions(drjit-core PRIVATE -DLZ4LIB_VISIBILITY=)

target_include_directories(drjit-core
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(drjit-core PRIVATE -DDRJIT_BUILD=1)
target_link_libraries(drjit-core PRIVATE nanothread)

if (DRJIT_DYNAMIC_CUDA OR APPLE)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_DYNAMIC_CUDA=1)
else()
  find_package(CUDA REQUIRED)
  target_include_directories(drjit-core PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(drjit-core PRIVATE cuda)
endif()

if (DRJIT_DYNAMIC_LLVM)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_DYNAMIC_LLVM=1)
else()
  find_package(LLVM REQUIRED)
  target_include_directories(drjit-core PRIVATE ${LLVM_INCLUDE_DIRS})
  target_link_libraries(drjit-core PRIVATE LLVM)
endif()

if (DRJIT_ENABLE_OPTIX)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_OPTIX=1)
  if (WIN32)
    target_link_libraries(drjit-core PRIVATE Cfgmgr32)
  endif()
endif()

# Generate ITT notifications (e.g. for VTune) if part of a larger project
# that provides an 'ittnotify' target
if (TARGET ittnotify)
  target_link_libraries(drjit-core PRIVATE ittnotify)
  target_include_directories(drjit-core PRIVATE ${ITT_INCLUDE_DIRS})
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_ITTNOTIFY=1)
endif()

# Similar notifications for NVIDIA profilers
if (DRJIT_ENABLE_NVTX)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_ENABLE_NVTX=1)
endif()

if (UNIX)
  target_link_libraries(drjit-core PRIVATE dl pthread)
endif()

set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE        TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL     TRUE)
set_target_properties(drjit-core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)

if (DRJIT_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
