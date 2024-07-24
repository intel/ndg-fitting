# ----------------------------------------------------------
# Top-level nanothread CMake file, needs a recent version
# ----------------------------------------------------------
cmake_minimum_required(VERSION 3.13...3.18)

project(nanothread
  DESCRIPTION
    "nanothread"
  LANGUAGES
    CXX C
)

# ----------------------------------------------------------
#  Optional features available to users
# ----------------------------------------------------------

option(NANOTHREAD_ENABLE_TESTS "Build test suite?" OFF)

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ext/cmake-defaults/CMakeLists.txt")
  message(FATAL_ERROR "The nanothread dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

# ----------------------------------------------------------
#  Build defaults for projects by the Realistic Graphics Lab
# ----------------------------------------------------------

include(ext/cmake-defaults/CMakeLists.txt)

# ----------------------------------------------------------
#  Compile the nanothread library
# ----------------------------------------------------------

add_library(
  nanothread SHARED
  include/nanothread/nanothread.h
  src/queue.cpp src/queue.h
  src/nanothread.cpp
)

target_compile_features(nanothread PRIVATE cxx_std_11)
target_include_directories(nanothread PRIVATE include)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(nanothread PRIVATE -mcx16)
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # GCC needs libatomic for 16 byte CSA
  find_library(LIBATOMIC NAMES libatomic.so libatomic.so.1)
  if (NOT LIBATOMIC)
    message(FATAL_ERROR "libatomic could not be found!")
  endif()
  target_link_libraries(nanothread PRIVATE ${LIBATOMIC})
  mark_as_advanced(LIBATOMIC)
endif()

target_include_directories(nanothread
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(nanothread PRIVATE -DNANOTHREAD_BUILD=1)
set_target_properties(nanothread PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

if (NANOTHREAD_ENABLE_TESTS)
   add_subdirectory(tests)
endif()
