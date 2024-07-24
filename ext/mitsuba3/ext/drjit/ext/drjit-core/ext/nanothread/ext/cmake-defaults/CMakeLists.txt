# ---------------------------------------------------------------------------
# Determine the project name and prefix
# ---------------------------------------------------------------------------

if (NOT P)
  string(TOUPPER "${PROJECT_NAME}" P)
  string(REPLACE "-" "_" P "${P}")
endif()

if (NOT PN)
  set(PN "${PROJECT_DESCRIPTION}")
endif()

# ---------------------------------------------------------------------------
# Is this the master project?
# ---------------------------------------------------------------------------

set(${P}_MASTER_PROJECT OFF)
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  set(${P}_MASTER_PROJECT ON)
endif()

# ---------------------------------------------------------------------------
# Try to extract the project version from the source code
# ---------------------------------------------------------------------------

# Extract project version from source
if (${P}_VERSION_FILE)
  file(STRINGS ${${P}_VERSION_FILE} TMP REGEX "#define ${P}_VERSION_(MAJOR|MINOR|PATCH) ")

  foreach(TMP2 ${TMP})
    if(TMP2 MATCHES "#define ${P}_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(${P}_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    endif()
  endforeach()

  unset(TMP)
  unset(TMP2)

  if(${P}_VERSION_PATCH MATCHES [[\.([a-zA-Z0-9]+)$]])
    set(${P}_VERSION_TYPE "${CMAKE_MATCH_1}")
  endif()

  string(REGEX MATCH "^[0-9]+" ${P}_VERSION_PATCH "${${P}_VERSION_PATCH}")
  set(${P}_VERSION "${${P}_VERSION_MAJOR}.${${P}_VERSION_MINOR}.${${P}_VERSION_PATCH}")
  message(STATUS "${PN} v${${P}_VERSION} ${${P}_VERSION_TYPE}")
  unset(${P}_VERSION_FILE)
endif()

# ---------------------------------------------------------------------------
# Only run the rest of this file a single time
# ---------------------------------------------------------------------------

if (RGL_CMAKE_DEFAULTS)
  unset(PN)
  unset(P)
  return()
endif()

set(RGL_CMAKE_DEFAULTS TRUE)

# ---------------------------------------------------------------------------
# 32 bit architectures unsupported
# ---------------------------------------------------------------------------

if(CMAKE_SIZEOF_VOID_P STREQUAL "4")
  message(FATAL_ERROR "${PN}: project requires a 64 bit architecture!")
endif()

# ---------------------------------------------------------------------------
#  Reject old compiler versions
# ---------------------------------------------------------------------------

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU)$")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
     message(FATAL_ERROR "Your version of GCC is too old (found version ${CMAKE_CXX_COMPILER_VERSION}. Please use at least GCC 8.0)")
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Your version of Clang is too old (found version ${CMAKE_CXX_COMPILER_VERSION}. Please use at least Clang 7.0)")
  endif()
endif()

if (MSVC)
  if (${MSVC_VERSION} LESS 1924)
    message(FATAL_ERROR "MSVC 1924 or higher is required. You are running version ${MSVC_VERSION}.")
  endif()
endif()

# ---------------------------------------------------------------------------
#  Warn about Unix Makefiles
# ---------------------------------------------------------------------------

if (NOT WIN32)
  # Variables for printing colorful CMake warnings
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(BoldRed "${Esc}[1;31m")
endif()

if (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
  message(WARNING "\n${BoldRed}You're using CMake's \"Unix Makefiles\" "
    "generator. Compilation speed is going to be impacted due to its limited "
    "ability to express build parallelism. We strongly recommend using Ninja "
    "for development on Linux and macOS. To do so, install \"ninja\" (brew, "
    "MacPorts, Arch) or \"ninja-build\" elsewhere, and then invoke CMake using "
    "\"-GNinja\". You will have to delete \"CMakeCache.txt\" and the "
    "\"CMakeFiles\" directory when switching generators.) ${ColorReset}")
endif()

# ---------------------------------------------------------------------------
# Do a release build if nothing was specified
# ---------------------------------------------------------------------------

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "${PN}: setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# ---------------------------------------------------------------------------
# Compile for the native system architecture (assumes AVX2 on windows). When
# enoki is compiled as part of a PyPI package via scikit-build, adopt a lower
# target architecture (Ivy Bridge, which has the AVX and F16C extensions). An
# M1-like architecture is assumed for arm64 builds on macOS.
# ---------------------------------------------------------------------------

if (MSVC)
  if (SKBUILD)
    # Reasonably portable binaries for PyPI
    set(${P}_NATIVE_FLAGS_DEFAULT "/arch:AVX")
  else()
    set(${P}_NATIVE_FLAGS_DEFAULT "/arch:AVX2")
  endif()
elseif (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(${P}_NATIVE_FLAGS_DEFAULT "-mcpu=apple-a12")
else()
  if (SKBUILD)
    # Reasonably portable binaries for PyPI
    set(${P}_NATIVE_FLAGS_DEFAULT "-march=ivybridge")
  else()
    set(${P}_NATIVE_FLAGS_DEFAULT "-march=native")
  endif()
endif()

if (SKBUILD)
  message(STATUS "${PN}: setting portable compilation defaults for scikit-build.")
else()
  message(STATUS "${PN}: targeting the native CPU architecture (specify ${P}_NATIVE_FLAGS to change this).")
endif()

set(${P}_NATIVE_FLAGS ${${P}_NATIVE_FLAGS_DEFAULT} CACHE STRING
    "Compilation flags used to target the host processor architecture.")

add_compile_options(${${P}_NATIVE_FLAGS})
unset(${P}_NATIVE_FLAGS_DEFAULT)

# ---------------------------------------------------------------------------
# Prefer libc++ in conjunction with Clang
# ---------------------------------------------------------------------------

include(CheckCXXSourceRuns)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }"
                        ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (APPLE OR HAS_LIBCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
    message(STATUS "${PN}: using libc++.")
  else()
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP_AND_CPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
    if (HAS_LIBCPP_AND_CPPABI)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      message(STATUS "${PN}: using libc++ and libc++abi.")
    else()
      message(FATAL_ERROR "When Clang is used to compile ${PN}, libc++ "
          "must be available -- GCC's libstdc++ is not supported! (please install "
          "the libc++ development headers, provided e.g. by the packages "
          "'libc++-dev' and 'libc++abi-dev' on Debian/Ubuntu).")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Symbol visibility = hidden by default
# ---------------------------------------------------------------------------

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# ---------------------------------------------------------------------------
# Compile with a few more compiler warnings turned on
# ---------------------------------------------------------------------------

if (MSVC)
  if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    add_compile_options(/W4)
  endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-local-typedefs)
endif()

# ---------------------------------------------------------------------------
# Sane defaults for MSVC
# ---------------------------------------------------------------------------

if (MSVC)
   add_definitions(
     -D_CRT_SECURE_NO_WARNINGS
     -D_SCL_SECURE_NO_WARNINGS
     -D_CRT_NONSTDC_NO_DEPRECATE
     -D_SILENCE_CXX17_UNCAUGHT_EXCEPTION_DEPRECATION_WARNING
     -DNOMINMAX
   )
endif()

# ---------------------------------------------------------------------------
# Sane math optimization defaults for MSVC and Clang
# ---------------------------------------------------------------------------

if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_definitions(-fno-math-errno -ffp-contract=fast -fno-trapping-math)
endif()

# ---------------------------------------------------------------------------
# Force colored output for the Ninja generator
# ---------------------------------------------------------------------------

if (CMAKE_GENERATOR STREQUAL "Ninja")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()

# ---------------------------------------------------------------------------
# Enable folders for projects in Visual Studio
# ---------------------------------------------------------------------------

if (CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# ---------------------------------------------------------------------------
# Find Python if the parent project includes Python bindings
# ---------------------------------------------------------------------------

if (${P}_ENABLE_PYTHON)
  if (NOT TARGET pybind11::module)
    if (SKBUILD)
      execute_process(
        COMMAND
        "${PYTHON_EXECUTABLE}" -c
        "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
      list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
      find_package(pybind11 CONFIG REQUIRED)
      set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    else()
      set(Python_FIND_FRAMEWORK LAST) # Prefer Brew/Conda to Apple framework python
      find_package(Python COMPONENTS Interpreter Development REQUIRED)

      # Can't have conflicting versions of pybind11...
      include(CheckIncludeFileCXX)
      set(CMAKE_REQUIRED_INCLUDES ${Python_INCLUDE_DIRS})
      if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set(CMAKE_REQUIRED_FLAGS "-std=c++11")
      endif()
      check_include_file_cxx("pybind11/pybind11.h" ${P}_PYBIND11_FOUND)
      if (${P}_PYBIND11_FOUND)
        unset(${P}_PYBIND11_FOUND CACHE)
        message(FATAL_ERROR "An existing version of pybind11 was found on "
                "the include path, which is going to conflict with the "
                "${PN}-provided one. Please uninstall it first (via brew, "
                "pip uninstall, your system package manager, or similar)")
      endif()

      unset(${P}_PYBIND11_FOUND)
      unset(CMAKE_REQUIRED_FLAGS)
      unset(CMAKE_REQUIRED_INCLUDES)

      set(${P}_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11"
        CACHE STRING "Path containing the 'pybind11' library used to compile ${PN}.")
      add_subdirectory(${${P}_PYBIND11_DIR} pybind11)
      mark_as_advanced(
        USE_PYTHON_INCLUDE_DIR PYBIND11_INSTALL PYBIND11_TEST PYBIND11_NOPYTHON
        PYBIND11_FINDPYTHON PYBIND11_PYTHON_VERSION PYBIND11_WERROR
        ${P}_PYBIND11_DIR)
      endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Clear temporary variables
# ---------------------------------------------------------------------------

unset(PN)
unset(P)
