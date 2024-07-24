cmake_minimum_required (VERSION 3.13..3.18)
project(libjpeg C)

option(LIBJPEG_BUILD_SHARED "Build libjpeg as a shared library?" ON)
option(LIBJPEG_BUILD_EXECUTABLES "Build libjpeg executables?" ON)

include(CheckIncludeFile)
check_include_file(stddef.h HAVE_STDDEF_H)
check_include_file(stdlib.h HAVE_STDLIB_H)
if(WIN32 AND NOT CYGWIN)
  set(TWO_FILE_COMMANDLINE true)
endif()
configure_file(jconfig.h.cmake jconfig.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(HEADERS jerror.h jmorecfg.h jpeglib.h ${CMAKE_CURRENT_BINARY_DIR}/jconfig.h)

set(SRC jmemnobs.c jaricom.c jcapimin.c jcapistd.c jcarith.c jccoefct.c jccolor.c
  jcdctmgr.c jchuff.c jcinit.c jcmainct.c jcmarker.c jcmaster.c jcomapi.c jcparam.c
  jcprepct.c jcsample.c jctrans.c jdapimin.c jdapistd.c jdarith.c jdatadst.c jdatasrc.c
  jdcoefct.c jdcolor.c jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c jdmaster.c
  jdmerge.c jdpostct.c jdsample.c jdtrans.c jerror.c jfdctflt.c jfdctfst.c jfdctint.c
  jidctflt.c jidctfst.c jidctint.c jquant1.c jquant2.c jutils.c jmemmgr.c cderror.h
  cdjpeg.h jdct.h jinclude.h jmemsys.h jpegint.h jversion.h transupp.h jpegdll.def)

if (MSVC)
  add_definitions("/D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4267")
endif()

# Set library type
if(LIBJPEG_BUILD_SHARED)
  set(LIBJPEG_LIBRARY_TYPE "SHARED")
  add_definitions(-DLIBJPEG_SHARED)
else()
  set(LIBJPEG_LIBRARY_TYPE "STATIC")
endif()

add_library(jpeg ${LIBJPEG_LIBRARY_TYPE} ${SRC} ${HEADERS})
set_target_properties(jpeg PROPERTIES COMPILE_DEFINITIONS "LIBJPEG_BUILD")

if(LIBJPEG_BUILD_EXECUTABLES)
  add_executable(cjpeg cdjpeg.c cjpeg.c rdbmp.c rdgif.c rdppm.c rdrle.c rdtarga.c rdswitch.c)
  add_executable(djpeg cdjpeg.c djpeg.c wrbmp.c wrgif.c wrppm.c wrrle.c wrtarga.c rdcolmap.c)
  add_executable(jpegtran jpegtran.c cdjpeg.c rdswitch.c transupp.c)
  add_executable(rdjpgcom rdjpgcom.c)
  add_executable(wrjpgcom wrjpgcom.c)
  target_link_libraries(cjpeg jpeg)
  target_link_libraries(djpeg jpeg)
  target_link_libraries(jpegtran jpeg)
endif()
