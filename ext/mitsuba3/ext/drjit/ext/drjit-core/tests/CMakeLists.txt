enable_testing()

set(TEST_FILES basics.cpp mem.cpp graphviz.cpp vcall.cpp loop.cpp reductions.cpp)

foreach (TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp test.h test.cpp ${TEST_NAME}.cpp)
  target_link_libraries(test_${TEST_NAME} PRIVATE drjit-core)
  target_compile_definitions(test_${TEST_NAME} PRIVATE -DTEST_NAME="${TEST_NAME}")
  target_compile_features(test_${TEST_NAME} PRIVATE cxx_std_11)

  if (DRJIT_ENABLE_OPTIX)
    target_compile_definitions(test_${TEST_NAME} PRIVATE -DDRJIT_ENABLE_OPTIX=1)
  endif()
  add_test(
    NAME ${TEST_NAME}
    COMMAND test_${TEST_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endforeach()

if (NOT TARGET check)
  add_custom_target(check
          ${CMAKE_COMMAND} -E echo CWD=${CMAKE_BINARY_DIR}
          COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
          COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
          COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
              ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      USES_TERMINAL
  )
endif()

set_property(TARGET test_basics PROPERTY CXX_STANDARD 17)
set_property(TARGET test_vcall PROPERTY CXX_STANDARD 17)
set_property(TARGET test_loop PROPERTY CXX_STANDARD 17)

if (DRJIT_ENABLE_OPTIX)
 # target_sources(test_vcall PRIVATE optix_stubs.h optix_stubs.cpp)
 add_executable(triangle triangle.cpp optix_stubs.h optix_stubs.cpp)
 set_property(TARGET triangle PROPERTY CXX_STANDARD 17)
 # target_compile_definitions(test_vcall PRIVATE -DDRJIT_ENABLE_OPTIX=1)
 target_link_libraries(triangle PRIVATE drjit-core)
endif()
