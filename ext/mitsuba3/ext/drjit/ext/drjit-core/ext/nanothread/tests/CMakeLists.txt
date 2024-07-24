add_executable(test_01 test_01.c)
target_link_libraries(test_01 PRIVATE nanothread)

add_executable(test_02 test_02.cpp)
target_link_libraries(test_02 PRIVATE nanothread)
target_compile_features(test_02 PRIVATE cxx_std_11)

add_executable(test_03 test_03.cpp)
target_link_libraries(test_03 PRIVATE nanothread)
target_compile_features(test_03 PRIVATE cxx_std_14)

add_executable(test_04 test_04.cpp)
target_link_libraries(test_04 PRIVATE nanothread)
target_compile_features(test_04 PRIVATE cxx_std_11)
