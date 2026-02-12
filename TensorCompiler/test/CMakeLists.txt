find_package(GTest REQUIRED)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(test-checker TestChecker.cpp)
target_link_libraries(test-checker
    PRIVATE
    Frontend
    GTest::GTest
)
add_dependencies(test-checker tensor-compiler)


set(RUN_TESTS ./test-checker --gtest_color=yes)
add_test(
    NAME test_checker_target
    COMMAND ${RUN_TESTS}
)