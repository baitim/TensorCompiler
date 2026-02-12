cmake_minimum_required(VERSION 3.11)
project(TensorCompiler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_TESTS "Enable testing" ON)
if(ENABLE_TESTS)
    enable_testing()
endif()

add_subdirectory(./TensorCompiler)