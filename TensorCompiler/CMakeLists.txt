cmake_minimum_required(VERSION 3.11)
project(tensor_compiler)

find_package(ONNX REQUIRED)
find_package(protobuf REQUIRED)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/Frontend)
set(FRONTEND_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Frontend)

add_library(Frontend STATIC ${FRONTEND_LIB_DIR}/Frontend.cpp)
target_include_directories(Frontend PUBLIC
    $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(Frontend
    PRIVATE
    onnx::onnx 
    protobuf::libprotobuf
)

set(TENSOR_COMPILER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/TensorCompiler)
add_executable(tensor-compiler ${TENSOR_COMPILER_LIB_DIR}/TensorCompiler.cpp)
target_link_libraries(tensor-compiler Frontend)