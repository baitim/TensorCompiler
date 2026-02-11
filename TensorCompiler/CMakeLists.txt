cmake_minimum_required(VERSION 3.11)
project(tensor_compiler)

find_package(ONNX REQUIRED)
find_package(protobuf REQUIRED)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/Frontend)
set(FRONTEND_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Frontend)

add_library(Frontend STATIC 
    ${FRONTEND_LIB_DIR}/Common.cpp
    ${FRONTEND_LIB_DIR}/Parser.cpp
    ${FRONTEND_LIB_DIR}/Dump.cpp
)

target_include_directories(Frontend PUBLIC
    $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(Frontend
    PUBLIC
    onnx::onnx 
    protobuf::libprotobuf
)

set(TENSOR_COMPILER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/TensorCompiler)
add_executable(tensor-compiler ${TENSOR_COMPILER_LIB_DIR}/TensorCompiler.cpp)
target_link_libraries(tensor-compiler Frontend)

if(DEFINED ENV{CONAN_PACKAGE})
    include(GNUInstallDirs)

    set_target_properties(Frontend PROPERTIES EXPORT_NAME TensorCompiler)

    install(TARGETS Frontend EXPORT TensorCompilerTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Frontend)

    install(EXPORT TensorCompilerTargets
        FILE TensorCompilerConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorCompiler
    )
endif()