cmake_minimum_required(VERSION 3.11)
project(tensor_compiler)

find_package(ONNX REQUIRED)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --------------- Common ---------------

set(COMMON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Common)
add_library(Common STATIC
    ${COMMON_LIB_DIR}/Attribute.cpp
    ${COMMON_LIB_DIR}/Dump.cpp
    ${COMMON_LIB_DIR}/Node.cpp
    ${COMMON_LIB_DIR}/Tensor.cpp
)

target_include_directories(Common PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# -------------- Frontend --------------

set(FRONTEND_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Frontend)
add_library(Frontend STATIC
    ${FRONTEND_LIB_DIR}/Parser.cpp
)

target_include_directories(Frontend PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(Frontend
    PUBLIC
    Common
    onnx::onnx
)

# -------------- Execution -------------

set(TENSOR_COMPILER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/TensorCompiler)
add_executable(tensor-compiler ${TENSOR_COMPILER_LIB_DIR}/TensorCompiler.cpp)
target_link_libraries(tensor-compiler Frontend)

# -------------- Package ---------------

if(DEFINED ENV{CONAN_PACKAGE})
    include(GNUInstallDirs)

    set_target_properties(Frontend Common PROPERTIES EXPORT_NAME TensorCompiler)

    install(TARGETS Frontend Common EXPORT TensorCompilerTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT TensorCompilerTargets
        FILE TensorCompilerConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorCompiler
    )
endif()