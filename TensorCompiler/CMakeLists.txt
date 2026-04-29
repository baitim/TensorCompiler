find_package(ONNX REQUIRED)
if(DEFINED LLVM_DIR)
    set(LLVM_DIR ${LLVM_DIR} CACHE PATH "Path to LLVM cmake" FORCE)
endif()
if(DEFINED MLIR_DIR)
    set(MLIR_DIR ${MLIR_DIR} CACHE PATH "Path to MLIR cmake" FORCE)
endif()
find_package(MLIR REQUIRED)
find_package(LLVM REQUIRED)

set(LLVM_USE_SHARED_LIBS ON CACHE BOOL "Use shared LLVM libraries")
set(MLIR_USE_SHARED_LIBS ON CACHE BOOL "Use shared MLIR libraries")

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

# -------------- Backend --------------

set(BACKEND_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Backend)
add_library(Backend STATIC
    ${BACKEND_LIB_DIR}/MlirEmitter.cpp
)
target_include_directories(Backend PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(Backend PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(Backend
    PUBLIC
    Frontend
    LLVM
    MLIR
)

# --------------- Executor -------------

set(EXECUTOR_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/Executor)
add_library(Executor STATIC
    ${EXECUTOR_LIB_DIR}/Executor.cpp
)
target_include_directories(Executor PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(Executor PUBLIC Common)

# -------------- Execution -------------

set(TENSOR_COMPILER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/TensorCompiler)
add_executable(tensor-compiler ${TENSOR_COMPILER_LIB_DIR}/TensorCompiler.cpp)
target_link_libraries(tensor-compiler Frontend Backend Executor)

# ---------------- Test ----------------

if(ENABLE_TESTS)
    add_subdirectory(./test)
endif()

# -------------- Package ---------------

if(DEFINED ENV{CONAN_PACKAGE})
    include(GNUInstallDirs)

    set_target_properties(Common Frontend Backend Executor PROPERTIES EXPORT_NAME TensorCompiler)

    install(TARGETS Common Frontend Backend Executor EXPORT TensorCompilerTargets
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