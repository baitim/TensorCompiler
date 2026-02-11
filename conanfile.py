import os
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class TensorCompilerRecipe(ConanFile):
    name = "tensor_compiler"
    version = "1.0"
    user = "baitim"

    license = ""
    author = ""
    url = ""
    description = "tensor_compiler"
    topics = ""

    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    requires = "onnx/1.20.0"
    test_requires = "gtest/1.15.0"

    exports_sources = "CMakeLists.txt", "TensorCompiler/*"

    def configure(self):
        if self.settings.compiler == "msvc":
            self.settings.compiler.cppstd = "20"
        elif "gnu" in str(self.settings.compiler):
            self.settings.compiler.cppstd = "gnu20"
        else:
            self.settings.compiler.cppstd = "20"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

        cmake_user_presets = os.path.join(self.source_folder, "CMakeUserPresets.json")
        if os.path.exists(cmake_user_presets):
            os.remove(cmake_user_presets)

    def build(self):
        os.environ["CONAN_PACKAGE"] = "1"
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "config")
        self.cpp_info.set_property("cmake_file_name", "TensorCompiler")
        self.cpp_info.set_property("cmake_target_name", "TensorCompiler::TensorCompiler")
        self.cpp_info.libs = ["TensorCompiler"]
        self.cpp_info.includedirs = ["include"]