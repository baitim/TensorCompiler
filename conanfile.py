from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout

class TensorCompilerRecipe(ConanFile):
    name = "TensorCompiler"
    version = "1.0"
    
    settings = "os", "compiler", "build_type", "arch"
    
    def requirements(self):
        self.requires("onnx/1.20.0")
        self.requires("protobuf/6.32.1")
    
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
    
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()